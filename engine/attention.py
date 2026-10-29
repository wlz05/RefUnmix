# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
from difflib import SequenceMatcher
import math
import torch
import torch.nn.functional as functional

def tokens(tokenizer, source, target, length, words=None):
    # Include neighboring tokens for phrase matching.
    encoded = tokenizer([source, target], padding="max_length", truncation=True, max_length=length, return_tensors="pt").input_ids
    special = set(getattr(tokenizer, "all_special_ids", []))
    positions = [[index for index, value in enumerate(row) if int(value) not in special] for row in encoded]
    values = [[int(encoded[side, index]) for index in positions[side]] for side in range(2)]
    selected = [set(), set()]
    if words:
        requested = tokenizer([words], add_special_tokens=False, return_tensors="pt").input_ids.flatten().tolist()
        allowed = set(requested) - special
        for side in range(2):
            selected[side].update(index for index in positions[side] if int(encoded[side, index]) in allowed)
    else:
        for tag, left, endleft, right, endright in SequenceMatcher(a=values[0], b=values[1], autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            for side, start, stop in ((0, left, endleft), (1, right, endright)):
                if stop > start:
                    selected[side].update(positions[side][max(0, start - 1):min(len(values[side]), stop + 1)])
    return [sorted(side) for side in selected]

class Collector:
    def __init__(self, model, indices, shape, count=1, maximum=1024):
        if len(indices) != 2 or any(any(type(index) is not int or index < 0 for index in side) for side in indices):
            raise ValueError("Attention indices must contain source and target token positions.")
        if type(count) is not int or count < 1 or type(maximum) is not int or maximum < 1:
            raise ValueError("Attention sample and query limits must be positive integers.")
        self.model, self.indices, self.shape = model, indices, tuple(shape)
        self.count, self.maximum = count, maximum
        self.handles, self.maps, self.skipped = [], [], {}

    def skip(self, reason):
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def __enter__(self):
        self.maps, self.skipped = [], {}
        try:
            for name, module in self.model.named_modules():
                if name.endswith("attn2") and all(hasattr(module, key) for key in ("to_q", "to_k", "heads")):
                    self.handles.append(module.register_forward_pre_hook(self.capture, with_kwargs=True))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, kind, value, trace):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    @torch.no_grad()
    def capture(self, module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        context = kwargs.get("encoder_hidden_states", args[1] if len(args) > 1 else None)
        masked = kwargs.get("attention_mask", args[2] if len(args) > 2 else None)
        unsupported = any(getattr(module, name, None) is not None for name in ("spatial_norm", "norm_q", "norm_k", "add_k_proj"))
        if hidden is None or context is None or hidden.ndim != 3 or context.ndim != 3 or masked is not None or unsupported:
            self.skip("unsupported attention layout")
            return
        batch, queries, _ = hidden.shape
        if batch != 2 * self.count or queries > self.maximum or not any(self.indices):
            self.skip("batch, query limit, or empty tokens")
            return
        height = round(math.sqrt(queries * self.shape[0] / self.shape[1]))
        width = queries // max(height, 1)
        if height < 1 or height * width != queries or height * self.shape[1] != width * self.shape[0]:
            self.skip("ambiguous spatial geometry")
            return
        if getattr(module, "group_norm", None) is not None:
            hidden = module.group_norm(hidden.transpose(1, 2)).transpose(1, 2)
        if getattr(module, "norm_cross", None):
            context = module.norm_encoder_hidden_states(context)
        query, key = module.to_q(hidden), module.to_k(context)
        heads = int(module.heads)
        if heads < 1 or query.shape[-1] != key.shape[-1] or query.shape[-1] % heads:
            self.skip("unsupported head dimensions")
            return
        query = query.reshape(batch, queries, heads, -1).transpose(1, 2).float()
        key = key.reshape(batch, context.shape[1], heads, -1).transpose(1, 2).float()
        scores = (query @ key.transpose(-1, -2)) * float(getattr(module, "scale", query.shape[-1] ** -.5))
        probabilities = scores.softmax(-1).mean(1)
        maps = []
        for side, indices in enumerate(self.indices):
            valid = [index for index in indices if index < context.shape[1]]
            if valid:
                values = probabilities[side * self.count:(side + 1) * self.count, :, valid].mean(dim=(0, 2))
                maps.append(values.reshape(1, 1, height, width))
        if maps:
            value = torch.stack(maps).amax(0)
            if not torch.isfinite(value).all().item():
                raise RuntimeError("Attention localization produced nonfinite values.")
            self.maps.append(functional.interpolate(value, self.shape, mode="bilinear", align_corners=False).detach().cpu())

    def result(self):
        return torch.stack(self.maps).mean(0) if self.maps else None

    def report(self):
        return {"tokens": self.indices, "maps": len(self.maps), "skipped": self.skipped, "maximumqueries": self.maximum, "note": "Read-only attention reconstruction with token-neighbor phrase heuristics."}

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
