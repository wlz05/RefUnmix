# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from contextlib import contextmanager
from contextvars import ContextVar
import math
import torch

active = ContextVar("refunmixreference", default=(0., 0., 2))
observed = ContextVar("refunmixobserved", default=None)

@contextmanager
def route(keys=0., values=0., layout=2):
    # Keep reference context local to one shared model dispatch.
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (keys, values)):
        raise ValueError("Reference strengths must be finite and in [0, 1].")
    if layout not in (2, 3):
        raise ValueError("Reference layouts must contain two or three endpoints.")
    token = active.set((keys, values, layout))
    counts = [0, 0]
    countertoken = observed.set(counts)
    try:
        yield counts
    finally:
        observed.reset(countertoken)
        active.reset(token)

def hook(index):
    def apply(module, inputs, output):
        strength = active.get()[index]
        if strength == 0:
            return output
        layout = active.get()[2]
        if not isinstance(output, torch.Tensor) or output.ndim != 3 or output.shape[0] % layout:
            raise ValueError("Reference contexts require complete endpoint groups.")
        counts = observed.get()
        if counts is not None:
            counts[index] += 1
        result = output.clone()
        source, target = (0, 1) if layout == 2 else (1, 2)
        result[target::layout] = output[target::layout].lerp(output[source::layout], strength)
        return result
    return apply

def install(model):
    # Reuse source attention coordinates while retaining target queries.
    if hasattr(model, "referencehandles"):
        return len(model.referencehandles)
    handles = []
    for name, module in model.named_modules():
        if name.endswith("attn1") and not getattr(module, "is_cross_attention", False):
            for index, label in enumerate(("to_k", "to_v")):
                layer = getattr(module, label, None)
                if layer is not None:
                    handles.append(layer.register_forward_hook(hook(index)))
    if not handles:
        raise ValueError("The backbone exposes no compatible self-attention projections.")
    model.referencehandles = handles
    return len(handles)

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
