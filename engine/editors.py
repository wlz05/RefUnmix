# -*- coding: utf-8 -*-
# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

import ast
import inspect
from pathlib import Path
import runpy
import sys
import numpy as np
import torch
from engine.bridge import activate

def settings(method, source):
    entry = Path(source) / ("src/pipeline/run_pipeline.py" if method == "rrls" else "run_pie_bench.py")
    for node in ast.parse(entry.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "DEFAULT_EDIT_CONFIG" for target in node.targets):
            result = ast.literal_eval(node.value)
            if not isinstance(result, dict):
                break
            return result
    raise ValueError("The released default edit configuration is missing.")

class Official:
    def __init__(self, method, backbone, source, model, clip, device="cuda:0"):
        if method not in {"chord", "rrls", "where"} or backbone not in {"sdturbo", "swiftbrush", "instaflow"}:
            raise ValueError("Unsupported native method-backbone pair.")
        source, model = Path(source), Path(model)
        sys.path.insert(0, str(source))
        entry = source / ("src/pipeline/run_pipeline.py" if method == "rrls" else "run_pie_bench.py")
        released = runpy.run_path(str(entry))
        factory = released["WhereEditPipeline" if method == "where" else "ChordEditPipeline"]
        self.parameters = settings(method, source)
        paths = {part + "_path": str(model / part) for part in ("unet", "vae", "scheduler", "tokenizer", "text_encoder")}
        options = {"default_edit_config": self.parameters, "device": device, "torch_dtype": torch.float32, "compute_dtype": torch.float32, "image_size": 512, "use_safety_checker": False}
        accepted = inspect.signature(factory.from_local_weights).parameters
        self.pipeline = factory.from_local_weights(paths, **{k: v for k, v in options.items() if k in accepted})
        self.native = activate(self.pipeline) if backbone == "instaflow" else {"nativeoutput": "epsilon", "clock": "released VP schedule"}
        self.selector = self.scorer = None
        if method == "rrls":
            self.selector = runpy.run_path(str(source / "src/pipeline/residual_selector_clip.py"))
            self.scorer = self.selector["ClipScorer"](str(clip), torch.device(device))
        self.queries = []
        def meter(module, args, kwargs):
            sample = kwargs.get("sample", args[0] if args else None)
            if sample is None or not torch.isfinite(sample).all():
                raise FloatingPointError("Invalid native model query.")
            self.queries.append({"samples": sample.shape[0]})
        def finite(module, args, output):
            sample = output.sample if hasattr(output, "sample") else output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(sample) or not torch.isfinite(sample).all():
                raise FloatingPointError("The native model returned nonfinite predictions.")
        self.pipeline.unet.register_forward_pre_hook(meter, with_kwargs=True)
        self.pipeline.unet.register_forward_hook(finite)
        self.pipeline.vae.decoder.register_forward_hook(finite)

    @torch.no_grad()
    def __call__(self, item, image, seed, local=""):
        self.queries.clear()
        base = self.pipeline(image=image, source_prompt=item.source, target_prompt=item.target, seed=seed, output_type="pil").images[0]
        result, choice = base, None
        if self.selector is not None:
            strong = self.pipeline(image=image, source_prompt=item.source, target_prompt=item.target, seed=seed, edit_config=dict(self.parameters, step_scale=1.5), output_type="pil").images[0]
            names, images, costs = self.selector["candidate_images"](image, base, strong, [.55, .65, .75, .85])
            scores = self.scorer.scores(images, item.target)
            objective = np.asarray([score - 40 * cost for score, cost in zip(scores, costs)], dtype=np.float32)
            if not np.isfinite(objective).all():
                raise FloatingPointError("The RRLS selector objective is nonfinite.")
            winner = int(np.argmax(objective))
            result, choice = images[winner], names[winner]
        if not self.queries:
            raise RuntimeError("The official editor issued no native queries.")
        return result, {"parameters": self.parameters, "native": self.native, "choice": choice, "calls": len(self.queries), "samples": sum(q["samples"] for q in self.queries)}

# Author: Weiliang Zhu; Other main author: Yeming Dai, Yaming Lv
# Email: wlzchina05@gmail.com
