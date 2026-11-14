# -*- coding: utf-8 -*-
# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

from contextlib import nullcontext
import importlib.util
import math
from pathlib import Path
import sys
import torch
from engine.reproduce import digest

DEFAULTS = {"sd3": {"T_steps": 50, "n_avg": 1, "src_guidance_scale": 3.5, "tar_guidance_scale": 13.5, "n_min": 0, "n_max": 33}, "flux": {"T_steps": 28, "n_avg": 1, "src_guidance_scale": 1.5, "tar_guidance_scale": 5.5, "n_min": 0, "n_max": 24}}

def settings(family, changes=None):
    if family not in DEFAULTS or changes is not None and not isinstance(changes, dict):
        raise ValueError("FlowEdit requires the SD3 or FLUX configuration.")
    result = dict(DEFAULTS[family])
    if changes:
        if set(changes) - result.keys():
            raise ValueError("Unknown FlowEdit setting.")
        result.update(changes)
    for key in ("T_steps", "n_avg", "n_min", "n_max"):
        if type(result[key]) is not int:
            raise ValueError("FlowEdit step counts must be integers.")
    if not 1 <= result["T_steps"] <= 1000 or not 1 <= result["n_avg"] <= 64 or not 0 <= result["n_min"] < result["n_max"] <= result["T_steps"]:
        raise ValueError("Invalid FlowEdit time window or averaging count.")
    for key in ("src_guidance_scale", "tar_guidance_scale"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("FlowEdit guidance scales must be finite and nonnegative.")
        if family == "sd3" and value <= 1:
            raise ValueError("The released SD3 implementation requires classifier-free guidance.")
    if family == "sd3" and result["n_min"]:
        raise ValueError("The pinned SD3 continuation branch has an undefined upstream variable; use its released n_min=0 setting.")
    return result

def implementation(source, family):
    settings(family)
    path = Path(source) / "FlowEdit_utils.py"
    checksum = digest(path)
    name = "officialflowedit_" + checksum
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("The official FlowEdit module cannot be loaded.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        function = getattr(module, "FlowEditSD3" if family == "sd3" else "FlowEditFLUX", None)
        if not callable(function):
            raise ValueError("The released FlowEdit editing function is missing.")
        return function, {"file": path.as_posix(), "sha256": checksum}
    except BaseException:
        sys.modules.pop(name, None)
        raise

class FlowEdit:
    def __init__(self, pipeline, function, family, changes=None):
        self.pipeline, self.function = pipeline, function
        self.parameters = settings(family, changes)
        self.calls = self.samples = 0
        def meter(module, args, kwargs):
            states = kwargs.get("hidden_states", args[0] if args else None)
            if not torch.is_tensor(states) or not torch.isfinite(states).all():
                raise FloatingPointError("FlowEdit issued invalid transformer states.")
            self.calls += 1
            self.samples += states.shape[0]
        pipeline.transformer.register_forward_pre_hook(meter, with_kwargs=True)

    @torch.no_grad()
    def __call__(self, item, image, seed, local=""):
        if image.size != (512, 512):
            raise ValueError("PIE-Bench FlowEdit inputs must be 512 by 512.")
        self.calls = self.samples = 0
        pipe = self.pipeline
        device = torch.device(pipe._execution_device)
        pixels = pipe.image_processor.preprocess(image).to(device=device, dtype=pipe.vae.dtype)
        context = torch.autocast("cuda") if device.type == "cuda" else nullcontext()
        with context:
            latent = pipe.vae.encode(pixels).latent_dist.mode()
            scaling, shift = pipe.vae.config.scaling_factor, pipe.vae.config.shift_factor
            if not math.isfinite(scaling) or scaling <= 0 or not math.isfinite(shift):
                raise ValueError("The FlowEdit VAE scale and shift are invalid.")
            latent = (latent - shift) * scaling
            edited = self.function(pipe, pipe.scheduler, latent, item.source, item.target, "", **self.parameters)
            if not torch.is_tensor(edited) or edited.shape != latent.shape or not torch.isfinite(edited).all():
                raise FloatingPointError("FlowEdit returned invalid edited latents.")
            decoded = pipe.vae.decode(edited / scaling + shift, return_dict=False)[0]
        if not torch.isfinite(decoded).all() or self.calls < 1:
            raise FloatingPointError("FlowEdit decoding or query accounting failed.")
        result = pipe.image_processor.postprocess(decoded)
        if len(result) != 1:
            raise ValueError("FlowEdit must return one result per benchmark request.")
        return result[0], {"parameters": self.parameters, "calls": self.calls, "samples": self.samples, "negative_prompt": ""}

def load(family, source, model, device, offload=False):
    settings(family)
    function, metadata = implementation(source, family)
    from diffusers import FluxPipeline, StableDiffusion3Pipeline
    factory = StableDiffusion3Pipeline if family == "sd3" else FluxPipeline
    dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
    pipeline = factory.from_pretrained(str(model), torch_dtype=dtype, local_files_only=True)
    if offload:
        if torch.device(device).type != "cuda":
            raise ValueError("Model CPU offload requires a CUDA execution device.")
        pipeline.enable_model_cpu_offload(gpu_id=torch.device(device).index or 0)
    else:
        pipeline.to(device)
    result = FlowEdit(pipeline, function, family)
    result.metadata = {**metadata, "family": family, "dtype": str(dtype), "offload": offload}
    return result

# Author: Weiliang Zhu; Other main author: Yeming Dai, Yaming Lv
# Email: wlzchina05@gmail.com
