# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import hashlib
import importlib.util
import math
from pathlib import Path
import numpy as np
from PIL import Image
from data import clean, locate

def inputs(origin, edited, mask):
    if not isinstance(origin, Image.Image) or not isinstance(edited, Image.Image) or origin.size != edited.size:
        raise ValueError("Metric inputs must be equal-size PIL images.")
    first, second = np.asarray(origin.convert("RGB"), dtype=np.float32) / 255, np.asarray(edited.convert("RGB"), dtype=np.float32) / 255
    if mask is None:
        return first, second, None
    if not isinstance(mask, Image.Image) or mask.size != origin.size:
        raise ValueError("Evaluation masks must match the source image.")
    values = np.asarray(mask.convert("L"), dtype=np.uint8)
    if not np.all((values == 0) | (values == 255)):
        raise ValueError("Official metric masks must be binary.")
    return first, second, (values == 255).astype(np.float32)[..., None].repeat(3, axis=2)

def pixels(origin, edited, mask):
    # Match the full-canvas denominator after masking both images.
    first, second, area = inputs(origin, edited, mask)
    result = {"psnr": None, "mse": None}
    status = "missing mask" if area is None else "empty protected region"
    if area is not None and np.any(area < 1):
        loss = float(np.mean(((1 - area) * (second - first)) ** 2))
        result = {"mse": loss, "psnr": -10 * math.log10(loss) if loss > 0 else None}
        status = "finite" if loss > 0 else "positive infinity"
    return result, {"psnr": status, "mse": "finite" if result["mse"] is not None else status}

class Protocol:
    def __init__(self, calculator=None, metadata=None):
        self.calculator, self.metadata = calculator, metadata or {"backend": "pixel-only", "complete": False}

    @classmethod
    def load(cls, path=None, device="cpu", network=False):
        if path is None:
            return cls()
        source = locate(Path("."), clean(path, "protocol"))
        if source.suffix != ".py" or not source.is_file():
            raise FileNotFoundError("Protocol must point to the official Python metric implementation.")
        if not network:
            raise ValueError("The external backend may download pretrained assets; pass --network explicitly or use pixel-only mode.")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        spec = importlib.util.spec_from_file_location("benchmarkmetrics", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        calculator = module.MetricsCalculator(device)
        names = ("calculate_psnr", "calculate_mse", "calculate_lpips", "calculate_ssim", "calculate_structure_distance", "calculate_clip_similarity")
        if any(not callable(getattr(calculator, name, None)) for name in names):
            raise ValueError("Metric backend does not implement the required public interface.")
        return cls(calculator, {"backend": str(source), "sha256": digest, "complete": True, "networkauthorized": True, "device": str(device)})

    def evaluate(self, origin, edited, mask, target):
        import torch
        if not isinstance(target, str) or not target.strip():
            raise ValueError("Target text must not be empty.")
        _, _, area = inputs(origin, edited, mask)
        raw, status = pixels(origin, edited, mask)
        raw.update({"lpips": None, "ssim": None, "structure": None, "whole": None, "edited": None})
        if self.calculator is not None:
            backend = self.calculator

            def call(name, *args):
                for attribute in vars(backend).values():
                    if callable(getattr(attribute, "reset", None)):
                        attribute.reset()
                with torch.no_grad():
                    value = float(getattr(backend, name)(*args))
                if not math.isfinite(value):
                    raise RuntimeError("A defined official metric returned a nonfinite value: " + name)
                return value

            raw["structure"] = call("calculate_structure_distance", origin, edited)
            raw["whole"] = call("calculate_clip_similarity", edited, target)
            if area is not None and np.any(area):
                raw["edited"] = call("calculate_clip_similarity", edited, target, area)
            if area is not None and np.any(area < 1):
                protected = 1 - area
                measured = call("calculate_mse", origin, edited, protected, protected)
                if not math.isclose(measured, raw["mse"], rel_tol=1e-4, abs_tol=1e-8):
                    raise RuntimeError("External metric masking differs from the expected pixel protocol.")
                for name in ("lpips", "ssim"):
                    raw[name] = call("calculate_" + name, origin, edited, protected, protected)
                if measured > 0:
                    raw["psnr"] = call("calculate_psnr", origin, edited, protected, protected)
        factors = {"mse": 1000, "lpips": 1000, "ssim": 100, "structure": 1000}
        for name, value in raw.items():
            if name not in status:
                status[name] = "finite" if value is not None else "missing backend or required region"
        displayed = {name: None if value is None else value * factors.get(name, 1) for name, value in raw.items()}
        return {"raw": raw, "table": displayed, "status": status, "backend": self.metadata}

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
