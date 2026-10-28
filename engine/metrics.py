# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
from contextlib import contextmanager
import hashlib
import importlib.util
import math
import os
from pathlib import Path, PureWindowsPath
import shutil
import socket
import sys
import tempfile
import numpy as np
from PIL import Image
from data import clean, destination, locate

def relocate(hub):
    # Replace stale absolute snapshot links with local blob copies.
    hub = Path(hub)
    repaired = 0
    for snapshots in hub.glob("*--*/snapshots"):
        repository = snapshots.parent
        if not snapshots.resolve().is_relative_to(hub.resolve()):
            raise ValueError("Metric snapshot directories must stay inside their cache.")
        for link in snapshots.rglob("*"):
            if not link.is_symlink():
                continue
            target = os.readlink(link)
            absolute = Path(target).is_absolute() or bool(PureWindowsPath(target).drive)
            if not absolute:
                if not link.exists() or not link.resolve().is_relative_to(repository.resolve()):
                    raise FileNotFoundError("A metric snapshot link is incomplete; copy its repository blobs.")
                continue
            token = str(target).replace("\\", "/").rsplit("/", 1)[-1]
            blob = repository / "blobs" / token
            if not token or not blob.is_file() or not blob.resolve().is_relative_to(repository.resolve()) or not link.parent.resolve().is_relative_to(snapshots.resolve()):
                raise FileNotFoundError("A relocated metric snapshot needs its matching local blob.")
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=link.parent, suffix=".tmp", delete=False) as output, blob.open("rb") as source:
                    temporary = Path(output.name)
                    shutil.copyfileobj(source, output)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, link)
                repaired += 1
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    return repaired

@contextmanager
def connection(offline):
    if not offline:
        yield
        return
    original, extended = socket.socket.connect, socket.socket.connect_ex
    lookup = socket.getaddrinfo

    def blocked(*args, **kwargs):
        raise RuntimeError("Offline metric assets are incomplete; populate the cache with --network first.")

    socket.socket.connect = socket.socket.connect_ex = blocked
    socket.getaddrinfo = blocked
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex = original, extended
        socket.getaddrinfo = lookup

def storage(value, offline=False):
    # Keep downloaded metric assets inside the working directory.
    if not isinstance(offline, bool):
        raise ValueError("Offline cache mode must be boolean.")
    root = destination(value, "cache")
    parts = {"HF_HOME": "huggingface", "HF_HUB_CACHE": "huggingface/hub", "HF_ASSETS_CACHE": "huggingface/assets", "HF_XET_CACHE": "huggingface/xet", "TORCH_HOME": "torch", "XDG_CACHE_HOME": "."}
    paths = {key: locate(root, Path(part)).as_posix() for key, part in parts.items()}
    hub = locate(root, Path("torch/hub")).as_posix()
    legacy = {"HUGGINGFACE_HUB_CACHE": paths["HF_HUB_CACHE"], "HUGGINGFACE_ASSETS_CACHE": paths["HF_ASSETS_CACHE"], "TRANSFORMERS_CACHE": paths["HF_HUB_CACHE"], "PYTORCH_TRANSFORMERS_CACHE": paths["HF_HUB_CACHE"], "PYTORCH_PRETRAINED_BERT_CACHE": paths["HF_HUB_CACHE"]}
    import torch
    root.mkdir(parents=True, exist_ok=True)
    relocate(Path(paths["HF_HUB_CACHE"]))
    for key in legacy:
        os.environ.pop(key, None)
    os.environ.update(paths)
    os.environ.update({"HF_HUB_OFFLINE": "1" if offline else "0", "TRANSFORMERS_OFFLINE": "1" if offline else "0"})
    # Redirect cache constants when a caller already imported the libraries.
    for name in ("huggingface_hub.constants", "transformers.utils.hub", "transformers.file_utils"):
        module = sys.modules.get(name)
        if module is not None:
            for key, path in {**paths, **legacy, "default_cache_path": paths["HF_HUB_CACHE"], "hf_cache_home": paths["HF_HOME"], "HF_HUB_OFFLINE": offline, "_is_offline_mode": offline}.items():
                if hasattr(module, key):
                    setattr(module, key, path)
    torch.hub.set_dir(hub)
    return {"root": root.as_posix(), "huggingface": paths["HF_HUB_CACHE"], "torch": hub}

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
        self.calculator, self.metadata = calculator, metadata if metadata is not None else {"backend": "external" if calculator is not None else "pixel-only", "complete": calculator is not None}

    @classmethod
    def load(cls, path=None, device="cpu", network=False, cache="./cache/metrics", offline=False):
        clean(cache, "cache")
        if not isinstance(offline, bool) or not isinstance(network, bool) or offline and network:
            raise ValueError("Choose exactly one external metric mode: --network or --offline.")
        if path is None:
            if network or offline:
                raise ValueError("Pixel diagnostics do not use --network or --offline.")
            return cls()
        source = locate(Path("."), clean(path, "protocol"))
        if source.suffix != ".py" or not source.is_file():
            raise FileNotFoundError("Protocol must point to the official Python metric implementation.")
        if not network and not offline:
            raise ValueError("The external backend requires --network or --offline; pixel-only mode needs neither.")
        cached = storage(cache, offline)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        spec = importlib.util.spec_from_file_location("benchmarkmetrics", source)
        module = importlib.util.module_from_spec(spec)
        with connection(offline):
            spec.loader.exec_module(module)
            if not callable(getattr(module, "MetricsCalculator", None)):
                raise ValueError("Metric module must expose a MetricsCalculator constructor.")
            calculator = module.MetricsCalculator(device)
        names = ("calculate_psnr", "calculate_mse", "calculate_lpips", "calculate_ssim", "calculate_structure_distance", "calculate_clip_similarity")
        if any(not callable(getattr(calculator, name, None)) for name in names):
            raise ValueError("Metric backend does not implement the required public interface.")
        return cls(calculator, {"backend": str(source), "sha256": digest, "complete": True, "networkauthorized": network, "offline": offline, "device": str(device), "cache": cached})

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
                with torch.no_grad(), connection(self.metadata.get("offline", False)):
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
                if (measured == 0) != (raw["mse"] == 0) or not math.isclose(measured, raw["mse"], rel_tol=1e-4, abs_tol=1e-12):
                    raise RuntimeError("External metric masking differs from the expected pixel protocol.")
                for name in ("lpips", "ssim"):
                    raw[name] = call("calculate_" + name, origin, edited, protected, protected)
                if measured > 0:
                    raw["psnr"] = call("calculate_psnr", origin, edited, protected, protected)
                    if not math.isclose(raw["psnr"], -10 * math.log10(measured), rel_tol=1e-5, abs_tol=1e-4):
                        raise RuntimeError("External PSNR differs from the expected pixel protocol.")
        factors = {"mse": 1000, "lpips": 1000, "ssim": 100, "structure": 1000}
        for name, value in raw.items():
            if name not in status:
                status[name] = "finite" if value is not None else "missing backend or required region"
        displayed = {name: None if value is None else value * factors.get(name, 1) for name, value in raw.items()}
        return {"raw": raw, "table": displayed, "status": status, "backend": self.metadata}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
