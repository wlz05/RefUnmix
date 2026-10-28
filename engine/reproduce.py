# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import hashlib
from functools import lru_cache
from importlib import metadata
import os
from pathlib import Path
import platform
import random
import sys
from engine.format import POLICY

def worker(index):
    import numpy as np
    import torch
    value = torch.initial_seed() % (2 ** 32)
    random.seed(value)
    np.random.seed(value)

def seed(value, deterministic=False):
    if type(value) is not int or not -(2 ** 63) <= value < 2 ** 64 or not isinstance(deterministic, bool):
        raise ValueError("Seed and deterministic controls are invalid.")
    if deterministic:
        configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if configured is not None and configured not in {":4096:8", ":16:8"}:
            raise ValueError("Deterministic CUDA requires a supported CUBLAS workspace setting.")
        loaded = sys.modules.get("torch")
        if configured is None and loaded is not None and loaded.cuda.is_initialized():
            raise ValueError("Configure deterministic CUDA before initialization and restart the process.")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    random.seed(value)
    np.random.seed(value % (2 ** 32))
    torch.manual_seed(value)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            result.update(chunk)
    return result.hexdigest()

@lru_cache(maxsize=1)
def versions():
    result = {}
    for package in ("torch", "torchvision", "torchmetrics", "lpips", "numpy", "Pillow", "scipy", "diffusers", "transformers", "huggingface_hub", "safetensors", "accelerate", "datasets", "pyarrow", "sentencepiece"):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result

@lru_cache(maxsize=1)
def source():
    root = Path(__file__).resolve().parent.parent
    result = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        result.update(path.relative_to(root).as_posix().encode("utf-8"))
        result.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return result.hexdigest()

def snapshot():
    # Record actual runtime controls without promising cross-device equality.
    result = {"python": platform.python_version(), "platform": platform.platform(), "packages": dict(versions()), "codesha256": source(), "format": dict(POLICY), "cublas": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "pythonhashseed": os.environ.get("PYTHONHASHSEED"), "scope": "Same code, inputs, weights and runtime are required; cross-hardware bitwise equality is not guaranteed."}
    torch = sys.modules.get("torch")
    if torch is not None:
        result["torch"] = {"cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "deterministic": torch.are_deterministic_algorithms_enabled(), "warnonly": torch.is_deterministic_algorithms_warn_only_enabled(), "benchmark": torch.backends.cudnn.benchmark, "matmultf32": torch.backends.cuda.matmul.allow_tf32, "cudnntf32": torch.backends.cudnn.allow_tf32, "threads": torch.get_num_threads(), "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())] if torch.cuda.is_available() else []}
    return result

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
