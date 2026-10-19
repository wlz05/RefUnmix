# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
from importlib import import_module, metadata
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

def parser():
    result = argparse.ArgumentParser(description="Check local reproduction inputs without loading model weights.", epilog="Paths resolve from the working directory; run outside the Python source tree. Use python -B to avoid bytecode caches.")
    result.add_argument("--dataset", default=None)
    result.add_argument("--mapping", default="./mapping_file.json")
    result.add_argument("--images", default="./annotation_images")
    result.add_argument("--manifest", default=None)
    result.add_argument("--split", type=str.casefold, choices=("all", "train", "val", "test"), default="all")
    result.add_argument("--model", default=None)
    result.add_argument("--clip", default=None)
    result.add_argument("--backbone", type=str.casefold, choices=("vp", "flux", "sd3"), default="vp")
    result.add_argument("--device", type=str.casefold, default="cpu")
    result.add_argument("--precision", type=str.casefold, choices=("fp32", "fp16", "bf16"), default="fp32")
    result.add_argument("--limit", type=int, default=None)
    result.add_argument("--report", default=None)
    return result

def dependencies(model=False, clip=False):
    names = {"numpy": "numpy", "Pillow": "PIL", "torch": "torch", "scipy": "scipy"}
    if model:
        names.update({"diffusers": "diffusers", "transformers": "transformers", "safetensors": "safetensors"})
    elif clip:
        names.update({"transformers": "transformers", "safetensors": "safetensors"})
    result, errors = {}, []
    for package, module in names.items():
        try:
            version = metadata.version(package)
            import_module(module)
            result[package] = {"version": version, "import": True}
        except Exception as error:
            result[package] = {"version": None, "import": False}
            errors.append(f"Dependency {package}: {type(error).__name__}: {error}")
    return result, errors

def weights(root, family):
    from data import clean, locate
    path = locate(Path("."), clean(root, "model"))
    if not path.is_dir():
        raise NotADirectoryError(path)
    names = ("unet", "scheduler", "vae", "tokenizer", "text") if family == "vp" else ("transformer", "scheduler", "vae", "tokenizer", "text_encoder", "text_encoder_2")
    # Check checkpoint components before loading any weights.
    parts = {name: locate(path, Path(name)) for name in names}
    if family == "vp" and not parts["text"].is_dir():
        parts["text"] = locate(path, Path("text_encoder"))
    missing = [name for name, folder in parts.items() if not folder.is_dir()]
    if missing:
        raise ValueError(f"Missing model folders: {missing}")
    learned = {"unet", "transformer", "vae", "text", "text_encoder", "text_encoder_2"}
    empty = [name for name, folder in parts.items() if name in learned and not any(file.is_file() and file.suffix.casefold() in {".bin", ".safetensors", ".pt"} for file in folder.rglob("*"))]
    if empty:
        raise ValueError(f"Model folders contain no local weights: {empty}")
    return {"path": path.as_posix(), "family": family, "parts": {name: folder.as_posix() for name, folder in parts.items()}, "scope": "Directory and file presence only; tensor compatibility is checked by the native loader."}

def check(args):
    result = {"schema": 1, "settings": vars(args).copy(), "errors": [], "warnings": [], "scope": "Offline preflight only; no model inference or quality claim."}
    if sys.version_info < (3, 11):
        result["errors"].append("Python 3.11 or later is required.")
        result["ok"] = False
        return result
    result["dependencies"], errors = dependencies(bool(args.model), bool(args.clip))
    result["errors"].extend(errors)
    if errors or sys.version_info < (3, 11):
        result["ok"] = False
        return result
    from data import clean, locate, records
    from dataset import annotation
    from engine.format import orient, read
    from engine.reproduce import digest, snapshot
    from split import select
    import torch
    result["reproducibility"] = snapshot()
    if Path.cwd().resolve().is_relative_to(Path(__file__).resolve().parent):
        result["warnings"].append("Run from a parent workspace; runtime artifacts are forbidden inside the code tree.")
    if args.limit is not None and args.limit < 1:
        result["errors"].append("The inspection limit must be positive.")
    try:
        device = torch.device(args.device)
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("Only CPU and CUDA are supported.")
        if device.type == "cuda" and (not torch.cuda.is_available() or device.index is not None and device.index >= torch.cuda.device_count()):
            raise ValueError("The selected CUDA device is unavailable.")
        if device.type == "cpu" and args.precision != "fp32":
            raise ValueError("CPU reproduction requires fp32.")
        if device.type == "cuda" and args.precision == "bf16":
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("The selected CUDA device does not support bf16.")
    except (ValueError, RuntimeError) as error:
        result["errors"].append(str(error))
    for key in ("model", "clip"):
        value = getattr(args, key)
        if value is None:
            continue
        try:
            if key == "model":
                result[key] = weights(value, args.backbone)
            else:
                root = locate(Path("."), clean(value, key))
                if not root.is_dir() or not any(file.is_file() and file.suffix.casefold() in {".bin", ".safetensors"} for file in root.rglob("*")):
                    raise ValueError("CLIP requires a local checkpoint directory with weights.")
                result[key] = {"path": root.as_posix(), "scope": "Presence check only."}
        except (ValueError, OSError) as error:
            result["errors"].append(f"{key}: {error}")
    if args.dataset is None and (args.manifest is not None or args.split != "all"):
        result["errors"].append("Dataset selection requires --dataset.")
    if args.dataset is not None and (args.limit is None or args.limit > 0):
        try:
            selected = select(records(args.dataset, args.mapping, args.images), args.manifest, args.split)
            if not selected:
                raise ValueError("The selected dataset is empty.")
            chosen = selected if args.limit is None else selected[:args.limit]
            report = {"total": len(selected), "inspected": len(chosen), "complete": len(chosen) == len(selected), "pairs": 0, "masks": 0, "oriented": 0, "sources": []}
            for item in chosen:
                try:
                    image = read(item.image)
                    region = annotation(item)
                    if region is not None:
                        if item.maskpath is not None and region.size != image.size:
                            raise ValueError("Raster mask geometry differs from raw source geometry.")
                        from PIL import Image
                        region = region.resize(image.size, Image.Resampling.NEAREST)
                        report["masks"] += 1
                    report["oriented"] += int(image.getexif().get(274, 1) != 1)
                    fitted, aligned = orient(image, region)
                    if item.edited is not None:
                        orient(read(item.edited))
                        report["pairs"] += 1
                    report["sources"].append({"key": item.key, "sha256": digest(item.image), "size": list(fitted.size)})
                except (ValueError, OSError) as error:
                    result["errors"].append(f"Record {item.key}: {error}")
            result["dataset"] = report
            if not report["complete"]:
                result["warnings"].append("Only a limited subset was inspected; full-data validity is not certified.")
        except (ValueError, OSError) as error:
            result["errors"].append(f"Dataset: {error}")
    result["ok"] = not result["errors"]
    return result

def main():
    args = parser().parse_args()
    result = check(args)
    if args.report is not None:
        from pathlib import PureWindowsPath
        from engine.serial import write
        value = args.report.strip().replace("\\", "/")
        path = Path(value)
        if not value or path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts:
            raise ValueError("Report paths must be local and relative.")
        if not path.resolve().is_relative_to(Path.cwd().resolve()) or path.resolve().is_relative_to(Path(__file__).resolve().parent):
            raise ValueError("Reports must stay in the workspace outside the code tree.")
        write(path, result)
    print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))
    if not result["ok"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
