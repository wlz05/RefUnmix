# -*- coding: utf-8 -*-
# RefUnmix: Separating Prompt Response from Shared Transport in One-Step Image Editing

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time
import numpy as np
from PIL import Image
import torch
import conduct
from data import records
from engine.bench import guard
from engine.native import load
from engine.serial import write
from engine.session import toimage
from evaluate import datasetargs
from run import common, run, validate

def selection(items):
    if len(items) != 700 or len({item.key for item in items}) != 700:
        raise ValueError("Timing requires the full unique 700-request population.")
    groups = {}
    for item in items:
        groups.setdefault(item.category, []).append(item)
    if set(groups) != set("0123456789") or any(len(group) < 5 for group in groups.values()):
        raise ValueError("Timing requires five requests from each of ten categories.")
    return [item for category in sorted(groups) for item in sorted(groups[category], key=lambda item: hashlib.sha256(("runtime:" + item.key).encode()).hexdigest())[:5]]

def exclusive():
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    if any(line.strip().isdigit() and int(line.strip()) != os.getpid() for line in output.splitlines()):
        raise RuntimeError("Timing requires an otherwise idle GPU.")

def verify(rows, selected, seed):
    if len(rows) != 50 or len({row["key"] for row in rows}) != 50:
        raise ValueError("Timing requires 50 distinct completed requests.")
    if [row["key"] for row in rows] != [item.key for item in selected]:
        raise ValueError("Timing request identities or order changed.")
    if Counter(item.category for item in selected) != {key: 5 for key in "0123456789"}:
        raise ValueError("Timing categories are not balanced.")
    if any(row["seed"] != seed or type(row["seconds"]) not in (int, float) or not math.isfinite(row["seconds"]) or row["seconds"] <= 0 for row in rows):
        raise ValueError("Timing has an invalid seed or duration.")
    return statistics.fmean(row["seconds"] for row in rows)

def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure complete RefUnmix latency on one exclusive GPU.")
    parser.add_argument("--model", type=conduct.relative, default=Path("models"))
    parser.add_argument("--backbone", choices=("sdturbo", "swiftbrush", "instaflow"), default="sdturbo")
    parser.add_argument("--dataset", type=conduct.relative, default=Path("data"))
    parser.add_argument("--mapping", type=conduct.relative, default=Path("mapping_file.json"))
    parser.add_argument("--images", type=conduct.relative, default=Path("annotation_images"))
    parser.add_argument("--output", type=conduct.relative, default=Path("outputs/timing"))
    parser.add_argument("--cache", type=conduct.relative, default=Path("cache"))
    parser.add_argument("--seed", type=int, choices=conduct.SEEDS, default=42)
    args = parser.parse_args(argv)
    target = args.output / args.backbone / (str(args.seed) + ".json")
    if target.exists():
        raise FileExistsError("A completed timing result is immutable.")
    preset = conduct.parser().parse_args(["--backbone", args.backbone, "--model", str(args.model), "--dataset", str(args.dataset), "--mapping", str(args.mapping), "--images", str(args.images), "--batch", "1", "--seeds", str(args.seed)])
    command = conduct.generation(preset, args.seed, args.output / "unused")
    options = datasetargs(common(argparse.ArgumentParser())).parse_args(command[3:])
    validate(options)
    selected = selection(records(str(args.dataset), str(args.mapping), str(args.images)))
    torch.set_num_threads(1)
    with guard(args.cache / "inference.lock"), guard(args.cache / "timing.lock"):
        exclusive()
        editor = load(options.model, options.backbone, options.device, options.precision, options.size)
        rows = []
        tasks = [(True, item) for item in selected[:5]] + [(False, item) for item in selected]
        for warmup, item in tasks:
            exclusive()
            torch.cuda.synchronize()
            started = time.perf_counter()
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            with Image.open(item.image) as source:
                origin = source.convert("RGB").resize((512, 512))
            with torch.inference_mode():
                pixels, report, session = run(editor, None, origin, item.source, item.target, options)
                result = toimage(pixels).convert("RGB")
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if result.size != (512, 512) or tuple(report["stats"][key] for key in ("samples", "calls", "decodes", "encodes")) != (5, 2, 3, 1):
                raise RuntimeError("Timed execution differs from the complete main method.")
            if not warmup:
                rows.append({"key": item.key, "seed": args.seed, "seconds": elapsed})
            del result, pixels, report, session, origin
        exclusive()
        mean = verify(rows, selected, args.seed)
        value = {"method": "refunmix", "backbone": args.backbone, "seed": args.seed, "hardware": torch.cuda.get_device_name(0), "isolated": True, "protocol": {"unit": "seconds/image", "batch": 1, "warmup": 5, "requestsperseed": 50, "scope": "Input loading, complete generation and ready RGB output; excludes weights, output writes and report evaluation."}, "argv": command, "source": conduct.source(), "rows": rows, "mean": mean}
        if target.exists():
            raise FileExistsError("A completed timing result is immutable.")
        target.parent.mkdir(parents=True, exist_ok=True)
        write(target, value)
        print(json.dumps({"path": str(target), "seconds": mean}, allow_nan=False))

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
