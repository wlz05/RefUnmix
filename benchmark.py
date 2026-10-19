# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from data import clean, records
from dataset import annotation
from split import select
from engine.feedback import options, settings

def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number

def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RefUnmix benchmark export.")
    parser.add_argument("--model", default="./models")
    parser.add_argument("--dataset", default="./pie")
    parser.add_argument("--mapping", default="./mapping_file.json")
    parser.add_argument("--images", default="./annotation_images")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", type=str.casefold, choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--output", default="./results")
    parser.add_argument("--precision", type=str.casefold, choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--method", type=str.casefold, choices=("thread", "base", "frequency", "feedback"), default="thread")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise", type=positive, default=1)
    parser.add_argument("--start", type=float, default=0.90)
    parser.add_argument("--end", type=float, default=0.30)
    parser.add_argument("--delta", type=float, default=0.15)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--mode", type=str.casefold, choices=("surprisal", "relative", "invariant", "magnitude"), default="surprisal")
    parser.add_argument("--support", type=float, default=0.50)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--floor", type=float, default=0.10)
    parser.add_argument("--cap", type=float, default=1.50)
    parser.add_argument("--preserve", type=float, default=0.50)
    parser.add_argument("--couple", type=float, default=0.25)
    parser.add_argument("--prox", type=float, default=0.50)
    parser.add_argument("--relay", type=float, default=1.00)
    parser.add_argument("--reach", type=int, default=8)
    parser.add_argument("--band", type=int, default=3)
    parser.add_argument("--feather", type=int, default=2)
    parser.add_argument("--pivot", type=float, default=None)
    parser.add_argument("--signal", type=float, default=None)
    parser.add_argument("--rough", type=float, default=None)
    parser.add_argument("--path", type=float, default=1.0)
    parser.add_argument("--grid", type=float, default=0.0)
    parser.add_argument("--route", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--edge", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scalar", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cleanup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=positive, default=None)
    parser.add_argument("--maps", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    options(parser)
    return parser.parse_args()

def seam(image: Image.Image, mask: torch.Tensor) -> float:
    # Measure boundary gradients relative to the image interior.
    pixels = torch.from_numpy(np.asarray(image, dtype="float32") / 255.0).permute(2, 0, 1).mean(dim=0)
    latent = mask.float()
    grown = functional.max_pool2d(latent, 3, 1, 1)
    shrunk = -functional.max_pool2d(-latent, 3, 1, 1)
    ring = functional.interpolate(((grown - shrunk) > 0.5).float(), size=pixels.shape, mode="nearest")[0, 0] > 0.5
    vertical = (pixels[1:, :] - pixels[:-1, :]).abs()
    horizontal = (pixels[:, 1:] - pixels[:, :-1]).abs()
    magnitude = torch.zeros_like(pixels)
    magnitude[:-1, :] += vertical
    magnitude[:, :-1] += horizontal
    if not bool(ring.sum()) or not bool((~ring).sum()):
        return float("nan")
    return float(magnitude[ring].mean() / (magnitude[~ring].mean() + 1e-6))

def table(path, header: str, rows: list[str], overwrite: bool) -> None:
    if not rows:
        return
    content = "\n".join(rows) + "\n"
    if path.exists() and not overwrite:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(content)
    else:
        path.write_text(header + content, encoding="utf-8")

def main() -> None:
    args = arguments()
    from engine.model import EditConfig, Editor
    items = select(records(args.dataset, args.mapping, args.images), args.manifest, args.split)
    if args.limit is not None:
        items = items[:args.limit]
    config = EditConfig(
        method=args.method,
        noise=args.noise,
        start=args.start,
        end=args.end,
        delta=args.delta,
        scale=args.scale,
        cleanup=args.cleanup,
        mode=args.mode,
        support=args.support,
        gain=args.gain,
        floor=args.floor,
        cap=args.cap,
        preserve=args.preserve,
        couple=args.couple,
        prox=args.prox,
        relay=args.relay,
        pivot=args.pivot,
        signal=args.signal,
        rough=args.rough,
        path=args.path,
        grid=args.grid,
        reach=args.reach,
        band=args.band,
        feather=args.feather,
        route=args.route,
        edge=args.edge,
        scalar=args.scalar,
    )
    config.validate()
    feedback = settings(args)
    if args.oracle and args.method not in {"frequency", "feedback"}:
        raise ValueError("Oracle masks require frequency or feedback mode.")
    if args.oracle and any(item.mask is None for item in items):
        raise ValueError("Oracle mode requires an edit mask for every selected record.")
    judge = None
    if args.method == "feedback":
        if args.clip is None:
            raise ValueError("Feedback mode requires a local --clip folder.")
        from engine.clip import ClipJudge
        judge = ClipJudge.load(args.clip, args.device or "cpu")
    root = clean(args.output, "output")
    jobs = []
    for index, item in enumerate(items, start=1):
        target = root / "images" / item.relative
        if target.resolve() == item.image.resolve():
            raise ValueError("output must not overwrite a dataset image.")
        if target.exists() and not args.overwrite:
            continue
        jobs.append((index, item, target))
    if not jobs:
        print("No pending records.")
        return
    editor = Editor.load(args.model, args.device, args.precision)
    seams: list[tuple[str, float]] = []
    metrics: list[list[object]] = []
    frequencyrows: list[list[object]] = []
    for index, item, target in jobs:
        from engine.format import read
        image = read(item.image)
        region = annotation(item) if args.oracle else None
        if region is not None and region.size != image.size:
            region = region.resize(image.size, Image.Resampling.NEAREST)
        if editor.device.type == "cuda":
            torch.cuda.synchronize(editor.device)
        started = time.perf_counter()
        result = editor.edit(image, item.source, item.target, config, args.seed, region=region, judge=judge, feedback=feedback, localtext=args.localtext)
        if editor.device.type == "cuda":
            torch.cuda.synchronize(editor.device)
        elapsed = time.perf_counter() - started
        target.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(target)
        report = result.report or {}
        report.update({"image": item.key, "category": item.category, "split": args.split, "oracle": args.oracle, "seconds": elapsed})
        destination = (root / "reports" / item.relative).with_suffix(".json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        if args.maps:
            mapping = (root / "maps" / item.relative).with_suffix(".pt")
            mapping.parent.mkdir(parents=True, exist_ok=True)
            torch.save(result.maps, mapping)
        seamvalue = seam(result.images[0], result.maps["mask"]) if config.cleanup and config.method == "thread" else float("nan")
        if seamvalue == seamvalue:
            seams.append((item.relative.as_posix(), seamvalue))
        basenorm = float(result.maps["base"].float().norm())
        fieldnorm = float(result.maps["field"].float().norm())
        gate = float(result.maps["gate"].float().mean()) if "gate" in result.maps else float("nan")
        trust = float(result.maps["trust"].float().mean()) if "trust" in result.maps else float("nan")
        mask = float(result.maps["mask"].float().mean()) if "mask" in result.maps else float("nan")
        relay = float((result.maps["relay"] > 0.0).float().mean()) if "relay" in result.maps else float("nan")
        metrics.append([item.relative.as_posix(), basenorm, fieldnorm, fieldnorm / (basenorm + 1e-8), gate, trust, mask, relay, seamvalue])
        if config.method == "frequency":
            frequencyrows.append([item.relative.as_posix()] + [float(result.maps[key].float().mean()) for key in ("lowtime", "hightime", "lowweight", "highweight", "proxyhhpre", "proxyhlpre", "proxyllpre", "proxyhh", "proxyhl", "proxyll", "protected")])
        print(f"{index}/{len(items)} {target}")
    usable = [value for _, value in seams if value == value]
    if usable:
        table(root / "seam.csv", "image,seam\n", [f"{name},{value}" for name, value in seams], args.overwrite)
        print(f"Mean seam {sum(usable) / len(usable):.4f} over {len(usable)} images.")
    table(root / "metrics.csv", "image,basenorm,fieldnorm,normratio,gate,trust,mask,relay,seam\n", [",".join(str(value) for value in row) for row in metrics], args.overwrite)
    if frequencyrows:
        table(root / "frequency.csv", "image,lowtime,hightime,lowweight,highweight,proxyhhpre,proxyhlpre,proxyllpre,proxyhh,proxyhl,proxyll,protected\n", [",".join(str(value) for value in row) for row in frequencyrows], args.overwrite)

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
