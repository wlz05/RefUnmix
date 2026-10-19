# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import argparse
import json
from pathlib import Path
from PIL import Image
from data import clean, locate
from engine.feedback import options, settings

def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number

def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run historical field-routing controls; use main.py for RefUnmix.")
    parser.add_argument("--model", default="./models")
    parser.add_argument("--image", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", default="./output.png")
    parser.add_argument("--method", type=str.casefold, choices=("thread", "base", "frequency", "feedback", "response"), default="thread")
    parser.add_argument("--region", default=None)
    parser.add_argument("--polarity", type=str.casefold, choices=("edit", "protect"), default="edit")
    parser.add_argument("--report", default=None)
    parser.add_argument("--precision", type=str.casefold, choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--device", default=None)
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
    options(parser)
    return parser.parse_args()

def main() -> None:
    args = arguments()
    from engine.model import EditConfig, Editor
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
    judge = None
    if args.method == "feedback":
        if args.clip is None:
            raise ValueError("Feedback mode requires a local --clip folder.")
        from engine.clip import ClipJudge
        judge = ClipJudge.load(args.clip, args.device or "cpu")
    image = locate(Path("."), clean(args.image, "image"))
    if not image.is_file():
        raise FileNotFoundError(image)
    from engine.format import read
    from data import destination
    from run import writejson
    source = read(image)
    output = destination(args.output, "output")
    if output.resolve() == image.resolve():
        raise ValueError("output must differ from the input image.")
    if output.suffix != ".png":
        raise ValueError("Use a lossless .png output.")
    if output.exists():
        raise FileExistsError(output)
    region = None
    if args.region is not None:
        path = locate(Path("."), clean(args.region, "region"))
        if path.resolve() == output.resolve():
            raise ValueError("output must differ from the region mask.")
        region = read(path, "L")
    report = destination(args.report, "report") if args.report is not None else None
    if report is not None and report.exists():
        raise FileExistsError(report)
    protectedpaths = {image.resolve(), output.resolve()}
    if args.region is not None:
        protectedpaths.add(path.resolve())
    if report is not None and report.resolve() in protectedpaths:
        raise ValueError("report must differ from image paths.")
    editor = Editor.load(args.model, args.device, args.precision)
    result = editor.edit(source, args.source, args.target, config, args.seed, region=region, polarity=args.polarity, judge=judge, feedback=feedback, localtext=args.localtext)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(output)
    print(output)
    if args.method == "frequency":
        for key in ("lowtime", "hightime", "lowweight", "highweight", "proxyhhpre", "proxyhlpre", "proxyllpre", "proxyhh", "proxyhl", "proxyll", "protected"):
            print(f"{key}: {float(result.maps[key].float().mean()):.6f}")
    if result.report is not None:
        print(json.dumps({key: value for key, value in result.report.items() if key != "history"}, allow_nan=False))
        if report is not None:
            writejson(report, result.report)

if __name__ == "__main__":
    import sys
    method = next((value.split("=", 1)[1].casefold() if value.startswith("--method=") else sys.argv[index + 1].casefold() for index, value in enumerate(sys.argv) if value.startswith("--method=") or value == "--method" and index + 1 < len(sys.argv)), None)
    legacy = "--legacy" in sys.argv or method in {"frequency", "feedback"}
    if "--legacy" in sys.argv:
        sys.argv.remove("--legacy")
    if legacy:
        main()
    else:
        if method == "response":
            position = next(index for index, value in enumerate(sys.argv) if value.startswith("--method=") or value == "--method")
            if "=" in sys.argv[position]:
                sys.argv[position] = "--method=thread"
            else:
                sys.argv[position + 1] = "thread"
        from run import main as responsive
        responsive()

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
