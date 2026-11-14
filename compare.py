# -*- coding: utf-8 -*-
# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

import argparse
import json
from pathlib import Path
import subprocess
import sys
from engine.comparison import BACKBONES, METHODS, Registry

def main():
    parser = argparse.ArgumentParser(description="Plan the method-by-backbone matrix, or execute its official adapters with --run.")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS[:-1]))
    parser.add_argument("--backbones", nargs="+", choices=BACKBONES, default=list(BACKBONES))
    parser.add_argument("--adapters", nargs="*", default=[])
    parser.add_argument("--turbo", default="models")
    parser.add_argument("--brush", default="models/brush")
    parser.add_argument("--insta", default="models/insta")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, choices=(42, 52, 3047, 3407), default=[42])
    parser.add_argument("--sources", default="baselines")
    parser.add_argument("--weights", default="models/baselines")
    parser.add_argument("--dataset", default="data")
    parser.add_argument("--mapping", default="mapping_file.json")
    parser.add_argument("--images", default="annotation_images")
    parser.add_argument("--protocol", default="protocol/metrics.py")
    parser.add_argument("--cache", default="cache/metrics")
    parser.add_argument("--clip", default="clip")
    parser.add_argument("--output", default="outputs/matrix")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python")
    parser.add_argument("--network", action="store_true")
    args = parser.parse_args()
    registry = Registry()
    for path in args.adapters:
        registry.load(path)
    plan = registry.plan(args.methods, args.backbones, {"sdturbo": args.turbo, "swiftbrush": args.brush, "instaflow": args.insta})
    plan["seeds"] = args.seeds
    if args.run:
        execute(args, plan)
        return
    print(json.dumps(plan, indent=2, allow_nan=False))

def execute(args, plan):
    from engine.bench import SEEDS, compatible, guard, idle, main as baseline, meanpath, options, parser
    from engine.serial import read, write
    from engine.sources import relative
    if args.adapters:
        raise ValueError("Custom adapters execute through Registry.apply; --run uses the pinned official method adapters.")
    if args.seeds != [seed for seed in SEEDS if seed in args.seeds] or not args.seeds:
        raise ValueError("Select distinct seeds in the declared benchmark order.")
    output = relative(args.output, "matrix output")
    roots = {"sdturbo": args.turbo, "swiftbrush": args.brush, "instaflow": args.insta}
    rows = []
    for row in plan["rows"]:
        method, backbone = row["method"], row["backbone"]
        if method in {"native", "naive"}:
            root = output / method / backbone
            command = [str(relative(args.python, "interpreter")) if args.python else sys.executable, "-B", str(Path(__file__).with_name("conduct.py")), "--profile", method, "--backbone", backbone, "--model", str(relative(roots[backbone], "model")), "--batch", "1", "--seeds", *map(str, args.seeds), "--output", str(root)]
            for name in ("dataset", "mapping", "images", "protocol", "cache", "device"):
                command.extend(("--" + name, str(getattr(args, name))))
            if args.network:
                command.append("--network")
            lockargs = options(parser().parse_args(["--method", "refunmix", "--backbone", backbone, "--model", roots[backbone], "--output", str(output), "--device", args.device, "--batch", "1"]))
            with guard(lockargs.cache.parent / "inference.lock"):
                idle(lockargs)
                subprocess.run(command, check=True)
            from aggregate import combine
            value = combine([root / method / str(seed) / "metrics.json" for seed in args.seeds], args.seeds, args.dataset, args.mapping, args.images)
            rows.append({"method": method, "backbone": backbone, "state": "complete", "output": str(root / method), "table": value["table"]})
        elif not compatible(method, backbone):
            rows.append({"method": method, "backbone": backbone, "state": "unsupported released weights", "metrics": None})
        else:
            command = ["--method", method, "--backbone", backbone, "--action", "run", "--model", roots[backbone], "--inverse-model", args.turbo, "--seeds", *map(str, args.seeds)]
            for name in ("sources", "weights", "dataset", "mapping", "images", "protocol", "cache", "clip", "output", "device", "python"):
                value = getattr(args, name)
                if value is not None:
                    command.extend(("--" + name, str(value)))
            if args.network:
                command.append("--network")
            baseline(command)
            selected = options(parser().parse_args(command))
            metrics = meanpath(selected)
            value = read(metrics)
            rows.append({"method": method, "backbone": backbone, "state": "complete", "metrics": str(metrics), "table": value["table"]})
    target = output / ("matrix" + "-".join(map(str, args.seeds)) + ".json")
    value = {"seeds": args.seeds, "requestsperseed": 700, "rows": rows}
    if target.exists():
        if read(target) != value:
            raise ValueError("The saved matrix has different input reports.")
    else:
        write(target, value)
    print(json.dumps(value, indent=2, allow_nan=False))

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yeming Dai, Yaming Lv
# Email: wlzchina05@gmail.com
