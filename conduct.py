# -*- coding: utf-8 -*-
# RefUnmix: Separating Prompt Response from Shared Transport in One-Step Image Editing

import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import subprocess
import sys

SEEDS = (42, 52, 3047, 3407)
LEGACY = ("calibration", "bias", "locality", "condition")
PROFILES = ("main", "assignment", "reference", "amplification", "gating", "target", "grounding", "attraction", "native", "calibration", "bias", "locality", "condition", "naive")

def relative(value):
    path = Path(value)
    if path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts:
        raise argparse.ArgumentTypeError("Use a local relative path.")
    if not path.resolve().is_relative_to(Path.cwd().resolve()):
        raise argparse.ArgumentTypeError("The path escapes the project directory.")
    return path

def parser():
    result = argparse.ArgumentParser(description="Run the fixed RefUnmix benchmark and paired controls.")
    result.add_argument("--profile", choices=PROFILES, default="main")
    result.add_argument("--action", choices=("run", "generate", "score", "aggregate"), default="run")
    result.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=[42])
    result.add_argument("--dataset", type=relative, default=Path("data"))
    result.add_argument("--mapping", type=relative, default=Path("mapping_file.json"))
    result.add_argument("--images", type=relative, default=Path("annotation_images"))
    result.add_argument("--model", type=relative, default=Path("models"))
    result.add_argument("--backbone", choices=("vp", "sdturbo", "swiftbrush", "instaflow"), default="vp")
    result.add_argument("--cache", type=relative, default=Path("cache/metrics"))
    result.add_argument("--protocol", type=relative, default=Path("protocol/metrics.py"))
    result.add_argument("--output", type=relative, default=Path("outputs"))
    result.add_argument("--batch", type=int, choices=(1, 2, 3, 4), default=1)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--network", action="store_true")
    return result

def entry(name):
    # Resolve script locations without changing the data working directory.
    return Path(__file__).resolve().with_name(name).relative_to(Path.cwd().resolve()).as_posix()

def generation(args, seed, target):
    if args.backbone == "instaflow" and (args.batch != 1 or args.profile in {"calibration", "bias", "locality"}):
        raise ValueError("Use InstaFlow with --batch 1 and calibration disabled.")
    if args.profile == "native":
        return native(args, seed, target)
    if args.profile == "naive":
        return naive(args, seed, target)
    command = [sys.executable, "-B", "evaluate.py", "--dataset", (args.dataset).as_posix(), "--mapping", (args.mapping).as_posix(), "--images", (args.images).as_posix(), "--split", "all", "--model", (args.model).as_posix(), "--backbone", "vp", "--device", args.device, "--precision", "fp32", "--size", "512", "--seed", str(seed), "--seeds", str(seed), "--batch", str(args.batch), "--output", (target).as_posix(), "--single", "--singlemode", "contrast", "--singlecondition", "extrapolated", "--conditiongain", "1.25", "--singlegain", "1.5", "--innovationgain", "2", "--singleend", "responsibility", "--start", ".75", "--delta", "0", "--end", ".30", "--noise", "1", "--scale", "1", "--cleanup", "--recenter", "0", "--no-singlebias", "--biasgeometry", "local", "--biaswindow", "5", "--biasridge", ".1", "--responsepower", ".5", "--responsepenalty", "1", "--method", "raw", "--field", "chord", "--poststeps", "1", "--guidance", "1", "--no-cfg", "--prior", "off", "--no-control", "--no-compensation", "--no-frontier", "--no-heldout", "--maxsamples", "64", "--maxcalls", "64"]
    if args.profile not in LEGACY:
        command[command.index("--singlegain") + 1] = "1"
        command[command.index("--innovationgain") + 1] = "3"
        command.extend(("--referencelocal", "--targetpull", ".15", "--retainedcontext", ".10"))
    changes = {"calibration": ("--recenter", ".5"), "locality": ("--biasgeometry", "global"), "amplification": ("--innovationgain", "1"), "condition": ("--conditiongain", "1")}
    if args.profile in changes:
        flag, value = changes[args.profile]
        command[command.index(flag) + 1] = value
    if args.profile == "calibration":
        command[command.index("--singlemode") + 1] = "recenter"
    if args.profile in {"calibration", "bias", "locality"}:
        command[command.index("--no-singlebias")] = "--singlebias"
    if args.profile in {"assignment", "reference", "gating", "target", "grounding", "attraction"}:
        command.extend(("--ablation", args.profile))
    command[command.index("--backbone") + 1] = args.backbone
    command[2] = entry("evaluate.py")
    return command

def native(args, seed, target):
    # Select the direct predictor with all editing operations disabled.
    command = naive(args, seed, target)
    command[command.index("--naive")] = "--native"
    command[command.index("--start") + 1] = ".30"
    command.extend(("--targetpull", "0", "--retainedcontext", "1"))
    return command

def naive(args, seed, target):
    # Match ChordEdit controls while disabling its temporal smoothing.
    return [sys.executable, "-B", entry("evaluate.py"), "--dataset", args.dataset.as_posix(), "--mapping", args.mapping.as_posix(), "--images", args.images.as_posix(), "--split", "all", "--model", args.model.as_posix(), "--backbone", args.backbone, "--device", args.device, "--precision", "fp32", "--size", "512", "--seed", str(seed), "--seeds", str(seed), "--batch", str(args.batch), "--output", target.as_posix(), "--naive", "--no-single", "--method", "raw", "--field", "chord", "--start", ".90", "--delta", "0", "--end", ".30", "--scale", "1", "--noise", "1", "--cleanup", "--poststeps", "1", "--guidance", "1", "--no-cfg", "--prior", "off", "--no-control", "--no-heldout", "--no-compensation", "--no-frontier", "--singlemode", "contrast", "--singlecondition", "raw", "--conditiongain", "1", "--singlegain", "1", "--innovationgain", "1", "--recenter", "0", "--no-singlebias", "--maxsamples", "64", "--maxcalls", "64"]

def scoring(args, seed, target):
    return [sys.executable, "-B", entry("assess.py"), "--input", (target / "summary.json").as_posix(), "--dataset", (args.dataset).as_posix(), "--mapping", (args.mapping).as_posix(), "--images", (args.images).as_posix(), "--scope", "benchmark", "--seeds", str(seed), "--protocol", (args.protocol).as_posix(), "--device", args.device, "--network" if args.network else "--offline", "--cache", (args.cache).as_posix(), "--output", (target / "metrics.json").as_posix()]

def complete(path, seed):
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("rows", [])
    if len(rows) != 700 or len({row.get("key") for row in rows}) != 700:
        raise ValueError("A completed seed must contain 700 distinct requests.")
    if any(row.get("seed") != seed for row in rows):
        raise ValueError("A result contains an unexpected seed.")
    return document

def source():
    # Keep the same runtime source throughout the seed sweep.
    root = Path(__file__).resolve().parent
    files = list(root.glob('*.py')) + list((root / 'engine').glob('*.py'))
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for path in sorted(files)}

def matching(path, command, revision):
    saved = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict) or saved.get("argv", [])[1:] != command[1:] or saved.get("source") != revision:
        raise ValueError("The saved run uses different code or settings; choose a fresh output directory.")

def main():
    args = parser().parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Each seed may appear only once.")
    if args.seeds != [seed for seed in SEEDS if seed in args.seeds]:
        raise ValueError("Run seeds in the order 42, 52, 3047, 3407.")
    from data import records
    items = records((args.dataset).as_posix(), (args.mapping).as_posix(), (args.images).as_posix())
    if len(items) != 700 or {item.category for item in items} != set("0123456789"):
        raise ValueError("Use the complete official 700-request PIE benchmark.")
    root = args.output / args.profile
    root.mkdir(parents=True, exist_ok=True)
    revision = source()
    for seed in args.seeds:
        target = root / str(seed)
        recipe = root / (str(seed) + ".json")
        command = generation(args, seed, target)
        if recipe.exists():
            matching(recipe, command, revision)
        if args.action in {"run", "generate"}:
            if (target / "summary.json").exists():
                complete(target / "summary.json", seed)
            else:
                if target.exists() and any(target.iterdir()):
                    raise FileExistsError("An unfinished run exists; preserve it and choose a fresh output directory.")
                recipe.write_text(json.dumps({"argv": command, "source": revision}, indent=2) + "\n", encoding="utf-8")
                subprocess.run(command, check=True)
                complete(target / "summary.json", seed)
        if args.action in {"run", "score"}:
            if not recipe.exists():
                raise FileNotFoundError("The generation recipe is required before scoring.")
            complete(target / "summary.json", seed)
            if not (target / "metrics.json").exists():
                subprocess.run(scoring(args, seed, target), check=True)
            complete(target / "metrics.json", seed)
    if args.action in {"run", "aggregate"}:
        paths = [root / str(seed) / "metrics.json" for seed in args.seeds]
        if not all(path.is_file() for path in paths):
            if args.action == "aggregate":
                raise FileNotFoundError("Every requested seed report is required.")
            print("Per-seed results saved; requested seed reports are incomplete.")
            return
        # Recompute each requested seed before equal-weight aggregation.
        for seed in args.seeds:
            matching(root / (str(seed) + ".json"), generation(args, seed, root / str(seed)), revision)
        from aggregate import combine
        result = combine(paths, args.seeds, (args.dataset).as_posix(), (args.mapping).as_posix(), (args.images).as_posix())
        destination = root / ("mean.json" if args.seeds == list(SEEDS) else "mean" + "-".join(map(str, args.seeds)) + ".json")
        destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({name: value["mean"] for name, value in result["table"].items()}, indent=2, allow_nan=False))

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
