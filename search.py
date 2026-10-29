# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from data import clean, locate, records
from split import grouped, identities, select

def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number

def arguments():
    parser = argparse.ArgumentParser(description="Select global editing times on validation data and freeze them for testing.")
    parser.add_argument("--phase", type=str.casefold, choices=("fit", "test"), default="fit")
    parser.add_argument("--dataset", default="./pie")
    parser.add_argument("--mapping", default="./mapping_file.json")
    parser.add_argument("--images", default="./annotation_images")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--recipe", default=None)
    parser.add_argument("--output", default="./runs/timing")
    parser.add_argument("--model", default=None)
    parser.add_argument("--clip", default=None)
    parser.add_argument("--device", type=str.casefold, default=None)
    parser.add_argument("--judge", type=str.casefold, default="cpu")
    parser.add_argument("--precision", type=str.casefold, choices=("fp32", "fp16", "bf16"), default=None)
    parser.add_argument("--size", type=positive, default=None)
    parser.add_argument("--noise", type=positive, default=None)
    parser.add_argument("--method", type=str.casefold, choices=("base", "thread", "frequency"), default=None)
    parser.add_argument("--scan", type=str.casefold, choices=("time", "bands"), default=None)
    parser.add_argument("--study", action="store_true", default=None, help="Compare single-time and chord families with matched endpoint controls.")
    parser.add_argument("--times", type=float, nargs="+", default=None)
    parser.add_argument("--pivot", type=float, default=None, help="Fixed opposite-band time for conditional comparisons.")
    parser.add_argument("--bandbudget", type=float, default=None, help="Allowed protected-band distortion increase in dB.")
    parser.add_argument("--starts", type=float, nargs="+", default=None)
    parser.add_argument("--deltas", type=float, nargs="+", default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--cleanup", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--local", type=float, default=None)
    parser.add_argument("--budget", type=float, default=None, help="Allowed protection PSNR loss in dB.")
    parser.add_argument("--risk", type=float, default=None, help="Allowed fraction of image-seed pairs exceeding the loss budget.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--limit", type=positive, default=None)
    parser.add_argument("--dry", action="store_true", help="Validate the dataset and print the workload without inference.")
    parser.add_argument("--resume", action="store_true", help="Resume only an identical saved experiment plan.")
    return parser.parse_args()

def read(path):
    from engine.serial import read
    return read(path)

def save(path, value):
    # Never replace a different experiment artifact.
    if path.exists():
        if read(path) != value:
            raise FileExistsError(path)
        return
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def fingerprint(items, digests):
    return digest([{ "key": item.key, "image": digests[item.key], "source": item.source, "target": item.target, "instruction": item.instruction, "category": item.category, "group": item.group, "mask": item.mask, "masksize": item.masksize} for item in sorted(items, key=lambda value: value.key)])

def settings(args, recipe=None):
    defaults = {"model": "./models", "clip": "./clip", "precision": "fp32", "size": 512, "noise": 1, "local": 0.5, "budget": 1.0, "risk": 0.1, "bandbudget": None}
    if recipe is not None:
        defaults = recipe["settings"]
    result = {}
    for name, default in defaults.items():
        value = getattr(args, name)
        if name in {"model", "clip"}:
            default = clean(default, name).as_posix()
            value = clean(value, name).as_posix() if value is not None else None
        if recipe is not None and value is not None and value != default:
            raise ValueError(f"Testing must retain the frozen {name} setting.")
        result[name] = default if value is None else value
    if result["size"] % 8:
        raise ValueError("size must be a positive multiple of eight.")
    for name in ("local", "budget", "risk"):
        value = result[name]
        if not math.isfinite(value) or value < 0 or (name != "budget" and value > 1):
            raise ValueError(f"Invalid {name} setting.")
    bandbudget = result.get("bandbudget")
    if bandbudget is not None and (not math.isfinite(bandbudget) or bandbudget < 0):
        raise ValueError("The band damage budget must be finite and nonnegative.")
    for name in ("model", "clip"):
        clean(result[name], name)
    return result

def prepare(args):
    from engine.spectral import grid as bandgrid, validate
    from engine.timing import grid, subset
    items = records(args.dataset, args.mapping, args.images)
    selected = select(items, args.manifest, "val" if args.phase == "fit" else "test")
    selected = subset(selected, args.limit)
    if not selected or any(item.mask is None or not any(item.mask[1::2]) for item in selected):
        raise ValueError("Selected records must have nonempty evaluation edit masks.")
    digests = identities(items)
    identity = fingerprint(items, digests)
    manifest = read(locate(Path("."), clean(args.manifest, "manifest")))
    groups = {key: str(index) for index, group in enumerate(grouped(items, digests)) for key in group}
    recipe = None
    if args.phase == "test":
        if args.recipe is None:
            raise ValueError("Testing requires a frozen --recipe from the fit phase.")
        recipe = read(locate(Path("."), clean(args.recipe, "recipe")))
        if recipe.get("schema") != 1 or recipe.get("phase") != "fit":
            raise ValueError("The recipe is not a completed validation selection.")
        if recipe["fingerprint"] != identity or recipe["manifest"] != digest(manifest):
            raise ValueError("The recipe does not match this dataset and split manifest.")
        if set(recipe["keys"]) & {item.key for item in selected}:
            raise ValueError("Development and test images must be disjoint.")
        if any(getattr(args, name) is not None for name in ("method", "starts", "deltas", "end", "cleanup", "times", "pivot")):
            raise ValueError("Testing cannot change frozen editing parameters.")
    elif args.recipe is not None:
        raise ValueError("A recipe is only accepted in the test phase.")
    runtime = settings(args, recipe)
    scan = recipe.get("scan", "time") if recipe else (args.scan or "time")
    if scan not in {"time", "bands"}:
        raise ValueError("The recipe contains an unsupported search mode.")
    if args.scan is not None and args.scan != scan:
        raise ValueError("Testing must retain the frozen search mode.")
    study = recipe.get("study", False) if recipe else bool(args.study)
    if args.study is not None and args.study != study:
        raise ValueError("Testing must retain the frozen study protocol.")
    if study and (scan != "time" or args.method not in (None, "base")):
        raise ValueError("The time study requires --scan time and the base editor.")
    pivot = recipe.get("pivot") if recipe else (0.75 if args.pivot is None else args.pivot)
    if recipe is None:
        defaulttimes = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        end, cleanup = 0.30 if args.end is None else args.end, True if args.cleanup is None else args.cleanup
        if scan == "bands":
            if args.starts is not None or args.deltas is not None or args.method not in (None, "base"):
                raise ValueError("Band scans use --times and the base editor, not --starts or --deltas.")
            configs = bandgrid(args.times if args.times is not None else defaulttimes, pivot, end, cleanup, runtime["noise"])
            skipped = []
            runtime["bandbudget"] = 1.0 if runtime["bandbudget"] is None else runtime["bandbudget"]
        else:
            if args.times is not None or args.pivot is not None or args.bandbudget is not None:
                raise ValueError("Frequency options require --scan bands.")
            pivot = None
            if study:
                from engine.experiment import candidates
                configs, skipped = candidates(args.starts, args.deltas, end, cleanup, runtime["noise"])
            else:
                configs, skipped = grid(args.starts if args.starts is not None else defaulttimes, args.deltas if args.deltas is not None else [0.0, 0.05, 0.10, 0.15], args.method or "base", end, cleanup, runtime["noise"])
        locked = None
    else:
        locked = recipe["selected"]
        configs = {"native": recipe["reference"], **recipe.get("controls", {}), locked: recipe["config"]}
        skipped = []
        for value in configs.values():
            validate(value)
    seeds = args.seeds if args.seeds is not None else ([81, 82, 83] if args.phase == "fit" else [84, 85])
    if len(set(seeds)) != len(seeds) or any(not -(2 ** 63) <= seed < 2 ** 64 for seed in seeds):
        raise ValueError("Seeds must be unique supported integers.")
    if len({seed % (2 ** 64) for seed in seeds}) != len(seeds):
        raise ValueError("Seeds must remain distinct after generator normalization.")
    plan = {"schema": 1, "revision": "spectral", "phase": args.phase, "scan": scan, "pivot": pivot, "roles": recipe.get("roles", {}) if recipe else {}, "fingerprint": identity, "manifest": digest(manifest), "keys": [item.key for item in selected], "settings": runtime, "configs": configs, "seeds": seeds, "skipped": skipped, "locked": locked, "device": args.device, "judge": args.judge, "maskpolicy": "evaluation only; no annotation enters model inference", "metric": "custom CLIP cosine x100: whole image and neutral masked tight crop; not the official PIE evaluator", "aggregation": "mean over seeds then editing records; source-group bootstrap", "groups": {item.key: groups[item.key] for item in selected}}
    if study:
        plan.update({"study": True, "revision": "experiment"})
    return selected, plan

def pending(root, plan, resume):
    # Resume complete records from the same experiment.
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError("The output folder is not empty; use --resume or a new folder.")
    root.mkdir(parents=True, exist_ok=True)
    path = root / "plan.json"
    if resume and not path.exists():
        raise FileNotFoundError("Resuming requires an existing plan.json.")
    save(path, plan)
    rows = []
    path = root / "rows.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                from engine.serial import loads
                rows.append(loads(line))
    completed = {(row["key"], row["seed"], row["config"]) for row in rows}
    if len(completed) != len(rows) or any(key not in plan["keys"] or seed not in plan["seeds"] or name not in plan["configs"] for key, seed, name in completed):
        raise ValueError("Saved rows are duplicated or incompatible with the plan.")
    return rows, completed

def main():
    args = arguments()
    from engine.spectral import controls, describe, predictions, profiles
    from engine.timing import collect, summarize
    items, plan = prepare(args)
    total = len(items) * len(plan["seeds"]) * len(plan["configs"])
    print(json.dumps({"phase": args.phase, "scan": plan["scan"], "study": plan.get("study", False), "records": len(items), "candidates": len(plan["configs"]), "seeds": plan["seeds"], "edits": total, "skipped": plan["skipped"], "metric": plan["metric"]}))
    if args.dry:
        print(json.dumps({name: describe(value) for name, value in plan["configs"].items()}, indent=2))
        return
    root = locate(Path("."), clean(args.output, "output"))
    code = Path(__file__).resolve().parent
    if root.resolve() == code or code in root.resolve().parents:
        raise ValueError("Experiment outputs must stay outside the code folder.")
    rows, completed = pending(root, plan, args.resume)
    runtime = plan["settings"]
    if len(rows) < total:
        from engine.clip import ClipJudge
        from engine.model import Editor
        editor = Editor.load(runtime["model"], args.device, runtime["precision"], runtime["size"])
        judge = ClipJudge.load(runtime["clip"], args.judge)
        with (root / "rows.jsonl").open("a", encoding="utf-8") as handle:
            for row in collect(editor, judge, items, plan["configs"], plan["seeds"], plan["groups"], runtime["local"], completed, plan["scan"]):
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
                rows.append(row)
                print(f"{len(rows)}/{total} {row['key']} seed={row['seed']} {row['config']} score={row['semantic']:.4f}", flush=True)
    report = summarize(rows, plan["configs"], runtime["budget"], runtime["risk"], plan["locked"], bandbudget=runtime.get("bandbudget"), seeds=plan["seeds"])
    if plan["scan"] == "bands":
        report["bands"] = profiles(report, plan["pivot"], selecting=args.phase == "fit")
        report["predictions"] = predictions(rows)
        report["frozenroles"] = plan["roles"]
        report["fieldnote"] = "Each band uses one raw target-minus-source clean-latent prediction; native alone retains its two-time chord field."
        report["bandnote"] = "Band budgets compare final protected RGB distortion with native in dB; missing interior support is not zero damage."
        report["costnote"] = "Calls and samples are actual cached sweep costs; requiredsamples describes a cold candidate and excludes scoring."
    if plan.get("study"):
        from engine.experiment import analyze, export
        report["study"] = analyze(report, rows, frozen=plan["roles"] if args.phase == "test" else None)
    report.update({"phase": args.phase, "plan": digest(plan), "maskpolicy": plan["maskpolicy"], "metric": plan["metric"], "edits": total, "seconds": sum(row["seconds"] for row in rows), "scoreseconds": sum(row["scoreseconds"] for row in rows), "calls": sum(row["calls"] for row in rows), "samples": sum(row["samples"] for row in rows)})
    save(root / "summary.json", report)
    if plan.get("study"):
        export(root / "curves.csv", report["study"])
    if args.phase == "fit":
        chosen = report["selected"]
        frozen = {name: plan["configs"][name] for name in controls(report, plan["pivot"])} if plan["scan"] == "bands" else {}
        roles = {key: report["bands"][key] for key in ("lowchoice", "highchoice", "samechoice", "independent")} if plan["scan"] == "bands" else {}
        if plan.get("study"):
            roles = report["study"]["roles"]
            frozen = {name: plan["configs"][name] for name in sorted({value for value in roles.values() if value is not None and value != "native"})}
        recipe = {"schema": 1, "phase": "fit", "scan": plan["scan"], "pivot": plan["pivot"], "controls": frozen, "roles": roles, "fingerprint": plan["fingerprint"], "manifest": plan["manifest"], "keys": plan["keys"], "settings": runtime, "selected": chosen, "config": plan["configs"][chosen], "reference": plan["configs"]["native"]}
        if plan.get("study"):
            recipe["study"] = True
        save(root / "recipe.json", recipe)
    selected = next(row for row in report["table"] if row["id"] == report["selected"])
    print(json.dumps({"selected": report["selected"], "locked": report["locked"], **describe(selected["config"]), "gain": selected["gain"], "interval": selected["interval"], "violation": selected["violation"], "bandviolation": selected["bandviolation"], "feasible": selected["feasible"], "summary": (root / "summary.json").as_posix()}, indent=2))
    if plan["scan"] == "bands":
        print(json.dumps({"developmentchoices": {key: report["bands"][key] for key in ("lowchoice", "highchoice", "samechoice", "independent")} if args.phase == "fit" else plan["roles"]}, indent=2))
    if plan.get("study"):
        print(json.dumps({"roles": {role: describe(plan["configs"][name]) if name is not None else None for role, name in report["study"]["roles"].items()}, "comparisons": report["study"]["comparisons"], "curves": (root / "curves.csv").as_posix()}, indent=2))

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
