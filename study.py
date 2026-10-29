# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from data import clean, locate
from evaluate import AGGREGATION, aggregate, datasetargs, declared, evaluator, identifier, items, masktensor, metrics, regions, summary
from run import common, load, localization, times, writejson

def configurations(args):
    values = times(args.times)
    result = [{"kind": "base", "start": args.start, "delta": args.delta}]
    if args.scan == "time":
        result.extend({"kind": "single", "start": value, "delta": 0.} for value in values)
        result.extend({"kind": "chord", "start": value, "delta": args.delta} for value in values if args.delta > 0 and value > args.delta)
    else:
        delta = args.delta if args.proposal == "chord" else 0.
        if any(value <= delta for value in values):
            raise ValueError("Band proposal times must exceed their chord interval.")
        result.extend({"kind": "bands", "low": low, "high": high, "delta": delta} for low in values for high in values)
    return result

def signature(config):
    return json.dumps(config, sort_keys=True, separators=(",", ":"))

def field(session, config):
    kind = config.get("kind")
    if kind in {"base", "single", "chord"}:
        start, delta = float(config["start"]), float(config["delta"])
        if not math.isfinite(delta) or not 0 <= delta < start:
            raise ValueError("Invalid frozen chord interval.")
        return session.field(times([start])[0], delta)
    if kind == "bands":
        from engine.wavelet import split
        low, high = times([config["low"]]), times([config["high"]])
        delta = float(config.get("delta", 0.))
        if not math.isfinite(delta) or not 0 <= delta < min(low[0], high[0]):
            raise ValueError("Invalid frozen band interval.")
        lowfield, highfield = session.field(low[0], delta), session.field(high[0], delta)
        return split(lowfield)[0] + split(highfield)[1]
    raise ValueError("Unknown frozen configuration.")

def choose(rows, configs, budget, seeds=None):
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("Budget must be finite and nonnegative.")
    if not configs or configs[0].get("kind") != "base" or len({signature(config) for config in configs}) != len(configs):
        raise ValueError("Unique configurations must begin with the baseline.")
    grouped = {signature(config): [] for config in configs}
    for row in rows:
        if not math.isfinite(row["selectionscore"]) or row.get("protectedvalid", True) and (row["selectionloss"] is None or not math.isfinite(row["selectionloss"])):
            raise ValueError("Validation scores and valid losses must be finite.")
        grouped[signature(row["configuration"])].append(row)
    base = grouped[signature(configs[0])]
    if not base:
        raise ValueError("Validation requires baseline rows.")
    count = len(base)
    pairs = {(row["key"], row["seed"]) for row in base}
    if len(pairs) != count:
        raise ValueError("Duplicate validation image and seed pairs.")
    protected = [row for row in base if row.get("protectedvalid", True)]
    basepoint = aggregate([{**row, "metrics": {"loss": row["selectionloss"] if row.get("protectedvalid", True) else None}} for row in base])
    baseloss = basepoint["metrics"]["loss"]["mean"]
    scores = []
    for config in configs:
        values = grouped[signature(config)]
        declared({"rows": values, **({"seeds": seeds} if seeds is not None else {})})
        if len(values) != count or {(row["key"], row["seed"]) for row in values} != pairs:
            raise ValueError("Each configuration requires all validation image and seed pairs.")
        point = aggregate([{**row, "metrics": {"score": row["selectionscore"], "loss": row["selectionloss"] if row.get("protectedvalid", True) else None}} for row in values])
        score = point["metrics"]["score"]["mean"]
        valid = [row for row in values if row.get("protectedvalid", True)]
        if {(row["key"], row["seed"]) for row in valid} != {(row["key"], row["seed"]) for row in protected}:
            raise ValueError("Protection support must be fixed across candidates.")
        loss = point["metrics"]["loss"]["mean"]
        feasible = None if baseloss is None else loss <= baseloss + budget
        scores.append({"configuration": config, "score": score, "loss": loss, "byseed": point["byseed"], "seeds": point["seeds"], "aggregation": point["aggregation"], "protectedcount": len(valid), "feasible": feasible})
    feasible = [value for value in scores if value["feasible"] is not False]
    best = max(feasible, key=lambda value: value["score"])
    return best["configuration"], scores

def frozen(args):
    path = locate(Path("."), clean(args.recipe, "recipe"))
    from engine.serial import read
    payload = read(path)
    if payload.get("schema") != 1 or payload.get("split") != "val" or "selected" not in payload:
        raise ValueError("Frozen selection must come from a validation study.")
    configs = payload.get("configurations", [])
    if not configs or configs[0].get("kind") != "base" or payload["selected"] not in configs:
        raise ValueError("Frozen selection is missing its tested configuration.")
    if args.manifest is None or payload.get("manifest") is None:
        raise ValueError("Frozen evaluation requires a grouped split manifest.")
    supplied = locate(Path("."), clean(args.manifest, "manifest")).read_bytes()
    import hashlib
    if hashlib.sha256(supplied).hexdigest() != payload.get("manifestdigest"):
        raise ValueError("Frozen selection and evaluation must use the same split manifest.")
    fixed = payload["settings"]
    reference = "raw" if args.reference == "auto" else args.reference
    previous = payload.get("renderer", "raw" if fixed.get("reference", "auto") == "auto" else fixed["reference"])
    if reference != previous:
        raise ValueError("Frozen evaluation changed the output reference operator.")
    for key in ("model", "clip", "size", "noise", "end", "cleanup", "scale", "oracle", "start", "delta", "precision", "localtext", "proposal"):
        if getattr(args, key) != fixed[key]:
            raise ValueError("Frozen evaluation changed setting: " + key)
    if fixed.get("localization", "baseline") != args.localization:
        raise ValueError("Frozen evaluation changed localization mode.")
    if args.localization != "baseline":
        for key in ("times", "attentionwords", "attentionqueries", "blend", "threshold", "signalfloor", "components", "minimum", "small", "dilation"):
            if getattr(args, key) != fixed.get(key):
                raise ValueError("Frozen localization changed setting: " + key)
    return [payload["configurations"][0], payload["selected"]]

def main():
    parser = datasetargs(common(argparse.ArgumentParser(description="Freeze time or frequency choices on validation sources.")))
    parser.set_defaults(split="val", output="./study", method="base", field="chord")
    parser.add_argument("--scan", choices=("time", "bands"), type=str.casefold, default="time")
    parser.add_argument("--recipe", default=None)
    args = parser.parse_args()
    reference = "raw" if args.reference == "auto" else args.reference
    if args.split not in {"val", "test"} or args.manifest is None:
        raise ValueError("Studies require a grouped manifest and a val or test split.")
    if args.split == "test" and args.recipe is None:
        raise ValueError("Test studies require --recipe from validation.")
    if args.split == "val" and args.recipe is not None:
        raise ValueError("Validation studies do not accept frozen selections.")
    if not args.clip:
        raise ValueError("Studies require a local --clip scorer.")
    chosen = items(args)
    configs = frozen(args) if args.recipe else configurations(args)
    configs = list({signature(config): config for config in configs}.values())
    output = clean(args.output, "output")
    if output.exists():
        raise FileExistsError(output)
    editor, judge = load(args)
    evaluation, independent = evaluator(args, judge)
    from PIL import Image
    from engine.response import Protection
    from engine.locality import locate as support
    from engine.session import toimage
    output.mkdir(parents=True)
    rows = []
    for item in chosen:
        from engine.format import read
        from engine.reproduce import seed as reseed
        image = read(item.image)
        annotationmask, fittedmask = regions(item, image, args.size)
        for seed in sorted(set(args.seeds or [args.seed])):
            reseed(seed, getattr(args, "deterministic", False))
            session = editor.prepare(image, item.source, item.target, seed, args.noise, args.end, args.cleanup, args.scale, region=annotationmask if args.oracle else None)
            if session.mask is None and args.localization in {"attention", "hybrid"}:
                session.localization(item.source, item.target, args.attentionwords, args.attentionqueries)
            base = field(session, configs[0])
            pool = {value: session.field(value, args.delta if args.proposal == "chord" else 0.) for value in args.times} if args.localization != "baseline" else {}
            attention = session.attention() if args.localization in {"attention", "hybrid"} else None
            editmask, localizationreport = support(base, pool, session.mask, attention, localization(args))
            score = judge.prepare(session.origin, item.source, item.target, editmask, args.localtext)
            protection = Protection(session.origin, editmask)
            annotationtensor = masktensor(fittedmask, session.origin)
            evaluate = evaluation.prepare(session.origin, item.source, item.target, annotationtensor, args.localtext) if evaluation is not None else None
            for index, config in enumerate(configs):
                candidate = field(session, config)
                edited = session.render(candidate) if reference == "raw" else session.render(candidate, reference=reference)
                values = metrics(edited, session.origin, annotationtensor)
                if evaluate is not None:
                    values.update({"clip" + key: value for key, value in evaluate(edited).items()})
                loss = protection.energy(edited)
                name = identifier(item.key, seed) + "-" + str(index)
                toimage(edited).save(output / (name + ".png"))
                control = session.diagnostic(candidate, editmask) if args.control else None
                row = {"key": item.key, "group": item.group or item.relative.as_posix(), "seed": seed, "configuration": config, "renderer": reference, "guardapplied": False, "control": control, "selectionscore": score(edited)["score"], "selectionloss": loss, "protectedvalid": loss is not None, "localization": localizationreport, "metrics": values, "stats": session.stats(), "statsmode": "cumulative shared-session candidate-cache cost", "image": name + ".png"}
                rows.append(row)
                writejson(output / (name + ".json"), row)
                print(json.dumps({"key": item.key, "seed": seed, "configuration": config, "score": row["selectionscore"]}, allow_nan=False), flush=True)
    selected, table = choose(rows, configs, args.budget, sorted(set(args.seeds or [args.seed]))) if args.split == "val" else (configs[-1], None)
    import hashlib
    digest = hashlib.sha256(locate(Path("."), clean(args.manifest, "manifest")).read_bytes()).hexdigest()
    seeds = sorted(set(args.seeds or [args.seed]))
    points = {}
    for config in configs:
        values = [row for row in rows if row["configuration"] == config]
        declared({"rows": values, "seeds": seeds})
        points[signature(config)] = aggregate(values)
    groups = {key: value["metrics"] for key, value in points.items()}
    payload = {"schema": 1, "split": args.split, "renderer": reference, "manifest": args.manifest, "manifestdigest": digest, "selected": selected, "configurations": configs, "selectiontable": table, "independentcheckpoint": independent, "settings": vars(args), "seeds": seeds, "aggregation": AGGREGATION, "summary": groups, "byseed": {key: value["byseed"] for key, value in points.items()}, "rows": rows, "note": "Pure operator scans do not apply the projection raw guard; validation means select configurations and test settings are frozen before inference."}
    writejson(output / "summary.json", payload)

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
