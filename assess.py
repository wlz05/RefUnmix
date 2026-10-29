# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from data import clean, destination, locate, records
from dataset import annotation
from engine.metrics import Protocol, inputs as metricinputs
from evaluate import AGGREGATION, averaged, summary
from run import writejson
from engine.format import read
from engine.reproduce import digest, versions
from engine.serial import read as readjson

def parser():
    result = argparse.ArgumentParser(description="Score saved editing outputs with the public benchmark protocol.")
    result.add_argument("--input", required=True)
    result.add_argument("--dataset", default="./pie")
    result.add_argument("--mapping", default="./mapping_file.json")
    result.add_argument("--images", default="./annotation_images")
    backend = result.add_mutually_exclusive_group(required=True)
    backend.add_argument("--protocol", help="Path to the released PIE metric calculator.")
    backend.add_argument("--pixels", action="store_true", help="Run explicit pixel-only diagnostics.")
    result.add_argument("--scope", type=str.casefold, choices=("benchmark", "subset"), default="benchmark")
    result.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 3047, 3407])
    result.add_argument("--device", default="cpu")
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--network", action="store_true")
    mode.add_argument("--offline", action="store_true", help="Load the external calculator from complete local caches.")
    result.add_argument("--cache", default="./cache/metrics", help="Relative directory for downloaded metric assets.")
    result.add_argument("--output", default="./assessment.json")
    return result

def population(payload, items, seeds, scope):
    # Validate the complete request-seed grid before loading metric weights.
    if scope not in {"benchmark", "subset"}:
        raise ValueError("Assessment scope must be benchmark or subset.")
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int or not -(2 ** 63) <= seed < 2 ** 64 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Assessment seeds must be unique supported integers.")
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list) or not payload["rows"]:
        raise ValueError("Input must contain a nonempty list of evaluation rows.")
    recorded = payload.get("seeds")
    if recorded is not None and (not isinstance(recorded, list) or any(type(seed) is not int for seed in recorded) or sorted(recorded) != sorted(seeds)):
        raise ValueError("Declared assessment seeds differ from the inference summary.")
    if scope == "benchmark" and (len(items) != 700 or {item.category for item in items.values()} != set("0123456789")):
        raise ValueError("Benchmark scoring requires all 700 PIE requests across categories 0 through 9; use --scope subset for diagnostics.")
    seen, keys, paths = set(), set(), set()
    for row in payload["rows"]:
        if not isinstance(row, dict) or "configuration" in row:
            raise ValueError("Input must describe one method, not a time-search study.")
        key, seed = row.get("key"), row.get("seed")
        if not isinstance(key, str) or key not in items or type(seed) is not int or seed not in seeds:
            raise ValueError("Assessment rows require a known request key and a declared integer seed.")
        pair = (key, seed)
        if pair in seen:
            raise ValueError("Assessment request-seed pairs must be unique.")
        path = clean(row.get("image"), "prediction").as_posix()
        if path in paths:
            raise ValueError("Assessment rows must reference separate prediction files.")
        if not isinstance(row.get("report", {}), dict):
            raise ValueError("Assessment row reports must be objects.")
        seen.add(pair)
        keys.add(key)
        paths.add(path)
    expected = set(items) if scope == "benchmark" else keys
    missing = len(expected) * len(seeds) - len(seen)
    if keys != expected or missing:
        raise ValueError(f"The assessment request-seed grid is incomplete: {missing} missing rows.")
    categories = {category: sum(items[key].category == category for key in keys) for category in sorted({items[key].category for key in keys})}
    return {"scope": scope, "complete": True, "benchmarkcomplete": scope == "benchmark", "requests": len(keys), "datasetrequests": len(items), "rows": len(seen), "seeds": sorted(seeds), "categories": categories}

def prediction(row, source):
    path = locate(source.parent, clean(row["image"], "prediction"))
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = row.get("resultsha256")
    if expected is not None and (not isinstance(expected, str) or digest(path) != expected):
        raise ValueError("Saved prediction no longer matches its inference hash.")
    return path

def geometry(origin, edited, item):
    if origin.size != (512, 512) or edited.size != (512, 512) or item.masksize != (512, 512):
        raise ValueError("PIE scoring requires 512 by 512 source, output, and annotation geometry.")
    if origin.getexif().get(274, 1) != 1 or edited.getexif().get(274, 1) != 1:
        raise ValueError("PIE scoring requires normalized source and output orientation.")

def verified(row, source, item):
    image, edited = read(item.image), read(prediction(row, source))
    geometry(image, edited, item)
    if row.get("inputsha256") is not None and digest(item.image) != row["inputsha256"]:
        raise ValueError("Source image no longer matches its inference hash.")
    settings = row.get("report", {})
    if settings.get("target", item.target) != item.target or settings.get("source", item.source) != item.source:
        raise ValueError("Generated prompts no longer match the evaluation dataset.")
    if settings.get("imagesha256") is not None:
        import numpy as np
        sourcepixels = (np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255).copy(order="C")
        if hashlib.sha256(sourcepixels.tobytes()).hexdigest() != settings["imagesha256"]:
            raise ValueError("Generated source pixels no longer match the evaluation dataset.")
    mask = annotation(item, border=True)
    if mask is None:
        raise ValueError("PIE assessment requires an annotation mask for every request.")
    metricinputs(image, edited, mask)
    return image, edited, mask

def grouped(rows, field, separate=None):
    # Preserve the seed-first reporting order in official scores.
    means = [{"key": row["key"], "metrics": row[field], "status": row.get("status", {}), **({"seed": row["seed"]} if "seed" in row else {})} for row in rows]
    result = summary(means) if separate is None else averaged(means, separate)
    return result

def coverage(rows):
    # Keep undefined regions visible in every metric denominator.
    metrics = sorted({key for row in rows for key in row.get("status", {})})
    result = {}
    for metric in metrics:
        states = sorted({row.get("status", {}).get(metric, "missing status") for row in rows})
        result[metric] = {state: {"rows": sum(row.get("status", {}).get(metric, "missing status") == state for row in rows), "sources": sorted({str(row["key"]) for row in rows if row.get("status", {}).get(metric, "missing status") == state})} for state in states}
    return result

def main():
    args = parser().parse_args()
    source = locate(Path("."), clean(args.input, "input"))
    target = destination(args.output, "output")
    if target.exists():
        raise FileExistsError(target)
    sourcehash = digest(source)
    payload = readjson(source)
    items = {item.key: item for item in records(args.dataset, args.mapping, args.images)}
    if any(item.benchmark != "pie" for item in items.values()):
        raise ValueError("This scoring interface implements the PIE protocol only.")
    members = population(payload, items, args.seeds, args.scope)
    if args.pixels and (args.network or args.offline):
        raise ValueError("Pixel diagnostics do not use --network or --offline.")
    inputs = payload.get("inputs", {})
    if not isinstance(inputs, dict) or not isinstance(inputs.get("mapping", {}), dict):
        raise ValueError("Inference input provenance must contain mapping metadata.")
    mapping = locate(locate(Path("."), clean(args.dataset, "dataset")), clean(args.mapping, "mapping"))
    mappinghash = digest(mapping)
    if inputs.get("mapping", {}).get("sha256", mappinghash) != mappinghash:
        raise ValueError("The annotation mapping differs from the inference run.")
    # Check every saved output before constructing the external calculator.
    files = set()
    for row in payload["rows"]:
        path = prediction(row, source).resolve()
        if path in files:
            raise ValueError("Assessment rows must not alias the same prediction file.")
        files.add(path)
        verified(row, source, items[row["key"]])
    backend = Protocol.load(args.protocol, args.device, args.network, cache=args.cache, offline=args.offline)
    output = []
    for row in payload["rows"]:
        pair = (row["key"], row["seed"])
        item = items[pair[0]]
        image, edited, mask = verified(row, source, item)
        result = backend.evaluate(image, edited, mask, item.target)
        output.append({"key": item.key, "seed": pair[1], "category": item.category, "image": row["image"], **result})
        print(json.dumps({"key": item.key, "seed": pair[1], "table": result["table"]}, allow_nan=False), flush=True)
    if digest(source) != sourcehash or digest(mapping) != mappinghash:
        raise ValueError("Inference summary or annotation mapping changed during assessment.")
    result = {"schema": 1, "protocol": "PIE external metric adapter" if args.protocol else "PIE pixel diagnostics", "settings": vars(args), "inputsha256": sourcehash, "mappingsha256": mappinghash, "population": members, "packages": dict(versions()), "backend": backend.metadata, "rows": output, "note": "Metrics are averaged across requests within each seed, then across seed results; undefined and infinite PSNR values remain null with explicit status."}
    result["byseed"] = {str(seed): {"raw": grouped([row for row in output if row["seed"] == seed], "raw"), "table": grouped([row for row in output if row["seed"] == seed], "table"), "coverage": coverage([row for row in output if row["seed"] == seed])} for seed in sorted(args.seeds)}
    for field in ("raw", "table"):
        result[field] = grouped(output, field, {seed: value[field] for seed, value in result["byseed"].items()})
    result["aggregation"] = AGGREGATION
    result["coverage"] = coverage(output)
    writejson(target, result)

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
