# -*- coding: utf-8 -*-
# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

from assess import grouped, coverage
from data import records
from dataset import annotation

def finite(rows, field):
    # Exclude undefined or nonfinite values only from their own metric.
    result = {}
    for metric in sorted({key for row in rows for key in row[field]}):
        values = [row[field].get(metric) for row in rows]
        valid = [value for value in values if type(value) in (int, float) and math.isfinite(value)]
        excluded = sum(row.get("status", {}).get(metric) in {"positive infinity", "negative infinity", "nan", "nonfinite"} or type(value) in (int, float) and not math.isfinite(value) for row, value in zip(rows, values))
        result[metric] = {"mean": statistics.fmean(valid) if valid else None, "count": len(valid), "excludednonfinite": excluded}
    return result

def equal(reports):
    result = {}
    for metric in sorted({key for report in reports.values() for key in report["table"]}):
        values = [report["table"][metric] for report in reports.values()]
        missing = [seed for seed, report in reports.items() if report["table"][metric]["mean"] is None]
        result[metric] = {"mean": None if missing else statistics.fmean(value["mean"] for value in values), "count": sum(value["count"] for value in values), "excludednonfinite": sum(value["excludednonfinite"] for value in values)}
        if missing:
            result[metric].update(status="undefined seed", undefinedseeds=missing)
    return result

def combine(paths, seeds, dataset="data", mapping="mapping_file.json", images="annotation_images"):
    # Require the same complete request grid and frozen field profile.
    items = records(dataset, mapping, images)
    regions = {item.key: annotation(item, border=True).getextrema() for item in items}
    expected = set(regions)
    if len(expected) != 700 or len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Aggregation requires unique seeds and the complete benchmark.")
    reports, rows, profile, backend, mapping = {}, [], None, None, None
    excluded = {"seed", "seeds", "output"}
    for path in paths:
        document = json.loads(Path(path).read_text())
        population = document.get("population", {})
        declared = population.get("seeds")
        if not population.get("benchmarkcomplete") or not isinstance(declared, list) or len(declared) != 1:
            raise ValueError("Each input must be one completed benchmark seed.")
        seed = declared[0]
        if seed not in seeds or seed in reports:
            raise ValueError("Input seeds must match the requested unique seeds.")
        samples = document["rows"]
        if len(samples) != 700 or {row["key"] for row in samples} != expected or any(row["seed"] != seed for row in samples):
            raise ValueError("The request-seed grid is incomplete or duplicated.")
        inferencepath = Path(document["settings"]["input"])
        inferencebytes = inferencepath.read_bytes()
        if hashlib.sha256(inferencebytes).hexdigest() != document["inputsha256"]:
            raise ValueError("The scored inference summary has changed.")
        inference = json.loads(inferencebytes)
        settings = inference["rows"][0]["report"]["settings"]
        current = {key: value for key, value in settings.items() if key not in excluded}
        current["selection"] = inference["rows"][0]["report"].get("selected")
        current["responsecontrol"] = inference.get("responsecontrol")
        calculator = document["backend"].get("sha256")
        if calculator != "4dee71873d6eadb0ce5eeed22c82ccd2690938c94cea105126ec11cfa5771343":
            raise ValueError("The metric calculator differs from the audited official file.")
        if profile is not None and (current != profile or calculator != backend or document["mappingsha256"] != mapping):
            raise ValueError("Seed runs use different profiles, evaluators, or request mappings.")
        profile, backend, mapping = current, calculator, document["mappingsha256"]
        for row in samples:
            if row["backend"].get("sha256") != calculator:
                raise ValueError("A metric row uses a different calculator.")
            if set(row["table"]) != {"psnr", "mse", "ssim", "lpips", "whole", "edited", "structure"}:
                raise ValueError("The metric set is incomplete.")
            for metric, value in row["table"].items():
                status = row["status"].get(metric)
                raw = row["raw"].get(metric)
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                    raise ValueError("Displayed metrics require finite numeric values or explicit null status.")
                scale = {"mse": 1000., "ssim": 100., "lpips": 1000., "structure": 1000.}.get(metric, 1.)
                if value is not None and (isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw) or not math.isclose(value, raw * scale, rel_tol=1e-10, abs_tol=1e-10)):
                    raise ValueError("Raw and displayed metric scales differ.")
                if value is None and raw is not None:
                    raise ValueError("Raw and displayed undefined metric values differ.")
                if status == "finite":
                    if metric in {"psnr", "mse", "ssim", "lpips"} and regions[row["key"]][0] != 0:
                        raise ValueError("A background metric is defined on an empty protected region.")
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError("Finite metric status requires a finite numeric value.")
                else:
                    protected = regions[row["key"]][0] == 0
                    background = metric in {"psnr", "mse", "ssim", "lpips"}
                    absent = background and not protected
                    exact = metric == "psnr" and protected and status == "positive infinity" and row["table"]["mse"] == 0
                    if value is not None or not (absent or exact):
                        raise ValueError("A required metric is missing or has invalid infinite status.")
                    if absent and status not in {"empty protected region", "missing backend or required region"}:
                        raise ValueError("The undefined metric status differs from its region coverage.")
        byseed = {"raw": grouped(samples, "raw"), "table": grouped(samples, "table"), "coverage": coverage(samples)}
        if byseed != document["byseed"].get(str(seed)):
            raise ValueError("Stored byseed values do not match the per-request metrics.")
        reports[str(seed)] = {"raw": finite(samples, "raw"), "table": finite(samples, "table"), "coverage": byseed["coverage"]}
        rows.extend(samples)
    if set(map(int, reports)) != set(seeds):
        raise ValueError("A requested seed is missing.")
    table = equal(reports)
    return {"seeds": seeds, "requestsperseed": 700, "byseed": reports, "table": table, "coverage": coverage(rows), "aggregation": "Exclude nonfinite values separately per metric; arithmetic mean within each seed, then equal-weight mean across all requested seeds. Raw calculator reports remain unchanged."}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--dataset", default="data")
    parser.add_argument("--mapping", default="mapping_file.json")
    parser.add_argument("--images", default="annotation_images")
    args = parser.parse_args()
    result = combine(args.input, args.seeds, args.dataset, args.mapping, args.images)
    compact = {"seeds": result["seeds"], "requestsperseed": 700, "byseed": {seed: value["table"] for seed, value in result["byseed"].items()}, "table": result["table"], "coverage": {metric: {status: value["rows"] for status, value in states.items()} for metric, states in result["coverage"].items()}, "aggregation": result["aggregation"]}
    print(json.dumps(compact, indent=2, allow_nan=False))

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
