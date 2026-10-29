# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import argparse
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
from data import clean, destination, locate, records
from run import common, load, run, saveartifacts, writejson
from split import select
from engine.format import fit, read
from engine.reproduce import digest, snapshot
from engine.serial import read as readjson
from engine.protocol import signature

AGGREGATION = "Arithmetic mean across requests within each seed, then arithmetic mean across seed results."

def datasetargs(parser):
    parser.add_argument("--dataset", default="./pie")
    parser.add_argument("--mapping", default="./mapping_file.json")
    parser.add_argument("--images", default="./annotation_images")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", type=str.casefold, choices=("all", "train", "val", "test"), default="test")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--evalclip", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--categories", nargs="+", type=str.casefold, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--batch", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--output", default="./evaluation")
    return parser

def items(args):
    if args.limit is not None and args.limit < 1:
        raise ValueError("Limit must be positive.")
    values = select(records(args.dataset, args.mapping, args.images), args.manifest, args.split)
    if args.categories:
        requested = set(value.strip().casefold() for value in args.categories)
        values = [item for item in values if item.category.casefold() in requested]
    values = values[:args.limit] if args.limit is not None else values
    if not values:
        raise ValueError("The requested subset is empty.")
    if args.oracle:
        from dataset import annotation
        for item in values:
            mask = annotation(item)
            if mask is None or mask.getbbox() is None:
                raise ValueError("Oracle selection requires a nonempty annotation for every record.")
    return values

def evaluator(args, judge):
    if not args.evalclip:
        return None, False
    from engine.judge import ClipJudge
    root = locate(Path("."), clean(args.evalclip, "evalclip"))
    independent = args.clip is None or root.resolve() != locate(Path("."), clean(args.clip, "clip")).resolve()
    if not independent:
        raise ValueError("Report scorers must not reuse the control checkpoint.")
    loaded = ClipJudge.load(str(root), args.device or "cpu")
    controlhash = getattr(judge, "metadata", {}).get("weights")
    reporthash = getattr(loaded, "metadata", {}).get("weights")
    if controlhash is not None and controlhash == reporthash:
        raise ValueError("Renamed control weights are not independent report weights.")
    return loaded, bool(controlhash and reporthash and controlhash != reporthash)

def masktensor(mask, reference):
    if mask is None:
        return None
    import numpy as np
    import torch
    import torch.nn.functional as functional
    result = torch.from_numpy(np.asarray(mask, dtype=np.float32).copy() / 255.).to(reference.device)[None, None]
    return functional.interpolate(result, size=reference.shape[-2:], mode="nearest")

def regions(item, source, size):
    from PIL import Image, ImageOps
    from dataset import annotation
    from engine.session import orient
    mask = annotation(item)
    if mask is None:
        return None, None
    if item.maskpath is not None and mask.size != source.size:
        raise ValueError("Raster masks must match raw source geometry.")
    mask = mask.resize(source.size, Image.Resampling.NEAREST)
    _, aligned = orient(source, mask)
    fitted = ImageOps.fit(aligned, (size, size), method=Image.Resampling.NEAREST)
    return mask, fitted

def metrics(image, origin, mask=None):
    from engine.wavelet import split
    difference = image.float() - origin.float()
    result = {"mse": float(difference.square().mean()), "l1": float(difference.abs().mean())}
    low, high = split(difference)
    result.update({"lowmse": float(low.square().mean()), "highmse": float(high.square().mean())})
    for name, region in (("edited", mask), ("protected", None if mask is None else 1 - mask)):
        mass = 0. if region is None else float(region.sum()) * difference.shape[1]
        mse = None if mass <= 0 else float((difference.square() * region).sum()) / mass
        result[name + "mse"] = mse
        result[name + "psnr"] = None if mse is None else -10 * math.log10(max(mse, 1e-12))
    return result

def paired(image, item):
    # Compare against an available paired reference.
    if item.edited is None:
        return {}
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    reference = fit(read(item.edited), (image.shape[-1], image.shape[-2]))
    pixels = torch.from_numpy(np.asarray(reference, dtype=np.float32).copy()).permute(2, 0, 1)[None].to(image.device) / 255
    difference = image.float() - pixels
    return {"pairmse": float(difference.square().mean()), "pairl1": float(difference.abs().mean())}

def seeded(rows):
    groups = {}
    for row in rows:
        if type(row.get("seed")) is not int:
            raise ValueError("Seed summaries require an integer seed on every row.")
        groups.setdefault(row["seed"], []).append({key: value for key, value in row.items() if key != "seed"})
    return {str(seed): summary(values) for seed, values in sorted(groups.items())}

def averaged(rows, separate):
    # Reduce the same seed results that are saved in the report.
    if set(separate) != {str(row["seed"]) for row in rows}:
        raise ValueError("Seed summaries do not match the evaluated seed set.")
    result = {}
    for key in sorted({key for row in rows for key in row["metrics"]}):
        values = [separate[str(seed)].get(key, {}).get("mean") for seed in sorted({row["seed"] for row in rows})]
        valid = {str(row.get("key", index)) for index, row in enumerate(rows) if isinstance(row["metrics"].get(key), (int, float)) and not isinstance(row["metrics"].get(key), bool) and math.isfinite(row["metrics"][key])}
        result[key] = {"mean": sum(values) / len(values) if values and all(value is not None for value in values) else None, "count": len(valid)}
    return result

def complete(rows, seeds, keys):
    if any(type(row.get("seed")) is not int or not isinstance(row.get("key"), str) for row in rows):
        raise ValueError("Completed results require request keys and integer seeds.")
    observed = {(row["key"], row["seed"]) for row in rows}
    expected = {(key, seed) for key in keys for seed in seeds}
    if len(observed) != len(rows) or observed != expected:
        raise ValueError("Completed results must contain exactly one row per requested image and seed.")

def summary(rows):
    keys = sorted({key for row in rows for key in row["metrics"]})
    if any("seed" in row for row in rows):
        # Average requests within each seed before averaging seed results.
        return averaged(rows, seeded(rows))
    result = {}
    for key in keys:
        grouped = {}
        for index, row in enumerate(rows):
            value = row["metrics"].get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                grouped.setdefault(str(row.get("key", index)), []).append(value)
        values = [sum(group) / len(group) for group in grouped.values()]
        result[key] = {"mean": sum(values) / len(values) if values else None, "count": len(values)}
    return result

def identifier(key, seed):
    return hashlib.sha256((str(key) + ":" + str(seed)).encode()).hexdigest()[:20]

def costs(rows):
    # Report seed-first means and per-request cost distributions.
    import numpy as np
    grouped, measurements = {}, []
    stages = {"front", "field", "backend", "prior", "diagnostic"}
    for row in rows:
        stages.update((row.get("report", {}).get("stats", {}).get("budget") or {}).get("stages", {}))
    for index, row in enumerate(rows):
        stats = row.get("report", {}).get("stats", {})
        judge = row.get("report", {}).get("judgestats", {})
        values = {"seconds": row.get("seconds"), "vram": row.get("vram"), "residentvram": row.get("residentvram"), "incrementalvram": row.get("incrementalvram"), **{key: stats.get(key) for key in ("samples", "calls", "decodes", "encodes", "textcalls")}, **{"control" + key: judge.get(key) for key in ("evaluations", "imagecalls", "imagesamples")}}
        if stats.get("budget") is not None:
            for stage in stages:
                counts = stats["budget"].get("stages", {}).get(stage, {})
                values.update({stage + key: counts.get(key, 0) for key in ("samples", "calls")})
        measured = {key: value for key, value in values.items() if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0}
        measurements.append({"key": row.get("key", index), "metrics": measured, **({"seed": row["seed"]} if "seed" in row else {})})
        for key, value in values.items():
            if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
                grouped.setdefault(key, {}).setdefault(str(row.get("key", index)), []).append(float(value))
    separate = seeded(measurements) if measurements and all("seed" in row for row in measurements) else {}
    result, means = {}, averaged(measurements, separate) if separate else summary(measurements)
    for key, groups in sorted(grouped.items()):
        values = [sum(group) / len(group) for group in groups.values()]
        result[key] = {"mean": means[key]["mean"], "median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "sources": len(values)}
    concurrent = any(row.get("concurrent") for row in rows)
    return {"metrics": result, "byseed": separate, "aggregation": "Means average requests within each seed, then seeds; median and p95 summarize per-request seed averages.", "scope": "Concurrent request latency includes queue waits, while execution.throughput measures complete dataset throughput." if concurrent else "Synchronized per-image inference excludes checkpoint loading and independent report scoring."}

def coverage(rows):
    # Count executed mechanisms separately from enabled switches.
    reports = [row.get("report", {}) for row in rows if row.get("report", {}).get("method") == "path"]
    result = {"requests": len(reports), "initialized": sum(report.get("initialization", {}).get("complete") is True for report in reports), "searched": sum(len(report.get("candidates", [])) > report.get("initialization", {}).get("evaluated", 0) for report in reports), "rawfallback": sum(report.get("fallback") is True for report in reports), "emptyprotection": sum(report.get("protectedvalid") is False for report in reports)}
    for key in ("release", "rescue"):
        records = [report.get(key, {}) for report in reports]
        result[key] = {"enabled": sum(record.get("enabled") is True for record in records), "queried": sum(record.get("queries", 0) > 0 for record in records), "accepted": sum(record.get("accepted", 0) > 0 for record in records), "queries": sum(record.get("queries", 0) for record in records)}
    records = [report["prioractivity"] for report in reports if "prioractivity" in report]
    result["prior"] = {"recorded": len(records), "missing": len(reports) - len(records), "enabled": sum(record.get("enabled") is True for record in records), "queried": sum(record.get("evaluations", 0) > 0 for record in records), "penalized": sum(record.get("penalized", 0) > 0 for record in records), "vetoed": sum(record.get("vetoed", 0) > 0 for record in records), "selectionchanged": sum(record.get("selectionchanged") is True for record in records), "reasons": {reason: sum(record.get("reason") == reason for record in records) for reason in sorted({record["reason"] for record in records})}}
    records = [report.get("rescue", {}) for report in reports]
    result["rescueactivity"] = {key: sum(record.get(key, 0) for record in records) for key in ("considered", "eligible", "proposed", "budgetblocked")}
    result["rescueactivity"]["selected"] = sum(record.get("selected") is True for record in records)
    heads = [key for report in reports for key in report.get("attainment", {}).get("bottlenecks", [])]
    result["bottlenecks"] = {key: heads.count(key) for key in sorted(set(heads))}
    result["scope"] = "All image-seed requests; counts describe execution rather than independent quality."
    return result

def sync(device):
    import torch
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)

def clusters(args):
    if args.manifest is None:
        return {}
    payload = readjson(locate(Path("."), clean(args.manifest, "manifest")))
    return {key: hashlib.sha256("|".join(sorted(group)).encode()).hexdigest() for group in payload.get("groups", []) for key in group}

def reconcile(rows, dispatch):
    # Match request budgets to the shared inference ledger.
    totals = {key: sum(row.get("report", {}).get("stats", {}).get(key, 0) for row in rows) for key in ("samples", "calls")}
    if totals["samples"] != dispatch["denoisersamples"] or totals["calls"] != dispatch["denoiserrequests"]:
        raise RuntimeError("Per-request denoiser counts do not match shared dispatch accounting.")
    return {"matched": True, "samples": totals["samples"], "logicalcalls": totals["calls"], "physicalcalls": dispatch["denoisercalls"]}

def request(args, item, seed, resources, output, grouping, concurrent=False):
    # Keep request state and counters separate from shared weights.
    args = deepcopy(args)
    editor, judge, evaluation = resources
    from engine.session import toimage
    import torch
    for key in ("attributes", "retain", "localtext"):
        value = getattr(item, key) or getattr(args, key)
        setattr(args, key, list(value) if key in {"attributes", "retain"} and value is not None else value)
    source = read(item.image)
    annotationmask, fittedmask = regions(item, source, args.size)
    sourcehash, args.seed = digest(item.image), seed
    if not concurrent:
        sync(editor.device)
        if str(editor.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(editor.device)
    resident = torch.cuda.memory_allocated(editor.device) / 2 ** 20 if not concurrent and str(editor.device).startswith("cuda") else None
    started = time.perf_counter()
    edited, report, session = run(editor, judge, source, item.source, item.target, args, annotationmask if args.oracle else None)
    sync(edited.device)
    elapsed = time.perf_counter() - started
    memory = torch.cuda.max_memory_allocated(editor.device) / 2 ** 20 if not concurrent and str(editor.device).startswith("cuda") else None
    mask = masktensor(fittedmask, session.origin)
    values = metrics(edited, session.origin, mask)
    values.update(paired(edited, item))
    reportstats = evaluation.stats() if evaluation is not None else {}
    if evaluation is not None:
        reporting = evaluation.prepare(session.origin, item.source, item.target, mask, args.localtext)
        values.update({"clip" + key: value for key, value in reporting(edited).items()})
        if session.artifacts:
            artifactevaluation = {name: reporting(pixels) for name, pixels in session.artifacts.items()}
            report["artifactevaluation"] = artifactevaluation
            if "mechanism" in report:
                report["mechanism"]["reportevaluation"] = artifactevaluation
        report["reportstats"] = {key: value - reportstats.get(key, 0) for key, value in evaluation.stats().items()}
    name = identifier(item.key, seed)
    toimage(edited).save(output / (name + ".png"))
    report["artifacts"] = saveartifacts(session, output / name)
    report["artifactsha256"] = {key: hashlib.sha256(Path(value).read_bytes()).hexdigest() for key, value in report["artifacts"].items()}
    row = {"key": item.key, "group": grouping.get(item.key, item.group or item.relative.as_posix()), "sourcegroup": item.group, "family": item.family, "instruction": item.instruction, "sourceimage": item.image.as_posix(), "category": item.category, "benchmark": item.benchmark, "partition": item.partition, "captionorigin": item.captionorigin, "seed": seed, "seconds": elapsed, "vram": memory, "image": name + ".png", "resultsha256": hashlib.sha256((output / (name + ".png")).read_bytes()).hexdigest(), "metrics": values, "report": report}
    row.update({"inputsha256": sourcehash, "residentvram": resident, "incrementalvram": max(0., memory - resident) if memory is not None else None, "concurrent": concurrent, "timing": "Shared-device request latency including queue waits." if concurrent else "Synchronized standalone inference latency."})
    writejson(output / (name + ".json"), row)
    return row

def execute(args, resources=None):
    from engine.batch import Batch, options
    import torch
    args = deepcopy(args)
    width, settings = options(args), deepcopy(vars(args))
    chosen, seeds = items(args), sorted(set(args.seeds or [args.seed]))
    if any(type(seed) is not int or not -(2 ** 63) <= seed < 2 ** 64 for seed in seeds):
        raise ValueError("All experiment seeds must be supported integers.")
    output = destination(args.output, "output")
    if output.exists():
        raise FileExistsError(output)
    editor, judge = load(args) if resources is None else resources[:2]
    evaluation, independent = evaluator(args, judge) if resources is None else resources[2:]
    grouping = clusters(args)
    inputfiles = {"mapping": locate(locate(Path("."), clean(args.dataset, "dataset")), clean(args.mapping, "mapping"))}
    if args.manifest is not None:
        inputfiles["manifest"] = locate(Path("."), clean(args.manifest, "manifest"))
    provenance = {key: {"path": path.as_posix(), "sha256": digest(path)} for key, path in inputfiles.items()}
    output.mkdir(parents=True)
    rows, jobs = [], [(item, seed) for item in chosen for seed in seeds]
    width = min(width, len(jobs))
    resources = (editor, judge, evaluation)
    sync(editor.device)
    gpu = str(editor.device).startswith("cuda")
    if gpu:
        torch.cuda.reset_peak_memory_stats(editor.device)
    resident = torch.cuda.memory_allocated(editor.device) / 2 ** 20 if gpu else None
    residentreserved = torch.cuda.memory_reserved(editor.device) / 2 ** 20 if gpu else None
    memorytrace = []
    started = time.perf_counter()
    try:
        with Batch(resources, width) if width > 1 else nullcontext(None) as parallel:
            def worker(value):
                current = parallel.replicas() if parallel is not None else resources
                return request(args, *value, current, output, grouping, concurrent=parallel is not None)

            for offset in range(0, len(jobs), width):
                group = jobs[offset:offset + width]
                completed = parallel.map(worker, group) if parallel is not None else [worker(group[0])]
                rows.extend(completed)
                for row in completed:
                    print(json.dumps({key: row[key] for key in ("key", "seed", "seconds", "metrics")}, allow_nan=False), flush=True)
                if getattr(editor, "metadata", {}).get("family") == "flow":
                    for name in ("pools", "textids"):
                        getattr(editor, name).clear()
                if gpu:
                    memorytrace.append({"completed": len(rows), "allocated": torch.cuda.memory_allocated(editor.device) / 2 ** 20, "reserved": torch.cuda.memory_reserved(editor.device) / 2 ** 20, "peakreserved": torch.cuda.max_memory_reserved(editor.device) / 2 ** 20})
            complete(rows, seeds, {item.key for item in chosen})
            calls = sum(row["report"].get("stats", {}).get("calls", 0) for row in rows)
            samples = sum(row["report"].get("stats", {}).get("samples", 0) for row in rows)
            dispatch = parallel.report() if parallel is not None else {"width": 1, "sharedweights": True, "denoisercalls": calls, "denoiserrequests": calls, "denoisersamples": samples}
            dispatch["accounting"] = {**reconcile(rows, dispatch), "checked": parallel is not None}
    except BaseException as error:
        failure = {"schema": 1, "error": type(error).__name__, "message": str(error), "settings": settings, "completed": sorted(path.name for path in output.glob("*.json")), "scope": "Only completed row JSON files belong to retained results; no complete summary is published."}
        try:
            writejson(output / "failure.json", failure)
        except (OSError, ValueError, TypeError) as reporterror:
            error.add_note("The failure report could not be saved: " + str(reporterror))
        raise
    sync(editor.device)
    elapsed = time.perf_counter() - started
    payload = {"schema": 1, "split": args.split, "manifest": args.manifest, "oracle": args.oracle, "independentcheckpoint": independent, "controlcheckpoint": getattr(judge, "metadata", None), "reportcheckpoint": getattr(evaluation, "metadata", None), "evaluatornote": "Different CLIP weights remain shared-family diagnostics, not independent task evidence.", "metricnote": "Pixel, band, and paired errors are diagnostics, not official benchmark protocols.", "settings": settings, "seeds": seeds, "inputs": provenance, "reproducibility": snapshot(), "categories": {category: summary([row for row in rows if row["category"] == category]) for category in sorted({row["category"] for row in rows})}, "rows": rows}
    payload["byseed"] = seeded(rows)
    payload["summary"] = averaged(rows, payload["byseed"])
    payload["aggregation"] = AGGREGATION
    payload["cost"] = costs(rows)
    peak = torch.cuda.max_memory_allocated(editor.device) / 2 ** 20 if gpu and width > 1 else max((row["vram"] for row in rows if row["vram"] is not None), default=None)
    payload["execution"] = {**dispatch, "requested": getattr(args, "batch", 1), "requests": len(rows), "seconds": elapsed, "throughput": len(rows) / elapsed, "residentvram": resident, "peakvram": peak, "residentreservedvram": residentreserved, "finalvram": torch.cuda.memory_allocated(editor.device) / 2 ** 20 if gpu else None, "reservedvram": torch.cuda.memory_reserved(editor.device) / 2 ** 20 if gpu else None, "peakreservedvram": max((row["peakreserved"] for row in memorytrace), default=None), "memorytrace": memorytrace, "scope": "End-to-end dataset throughput includes input loading, control, independent scoring, and output writes; checkpoint loading is excluded. Memory values are MiB; reserved memory includes allocator caches."}
    payload["controlsignature"] = signature(argparse.Namespace(**settings))
    payload["controller"] = coverage(rows)
    statuses = [row["report"].get("field", {}) for row in rows]
    payload["fields"] = {"requested": args.field, "requests": len(rows), "actual": {kind: sum(value.get("actual") == kind for value in statuses) for kind in sorted({value["actual"] for value in statuses if "actual" in value})}, "budgetfallback": sum(value.get("status") == "budget" for value in statuses), "budgetstopped": sum(bool(row["report"].get("budgetstopped")) for row in rows), "sourcefallback": sum(bool(row["report"].get("budgetstopped")) and row["report"].get("selected") == "source" for row in rows)}
    measurements = []
    for row in rows:
        field = row["report"].get("field", {})
        measured = {key: field.get(key) for key in ("rank", "actualfitmean", "relativefit", "responsecosine", "modelgapmean", "relativegap")}
        measured.update({"input" + key: value for key, value in field.get("alignment", {}).items() if key != "scope"})
        measured.update({"response" + key: field.get("capacity", {}).get(key) for key in ("rank", "lineerror", "spanerror", "directionroom", "linecosine")})
        measured.update({"temporal" + key: field.get("temporal", {}).get(key) for key in ("rank", "spanerror", "coneerror", "strength", "weight")})
        measurements.append({"key": row["key"], "seed": row["seed"], "metrics": measured})
    payload["fields"]["diagnostics"] = summary(measurements)
    writejson(output / "summary.json", payload)
    return payload

def main():
    parser = datasetargs(common(argparse.ArgumentParser(description="Evaluate editing with report-only independent scorers.")))
    execute(parser.parse_args())

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
