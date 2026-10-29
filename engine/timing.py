# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import hashlib
import math
import random
import statistics
import time
from dataclasses import asdict
import numpy as np
import torch
from PIL import Image, ImageOps
from dataset import annotation
from engine.model import EditConfig

def grid(starts, deltas, method="base", end=0.30, cleanup=True, noise=1):
    # Include the reference and deduplicate configurations.
    if not starts or not deltas:
        raise ValueError("Time grids must not be empty.")
    if any(not math.isfinite(value) or not 0 < value <= 1 for value in starts):
        raise ValueError("Start times must be finite and in (0, 1].")
    if any(not math.isfinite(value) or not 0 <= value < 1 for value in deltas):
        raise ValueError("Windows must be finite and in [0, 1).")
    reference = EditConfig(method="base", start=0.90, delta=0.15, end=end, cleanup=cleanup, noise=noise)
    reference.validate()
    configs = {"native": asdict(reference)}
    skipped = []
    valid = 0
    for start in sorted(set(starts)):
        for delta in sorted(set(deltas)):
            if delta >= start:
                skipped.append({"start": start, "delta": delta})
                continue
            config = EditConfig(method=method, start=start, delta=delta, end=end, cleanup=cleanup, noise=noise)
            config.validate()
            value = asdict(config)
            valid += 1
            if value not in configs.values():
                configs[f"trial{len(configs):03d}"] = value
    if not valid:
        raise ValueError("No valid time pairs remain with delta below start.")
    return configs, skipped

def subset(items, limit, seed=17):
    # Balance a pilot subset across available categories.
    if limit is None or limit >= len(items):
        return list(items)
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be positive.")
    categories = {}
    for item in items:
        categories.setdefault(item.category, []).append(item)
    rank = lambda value: hashlib.sha256((str(seed) + ":" + value).encode()).hexdigest()
    categories = {key: sorted(value, key=lambda item: rank(item.key)) for key, value in categories.items()}
    order = sorted(categories, key=rank)
    result = []
    while len(result) < limit:
        for category in order:
            if categories[category] and len(result) < limit:
                result.append(categories[category].pop(0))
    return result

def mean(values):
    values = [float(value) for value in values if value is not None]
    return statistics.fmean(values) if values else None

def interval(values, groups, repeats=2000, seed=17):
    # Resample source groups after averaging their repeated seeds.
    if len(values) != len(groups) or not values or repeats < 1:
        raise ValueError("Bootstrap inputs are invalid.")
    clusters = {}
    for value, group in zip(values, groups):
        clusters.setdefault(group, []).append(value)
    if len(clusters) < 2:
        return None
    groups = list(clusters.values())
    generator = random.Random(seed)
    samples = sorted(statistics.fmean(value for group in generator.choices(groups, k=len(groups)) for value in group) for _ in range(repeats))
    return [samples[int(0.025 * (repeats - 1))], samples[int(0.975 * (repeats - 1))]]

def summarize(rows, configs, budget=1.0, risk=0.1, locked=None, repeats=2000, bandbudget=None):
    # Select using development data and grouped seeds.
    if not math.isfinite(budget) or budget < 0 or not math.isfinite(risk) or not 0 <= risk <= 1:
        raise ValueError("budget and risk are invalid.")
    if bandbudget is not None and (not math.isfinite(bandbudget) or bandbudget < 0):
        raise ValueError("The band damage budget must be finite and nonnegative.")
    buckets = {name: {} for name in configs}
    for row in rows:
        key = (row["key"], row["seed"])
        if row["config"] not in buckets or key in buckets[row["config"]]:
            raise ValueError("Unknown configurations or duplicate rows were found.")
        for name in ("semantic", "global", "local", "mse", "psnr", "seconds", "calls", "samples"):
            if row.get(name) is None and name in {"local", "mse", "psnr"}:
                continue
            if not isinstance(row.get(name), (int, float)) or not math.isfinite(row[name]):
                raise ValueError("Scores must be finite and complete.")
        buckets[row["config"]][key] = row
    reference = buckets.get("native", {})
    if not reference or any(set(bucket) != set(reference) for bucket in buckets.values()):
        raise ValueError("Every candidate must cover the same image and seed pairs.")
    table = []
    for name, bucket in buckets.items():
        records = {}
        drops = []
        banddrops = {"low": [], "high": []}
        bandviolations = []
        for key, row in bucket.items():
            base = reference[key]
            if row["group"] != base["group"] or row["category"] != base["category"]:
                raise ValueError("Candidate metadata is inconsistent.")
            if (row["mse"] is None) != (base["mse"] is None) or (row["psnr"] is None) != (base["psnr"] is None):
                raise ValueError("Candidate protection coverage is inconsistent.")
            entry = records.setdefault(key[0], {"group": row["group"], "category": row["category"], "gains": []})
            entry["gains"].append(row["semantic"] - base["semantic"])
            if row["psnr"] is not None:
                drops.append(base["psnr"] - row["psnr"])
            if bandbudget is not None:
                point = []
                for band in ("low", "high"):
                    for candidate in (row, base):
                        if candidate.get("frequency", {}).get("space") != "rgb" or "protected" not in candidate["frequency"].get(band, {}):
                            raise ValueError("Band constraints require complete RGB frequency diagnostics.")
                    after = row["frequency"][band]["protected"]
                    before = base["frequency"][band]["protected"]
                    if (after is None) != (before is None):
                        raise ValueError("Band protection coverage is inconsistent.")
                    if after is not None:
                        if any(not math.isfinite(value) or value < 0 for value in (before, after)):
                            raise ValueError("Band distortion must be finite and nonnegative.")
                        change = 10.0 * math.log10((after + 1e-12) / (before + 1e-12))
                        banddrops[band].append(change)
                        point.append(change)
                if point:
                    bandviolations.append(float(max(point) > bandbudget + 1e-9))
        gains = [mean(entry["gains"]) for entry in records.values()]
        violation = mean([float(value > budget + 1e-9) for value in drops])
        feasible = not drops or (mean(drops) <= budget + 1e-9 and violation <= risk + 1e-9)
        bandviolation = mean(bandviolations)
        if bandviolations:
            feasible = feasible and bandviolation <= risk + 1e-9 and all(mean(value) <= bandbudget + 1e-9 for value in banddrops.values() if value)
        categories = {}
        for entry in records.values():
            categories.setdefault(entry["category"], []).append(mean(entry["gains"]))
        scores = {metric: mean(row[metric] for row in bucket.values()) for metric in ("semantic", "global", "local", "mse", "psnr", "seconds", "calls", "samples")}
        table.append({"id": name, "config": configs[name], **scores, "gain": mean(gains), "interval": interval(gains, [entry["group"] for entry in records.values()], repeats), "psnrdrop": mean(drops), "violation": violation, "lowdrop": mean(banddrops["low"]), "highdrop": mean(banddrops["high"]), "bandviolation": bandviolation, "bandpairs": len(bandviolations), "feasible": feasible, "protectedpairs": len(drops), "records": len(records), "groups": len({entry["group"] for entry in records.values()}), "wins": sum(value > 1e-8 for value in gains), "losses": sum(value < -1e-8 for value in gains), "categories": {key: {"records": len(value), "gain": mean(value)} for key, value in categories.items()}})
    feasible = [row for row in table if row["feasible"]]
    chosen = locked or min(feasible, key=lambda row: (-row["gain"], row["id"] != "native", row["mse"] or 0.0, row["id"]))["id"]
    if chosen not in configs:
        raise ValueError("The frozen candidate is missing.")
    pareto = []
    for row in table:
        dominated = any(other["semantic"] >= row["semantic"] and (other["mse"] or 0.0) <= (row["mse"] or 0.0) and (other["semantic"] > row["semantic"] or (other["mse"] or 0.0) < (row["mse"] or 0.0)) for other in table)
        if not dominated:
            pareto.append(row["id"])
    return {"selected": chosen, "locked": locked is not None, "budget": budget, "risk": risk, "bandbudget": bandbudget, "pareto": pareto, "table": table, "warning": "CLIP proxy selection is not a proof of optimality; development intervals are exploratory."}

class Metric:
    def __init__(self, judge, original, target, region, local=0.5):
        if not math.isfinite(local) or not 0 <= local <= 1:
            raise ValueError("The local score weight must be in [0, 1].")
        self.judge = judge
        self.original = original.convert("RGB")
        self.pixels = np.asarray(self.original, dtype=np.float64) / 255.0
        self.mask = np.asarray(region.convert("L")) > 127
        if self.mask.shape != self.pixels.shape[:2] or not self.mask.any():
            raise ValueError("Evaluation requires an aligned, nonempty edit mask.")
        rows, columns = np.nonzero(self.mask)
        self.box = (int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1)
        self.local = local
        with torch.no_grad():
            self.text = judge.texts([target])[0]

    @torch.no_grad()
    def __call__(self, image):
        # Score fixed crops and complete images.
        if image.size != self.original.size:
            raise ValueError("Candidate and source sizes must agree.")
        pixels = np.asarray(image.convert("RGB"))
        focused = Image.fromarray(np.where(self.mask[..., None], pixels, 127).astype(np.uint8)).crop(self.box)
        scores = (self.judge.images([image, focused]) @ self.text).float().cpu().tolist()
        globalvalue, localvalue = (100.0 * value for value in scores)
        protected = ~self.mask
        mse = float(np.square(pixels.astype(np.float64) / 255.0 - self.pixels)[protected].mean()) if protected.any() else None
        psnr = -10.0 * math.log10(max(mse, 1e-12)) if mse is not None else None
        return {"global": globalvalue, "local": localvalue, "semantic": (1.0 - self.local) * globalvalue + self.local * localvalue, "mse": mse, "psnr": psnr}

def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)

@torch.inference_mode()
def collect(editor, judge, items, configs, seeds, groups, local=0.5, completed=None, scan="time"):
    # Keep evaluation masks out of model inference.
    if scan == "bands":
        from engine.spectral import collect as spectral
        yield from spectral(editor, judge, items, configs, seeds, groups, local, completed)
        return
    if scan != "time":
        raise ValueError("Unknown time search mode.")
    completed = completed or set()
    for item in items:
        pending = [(seed, name) for seed in seeds for name in configs if (item.key, seed, name) not in completed]
        if not pending:
            continue
        from engine.format import orient, read
        source = read(item.image)
        region = annotation(item)
        if region is None:
            raise ValueError("Every evaluation record needs an edit mask.")
        region = region.resize(source.size, Image.Resampling.NEAREST)
        source, region = orient(source, region)
        original = ImageOps.fit(source, (editor.size, editor.size), method=Image.Resampling.LANCZOS)
        region = ImageOps.fit(region, original.size, method=Image.Resampling.NEAREST)
        metric = Metric(judge, original, item.target, region, local)
        originalscore = metric(original)["semantic"]
        for seed, name in pending:
            synchronize(editor.device)
            started = time.perf_counter()
            result = editor.edit(source, item.source, item.target, EditConfig(**configs[name]), seed)
            synchronize(editor.device)
            elapsed = time.perf_counter() - started
            synchronize(judge.device)
            started = time.perf_counter()
            scores = metric(result.images[0])
            synchronize(judge.device)
            scoring = time.perf_counter() - started
            if not all(value is None or math.isfinite(value) for value in scores.values()):
                raise RuntimeError("Nonfinite candidate metrics were found.")
            report = result.report or {}
            yield {"key": item.key, "group": groups[item.key], "category": item.category, "seed": seed, "config": name, **scores, "original": originalscore, "editgain": scores["semantic"] - originalscore, "seconds": elapsed, "scoreseconds": scoring, "calls": report.get("calls", editor.calls), "samples": report.get("samples", editor.samples)}

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
