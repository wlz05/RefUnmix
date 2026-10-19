# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import csv
import io
import math
from dataclasses import replace, asdict
from engine.model import EditConfig
from engine.timing import grid, interval, mean

def candidates(starts=None, deltas=None, end=0.30, cleanup=True, noise=1):
    # Add both single-time endpoints for every chord candidate.
    starts = [0.20, 0.30, 0.45, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] if starts is None else starts
    deltas = [0.0, 0.15] if deltas is None else deltas
    configs, skipped = grid(starts, deltas, "base", end, cleanup, noise)
    points = sorted({float(f"{point:.12g}") for value in configs.values() for point in (value["start"], value["start"] - value["delta"])})
    reference = EditConfig(**configs["native"])
    for point in points:
        value = asdict(replace(reference, start=point, delta=0.0))
        if not any(config["delta"] == 0 and math.isclose(config["start"], point, rel_tol=1e-12, abs_tol=0.0) for config in configs.values()):
            configs[f"trial{len(configs):03d}"] = value
    return configs, skipped

def point(table, time):
    return next((row for row in table if row["config"]["delta"] == 0 and math.isclose(row["config"]["start"], time, rel_tol=1e-10, abs_tol=1e-12)), None)

def roles(report):
    # Select families under development preservation limits.
    table = report["table"]
    anchor = point(table, 0.75)

    def best(values):
        feasible = [row for row in values if row["feasible"]]
        if not feasible:
            return None
        return min(feasible, key=lambda row: (-row["gain"], row["id"] not in {"native", anchor["id"] if anchor else None}, row["samples"], row["config"]["start"], row["config"]["delta"]))

    single = best([row for row in table if row["config"]["delta"] == 0])
    chord = best([row for row in table if row["config"]["delta"] > 0])
    current = point(table, chord["config"]["start"]) if chord else None
    previous = point(table, chord["config"]["start"] - chord["config"]["delta"]) if chord else None
    native = next(row for row in table if row["id"] == "native")
    values = {"native": native, "anchor": anchor, "single": single, "chord": chord, "current": current, "previous": previous, "nativestart": point(table, native["config"]["start"])}
    return {key: value["id"] if value else None for key, value in values.items()}

def paired(rows, first, second, repeats=2000):
    # Bootstrap paired differences directly.
    left = {(row["key"], row["seed"]): row for row in rows if row["config"] == first}
    right = {(row["key"], row["seed"]): row for row in rows if row["config"] == second}
    if len(left) != sum(row["config"] == first for row in rows) or len(right) != sum(row["config"] == second for row in rows):
        raise ValueError("Paired comparisons cannot contain duplicate rows.")
    if not left or set(left) != set(right):
        raise ValueError("Paired comparisons require matching image and seed pairs.")
    for key in left:
        if left[key]["group"] != right[key]["group"]:
            raise ValueError("Paired source groups disagree.")
    result = {"first": first, "second": second, "direction": "first minus second; positive semantic and PSNR or negative MSE favors first", "metrics": {}}
    for metric in ("semantic", "global", "local", "psnr", "mse"):
        values, groups = {}, {}
        for key, row in left.items():
            before, after = right[key][metric], row[metric]
            if (before is None) != (after is None):
                raise ValueError("Paired metric coverage disagrees.")
            if before is not None:
                if not math.isfinite(before) or not math.isfinite(after):
                    raise ValueError("Paired metrics must be finite.")
                values.setdefault(key[0], []).append(after - before)
                if key[0] in groups and groups[key[0]] != row["group"]:
                    raise ValueError("Repeated seeds must retain their source group.")
                groups[key[0]] = row["group"]
        gains = [mean(value) for value in values.values()]
        result["metrics"][metric] = {"mean": mean(gains), "interval": interval(gains, list(groups.values()), repeats) if gains else None, "records": len(gains)}
    return result

def analyze(report, rows, frozen=None, repeats=2000):
    selected = roles(report) if frozen is None else dict(frozen)
    table = {row["id"]: row for row in report["table"]}
    if set(selected) != {"native", "anchor", "single", "chord", "current", "previous", "nativestart"} or selected["native"] != "native":
        raise ValueError("Time study roles are incomplete or invalid.")
    if any(value is not None and value not in table for value in selected.values()):
        raise ValueError("A frozen time control is missing from the evaluation.")
    curves = {"single": [], "chord": []}
    for row in sorted(table.values(), key=lambda value: (value["config"]["delta"], value["config"]["start"])):
        config = row["config"]
        family = "single" if config["delta"] == 0 else "chord"
        value = {key: row[key] for key in ("id", "semantic", "global", "local", "psnr", "mse", "gain", "interval", "feasible", "violation", "seconds", "samples")}
        value.update({"start": config["start"], "second": float(f"{config['start'] - config['delta']:.12g}"), "delta": config["delta"]})
        curves[family].append(value)
    comparisons = {}
    for name, first, second in (("singlenative", "single", "native"), ("chordnative", "chord", "native"), ("chordsingle", "chord", "single"), ("anchornative", "anchor", "native"), ("chordcurrent", "chord", "current"), ("chordprevious", "chord", "previous")):
        if selected.get(first) is not None and selected.get(second) is not None:
            comparisons[name] = paired(rows, selected[first], selected[second], repeats)
    matched = []
    for row in table.values():
        config = row["config"]
        if config["delta"] <= 0:
            continue
        current, previous = point(list(table.values()), config["start"]), point(list(table.values()), config["start"] - config["delta"])
        matched.append({"chord": row["id"], "start": config["start"], "second": float(f"{config['start'] - config['delta']:.12g}"), "current": current["id"] if current else None, "previous": previous["id"] if previous else None, "currentgain": row["semantic"] - current["semantic"] if current else None, "previousgain": row["semantic"] - previous["semantic"] if previous else None, "currentpsnr": row["psnr"] - current["psnr"] if current and row["psnr"] is not None and current["psnr"] is not None else None, "previouspsnr": row["psnr"] - previous["psnr"] if previous and row["psnr"] is not None and previous["psnr"] is not None else None})
    return {"roles": selected, "locked": frozen is not None, "curves": curves, "comparisons": comparisons, "matched": matched, "primary": "chordsingle.semantic", "scope": "frozen development controls" if frozen is not None else "development grid", "warning": "Best means best among feasible tested candidates, not a universal optimal timestep; secondary intervals are not adjusted for multiple comparisons."}

def export(path, study):
    # Export plotting data without overwriting reports.
    buffer = io.StringIO(newline="")
    fields = ("family", "id", "start", "second", "delta", "semantic", "global", "local", "psnr", "mse", "gain", "lower", "upper", "feasible", "violation", "seconds", "samples")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for family, values in study["curves"].items():
        for value in values:
            bounds = value["interval"] or [None, None]
            writer.writerow({"family": family, "lower": bounds[0], "upper": bounds[1], **{key: item for key, item in value.items() if key != "interval"}})
    content = buffer.getvalue()
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(path)
        return
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(content)

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
