# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
from data import clean, locate
from run import writejson
from engine.protocol import changes
from evaluate import aggregate, declared, intervals, state

def read(path):
    from engine.serial import read as readjson
    payload = readjson(locate(Path("."), clean(path, "summary")))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or payload.get("schema") != 1 or not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty evaluation summary.")
    declared(payload)
    return payload

def values(row):
    report = row.get("report", {})
    result = {"seconds": row.get("seconds"), "vram": row.get("vram")}
    result.update({"metric:" + key: value for key, value in row.get("metrics", row.get("raw", {})).items()})
    field = report.get("field", {})
    result.update({"field:" + key: field.get(key) for key in ("rank", "actualfitmean", "relativefit", "responsecosine", "modelgapmean", "relativegap")})
    result.update({"field:input" + key: value for key, value in field.get("alignment", {}).items() if key != "scope"})
    result.update({"field:response" + key: field.get("capacity", {}).get(key) for key in ("rank", "lineerror", "spanerror", "directionroom", "linecosine")})
    result.update({"field:temporal" + key: field.get("temporal", {}).get(key) for key in ("rank", "spanerror", "coneerror", "strength", "weight")})
    result.update({"field:repair" + key: field.get("repair", {}).get(key) for key in ("accepted", "endpoints", "initialerror", "finalerror", "movement", "calls", "samples")})
    for key in ("gain", "damagechange", "evaluations"):
        result["selection:" + key] = report.get(key)
    for key in ("interactionenergy", "interactionratio", "damageinteraction", "damageoverlap", "damagenonlinear"):
        result["interaction:" + key] = report.get("interaction", {}).get(key)
    for key in ("rgb", "score"):
        result["calibration:" + key] = report.get("calibration", {}).get(key)
    for key in ("rank", "nullity", "semanticrank", "norm", "lockerror", "violation", "initial", "predicted", "objective", "modeldecrease", "descent", "faces", "feasiblefaces"):
        result["projection:" + key] = report.get("solver", {}).get(key)
    if report.get("method") == "projection":
        result["projection:directions"] = len(report.get("basis", {}).get("atoms", []))
        verified = report.get("verification", [])
        result["projection:rejection"] = sum(not row["accepted"] for row in verified) / len(verified) if verified else None
        result["projection:updates"] = report.get("feedback", {}).get("updates")
        result["projection:selectedmodel"] = report.get("selectedmodel")
        for key in ("pixelerror", "semanticerror", "semanticresponseerror", "responseerror", "responsedefect", "semanticdefect", "initialresponseerror", "initialsemanticerror"):
            observed = [row[key] for row in verified if key in row and math.isfinite(row[key])]
            result["projection:" + key] = float(np.mean(observed)) if observed else None
        for key in ("responseimprovement", "semanticimprovement"):
            observed = [row[key] for row in verified if row.get("model", 0) > 0 and key in row and math.isfinite(row[key])]
            result["projection:" + key] = float(np.mean(observed)) if observed else None
        for key in ("rgb", "protected", "relative", "transport", "refinementchange", "overlap", "nonlinear", "damageinteraction", "identityerror"):
            result["factorial:" + key] = report.get("factorial", {}).get(key)
        factorial = report.get("factorial", {})
        if isinstance(factorial.get("singlevalidjointinvalid"), bool):
            result["factorial:singlevalidjointinvalid"] = int(factorial["singlevalidjointinvalid"])
        for key, value in factorial.get("semanticinteraction", {}).items():
            result["factorial:semantic:" + key] = value
    if report.get("method") == "path":
        for key in ("damage", "rawdamage", "damagechange", "evaluations"):
            result["path:" + key] = report.get(key)
        candidates = report.get("candidates", [])
        result["path:rejection"] = sum(not row["feasible"] for row in candidates) / len(candidates) if candidates else None
        result["path:improved"] = int(not report.get("fallback", True))
        result["path:priorqueries"] = report.get("prior", {}).get("evaluations")
        result["path:heldout"] = report.get("heldout", {}).get("value")
        history = report.get("history", [])
        for key in ("bestgain", "gainfloor", "selectedgain", "gainloss", "bandcount", "eligible"):
            result["attainment:" + key] = report.get("attainment", {}).get(key)
        result["attainment:queries"] = sum(row.get("phase") == "attainment" for row in history)
        released = [row for row in history if row.get("kind") == "release"]
        result["release:queries"] = len(released)
        result["release:accepted"] = sum(row.get("accepted") is True for row in released)
        result["release:predictedgain"] = float(np.mean([row["solver"]["gain"] for row in released])) if released else None
        result["release:responseerror"] = float(np.mean([row["responseerror"] for row in released if "responseerror" in row])) if any("responseerror" in row for row in released) else None
        for key in ("predictedgain", "actualgain", "predicteddamagechange", "actualdamagechange"):
            measured = [row["verification"][key] for row in released if isinstance(row.get("verification", {}).get(key), (int, float)) and math.isfinite(row["verification"][key])]
            result["release:verified" + key] = float(np.mean(measured)) if measured else None
        for key in ("protectednorm", "uncompensatednorm", "damagecross", "damagequadratic", "repaircredit"):
            measured = [row["solver"][key] for row in released if key in row.get("solver", {})]
            result["release:" + key] = float(np.mean(measured)) if measured else None
        for geometry in ("output", "latent", "attenuate"):
            measured = [row for row in released if row.get("solver", {}).get("geometry") == geometry]
            result["geometry:" + geometry + ":queries"] = len(measured)
            result["geometry:" + geometry + ":accepted"] = sum(row.get("accepted") is True for row in measured)
        common = report.get("commonpool", {})
        if "attain" in common and "protect" in common:
            result["attainment:commonpoolchange"] = int(common["attain"]["selected"] != common["protect"]["selected"])
            result["attainment:commonpoolgain"] = common["attain"]["selectedgain"] - common["protect"]["selectedgain"]
        leverage = [row["leverage"] for row in history if row.get("leverage", {}).get("geometryresolved") and row["leverage"].get("fittrusted")]
        for key in ("jointnovelty", "protectionuncancellable", "protectionresidual", "directionnorm", "damageslope", "damagecurvature"):
            measured = [row[key] for row in leverage if isinstance(row.get(key), (int, float)) and math.isfinite(row[key])]
            result["leverage:" + key] = float(np.mean(measured)) if measured else None
        result["leverage:trustedmodels"] = len(leverage)
        result["leverage:queries"] = sum(row.get("kind") == "compensation" for row in history)
        result["path:initialqueries"] = report.get("initialization", {}).get("evaluated")
        result["path:initialcomplete"] = int(report.get("initialization", {}).get("complete", False))
        result["rescue:queries"] = report.get("rescue", {}).get("queries")
        result["rescue:accepted"] = report.get("rescue", {}).get("accepted")
        for key in ("considered", "eligible", "proposed", "budgetblocked"):
            result["rescue:" + key] = report.get("rescue", {}).get(key)
        for key in ("evaluations", "positiveexcess", "penalized", "vetoed"):
            result["prior:" + key] = report.get("prioractivity", {}).get(key)
        if isinstance(report.get("prioractivity", {}).get("selectionchanged"), bool):
            result["prior:selectionchanged"] = int(report["prioractivity"]["selectionchanged"])
        initial = report.get("initialleverage") or {}
        for key in ("jointnovelty", "protectionuncancellable", "protectionresidual", "directionnorm", "damageslope", "damagecurvature"):
            if initial.get("geometryresolved") and initial.get("fittrusted"):
                result["initialleverage:" + key] = initial.get(key)
        for name, observed in initial.get("neutralgain", {}).items():
            if initial.get("geometryresolved") and initial.get("fittrusted"):
                result["initialleverage:neutral:" + name] = observed.get("increment")
        frontier = initial.get("frontier", {})
        result["frontier:iterations"] = frontier.get("iterations")
        balanced = frontier.get("balanced", {})
        if balanced.get("verified"):
            for key in ("incrementlower", "incrementupper", "iterations"):
                result["frontier:balanced:" + key] = balanced.get(key)
        for name, observed in frontier.get("components", {}).items():
            if observed.get("verified"):
                for key in ("incrementlower", "incrementupper"):
                    result["frontier:" + name + ":" + key] = observed.get(key)
                result["frontier:" + name + ":positive"] = int(observed.get("positiveidentified", False))
        for name in {name for row in leverage for name in row.get("neutralgain", {})}:
            measured = [row["neutralgain"][name]["increment"] for row in leverage if name in row.get("neutralgain", {})]
            result["leverage:neutral:" + name] = float(np.mean(measured)) if measured else None
        certificates = [row["certificate"] for row in history if "certificate" in row]
        result["path:localinfeasible"] = sum(row.get("infeasible", False) for row in certificates)
        result["path:certificatemargin"] = max([row.get("normalizedmargin", 0.) for row in certificates] + [0.])
        result["path:solveriterations"] = sum(row.get("solver", {}).get("iterations", 0) for row in history)
        result["path:certificateiterations"] = sum(row.get("iterations", 0) for row in certificates)
        result["path:certifiedproposals"] = sum(row.get("solver", {}).get("optimalitycertified", False) for row in history)
        result["path:curvaturealerts"] = report.get("curvature", {}).get("alerts")
        for key in ("bias", "condition", "field", "interaction"):
            result["path:mechanism:" + key] = report.get("mechanism", {}).get("latent", {}).get(key)
        result["path:mechanism:identity"] = report.get("mechanism", {}).get("identityerror")
    for key in ("calls", "samples", "decodes", "encodes", "textcalls"):
        result["cost:" + key] = report.get("stats", {}).get(key)
    for key in ("rawdamage", "candidatedamage", "damagechange"):
        result["reference:" + key] = report.get("referenceguard", {}).get(key)
    for key in ("accepted", "semanticvalid", "improves"):
        value = report.get("referenceguard", {}).get(key)
        result["reference:" + key] = int(value) if isinstance(value, bool) else None
    for key in ("reconstruction", "refinementdrift", "zeropixelerror", "saturation", "rawdamage", "centerdamage", "neutraldamage"):
        result["control:" + key] = report.get("control", {}).get(key)
    for key in ("intercept", "response", "cross", "total", "identityerror"):
        result["control:protected:" + key] = report.get("control", {}).get("protected", {}).get(key)
    return {key: value if key.startswith("metric:") else float(value) for key, value in result.items() if key.startswith("metric:") or isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)}

def observations(rows):
    if any(not isinstance(row, dict) or not isinstance(row.get("key"), str) or type(row.get("seed")) is not int for row in rows):
        raise ValueError("Audit observations require request keys and integer seeds.")
    return [{"key": row["key"], "seed": row["seed"], "group": row.get("group", row["key"]), "metrics": values(row), "status": {"metric:" + key: value for key, value in row.get("status", {}).items()}} for row in rows]

def summarize(rows):
    point = aggregate(observations(rows))
    grouped = {}
    for row in rows:
        for key, value in values(row).items():
            if value is not None and math.isfinite(value):
                grouped.setdefault(key, {}).setdefault(str(row["key"]), []).append(value)
    result = {}
    for key, metric in point["metrics"].items():
        groups = grouped.get(key, {})
        means = np.array([np.mean(group) for group in groups.values()])
        result[key] = {**metric, "median": float(np.median(means)) if len(means) else None, "images": metric["count"], "observations": sum(map(len, groups.values()))}
    states = {}
    for key in ("fallback", "trusted", "constraintmet", "protectedvalid"):
        observed = [row.get("report", {}).get(key) for row in rows]
        states[key] = {"true": sum(value is True for value in observed), "false": sum(value is False for value in observed), "missing": sum(not isinstance(value, bool) for value in observed)}
    paths = [row.get("report", {}) for row in rows if row.get("report", {}).get("method") == "path"]
    initial = [report.get("initialleverage") for report in paths]
    present = [record for record in initial if isinstance(record, dict)]
    active = [record for record in present if record.get("active") is True]
    trusted = sum(record.get("geometryresolved") is True and record.get("fittrusted") is True for record in active)
    coverage = {"pathrows": len(paths), "recorded": len(present), "missing": len(paths) - len(present), "inactive": len(present) - len(active), "untrusted": len(active) - trusted, "usable": trusted, "fraction": trusted / len(paths) if paths else None, "scope": "First fitted model only; coverage counts observations, not independent source groups."}
    frontierrecords = [(record, record["frontier"]) for record in present if isinstance(record.get("frontier"), dict)]
    frontiers = [record for _, record in frontierrecords]
    checked = [record for record in frontiers if record.get("components") and all(value.get("verified") is True for value in record["components"].values())]
    partial = sum(bool(record.get("components")) and any(value.get("verified") is True for value in record["components"].values()) and record not in checked for record in frontiers)
    unresolved = sum(parent.get("fittrusted") is True and record.get("active") is True and not record.get("components") for parent, record in frontierrecords)
    unverified = sum(bool(record.get("components")) and not any(value.get("verified") is True for value in record["components"].values()) for record in frontiers)
    frontiercoverage = {"pathrows": len(paths), "recorded": len(frontiers), "unrecorded": len(paths) - len(frontiers), "inactive": sum(record.get("active") is False for record in frontiers), "untrusted": sum(parent.get("fittrusted") is not True for parent, _ in frontierrecords), "unresolved": unresolved, "unverified": unverified, "partiallyverified": partial, "fullyverified": len(checked), "fraction": len(checked) / len(paths) if paths else None, "scope": "First-model observation counts; status categories can overlap; disabled or unresolved entries are not zero capacity."}
    return {"metrics": result, "byseed": point["byseed"], "seeds": point["seeds"], "aggregation": point["aggregation"], "states": states, "diagnostics": {"initialleverage": coverage, "frontier": frontiercoverage}, "note": "Means average requests within each seed, then seeds; medians summarize per-request seed averages."}

def paired(left, right, repeats=2000, seed=42):
    if type(repeats) is not int or not 1 <= repeats <= 100000:
        raise ValueError("Bootstrap repeats must be from one to one hundred thousand.")
    declared({"rows": left})
    declared({"rows": right})
    first = {(str(row["key"]), int(row["seed"])): row for row in left}
    second = {(str(row["key"]), int(row["seed"])): row for row in right}
    if set(first) != set(second) or len(first) != len(left) or len(second) != len(right) or not first:
        raise ValueError("Paired summaries require identical unique source and seed sets.")
    measured, sourcegroups = [], {}
    for identity in sorted(first):
        a, b = first[identity], second[identity]
        group = str(a.get("group", identity[0]))
        if group != str(b.get("group", identity[0])):
            raise ValueError("Paired summaries disagree on source grouping.")
        if identity[0] in sourcegroups and sourcegroups[identity[0]] != group:
            raise ValueError("Source grouping must remain fixed across seeds.")
        sourcegroups[identity[0]] = group
        for key in ("source", "target", "imagesha256", "masksha256"):
            if a.get("report", {}).get(key) != b.get("report", {}).get(key):
                raise ValueError("Paired summaries differ in source, prompt, or mask provenance.")
        va, vb = observations([a])[0], observations([b])[0]
        record = {"key": identity[0], "seed": identity[1], "group": group, "metrics": {}, "status": {}}
        for key in va["metrics"].keys() | vb["metrics"].keys():
            sa, sb = state(va, key), state(vb, key)
            value = None if not sa or not sb or sa == sb and sa in {-1, 1} else (vb["metrics"][key] if sb == 2 else math.copysign(math.inf, sb)) - (va["metrics"][key] if sa == 2 else math.copysign(math.inf, sa))
            record["metrics"][key] = value if value is None or math.isfinite(value) else None
            if value is not None and not math.isfinite(value):
                record["status"][key] = "positive infinity" if value > 0 else "negative infinity"
        measured.append(record)
    point, bounds = aggregate(measured), intervals(measured, repeats, seed)
    result = {}
    for key, value in point["metrics"].items():
        valid = [row for row in measured if state(row, key)]
        result[key] = {**{name: value for name, value in value.items() if name != "mean"}, "difference": value["mean"], "interval": bounds[key], "images": len({row["group"] for row in valid}), "instances": value["count"], "pairs": len(valid)}
    provenance = all(all(key in row.get("report", {}) for key in ("source", "target", "imagesha256", "masksha256")) for row in [*left, *right])
    return {"direction": "comparison minus reference", "bootstrap": "source-grouped seed-first percentile intervals; not corrected for multiple comparisons", "provenancecomplete": provenance, "repeats": repeats, "seed": seed, "metrics": result, "byseed": point["byseed"], "seeds": point["seeds"], "aggregation": point["aggregation"]}

def agreement(left, right):
    # Check exact output equality instead of rounded metric equality.
    declared({"rows": left})
    declared({"rows": right})
    first = {(str(row["key"]), int(row["seed"])): row for row in left}
    second = {(str(row["key"]), int(row["seed"])): row for row in right}
    if len(first) != len(left) or len(second) != len(right) or first.keys() != second.keys() or not first:
        raise ValueError("Output agreement requires complete unique matched requests.")
    result = {"pairs": len(first), "hashed": 0, "identical": 0, "different": 0, "missinghash": 0, "samecandidatepath": 0, "recordedcandidatepath": 0}
    for key, row in first.items():
        other = second[key]
        hashes = row.get("resultsha256"), other.get("resultsha256")
        valid = all(isinstance(value, str) and len(value) == 64 and all(letter in "0123456789abcdef" for letter in value) for value in hashes)
        result["hashed" if valid else "missinghash"] += 1
        if valid:
            result["identical" if hashes[0] == hashes[1] else "different"] += 1
        reports = row.get("report", {}), other.get("report", {})
        if all(isinstance(report.get("candidates"), list) for report in reports):
            result["recordedcandidatepath"] += 1
            result["samecandidatepath"] += [value.get("point") for value in reports[0]["candidates"]] == [value.get("point") for value in reports[1]["candidates"]]
    result["scope"] = "Stored PNG hashes and visited coordinates; equal aggregate scores do not imply equal images."
    return result

def main():
    parser = argparse.ArgumentParser(description="Audit measured interactions and paired held-out outcomes.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--compare", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    reference = read(args.summary)
    result = {"schema": 1, "split": reference.get("split"), "summary": summarize(reference["rows"])}
    if args.compare:
        comparison = read(args.compare)
        if reference.get("split") != comparison.get("split") or reference.get("oracle") != comparison.get("oracle"):
            raise ValueError("Paired summaries must use the same split and mask protocol.")
        result["paired"] = paired(reference["rows"], comparison["rows"], args.repeats, args.seed)
        result["agreement"] = agreement(reference["rows"], comparison["rows"])
        identities = [payload.get("controlsignature") for payload in (reference, comparison)]
        result["configuration"] = {"recorded": all(isinstance(value, dict) and isinstance(value.get("settings"), dict) for value in identities)}
        if result["configuration"]["recorded"]:
            result["configuration"].update({"reference": identities[0]["sha256"], "comparison": identities[1]["sha256"], "changes": changes(identities[0]["settings"], identities[1]["settings"])})
    if args.output:
        output = clean(args.output, "output")
        if output.exists():
            raise FileExistsError(output)
        writejson(output, result)
    print(json.dumps(result, indent=2, allow_nan=False))

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
