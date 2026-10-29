# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import hashlib
import json
import math
from pathlib import Path
from audit import paired, read
from data import clean, destination, locate
from run import jsonfile, writejson

PROVENANCE = ("source", "target", "imagesha256", "masksha256", "fieldsha256", "protectionsha256", "modelcheckpoint", "adaptersha256", "configuration")
CONTROLS = ("backbone", "size", "noise", "start", "delta", "end", "scale", "cleanup", "poststeps", "guidance", "cfg", "referencepath", "precision", "localization", "deterministic", "field", "fieldridge", "fieldprobe", "fieldradius", "fieldtolerance", "fieldgeometry")
TASKS = ("success", "preserved", "artifact", "naturalness", "jointsuccess")

def identities(rows):
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("key"), str) or type(row.get("seed")) is not int:
            raise ValueError("Evidence rows require string keys and integer seeds.")
        identity = (row["key"], row["seed"])
        if identity in result:
            raise ValueError("Duplicate evidence identities are unsupported.")
        result[identity] = row
    return result

def taskrows(summary, path, annotations):
    # Verify independent labels against unchanged frozen outputs.
    labels = jsonfile(annotations)
    summarypath = locate(Path("."), clean(path, "summary"))
    if not isinstance(labels, dict) or labels.get("schema") != 1 or labels.get("split") != summary.get("split") or labels.get("summaryhash") != hashlib.sha256(summarypath.read_bytes()).hexdigest():
        raise ValueError("Task reports must bind to this exact evaluation summary.")
    observed = labels.get("observations")
    if not isinstance(observed, list):
        raise ValueError("Task reports require individual frozen observations.")
    reference, tasks = identities(summary["rows"]), identities(observed)
    if set(reference) != set(tasks):
        raise ValueError("Task evidence must cover every source and seed.")
    for identity, row in reference.items():
        label = tasks[identity]
        image = locate(summarypath.parent, clean(row["image"], "image"))
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest != row.get("resultsha256") or digest != label.get("imagesha256"):
            raise ValueError("Frozen result bytes differ from task evidence.")
        if row.get("group", identity[0]) != label.get("group", identity[0]):
            raise ValueError("Task grouping differs from evaluation grouping.")
        for key in ("source", "target", "imagesha256", "masksha256"):
            if row.get("report", {}).get(key) != label.get("report", {}).get(key):
                raise ValueError("Task provenance differs from frozen evaluation.")
        metrics = label.get("metrics", {})
        if any(isinstance(metrics.get(key), bool) or not isinstance(metrics.get(key), (int, float)) or not math.isfinite(metrics[key]) for key in TASKS):
            raise ValueError("Task evidence requires all finite primary outcomes.")
        if any(metrics[key] not in (0., 1.) for key in TASKS if key != "naturalness") or not 1 <= metrics["naturalness"] <= 5:
            raise ValueError("Task outcomes have invalid ranges.")
        expected = float(metrics["success"] == 1 and metrics["preserved"] == 1 and metrics["artifact"] == 0)
        if metrics["jointsuccess"] != expected:
            raise ValueError("Joint success must match its three component outcomes.")
    return observed

def compare(reference, comparison, taskreference=None, taskcomparison=None, contrast="reference", repeats=2000, seed=42):
    # Separate mechanism evidence from aggregate benchmark quality.
    if contrast not in {"reference", "outcome", "field"}:
        raise ValueError("Unknown evidence contrast.")
    left, right = read(reference), read(comparison)
    if left.get("split") != right.get("split") or left.get("oracle") != right.get("oracle"):
        raise ValueError("Evidence contrasts require identical split and mask protocols.")
    first, second = identities(left["rows"]), identities(right["rows"])
    if set(first) != set(second):
        raise ValueError("Evidence contrasts require identical source and seed coverage.")
    missing = set()
    intervention = {"pairs": len(first), "active": 0, "samepoint": 0, "outputchanged": 0}
    for identity, row in first.items():
        a, b = row.get("report", {}), second[identity].get("report", {})
        if contrast == "field":
            if a.get("method") != b.get("method"):
                raise ValueError("Field contrasts must use the same execution mode.")
            for key in ("maxsamples", "maxcalls", "seconds", "candidates", "axes", "representation", "algorithm", "release", "geometry", "selection", "prior", "priorprobes"):
                if a.get("settings", {}).get(key) != b.get("settings", {}).get(key):
                    raise ValueError("Field contrasts differ in fixed budget or controller " + key + ".")
        for key in PROVENANCE:
            if contrast == "field" and key == "fieldsha256":
                continue
            if key not in a or key not in b:
                missing.add("missing matched " + key)
            elif a[key] != b[key]:
                raise ValueError("Evidence contrasts differ in matched " + key + ".")
        for key in CONTROLS:
            if contrast == "field" and key in {"field", "fieldridge", "fieldprobe", "fieldradius", "fieldtolerance", "fieldgeometry"}:
                continue
            aa, bb = a.get("settings", {}), b.get("settings", {})
            if key not in aa or key not in bb:
                missing.add("missing matched control " + key)
            elif aa[key] != bb[key]:
                raise ValueError("Evidence contrasts differ in fixed control " + key + ".")
        checkpoint = a.get("modelcheckpoint")
        if not isinstance(checkpoint, dict) or checkpoint.get("weights") is None:
            missing.add("unverified model weight hashes")
        if contrast == "reference":
            for report in (a, b):
                if report.get("method") != "fixed" or report.get("selection") != "unfiltered intervention" or report.get("fallback") is not False or report.get("renderer") != "latent":
                    raise ValueError("Reference attribution requires unfiltered latent interventions.")
                point = report.get("requested")
                if not isinstance(point, list) or len(point) != 3 or point[:2] != [1., 1.]:
                    raise ValueError("Reference attribution must freeze both field strengths.")
            if a["requested"][2] == b["requested"][2]:
                raise ValueError("Reference attribution needs distinct requested reference positions.")
            points = (a.get("point"), b.get("point"))
            if any(not isinstance(point, list) or len(point) != 3 or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in point) for point in points):
                missing.add("actual canonical intervention points")
            else:
                intervention["active"] += int(points[0] != points[1])
                intervention["samepoint"] += int(points[0] == points[1])
            if not row.get("resultsha256") or not second[identity].get("resultsha256"):
                missing.add("frozen output hashes")
            else:
                intervention["outputchanged"] += int(row["resultsha256"] != second[identity]["resultsha256"])
    if left.get("split") != "test":
        missing.add("held-out test split")
    frozenleft, frozenright = left.get("settings", {}).get("frozenrecipehash"), right.get("settings", {}).get("frozenrecipehash")
    if not frozenleft or frozenleft != frozenright:
        missing.add("shared frozen validation recipe hash")
    result = {"schema": 1, "contrast": contrast, "split": left.get("split"), "interventions": intervention if contrast == "reference" else None, "diagnostics": paired(left["rows"], right["rows"], repeats, seed), "tasks": None, "externalvalidity": {"externalauditprovided": False, "scope": "These matched runs do not establish that another editor performs the same reference subtraction or shares its failure mechanism."}}
    if bool(taskreference) != bool(taskcomparison):
        raise ValueError("Provide task reports for both contrasted methods.")
    if taskreference:
        a = taskrows(left, reference, taskreference)
        b = taskrows(right, comparison, taskcomparison)
        result["tasks"] = paired(a, b, repeats, seed)
        if any(value["interval"] is None for value in result["tasks"]["metrics"].values()):
            missing.add("at least two independent source groups")
    else:
        missing.add("postselection independent task annotations")
    result.update({"missing": sorted(missing), "status": "incomplete evidence" if missing else "paired evidence available", "scope": "Evidence coverage and descriptive paired intervals only; no automatic causal, SOTA, noninferiority, or global optimality claim.", "required": "Predeclare contrasts and margins; blind independent raters; disclose failures, quality, mask access, and total cost."})
    return result

def main():
    parser = argparse.ArgumentParser(description="Check matched mechanism and independent task evidence.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--taskreference", default=None)
    parser.add_argument("--taskcomparison", default=None)
    parser.add_argument("--contrast", choices=("reference", "outcome", "field"), default="reference")
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = destination(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = compare(args.reference, args.comparison, args.taskreference, args.taskcomparison, args.contrast, args.repeats, args.seed)
    writejson(output, result)
    print(json.dumps({"status": result["status"], "missing": result["missing"]}, allow_nan=False))

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
