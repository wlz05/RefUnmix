# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import hashlib
import math
from pathlib import Path
import numpy as np
from audit import read
from data import clean, destination, locate
from run import jsonfile, writejson

def template(summary):
    # Bind unrated annotations to frozen image bytes.
    payload = read(summary)
    root = locate(Path("."), clean(summary, "summary")).parent
    labels = []
    for row in payload["rows"]:
        digest = hashlib.sha256(locate(root, clean(row["image"], "image")).read_bytes()).hexdigest()
        if row.get("resultsha256") != digest:
            raise ValueError("Annotation templates require unchanged hashed outputs.")
        labels.append({"key": str(row["key"]), "seed": int(row["seed"]), "imagesha256": digest, "success": None, "preserved": None, "artifact": None, "naturalness": None})
    return labels

def assess(summary, annotations, repeats=2000, seed=42):
    # Score frozen outputs using independent task annotations.
    payload, labels = read(summary), jsonfile(annotations)
    if not isinstance(labels, list) or not labels or type(repeats) is not int or not 1 <= repeats <= 100000:
        raise ValueError("Task labels and bootstrap repeats must be nonempty.")
    lookup = {(str(row["key"]), int(row["seed"])): row for row in payload["rows"]}
    seen, grouped, rows, failures = set(), {}, [], {"proxypassfailure": 0, "priorimprovedfailure": 0, "textureartifact": 0}
    for label in labels:
        if not isinstance(label, dict) or type(label.get("seed")) is not int or not isinstance(label.get("key"), str):
            raise ValueError("Task annotations require explicit source and seed identities.")
        identity = (label["key"], label["seed"])
        if identity not in lookup or identity in seen:
            raise ValueError("Task identities must match unique frozen evaluation rows.")
        seen.add(identity)
        row = lookup[identity]
        image = locate(locate(Path("."), clean(summary, "summary")).parent, clean(row["image"], "result"))
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if label.get("imagesha256") != digest:
            raise ValueError("Task annotations must bind to exact output image bytes.")
        if row.get("resultsha256") is not None and row["resultsha256"] != digest:
            raise ValueError("Frozen result images have changed.")
        if type(label.get("success")) is not bool or type(label.get("preserved")) is not bool or type(label.get("artifact")) is not bool:
            raise ValueError("Task success, preservation, and artifacts need boolean labels.")
        naturalness = label.get("naturalness")
        if isinstance(naturalness, bool) or not isinstance(naturalness, (int, float)) or not math.isfinite(naturalness) or not 1 <= naturalness <= 5:
            raise ValueError("Naturalness ratings must be between one and five.")
        values = {"success": float(label["success"]), "preserved": float(label["preserved"]), "naturalness": float(naturalness), "artifact": float(label["artifact"]), "jointsuccess": float(label["success"] and label["preserved"] and not label["artifact"])}
        rows.append({"key": identity[0], "seed": identity[1], "group": row.get("group", identity[0]), "imagesha256": digest, "metrics": values, "report": {key: row.get("report", {}).get(key) for key in ("source", "target", "imagesha256", "masksha256")}})
        for name, value in values.items():
            grouped.setdefault(name, {}).setdefault(row.get("group", identity[0]), {}).setdefault(identity[0], []).append(value)
        report = row.get("report", {})
        failures["proxypassfailure"] += int(report.get("constraintmet", False) and not label["success"])
        rawprior = report.get("raw", {}).get("prior")
        selected = report.get("raw") if report.get("selected") == "raw" else next((item for item in report.get("candidates", []) if item["key"] == report.get("selected")), None)
        prior = selected.get("prior") if selected else None
        failures["priorimprovedfailure"] += int(rawprior is not None and prior is not None and prior < rawprior and not label["success"])
        failures["textureartifact"] += int(label["artifact"])
    if seen != set(lookup):
        raise ValueError("Every frozen result needs one independent task annotation.")
    generator = np.random.default_rng(seed)
    metrics = {}
    for name, groups in grouped.items():
        means = np.array([np.mean([np.mean(values) for values in sources.values()]) for sources in groups.values()])
        distribution = [float(means[generator.integers(0, len(means), len(means))].mean()) for _ in range(repeats)]
        metrics[name] = {"mean": float(means.mean()), "interval": np.quantile(distribution, [.025, .975]).tolist() if len(means) >= 2 else None, "groups": len(means)}
    return {"schema": 1, "split": payload.get("split"), "metrics": metrics, "failurecounts": failures, "rows": len(seen), "observations": rows, "seed": seed, "repeats": repeats, "summaryhash": hashlib.sha256(locate(Path("."), clean(summary, "summary")).read_bytes()).hexdigest(), "annotationhash": hashlib.sha256(locate(Path("."), clean(annotations, "annotations")).read_bytes()).hexdigest(), "scope": "Postselection task labels; source-grouped bootstrap; failure overlap is not causal attribution."}

def main():
    parser = argparse.ArgumentParser(description="Report independent annotations on frozen editing outputs.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--template", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.template == bool(args.annotations):
        raise ValueError("Choose either an annotation template or completed annotations.")
    output = destination(args.output)
    if output.exists():
        raise FileExistsError(output)
    writejson(output, template(args.summary) if args.template else assess(args.summary, args.annotations, args.repeats, args.seed))

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
