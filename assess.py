# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image
from data import clean, locate, records
from dataset import annotation
from engine.metrics import Protocol
from evaluate import summary
from run import writejson
from engine.format import read
from engine.serial import read as readjson

def parser():
    result = argparse.ArgumentParser(description="Score saved editing outputs with the public benchmark protocol.")
    result.add_argument("--input", required=True)
    result.add_argument("--dataset", default="./pie")
    result.add_argument("--mapping", default="./mapping_file.json")
    result.add_argument("--images", default="./annotation_images")
    result.add_argument("--protocol", default=None)
    result.add_argument("--device", default="cpu")
    result.add_argument("--network", action="store_true")
    result.add_argument("--output", default="./assessment.json")
    return result

def grouped(rows, field):
    groups = {}
    for row in rows:
        groups.setdefault(str(row["key"]), []).append(row)
    # Average seeds within each source before aggregating metrics.
    means = [{"metrics": {key: value["mean"] for key, value in summary([{"metrics": row[field]} for row in values]).items()}} for values in groups.values()]
    result = summary(means)
    infinite = {key for key, values in groups.items() if any(row.get("status", {}).get("psnr") == "positive infinity" for row in values)}
    if infinite:
        finite = sum(any(row[field].get("psnr") is not None for row in values) for key, values in groups.items() if key not in infinite)
        result["psnr"] = {"mean": None, "count": finite + len(infinite), "infinite": len(infinite), "status": "positive infinity"}
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
    destination = clean(args.output, "output")
    if destination.exists():
        raise FileExistsError(destination)
    payload = readjson(source)
    if not payload.get("rows") or any("configuration" in row for row in payload["rows"]):
        raise ValueError("Input must be an evaluation summary for one method, not a time-search study.")
    items = {item.key: item for item in records(args.dataset, args.mapping, args.images)}
    if any(item.benchmark != "pie" for item in items.values()):
        raise ValueError("This scoring interface implements the PIE protocol only.")
    backend = Protocol.load(args.protocol, args.device, args.network)
    output, seen = [], set()
    for row in payload["rows"]:
        pair = (str(row["key"]), int(row["seed"]))
        if pair in seen or pair[0] not in items:
            raise ValueError("Assessment rows must be unique and present in the dataset.")
        seen.add(pair)
        item = items[pair[0]]
        prediction = locate(source.parent, clean(row["image"], "prediction"))
        edited = read(prediction)
        if edited.getexif().get(274, 1) != 1:
            raise ValueError("Official scoring requires normalized output orientation.")
        if edited.size != (512, 512):
            raise ValueError("The benchmark protocol requires 512 by 512 outputs.")
        image = read(item.image)
        if image.size != (512, 512) or item.masksize != (512, 512):
            raise ValueError("Official scoring does not silently rescale benchmark geometry.")
        if image.getexif().get(274, 1) != 1:
            raise ValueError("Normalize source EXIF orientation before the official benchmark export.")
        settings = row.get("report", {})
        if settings.get("target", item.target) != item.target or settings.get("source", item.source) != item.source:
            raise ValueError("Generated prompts no longer match the evaluation dataset.")
        if settings.get("imagesha256"):
            import numpy as np
            sourcepixels = (np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255).copy(order="C")
            if hashlib.sha256(sourcepixels.tobytes()).hexdigest() != settings["imagesha256"]:
                raise ValueError("Generated source pixels no longer match the evaluation dataset.")
        mask = annotation(item, border=True)
        result = backend.evaluate(image, edited, mask, item.target)
        output.append({"key": item.key, "seed": pair[1], "category": item.category, "image": row["image"], **result})
        print(json.dumps({"key": item.key, "seed": pair[1], "table": result["table"]}, allow_nan=False), flush=True)
    result = {"schema": 1, "protocol": "public PIE metric interface", "settings": vars(args), "inputsha256": hashlib.sha256(source.read_bytes()).hexdigest(), "backend": backend.metadata, "raw": grouped(output, "raw"), "table": grouped(output, "table"), "rows": output, "note": "Seeds are averaged per image; undefined and infinite PSNR values remain null with explicit status."}
    result["coverage"] = coverage(output)
    writejson(destination, result)

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
