# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import hashlib
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from audit import paired, read
from data import clean, destination, locate
from run import jsonfile, writejson
from engine.format import fit, read as readimage
from evaluate import aggregate, declared, intervals

def build(left, right, output, seed=42):
    # Export anonymous pairs without revealing method identities.
    first, second = read(left), read(right)
    if first.get("split") != second.get("split") or first.get("oracle") != second.get("oracle"):
        raise ValueError("Blind pairs require matching evaluation protocols.")
    audit = paired(first["rows"], second["rows"], repeats=1, seed=seed)
    if not audit["provenancecomplete"]:
        raise ValueError("Blind pairs require complete source provenance.")
    target = destination(output)
    if target.exists():
        raise FileExistsError(target)
    lookup = {(str(row["key"]), int(row["seed"])): row for row in second["rows"]}
    generator = random.Random(seed)
    public, private, panels = [], [], []
    for row in first["rows"]:
        identity = (str(row["key"]), int(row["seed"]))
        other = lookup[identity]
        shape = row["report"].get("imageshape")
        if not isinstance(shape, list) or len(shape) != 4 or shape[:2] != [1, 3]:
            raise ValueError("Blind pairs require fitted source geometry.")
        sourcepath = locate(Path("."), clean(row.get("sourceimage", ""), "source"))
        source = fit(readimage(sourcepath), (shape[3], shape[2]))
        pixels = (np.asarray(source, dtype=np.float32) / 255).transpose(2, 0, 1)[None].copy()
        if hashlib.sha256(pixels.tobytes()).hexdigest() != row["report"]["imagesha256"]:
            raise ValueError("Blind source pixels disagree with evaluation provenance.")
        images = []
        for path, item in ((left, row), (right, other)):
            root = locate(Path("."), clean(path, "summary")).parent
            imagepath = locate(root, clean(item["image"], "result"))
            if item.get("resultsha256") is not None and item["resultsha256"] != hashlib.sha256(imagepath.read_bytes()).hexdigest():
                raise ValueError("Frozen blind-comparison outputs have changed.")
            image = readimage(imagepath)
            if image.getexif().get(274, 1) != 1:
                raise ValueError("Saved blind-comparison outputs must have normalized orientation.")
            if image.size != source.size:
                raise ValueError("Blind result geometry must match the source.")
            images.append(image)
        swap = bool(generator.getrandbits(1))
        a, b = images[::-1] if swap else images
        identifier = hashlib.sha256((str(seed) + ":" + ":".join(map(str, identity))).encode()).hexdigest()[:24]
        panel = Image.new("RGB", (source.width * 3, source.height + 28), "white")
        draw = ImageDraw.Draw(panel)
        for column, (label, image) in enumerate((("Source", source), ("A", a), ("B", b))):
            draw.text((column * source.width + 6, 8), label, fill="black")
            panel.paste(image, (column * source.width, 28))
        panels.append((identifier, panel))
        public.append({"id": identifier, "panel": identifier + ".png", "instruction": row.get("instruction") or row["report"]["target"], "source": row["report"]["source"], "target": row["report"]["target"]})
        private.append({"id": identifier, "key": identity[0], "seed": identity[1], "group": row.get("group", identity[0]), "family": row.get("family", ""), "a": "right" if swap else "left"})
    target.mkdir(parents=True)
    for identifier, panel in panels:
        panel.save(target / (identifier + ".png"))
    generator.shuffle(public)
    writejson(target / "tasks.json", {"schema": 1, "questions": {"completion": "Which output completes the target instruction?", "preservation": "Which output preserves the protected source content?", "naturalness": "Which output looks more natural?"}, "choices": ["A", "B", "tie", "unsure"], "tasks": public})
    writejson(target / "key.json", {"schema": 1, "left": left, "right": right, "seed": seed, "seeds": declared(first), "assignments": private, "note": "Keep this file private from raters."})
    return public, private

def collect(key, responses, repeats=2000, seed=42):
    # Aggregate human ratings after outputs are frozen.
    if type(repeats) is not int or repeats < 1:
        raise ValueError("Bootstrap repeats must be positive.")
    payload, ratings = jsonfile(key), jsonfile(responses)
    if payload.get("schema") != 1 or not isinstance(ratings, list) or not ratings:
        raise ValueError("Expected a private key and nonempty rating list.")
    if not isinstance(payload.get("assignments"), list) or not payload["assignments"] or any(not isinstance(item, dict) or not isinstance(item.get("id"), str) or item.get("a") not in {"left", "right"} for item in payload["assignments"]):
        raise ValueError("Blind assignments require IDs and valid left/right mappings.")
    lookup = {item["id"]: item for item in payload["assignments"]}
    if len(lookup) != len(payload["assignments"]):
        raise ValueError("Blind assignment IDs must be unique.")
    declared({"rows": payload["assignments"], **({"seeds": payload["seeds"]} if "seeds" in payload else {})})
    seen, grouped = set(), {item: {metric: [] for metric in ("completion", "preservation", "naturalness")} for item in lookup}
    for row in ratings:
        if not isinstance(row, dict) or not isinstance(row.get("rater"), str) or not row["rater"].strip() or row.get("id") not in lookup:
            raise ValueError("Ratings need a known case and nonempty rater.")
        identity = (row["rater"].strip(), row["id"])
        if identity in seen:
            raise ValueError("Duplicate rater and case ratings are forbidden.")
        seen.add(identity)
        assignment = lookup[row["id"]]
        for metric in ("completion", "preservation", "naturalness"):
            choice = row.get(metric)
            if choice not in {"A", "B", "tie", "unsure"}:
                raise ValueError("Every question requires an explicit supported choice.")
            if choice == "unsure":
                continue
            right = assignment["a"] == "right" if choice == "A" else assignment["a"] == "left"
            value = .5 if choice == "tie" else float(right)
            grouped[row["id"]][metric].append(value)
    if {identity[1] for identity in seen} != set(lookup):
        raise ValueError("Every frozen blind case needs at least one rating.")
    rows = [{"key": item["key"], "seed": item["seed"], "group": item["group"], "metrics": {metric: float(np.mean(values)) if values else None for metric, values in grouped[identifier].items()}} for identifier, item in lookup.items()]
    point, bounds = aggregate(rows), intervals(rows, repeats, seed)
    result = {metric: {**{key: value for key, value in values.items() if key != "mean"}, "rightpreference": values["mean"], "interval": bounds[metric], "groups": len({row["group"] for row in rows if row["metrics"][metric] is not None})} for metric, values in point["metrics"].items()}
    return {"schema": 1, "metrics": result, "byseed": point["byseed"], "seeds": point["seeds"], "aggregation": point["aggregation"], "raters": len({rater for rater, _ in seen}), "ratings": len(seen), "seed": seed, "repeats": repeats, "scope": "Rater-averaged cases followed by seed-first means and source-group bootstrap; no reselection."}

def main():
    parser = argparse.ArgumentParser(description="Export and aggregate frozen anonymous editing comparisons.")
    parser.add_argument("--left")
    parser.add_argument("--right")
    parser.add_argument("--key")
    parser.add_argument("--responses")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=2000)
    args = parser.parse_args()
    if args.responses:
        if not args.key or args.left or args.right:
            raise ValueError("Rating aggregation requires only key and responses.")
        output = destination(args.output)
        if output.exists():
            raise FileExistsError(output)
        writejson(output, collect(args.key, args.responses, args.repeats, args.seed))
    else:
        if not args.left or not args.right or args.key:
            raise ValueError("Blind export requires left and right summaries.")
        build(args.left, args.right, args.output, args.seed)

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
