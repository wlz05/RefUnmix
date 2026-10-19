# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from data import clean, locate, records
from engine.serial import read

def identities(items) -> dict[str, str]:
    result = {}
    for item in items:
        with item.image.open("rb") as handle:
            result[item.key] = hashlib.file_digest(handle, "sha256").hexdigest()
    return result

def grouped(items, digests: dict[str, str]) -> list[list[str]]:
    parent = {item.key: item.key for item in items}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    aliases = {}
    for item in items:
        tokens = ["digest:" + digests[item.key], "path:" + item.image.resolve().as_posix().casefold()]
        if item.group:
            tokens.append("group:" + item.group)
        if item.family:
            tokens.append("family:" + item.family)
        for token in tokens:
            if token in aliases:
                parent[find(item.key)] = find(aliases[token])
            else:
                aliases[token] = item.key
    groups = {}
    for item in items:
        groups.setdefault(find(item.key), []).append(item.key)
    return [sorted(values) for values in groups.values()]

def create(items, seed: int = 17, ratios=(0.7, 0.15, 0.15)) -> dict:
    # Keep repeated source images in one custom split.
    if not items or any(not item.key for item in items) or len({item.key for item in items}) != len(items):
        raise ValueError("Split records must have unique nonempty keys.")
    if any(item.partition in {"val", "test"} for item in items):
        raise ValueError("Official evaluation partitions cannot be reshuffled.")
    if len(ratios) != 3 or any(not math.isfinite(value) or value <= 0 for value in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("Three positive split ratios must sum to one.")
    digests = identities(items)
    groups = grouped(items, digests)
    if len(groups) < 3:
        raise ValueError("At least three independent source groups are required.")
    groups.sort(key=lambda values: hashlib.sha256((str(seed) + ":" + "|".join(values)).encode()).hexdigest())
    total = len(groups)
    train = min(total - 2, max(1, round(total * ratios[0])))
    validation = min(total - train - 1, max(1, round(total * ratios[1])))
    divisions = (groups[:train], groups[train:train + validation], groups[train + validation:])
    subsets = {name: sorted(key for group in division for key in group) for name, division in zip(("train", "val", "test"), divisions)}
    return {"schema": 1, "protocol": "custom source grouped split; not an official training split", "seed": seed, "ratios": list(ratios), "digests": digests, "groups": groups, "subsets": subsets}

def official(items):
    # Preserve official partitions and reject source leakage.
    if not items or any(not item.key for item in items) or len({item.key for item in items}) != len(items):
        raise ValueError("Official records need unique nonempty keys.")
    if any(item.partition not in {"train", "val", "test"} for item in items):
        raise ValueError("Every record needs an explicit official partition.")
    digests = identities(items)
    groups = grouped(items, digests)
    membership = {item.key: item.partition for item in items}
    if any(len({membership[key] for key in group}) != 1 for group in groups):
        raise ValueError("Official partitions contain source group leakage.")
    subsets = {name: sorted(item.key for item in items if item.partition == name) for name in ("train", "val", "test")}
    return {"schema": 1, "protocol": "official partition tags; empty training subsets remain empty", "digests": digests, "groups": groups, "subsets": subsets}

def select(items, manifest: str | None = None, subset: str = "all"):
    subset = subset.strip().casefold()
    if subset == "all":
        return list(items)
    if subset not in {"train", "val", "test"} or manifest is None:
        raise ValueError("Custom subsets require a manifest and train, val, or test.")
    path = locate(Path("."), clean(manifest, "manifest"))
    payload = read(path)
    if payload.get("schema") != 1 or payload.get("digests") != identities(items):
        raise ValueError("Split manifest does not match the current source images.")
    divisions = payload.get("subsets", {})
    if set(divisions) != {"train", "val", "test"} or any(not isinstance(value, list) for value in divisions.values()):
        raise ValueError("Split manifest has invalid subsets.")
    keys = [key for values in divisions.values() for key in values]
    if len(set(keys)) != len(keys) or set(keys) != {item.key for item in items}:
        raise ValueError("Split manifest has overlapping or missing records.")
    membership = {key: name for name, values in divisions.items() for key in values}
    officialtags = str(payload.get("protocol", "")).startswith("official")
    if officialtags and any(item.partition not in {"train", "val", "test"} for item in items):
        raise ValueError("Official manifests require official row partitions.")
    if any((officialtags or item.partition in {"val", "test"}) and item.partition and item.partition != membership[item.key] for item in items):
        raise ValueError("Split manifest reassigns official evaluation records.")
    for group in grouped(items, payload["digests"]):
        if len({membership[key] for key in group}) != 1:
            raise ValueError("Source group leakage was detected.")
    wanted = set(divisions[subset])
    return [item for item in items if item.key in wanted]

def main() -> None:
    parser = argparse.ArgumentParser(description="Create grouped custom or verified official splits.")
    parser.add_argument("--dataset", default="./pie")
    parser.add_argument("--mapping", default="./mapping_file.json")
    parser.add_argument("--images", default="./annotation_images")
    parser.add_argument("--output", default="./split.json")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.7, 0.15, 0.15))
    parser.add_argument("--protocol", type=str.casefold, choices=("custom", "official"), default="custom")
    args = parser.parse_args()
    output = clean(args.output, "output")
    if output.exists():
        raise FileExistsError(output)
    items = records(args.dataset, args.mapping, args.images)
    payload = official(items) if args.protocol == "official" else create(items, args.seed, args.ratios)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: len(value) for key, value in payload["subsets"].items()}))

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
