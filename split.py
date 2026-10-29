# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from data import clean, locate, records
from engine.serial import read, write

def pixels(items, paired=False):
    # Match decoded RGB content across lossless re-encodings.
    from engine.format import orient, read as image
    result, cache = {}, {}
    for item in items:
        path = item.edited if paired else item.image
        if path is None:
            continue
        resolved = path.resolve()
        if resolved not in cache:
            value = orient(image(path))[0]
            digest = hashlib.sha256(str(value.size).encode("ascii") + b"\0" + value.tobytes()).hexdigest()
            cache[resolved] = digest
        result[item.key] = cache[resolved]
    return result

def identities(items) -> dict[str, str]:
    result = {}
    for item in items:
        with item.image.open("rb") as handle:
            result[item.key] = hashlib.file_digest(handle, "sha256").hexdigest()
    return result

def grouped(items, digests: dict[str, str], content=None, paired=None) -> list[list[str]]:
    parent = {item.key: item.key for item in items}
    content = pixels(items) if content is None else content
    paired = pixels(items, True) if paired is None else paired

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    aliases = {}
    for item in items:
        tokens = ["digest:" + digests[item.key], "pixel:" + content[item.key], "path:" + item.image.resolve().as_posix().casefold()]
        if item.group:
            tokens.append("group:" + item.group.strip().casefold())
        if item.family:
            tokens.append("family:" + item.family.strip().casefold())
        for token in tokens:
            if token in aliases:
                parent[find(item.key)] = find(aliases[token])
            else:
                aliases[token] = item.key
    for key, digest in paired.items():
        token = "pixel:" + digest
        if token in aliases:
            parent[find(key)] = find(aliases[token])
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
    content, paired = pixels(items), pixels(items, True)
    groups = grouped(items, digests, content, paired)
    if len(groups) < 3:
        raise ValueError("At least three independent source groups are required.")
    groups.sort(key=lambda values: hashlib.sha256((str(seed) + ":" + "|".join(values)).encode()).hexdigest())
    total = len(groups)
    train = min(total - 2, max(1, round(total * ratios[0])))
    validation = min(total - train - 1, max(1, round(total * ratios[1])))
    divisions = (groups[:train], groups[train:train + validation], groups[train + validation:])
    subsets = {name: sorted(key for group in division for key in group) for name, division in zip(("train", "val", "test"), divisions)}
    return {"schema": 1, "protocol": "custom source grouped split; not an official training split", "seed": seed, "ratios": list(ratios), "digests": digests, "pixels": content, "paired": paired, "groups": groups, "subsets": subsets}

def official(items):
    # Preserve official partitions and reject source leakage.
    if not items or any(not item.key for item in items) or len({item.key for item in items}) != len(items):
        raise ValueError("Official records need unique nonempty keys.")
    if any(item.partition not in {"train", "val", "test"} for item in items):
        raise ValueError("Every record needs an explicit official partition.")
    digests = identities(items)
    content, paired = pixels(items), pixels(items, True)
    groups = grouped(items, digests, content, paired)
    membership = {item.key: item.partition for item in items}
    if any(len({membership[key] for key in group}) != 1 for group in groups):
        raise ValueError("Official partitions contain source group leakage.")
    subsets = {name: sorted(item.key for item in items if item.partition == name) for name in ("train", "val", "test")}
    return {"schema": 1, "protocol": "official partition tags; empty training subsets remain empty", "digests": digests, "pixels": content, "paired": paired, "groups": groups, "subsets": subsets}

def select(items, manifest: str | None = None, subset: str = "all"):
    subset = subset.strip().casefold()
    if subset == "all":
        if manifest is not None:
            raise ValueError("An all-record benchmark cannot ignore a split manifest; select train, val, or test.")
        return list(items)
    if subset not in {"train", "val", "test"} or manifest is None:
        raise ValueError("Custom subsets require a manifest and train, val, or test.")
    path = locate(Path("."), clean(manifest, "manifest"))
    payload = read(path)
    if not isinstance(payload, dict) or type(payload.get("schema")) is not int or payload.get("schema") != 1 or payload.get("digests") != identities(items):
        raise ValueError("Split manifest does not match the current source images.")
    divisions = payload.get("subsets", {})
    if not isinstance(divisions, dict) or set(divisions) != {"train", "val", "test"} or any(not isinstance(value, list) or any(not isinstance(key, str) or not key for key in value) for value in divisions.values()):
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
    content, paired = pixels(items), pixels(items, True)
    if payload.get("pixels") != content or payload.get("paired") != paired:
        raise ValueError("Split pixel identities or paired targets changed; regenerate the manifest.")
    groups = grouped(items, payload["digests"], content, paired)
    known = payload.get("groups")
    if not isinstance(known, list) or any(not isinstance(group, list) or not group or any(not isinstance(key, str) or not key for key in group) for group in known):
        raise ValueError("Split manifests require complete source groups.")
    if sorted(tuple(sorted(group)) for group in known) != sorted(tuple(sorted(group)) for group in groups):
        raise ValueError("Split source groups changed; regenerate the manifest.")
    for group in groups:
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
    write(output, payload)
    print(json.dumps({key: len(value) for key, value in payload["subsets"].items()}))

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
