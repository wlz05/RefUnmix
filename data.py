# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import sys

sys.dont_write_bytecode = True
import json
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any
from engine.format import runs
from engine.serial import read

@dataclass(frozen=True)
class Record:
    key: str
    image: Path
    relative: Path
    source: str
    target: str
    instruction: str
    category: str = ""
    group: str = ""
    mask: tuple[int, ...] | None = None
    edited: Path | None = None
    masksize: tuple[int, int] = (512, 512)
    partition: str = ""
    benchmark: str = "pie"
    maskpath: Path | None = None
    captionorigin: str = "native"
    family: str = ""
    attributes: tuple[str, ...] = ()
    retain: tuple[str, ...] = ()
    localtext: str | None = None

# Normalize names for case-insensitive matching.
def name(value: object) -> str:
    return "".join(character for character in str(value).strip().casefold() if character.isalnum())

def lower(item: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in item.items():
        field = name(key)
        if field in result and result[field] != value:
            raise ValueError(f"Conflicting field names: {key}")
        result[field] = value
    return result

def clean(value: str, label: str) -> Path:
    # Keep every configured path local and relative.
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{label} must be a local relative path string.")
    text = str(value).strip().replace("\\", "/")
    path = Path(text)
    if not text or path.is_absolute() or PureWindowsPath(text).drive or text.startswith("/") or ".." in path.parts:
        raise ValueError(f"{label} must be a local relative path.")
    parts = [part.casefold() for part in path.parts if part not in {"", "."}]
    if not parts:
        raise ValueError(f"{label} must be a nonempty relative path.")
    result = Path(*parts)
    if not result.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError(f"{label} escapes the working directory.")
    return result

def locate(root: Path, path: Path) -> Path:
    # Match existing path components without case sensitivity.
    current = root
    for part in path.parts:
        direct = current / part
        if not current.is_dir():
            current = direct
            continue
        matches = [entry for entry in current.iterdir() if entry.name.casefold() == part.casefold()]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous path component: {part}")
        current = matches[0] if matches else direct
    if not current.resolve().is_relative_to(root.resolve()):
        raise ValueError("Resolved path escapes its root.")
    return current

def destination(value, label="output"):
    path = locate(Path("."), clean(value, label))
    if path.resolve().is_relative_to(Path(__file__).resolve().parent):
        raise ValueError("Runtime artifacts must stay outside the code tree.")
    return path

def choose(item: dict[str, Any], *names: str) -> str:
    for field in names:
        value = item.get(name(field))
        if value is not None:
            if type(value) is int and name(field) in {"editingtypeid", "category", "sourceid", "group"}:
                value = str(value)
            if not isinstance(value, str):
                raise ValueError(f"Field {field} requires text.")
        if value and value.strip():
            return str(value).strip()
    return ""

def descriptions(value):
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError("Semantic components require nonempty text lists.")
    return tuple(item.strip() for item in value)

def localdescription(value):
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Local descriptions must be nonempty strings.")
    return value.strip()

def records(dataset: str = "./pie", mapping: str = "./mapping_file.json", images: str = "./annotation_images") -> list[Record]:
    # Load valid records from object or array mappings.
    root = locate(Path("."), clean(dataset, "dataset"))
    source = locate(root, clean(mapping, "mapping"))
    imagebase = locate(root, clean(images, "images"))
    if not source.is_file():
        raise FileNotFoundError(source)
    content = read(source)
    if isinstance(content, dict):
        entries = content.items()
    elif isinstance(content, list):
        entries = enumerate(content)
    else:
        raise ValueError("Mapping content must be an object or array.")
    result: list[Record] = []
    seen = set()
    for key, item in sorted(entries, key=lambda pair: str(pair[0]).casefold()):
        if not isinstance(item, dict):
            raise ValueError(f"Record {key} must be an object.")
        item = lower(item)
        value = choose(item, "imagepath", "image")
        if not value:
            raise ValueError(f"Record {key} needs an image path.")
        try:
            relative = clean(value, "image")
        except ValueError as error:
            raise ValueError(f"Invalid image path for record {key}: {error}") from error
        image = locate(imagebase, relative)
        if not image.is_file():
            raise FileNotFoundError(image)
        original = choose(item, "originalprompt", "sourceprompt", "source")
        target = choose(item, "editingprompt", "editedprompt", "targetprompt", "target")
        if not original or not target:
            raise ValueError(f"Record {key} needs source and target captions.")
        instruction = choose(item, "editinginstruction", "editprompt", "instruction") or target
        normalized = str(key).strip().casefold()
        if not normalized or normalized in seen:
            raise ValueError(f"Duplicate record key: {normalized}")
        seen.add(normalized)
        encoded = item.get("mask")
        if encoded is not None and (not isinstance(encoded, list) or len(encoded) % 2 or any(type(value) is not int or value < 0 for value in encoded)):
            raise ValueError(f"Invalid run length mask for record {normalized}.")
        shape = item.get("masksize", [512, 512])
        if not isinstance(shape, (list, tuple)) or len(shape) != 2 or any(type(value) is not int or value < 1 for value in shape):
            raise ValueError(f"Invalid mask size for record {normalized}.")
        if encoded is not None:
            if name(item.get("maskorder", "rowmajor")) != "rowmajor" or type(item.get("maskbase", 0)) is not int or item.get("maskbase", 0) != 0:
                raise ValueError("RLE masks require explicit zero-based row-major coordinates.")
            runs(encoded, shape)
        paired = choose(item, "targetimage", "editedimage", "targetpath")
        edited = locate(root, clean(paired, "target image")) if paired else None
        if edited is not None and not edited.is_file():
            raise FileNotFoundError(edited)
        category = choose(item, "editingtypeid", "category").casefold()
        group = choose(item, "sourceid", "group").casefold()
        division = choose(item, "partition", "officialsplit").casefold()
        division = {"validation": "val", "valid": "val", "dev": "val", "training": "train", "testing": "test"}.get(division, division)
        if division not in {"", "train", "val", "test"}:
            raise ValueError(f"Invalid official partition for record {normalized}.")
        maskfile = choose(item, "maskpath")
        maskpath = locate(root, clean(maskfile, "mask")) if maskfile else None
        if maskpath is not None and (not maskpath.is_file() or encoded is not None):
            raise ValueError("Raster masks must exist without competing RLE masks.")
        benchmark = choose(item, "benchmark").casefold() or "pie"
        if benchmark == "pie":
            original, target = (value.replace("[", "").replace("]", "") for value in (original, target))
        result.append(Record(normalized, image, relative, original, target, instruction, category, group, tuple(encoded) if encoded is not None else None, edited, tuple(shape), division, benchmark, maskpath, choose(item, "captionorigin") or "native", choose(item, "family").casefold(), descriptions(item.get("attributes")), descriptions(item.get("retain")), localdescription(item.get("localtext"))))
    if not result:
        raise FileNotFoundError("No valid benchmark records were found.")
    return result

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
