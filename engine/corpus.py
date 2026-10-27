# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import hashlib
import csv
import io
import json
from pathlib import Path
import tempfile
from PIL import Image, ImageOps
import numpy as np
from data import clean, descriptions, locate, lower, name
from engine.format import check, mask as grayscale, orient, rgb, runs
from engine.serial import loads, read

PROFILES = {
    "pie": {"image": ("imagepath", "image"), "source": ("originalprompt", "source"), "target": ("editingprompt", "target"), "instruction": ("editinginstruction", "instruction"), "pair": ("targetimage", "editedimage"), "partition": ""},
    "emu": {"image": ("image",), "source": ("inputcaption", "source"), "target": ("outputcaption", "target"), "instruction": ("instruction",), "pair": (), "partition": ""},
    "magicbrush": {"image": ("sourceimg", "image"), "source": ("sourcecaption", "inputcaption", "source"), "target": ("targetcaption", "outputcaption", "target"), "instruction": ("instruction",), "pair": ("targetimg", "editedimage"), "partition": ""},
    "imgedit": {"image": ("id", "inputimages", "image"), "source": ("inputcaption", "source"), "target": ("outputcaption", "target"), "instruction": ("prompt", "instruction"), "pair": ("outputimages", "editedimage"), "partition": "test"},
    "gedit": {"image": ("inputimage", "sourceimage", "image"), "source": ("inputcaption", "source"), "target": ("outputcaption", "target"), "instruction": ("instruction",), "pair": (), "partition": "test"},
}

def profile(value):
    result = name(value)
    aliases = {"piebench": "pie", "emuedit": "emu", "imgeditbench": "imgedit", "geditbench": "gedit"}
    result = aliases.get(result, result)
    if result not in PROFILES:
        raise ValueError("Unknown benchmark profile.")
    return result

def pick(row, names):
    for key in names:
        value = row.get(key)
        if value is not None and (not isinstance(value, str) or value.strip()):
            return value
    return None

def partition(value):
    value = str(value or "").strip().casefold()
    value = {"validation": "val", "valid": "val", "dev": "val", "training": "train", "testing": "test"}.get(value, value)
    if value not in {"", "train", "val", "test"}:
        raise ValueError("Official partitions must be train, val, or test.")
    return value

def entries(content):
    if isinstance(content, dict) and "records" in content:
        content = content["records"]
    if isinstance(content, dict):
        values = content.items()
    elif isinstance(content, list):
        values = enumerate(content)
    else:
        raise ValueError("Annotations must be an object or array.")
    for key, value in values:
        if not isinstance(value, dict):
            raise ValueError("Every annotation must be a record object.")
        if "data" in value and isinstance(value["data"], list):
            raise ValueError("Flatten multiturn records with session groups first.")
        yield str(key), value

def rows(path):
    # Read local annotations without network access.
    path = locate(Path("."), clean(str(path), "annotations"))
    if path.is_dir():
        try:
            from datasets import DatasetDict, load_from_disk
        except ImportError as error:
            raise RuntimeError("Local Arrow exports require datasets.") from error
        loaded = load_from_disk(str(path))
        divisions = loaded.items() if isinstance(loaded, DatasetDict) else [("", loaded)]
        for split, values in divisions:
            for index, row in enumerate(values):
                yield str(index), {**row, "storage_split": split}
        return
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.casefold() == ".json":
        yield from entries(read(path))
    elif path.suffix.casefold() == ".jsonl":
        with path.open(encoding="utf-8-sig") as handle:
            for index, line in enumerate(handle):
                if line.strip():
                    try:
                        row = loads(line)
                    except ValueError as error:
                        raise ValueError(f"Invalid JSONL at {path}:{index + 1}: {error}") from error
                    if not isinstance(row, dict):
                        raise ValueError("JSONL lines must be record objects.")
                    yield str(index), row
    elif path.suffix.casefold() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            if not fields or any(not name(field) for field in fields) or len({name(field) for field in fields}) != len(fields):
                raise ValueError("CSV headers must have unique normalized names.")
            for index, row in enumerate(reader):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"CSV row {reader.line_num} has the wrong column count.")
                if not any(value.strip() for value in row.values()):
                    continue
                for field, value in row.items():
                    normalized = name(field)
                    if normalized in {"mask", "masksize", "attributes", "retain", "inputimages", "outputimages"} and value.strip().startswith(("[", "{")):
                        row[field] = loads(value)
                    elif normalized in {"turnindex", "idx"} and value.strip():
                        row[field] = int(value)
                yield str(index), row
    elif path.suffix.casefold() == ".parquet":
        try:
            from pyarrow.parquet import ParquetFile
        except ImportError as error:
            raise RuntimeError("Local Parquet exports require pyarrow.") from error
        index = 0
        for batch in ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                yield str(index), row
                index += 1
    else:
        raise ValueError("Use JSON, JSONL, CSV, Parquet, or saved Arrow data.")

def captions(path=None):
    if path is None:
        return {}
    result = {}
    for key, row in rows(path):
        row = lower(row)
        identity = pick(row, ("key",))
        identity = str(key if identity is None else identity).strip().casefold()
        if not identity or identity in result:
            raise ValueError("Caption keys must be unique and nonempty.")
        result[identity] = row
    return result

def single(value):
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("Only single-source editing is currently supported.")
        value = value[0]
    return value

def normalize(kind, key, record, overrides=None, split=None):
    # Keep edit instructions distinct from target captions.
    kind = profile(kind)
    config, row = PROFILES[kind], lower(record)
    if isinstance(row.get("data"), list):
        raise ValueError("Flatten multiturn records with session groups first.")
    native = pick(row, ("officialsplit", "partition", "split"))
    native = partition(native)
    if not config["partition"]:
        stored = partition(pick(row, ("storagesplit",)))
        if native and stored and native != stored:
            raise ValueError("Saved dataset partitions disagree with official row tags.")
        native = native or stored
    declared = partition(split)
    if declared and native and declared != native:
        raise ValueError("Explicit partitions cannot override official row tags.")
    division = declared or native or config["partition"]
    identity = pick(row, ("key", "idx"))
    if kind == "emu":
        identity = pick(row, ("key", "hash"))
        if identity is None:
            index = pick(row, ("idx",))
            identity = f"{division}-{key if index is None else index}"
    if kind == "magicbrush":
        imageid, turn = pick(row, ("imgid",)), pick(row, ("turnindex",))
        if imageid is None or isinstance(imageid, bool) or not str(imageid).strip() or turn is None:
            raise ValueError("MagicBrush requires img_id and turn_index.")
        if isinstance(turn, bool) or not isinstance(turn, int) or turn < 1:
            raise ValueError("MagicBrush turns must be positive integers.")
        identity = f"{imageid}-{turn}"
    identity = str(key if identity is None else identity).strip().casefold()
    if not identity:
        raise ValueError("Record keys cannot be empty.")
    language = str(pick(row, ("instructionlanguage", "language")) or "en").strip().casefold()
    language = {"english": "en", "chinese": "cn", "zh": "cn"}.get(language, language)
    if language != "en":
        return None
    override = (overrides or {}).get(identity, {})
    source = pick(override, ("source", "inputcaption", "sourcecaption")) or pick(row, config["source"])
    target = pick(override, ("target", "outputcaption", "targetcaption")) or pick(row, config["target"])
    if any(not isinstance(value, str) or not value.strip() for value in (source, target)):
        raise ValueError(f"Record {identity} needs source and target captions; provide --captions.")
    instruction = pick(row, config["instruction"]) or target
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Instructions must be nonempty text.")
    if kind == "emu" and division not in {"val", "test"}:
        raise ValueError("The Emu benchmark provides validation and test only.")
    if kind == "magicbrush" and not division:
        raise ValueError("MagicBrush records need an official partition.")
    if kind in {"imgedit", "gedit"} and division != "test":
        raise ValueError("Benchmark test data cannot become training data.")
    group = pick(row, ("sourceid", "group", "sessionid"))
    if kind == "magicbrush":
        imageid = pick(row, ("imgid",))
        group = group if imageid is None else imageid
    if kind == "emu":
        hashvalue = pick(row, ("hash",))
        group = str(hashvalue).rsplit("_", 1)[-1] if hashvalue is not None else group
    mask = pick(override, ("maskpath",)) or pick(row, ("maskpath", "maskimg", "mask"))
    if isinstance(mask, (list, tuple)):
        if name(row.get("maskorder", "rowmajor")) != "rowmajor" or type(row.get("maskbase", 0)) is not int or row.get("maskbase", 0) != 0:
            raise ValueError("RLE masks require zero-based row-major coordinates.")
        runs(mask, row.get("masksize", [512, 512]))
    if isinstance(mask, dict) and "counts" in mask:
        raise ValueError("Convert COCO masks to grayscale or zero-based row-major pairs first.")
    return {"key": identity, "image": single(pick(row, config["image"])), "source": source.strip(), "target": target.strip(), "instruction": instruction.strip(), "edited": single(pick(row, config["pair"])), "mask": mask, "masksize": row.get("masksize", [512, 512]), "group": str("" if group is None else group).strip().casefold(), "category": str(pick(row, ("edittype", "tasktype", "task", "category", "editingtypeid")) or "").strip().casefold(), "partition": division, "benchmark": kind, "captionorigin": "sidecar" if override else "native"}

def image(value, root):
    if isinstance(value, Image.Image):
        check(value)
        return value.copy()
    if isinstance(value, dict):
        content, path = value.get("bytes"), value.get("path")
        if content is not None:
            with Image.open(io.BytesIO(bytes(content))) as handle:
                check(handle)
                return handle.copy()
        value = path
    if not isinstance(value, (str, Path)):
        raise ValueError("Images require local relative paths or embedded bytes.")
    path = locate(root, clean(str(value), "image"))
    with Image.open(path) as handle:
        check(handle)
        return handle.copy()

def asset(value, root, destination, mask=False):
    pixels = image(value, root)
    if mask and pixels.getexif().get(274, 1) != 1:
        raise ValueError("Normalize mask orientation before importing.")
    pixels = grayscale(pixels) if mask else orient(rgb(pixels))[0]
    digest = hashlib.sha256(str((pixels.mode, pixels.size)).encode() + pixels.tobytes()).hexdigest()
    relative = Path("masks" if mask else "images") / (digest + ".png")
    path = destination / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pixels.save(path)
    return relative.as_posix(), pixels.size

def convert(kind, annotation, root, output, sidecar=None, split=None):
    # Export normalized records outside the source tree.
    kind = profile(kind)
    root = locate(Path("."), clean(str(root), "dataset"))
    if not root.is_dir():
        raise NotADirectoryError(root)
    output = clean(str(output), "output")
    code = Path(__file__).resolve().parent.parent
    if output.resolve() == code or code in output.resolve().parents or output.resolve() == root.resolve() or root.resolve() in output.resolve().parents:
        raise ValueError("Exports must stay outside source and input directories.")
    if output.exists():
        raise FileExistsError(output)
    overrides = captions(sidecar)
    output.parent.mkdir(parents=True, exist_ok=True)
    records, seen = {}, set()
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        destination = Path(directory)
        for key, row in rows(annotation):
            item = normalize(kind, key, row, overrides, split)
            if item is None:
                continue
            if item["key"] in seen:
                raise ValueError("Record keys must be unique across partitions.")
            seen.add(item["key"])
            original = image(item["image"], root)
            maskvalue = item["mask"]
            if maskvalue is not None and original.getexif().get(274, 1) != 1:
                if isinstance(maskvalue, (list, tuple)):
                    total = runs(maskvalue, item["masksize"])
                    values = np.zeros(total, dtype=np.uint8)
                    for start, length in zip(maskvalue[::2], maskvalue[1::2]):
                        values[start:start + length] = 255
                    region = Image.fromarray(values.reshape(tuple(item["masksize"]))).resize(original.size, Image.Resampling.NEAREST)
                else:
                    region = grayscale(image(maskvalue, root))
                    if region.size != original.size:
                        raise ValueError("Raster masks must match raw source image geometry.")
                original, maskvalue = orient(original, region)
            source, size = asset(original, root, destination)
            result = {key: item[key] for key in ("source", "target", "instruction", "category", "group", "partition", "benchmark", "captionorigin")}
            extra = lower(row)
            override = overrides.get(item["key"], {})
            result.update({key: list(descriptions(override.get(key, extra.get(key)))) for key in ("attributes", "retain")})
            result["family"] = str(override.get("family", extra.get("family", ""))).strip().casefold()
            result["localtext"] = override.get("localtext", extra.get("localtext"))
            result["image"] = Path(source).relative_to("images").as_posix()
            if item["edited"] is not None:
                result["editedimage"], _ = asset(item["edited"], root, destination)
            if isinstance(maskvalue, (list, tuple)):
                encoded, shape = maskvalue, item["masksize"]
                if len(encoded) % 2 or any(type(value) is not int or value < 0 for value in encoded):
                    raise ValueError("RLE masks need nonnegative integer pairs.")
                if not isinstance(shape, (list, tuple)) or len(shape) != 2 or any(type(value) is not int or value < 1 for value in shape):
                    raise ValueError("RLE mask sizes need positive integer dimensions.")
                runs(encoded, shape)
                result.update({"mask": list(maskvalue), "masksize": item["masksize"]})
            elif maskvalue is not None:
                result["maskpath"], geometry = asset(maskvalue, root, destination, mask=True)
                if geometry != size:
                    raise ValueError("Raster masks must match source image geometry.")
                result["masksize"] = [geometry[1], geometry[0]]
            records[item["key"]] = result
        if not records:
            raise ValueError("No English benchmark records were exported.")
        (destination / "mapping.json").write_text(json.dumps(records, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
        if output.exists():
            raise FileExistsError(output)
        destination.rename(output)
    from engine.format import POLICY
    return {"benchmark": kind, "records": len(records), "output": str(output), "mapping": str(output / "mapping.json"), "images": "./images", "format": dict(POLICY), "protocol": "Official tags are preserved; custom splits require explicit grouping."}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
