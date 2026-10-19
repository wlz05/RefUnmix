# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import json
import math
import os
from pathlib import Path
import tempfile

def pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result

def constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")

def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON numbers must be finite.")
    if isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)
    return value

def loads(value):
    return finite(json.loads(value.lstrip("\ufeff"), object_pairs_hook=pairs, parse_constant=constant))

def read(path):
    path = Path(path)
    try:
        return loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeError, ValueError) as error:
        raise ValueError(f"Invalid UTF-8 JSON in {path}: {error}") from error

def write(path, payload):
    # Publish complete reports without overwriting existing results.
    path = Path(path)
    content = json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
