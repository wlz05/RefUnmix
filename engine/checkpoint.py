# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import hashlib
from pathlib import Path

def fingerprint(root):
    # Identify weights by content rather than checkpoint filenames.
    files = sorted(path for path in Path(root).rglob("*") if path.is_file() and path.suffix.casefold() in {".bin", ".safetensors", ".pt"})
    if not files:
        return None
    hashes = []
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1048576), b""):
                digest.update(chunk)
        hashes.append(digest.hexdigest())
    return hashlib.sha256("|".join(sorted(hashes)).encode()).hexdigest()

def signature(model):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

def configuration(root):
    root = Path(root)
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.casefold() in {".json", ".txt", ".model", ".spm"})
    if not files:
        return None
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
