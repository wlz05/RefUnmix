# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from copy import deepcopy
import hashlib
import json
from pathlib import PurePosixPath

def defaults(args=None):
    # Share the main controls with every ablation reference.
    values = {"field": "pullback", "fieldgeometry": "coupled", "method": "path", "axes": "all", "eta": 0., "prior": "off", "heldout": False, "control": False, "compensation": False, "frontier": False, "representation": "latent", "selection": "attain", "geometry": "output", "algorithm": "trust", "release": True, "poststeps": 1, "cfg": False, "guidance": 1., "referencepath": "auto"}
    return {key: deepcopy(getattr(args, key, value)) for key, value in values.items()}

def configuration(args):
    # Exclude request identity and export destinations from control identity.
    excluded = {"image", "source", "target", "region", "polarity", "output", "report", "dataset", "mapping", "images", "manifest", "split", "limit", "categories", "seed", "seeds", "batch", "evalclip", "recipe", "suite", "dry", "port", "frozenrecipehash"}
    values = {key: deepcopy(value) for key, value in vars(args).items() if key not in excluded}
    values.setdefault("oracle", False)
    for key in ("model", "clip", "adapter", "constraints", "probabilities", "radiusprofile"):
        if values.get(key) is not None:
            values[key] = PurePosixPath(str(values[key]).replace("\\", "/")).as_posix()
    paths = {key: values[key] for key in ("constraints", "probabilities", "radiusprofile") if values.get(key) is not None}
    if paths:
        from pathlib import Path
        from data import clean, locate
        from engine.reproduce import digest
        values["configurationhashes"] = {key: digest(locate(Path("."), clean(value, key))) for key, value in paths.items()}
    return values

def changes(reference, comparison):
    return {key: {"reference": reference.get(key), "comparison": comparison.get(key)} for key in sorted(reference.keys() | comparison.keys()) if reference.get(key) != comparison.get(key) or (key in reference) != (key in comparison)}

def signature(args):
    values = configuration(args)
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "settings": values, "scope": "Effective control settings; seed, input, code, and checkpoint hashes are recorded separately."}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
