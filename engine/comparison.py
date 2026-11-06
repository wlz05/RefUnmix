# -*- coding: utf-8 -*-
# RefUnmix: Separating Prompt Response from Shared Transport in One-Step Image Editing

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from threading import Lock
import torch
from engine.session import relative, tensor

BACKBONES = ("instaflow", "swiftbrush", "sdturbo")
METHODS = ("native", "chord", "rrls", "where", "swiftedit", "refunmix", "naive")
PROFILES = {"native": "native", "naive": "naive", "refunmix": "main"}

def digest(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def unique(values, choices, label):
    if not isinstance(values, (list, tuple)) or any(not isinstance(value, str) for value in values):
        raise ValueError("Select a list or tuple of supported " + label + ".")
    values = tuple(values)
    if not values or len(set(values)) != len(values) or any(value not in choices for value in values):
        raise ValueError("Select distinct supported " + label + ".")
    return values

class Registry:
    def __init__(self):
        self.entries = {}
        self.prepared = {}
        self.lock = Lock()

    def register(self, name, edit, backbones, settings, checkpoints=None):
        # Official adapters declare their supported models and parameters.
        if name not in METHODS or name in PROFILES or name in self.entries or not callable(edit) or not isinstance(settings, dict):
            raise ValueError("Register a distinct external method with callable edit(session, settings, metadata).")
        backbones = unique(backbones, BACKBONES, "backbones")
        try:
            settings = json.loads(json.dumps(settings, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError("Official settings must be finite JSON values.") from error
        checkpoints = deepcopy({} if checkpoints is None else checkpoints)
        if not isinstance(checkpoints, dict) or any(key not in backbones or not isinstance(paths, dict) for key, paths in checkpoints.items()):
            raise ValueError("Checkpoint declarations must match the supported backbones.")
        for paths in checkpoints.values():
            for role, path in paths.items():
                if not isinstance(role, str) or not role or not isinstance(path, str):
                    raise ValueError("Checkpoint roles and paths must be nonempty strings.")
                relative(path)
        if name == "swiftedit" and any(not checkpoints.get(backbone) for backbone in backbones):
            raise ValueError("SwiftEdit adapters must declare trained components for each supported backbone.")
        self.entries[name] = {"edit": edit, "backbones": backbones, "settings": settings, "settingssha256": digest(settings), "checkpoints": checkpoints, "source": None}

    def load(self, value):
        path = relative(value)
        if path.suffix != ".py" or not path.stem.isascii() or not path.stem.isalpha() or not path.stem.islower():
            raise ValueError("Method adapters require lowercase single-word Python filenames.")
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        name = "refunmixcomparison_" + checksum
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError("The method adapter cannot be loaded.")
        module = importlib.util.module_from_spec(spec)
        saved = dict(self.entries)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            if not callable(getattr(module, "register", None)):
                raise ValueError("Method adapter files must expose register(registry).")
            module.register(self)
            if self.entries.keys() == saved.keys():
                raise ValueError("The adapter did not register an editing method.")
            for key in self.entries.keys() - saved.keys():
                self.entries[key]["source"] = {"path": path.as_posix(), "sha256": checksum}
        except BaseException:
            self.entries = saved
            sys.modules.pop(name, None)
            raise

    def status(self, name, backbone):
        if name not in METHODS or backbone not in BACKBONES:
            raise ValueError("Unknown method or backbone.")
        if name in PROFILES:
            return "builtin"
        entry = self.entries.get(name)
        if entry is None:
            return "adapter required"
        if backbone not in entry["backbones"]:
            return "unsupported"
        if any(not relative(path).exists() for path in entry["checkpoints"].get(backbone, {}).values()):
            return "weights required"
        return "adapter"

    def prepare(self, name, backbone):
        # Hash dedicated weights once, before image generation or timing.
        status = self.status(name, backbone)
        if status not in {"builtin", "adapter"}:
            raise ValueError("The method-backbone combination is not ready: " + status)
        if status == "builtin":
            return {}
        entry = self.entries[name]
        paths = entry["checkpoints"].get(backbone, {})
        manifest = {}
        for role, value in paths.items():
            path = relative(value)
            files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.casefold() in {".bin", ".pt", ".safetensors"})
            if not files or any(item.suffix.casefold() not in {".bin", ".pt", ".safetensors"} or item.stat().st_size == 0 for item in files):
                raise ValueError("Dedicated checkpoints require nonempty model weight files.")
            manifest[role] = [(item.as_posix(), item.stat().st_size, item.stat().st_mtime_ns) for item in files]
        key = (name, backbone)
        with self.lock:
            if key in self.prepared:
                if self.prepared[key]["manifest"] != manifest:
                    raise ValueError("Dedicated checkpoints changed after preparation.")
            else:
                from engine.checkpoint import fingerprint
                weights = {}
                for role, value in paths.items():
                    path = relative(value)
                    checksum = fingerprint(path) if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()
                    weights[role] = {"path": path.as_posix(), "sha256": checksum}
                self.prepared[key] = {"manifest": manifest, "weights": weights}
            return deepcopy(self.prepared[key]["weights"])

    def plan(self, methods=METHODS[:-1], backbones=BACKBONES, models=None):
        methods, backbones = unique(methods, METHODS, "methods"), unique(backbones, BACKBONES, "backbones")
        roots = {"sdturbo": "models", "swiftbrush": "models/brush", "instaflow": "models/insta"}
        if models is not None:
            if not isinstance(models, dict) or any(name not in BACKBONES for name in models):
                raise ValueError("Model roots must name supported backbones.")
            roots.update(models)
        roots = {name: relative(path).as_posix() for name, path in roots.items()}
        rows = []
        for name in methods:
            for backbone in backbones:
                entry = self.entries.get(name)
                command = None
                if name in PROFILES:
                    command = ["python", "-B", "conduct.py", "--backbone", backbone, "--model", roots[backbone], "--profile", PROFILES[name], "--batch", "1", "--output", "outputs/matrix/" + name + "/" + backbone]
                rows.append({"method": name, "backbone": backbone, "model": roots[backbone], "status": self.status(name, backbone), "command": command, "settings": deepcopy(entry["settings"]) if entry else None, "settingssha256": entry["settingssha256"] if entry else None, "checkpoints": deepcopy(entry["checkpoints"].get(backbone, {})) if entry else {}, "adapter": deepcopy(entry["source"]) if entry else None})
        return {"schema": 1, "seeds": [42], "requestsperseed": 700, "rows": rows, "action": "plan only"}

    @torch.no_grad()
    def apply(self, name, backbone, session, metadata):
        weights = self.prepare(name, backbone)
        architecture = getattr(session.editor, "metadata", {}).get("architecture")
        if architecture != backbone or session.mask is not None or session.area is not None:
            raise ValueError("Use the declared native backbone without annotation masks.")
        if session.calls or session.decodes or session.cache or session.queries:
            raise ValueError("Each method-backbone request requires a fresh metered session.")
        if not isinstance(metadata, dict) or any(not isinstance(metadata.get(key), str) or not metadata[key].strip() for key in ("source", "target")):
            raise ValueError("Comparison metadata requires the original source and target prompts.")
        metadata = deepcopy(metadata)
        metadata["comparison"] = {"method": name, "backbone": backbone}
        requestseed = getattr(session, "seed", metadata.get("seed", 42))
        if type(requestseed) is not int or "seed" in metadata and (type(metadata["seed"]) is not int or metadata["seed"] != requestseed):
            raise ValueError("The comparison seed must match its prepared session.")
        metadata["seed"] = requestseed
        origin = session.origin.detach().clone()
        if name in PROFILES:
            import argparse
            import conduct
            from evaluate import datasetargs
            from run import common, validate
            options = conduct.parser().parse_args(["--profile", PROFILES[name], "--backbone", backbone, "--batch", "1"])
            options.model = relative(getattr(session.editor, "metadata", {}).get("root", "models"))
            args = datasetargs(common(argparse.ArgumentParser())).parse_args(conduct.generation(options, requestseed, Path("outputs") / name / backbone)[3:])
            args.size, args.device = session.editor.size, str(session.anchor.device)
            validate(args)
            metadata["settings"] = dict(vars(args))
            metadata["seed"] = requestseed
            if session.end != args.end or not session.cleanup or session.scale != 1 or session.noises.shape[0] != 1:
                raise ValueError("Builtin profiles require one shared noise, endpoint time .30 and unit-scale refinement.")
            if name == "native":
                from engine.native import direct as edit
            elif name == "naive":
                from engine.native import naive as edit
            else:
                from engine.single import edit
            pixels, report, returned = edit(session, args, metadata)
            if returned is not session:
                raise ValueError("Editing methods must retain the metered session.")
        else:
            entry = self.entries[name]
            if digest(entry["settings"]) != entry["settingssha256"]:
                raise ValueError("The registered official settings have changed.")
            source = entry["source"]
            if source is not None and hashlib.sha256(relative(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError("The registered method adapter has changed.")
            pixels, report = entry["edit"](session, deepcopy(entry["settings"]), metadata)
            if not isinstance(report, dict):
                raise ValueError("Method adapters must return pixels and a report dictionary.")
            report.update({"adapter": deepcopy(source), "officialsettings": deepcopy(entry["settings"]), "officialsettingssha256": entry["settingssha256"], "auxiliarycheckpoints": weights})
        tensor(pixels, 3)
        if pixels.shape != session.origin.shape or pixels.device != session.origin.device or torch.any((pixels < 0) | (pixels > 1)):
            raise ValueError("Method outputs must match the source RGB geometry and range.")
        if not torch.equal(origin, session.origin):
            raise ValueError("Method adapters must preserve the source image.")
        if type(session.calls) is not int or type(session.samples) is not int or any(not isinstance(query, dict) or type(query.get("samples")) is not int or query["samples"] < 1 for query in session.queries):
            raise ValueError("The native query ledger must contain positive sample counts.")
        if len(session.queries) != session.calls or sum(query["samples"] for query in session.queries) != session.samples:
            raise ValueError("Method adapters must retain the native query ledger.")
        report.update({"comparison": {"method": name, "backbone": backbone}, "seed": requestseed, "stats": session.stats()})
        return pixels, report, session

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
