# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from data import clean, destination, locate
from evaluate import datasetargs, evaluator, execute, items
from run import common, jsonfile, load, writejson
from split import identities
from engine.serial import read

def variants(suite):
    if suite == "alignment":
        shared = {"field": "pullback", "prior": "off", "heldout": False, "control": False, "compensation": False, "frontier": False, "representation": "latent", "selection": "attain", "geometry": "output", "algorithm": "trust", "release": True, "axes": "all"}
        return {"aligned": {**shared, "fieldgeometry": "coupled", "method": "raw"}, "scalar": {**shared, "fieldgeometry": "scalar", "method": "raw"}, "controlled": {**shared, "fieldgeometry": "coupled", "method": "path"}, "matched": {**shared, "fieldgeometry": "scalar", "method": "path"}}
    if suite == "fields":
        shared = {"prior": "off", "heldout": False, "control": False, "compensation": False, "frontier": False, "representation": "latent", "selection": "attain", "geometry": "output", "algorithm": "trust"}
        result = {}
        for field in ("pullback", "chord", "instant", "average"):
            result[field + "raw"] = {**shared, "field": field, "fieldgeometry": "coupled", "method": "raw"}
            result[field + "fixed"] = {**shared, "field": field, "fieldgeometry": "coupled", "method": "fixed", "axes": "none", "eta": 0.}
            result[field + "full"] = {**shared, "field": field, "fieldgeometry": "coupled", "method": "path", "axes": "all", "release": True}
        return result
    if suite == "diagnosis":
        fixed = {"method": "fixed", "axes": "none", "prior": "off", "heldout": False, "representation": "latent", "compensation": False, "frontier": False}
        return {"raw": {"method": "raw", "prior": "off", "heldout": False, "compensation": False, "frontier": False}, "target": {**fixed, "eta": 0.}, "middle": {**fixed, "eta": .5}, "source": {**fixed, "eta": 1.}, "mixture": {**fixed, "eta": .5, "representation": "mix"}, "rgb": {**fixed, "eta": .5, "representation": "rgb"}, "random": {**fixed, "eta": .5, "representation": "random"}, "full": {}}
    core = {"raw": {"method": "raw", "compensation": False, "frontier": False}, "target": {"axes": "none", "eta": 0.}, "source": {"axes": "none", "eta": 1.}, "eta": {"axes": "path"}, "field": {"axes": "field", "eta": 0.}, "full": {}}
    extra = {"reference": {"rgb": {"representation": "rgb"}, "mixture": {"representation": "mix"}, "random": {"representation": "random"}, "conditional": {"representation": "cfg"}}, "prior": {"noprior": {"prior": "off"}, "hardprior": {"priorcontrol": "hard"}, "softprior": {"priorcontrol": "soft"}}, "search": {"poll": {"algorithm": "poll"}, "randomsearch": {"algorithm": "random"}, "unguarded": {"monitor": False}}, "mechanism": {"decoder": {"cleanup": False}, "multistep": {"poststeps": 5, "cfg": True, "referencepath": "prediction"}}, "guidance": {"cfgone": {"cfg": True, "guidance": 1., "referencepath": "prediction"}, "cfgtwo": {"cfg": True, "guidance": 2., "referencepath": "prediction"}, "cfgfour": {"cfg": True, "guidance": 4., "referencepath": "prediction"}}, "budget": {"small": {"candidates": 8}, "large": {"candidates": 24}}, "core": {}}
    if suite not in extra:
        if suite == "leverage":
            extra["leverage"] = {"compensated": {"compensation": True}, "uncompensated": {"compensation": False}}
        elif suite == "frontier":
            extra["frontier"] = {"capacity": {"frontier": True}, "unchecked": {"frontier": False}}
        elif suite == "attainment":
            extra["attainment"] = {"protected": {"selection": "protect"}, "attained": {"selection": "attain"}, "field": {"selection": "attain", "axes": "field", "eta": 0.}, "random": {"selection": "attain", "representation": "random"}, "poll": {"selection": "attain", "algorithm": "poll"}}
        elif suite == "release":
            extra["release"] = {"released": {"release": True}, "joint": {"release": False}, "random": {"representation": "random", "release": True}, "field": {"axes": "field", "eta": 0., "release": False}}
        elif suite == "geometry":
            extra["geometry"] = {"output": {"release": True, "geometry": "output"}, "latent": {"release": True, "geometry": "latent"}, "attenuated": {"release": True, "geometry": "attenuate"}, "joint": {"release": False, "geometry": "output"}}
        elif suite == "solver":
            extra["solver"] = {"nocertificate": {"certificate": False}, "shortsolve": {"solversteps": 25, "certificatestep": 25}, "unguarded": {"monitor": False}}
        elif suite != "cost":
            raise ValueError("Unknown ablation suite.")
        else:
            extra["cost"] = {"light": {"candidates": 3, "axes": "path", "prior": "off"}, "short": {"candidates": 7}, "noprior": {"prior": "off"}}
    return {**core, **extra[suite]}

def settings(args):
    excluded = {"output", "manifest", "split", "limit", "seeds", "seed", "dataset", "mapping", "images", "recipe", "suite"}
    result = {key: value for key, value in vars(args).items() if key not in excluded}
    result["configuration"] = {key: jsonfile(getattr(args, key)) for key in ("constraints", "probabilities", "radiusprofile")}
    result["adapterhash"] = hashlib.sha256(locate(Path("."), clean(args.adapter, "adapter")).read_bytes()).hexdigest() if args.adapter else None
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for source in sorted([*root.glob("*.py"), *root.joinpath("engine").glob("*.py")]):
        digest.update(source.relative_to(root).as_posix().encode("utf-8") + b"\0" + source.read_bytes() + b"\0")
    result["codehash"] = digest.hexdigest()
    return result

def frozen(args, chosen):
    source = locate(Path("."), clean(args.recipe, "recipe"))
    payload = read(source)
    if payload.get("schema") != 1 or payload.get("split") != "val" or not payload.get("variants") or payload.get("settings") != settings(args):
        raise ValueError("Frozen tests require unchanged validation control settings.")
    current = set(identities(chosen).values())
    if current & set(payload.get("digests", {}).values()):
        raise ValueError("Test images overlap validation sources.")
    for field in ("family", "group"):
        if {getattr(item, field) for item in chosen if getattr(item, field)} & set(payload.get(field, [])):
            raise ValueError("Test families or source groups overlap validation.")
    controls = payload["variants"]
    allowed = {"method", "axes", "eta", "representation", "prior", "priorcontrol", "algorithm", "cleanup", "poststeps", "cfg", "guidance", "referencepath", "candidates", "monitor", "compensation", "release", "geometry", "frontier", "certificate", "solversteps", "certificatestep", "heldout", "selection", "gainloss", "goals", "field", "fieldridge", "fieldprobe", "fieldradius", "fieldtolerance", "fieldgeometry", "control"}
    if not isinstance(controls, dict) or any(not key.isalpha() or key != key.casefold() or not isinstance(value, dict) or value.keys() - allowed for key, value in controls.items()):
        raise ValueError("Frozen variants contain unsupported controls.")
    return controls

def main():
    parser = datasetargs(common(argparse.ArgumentParser(description="Run frozen native reference and prior ablations.")))
    parser.set_defaults(split="val", output="./trials")
    parser.add_argument("--suite", type=str.casefold, choices=("alignment", "fields", "core", "reference", "diagnosis", "leverage", "frontier", "attainment", "release", "geometry", "prior", "mechanism", "search", "guidance", "budget", "cost", "solver"), default="core")
    parser.add_argument("--recipe", default=None)
    args = parser.parse_args()
    if args.split not in {"val", "test"} or not args.manifest or args.method != "path":
        raise ValueError("Native trials require path mode and a grouped val/test manifest.")
    if args.adapter:
        raise ValueError("Frozen trials require verified native weights; use external adapters separately.")
    if (args.split == "test") != bool(args.recipe):
        raise ValueError("Test trials require a frozen validation recipe.")
    chosen = items(args)
    controls = frozen(args, chosen) if args.recipe else variants(args.suite)
    output = destination(args.output, "output")
    if output.exists():
        raise FileExistsError(output)
    editor, judge = load(args)
    evaluation, independent = evaluator(args, judge)
    checkpoints = {"model": getattr(editor, "metadata", None), "control": getattr(judge, "metadata", None), "report": getattr(evaluation, "metadata", None)}
    if args.recipe:
        payload = jsonfile(args.recipe)
        if payload.get("checkpoints") != checkpoints or any(value is not None and value.get("weights") is None for value in checkpoints.values()):
            raise ValueError("Frozen tests require verified unchanged checkpoint weights.")
    output.mkdir(parents=True)
    results = {}
    for label, modification in controls.items():
        current = deepcopy(args)
        current.output = (output / label).as_posix()
        current.frozenrecipehash = hashlib.sha256(locate(Path("."), clean(args.recipe, "recipe")).read_bytes()).hexdigest() if args.recipe else None
        for key, value in modification.items():
            setattr(current, key, value)
        payload = execute(current, (editor, judge, evaluation, independent))
        results[label] = {"summary": payload["summary"], "cost": payload["cost"], "path": str(output / label / "summary.json"), "rows": len(payload["rows"])}
    if args.split == "val":
        recipe = {"schema": 1, "split": "val", "settings": settings(args), "checkpoints": checkpoints, "variants": controls, "digests": identities(chosen), "group": sorted({item.group for item in chosen if item.group}), "family": sorted({item.family for item in chosen if item.family}), "manifestdigest": hashlib.sha256(locate(Path("."), clean(args.manifest, "manifest")).read_bytes()).hexdigest(), "note": "Configurations are frozen; no test-based winner is selected."}
        writejson(output / "recipe.json", recipe)
    writejson(output / "summary.json", {"schema": 1, "split": args.split, "settings": vars(args), "variants": results, "note": "Controller proxies and consistency remain diagnostics; use independent evidence."})

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
