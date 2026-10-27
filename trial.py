# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from data import clean, destination, locate
from evaluate import datasetargs, evaluator, execute, items
from run import common, jsonfile, load, validate, writejson
from split import identities, pixels
from engine.serial import read
from engine.protocol import changes, configuration, defaults, signature
from engine.reproduce import digest

def variants(suite, consistency="ddpm", base=None):
    # Freeze one main configuration across all ablation suites.
    shared = defaults(base)
    if suite == "alignment":
        raw = {"method": "raw", "prior": "off", "heldout": False, "control": False, "compensation": False, "frontier": False}
        return {"aligned": {**shared, **raw, "fieldgeometry": "coupled"}, "scalar": {**shared, **raw, "fieldgeometry": "scalar"}, "controlled": {**shared, "fieldgeometry": "coupled", "method": "path"}, "matched": {**shared, "fieldgeometry": "scalar", "method": "path"}}
    if suite == "temporal":
        raw = {"method": "raw", "prior": "off", "heldout": False, "control": False, "compensation": False, "frontier": False}
        result = {"full": shared}
        for name, weight in (("first", 0.), ("early", .25), ("mean", .5), ("late", .75), ("last", 1.)):
            blend = {**shared, "field": "blend", "fieldgeometry": "coupled", "fieldweight": weight}
            result[name + "raw"] = {**blend, **raw}
            result[name + "full"] = blend
        ect = {**shared, "field": "ect", "fieldgeometry": "coupled", "start": .75, "delta": 0., "fieldshrink": .7}
        localizationtime = getattr(base, "localizationtime", None) if base is not None else None
        localizationtime = getattr(base, "start", .9) if localizationtime is None else localizationtime
        result.update({"ectraw": {**ect, **raw, "localizationtime": None}, "ectfull": {**ect, "localizationtime": localizationtime}, "ectdirect": {**ect, **raw, "cleanup": False, "localizationtime": None}, "chorddirect": {**shared, **raw, "field": "chord", "fieldgeometry": "coupled", "cleanup": False}})
        return result
    if suite == "fields":
        result = {}
        for field in ("pullback", "chord", "instant", "average"):
            result[field + "raw"] = {**shared, "field": field, "method": "raw", "prior": "off", "heldout": False, "control": False, "compensation": False, "frontier": False}
            result[field + "fixed"] = {**shared, "field": field, "method": "fixed", "axes": "none", "eta": 0., "prior": "off", "heldout": False, "compensation": False, "frontier": False}
            result[field + "full"] = {**shared, "field": field}
        return result
    if suite == "null":
        disabled = {"field": "zero", "fieldgeometry": "coupled", "prior": "off", "heldout": False, "control": False, "compensation": False, "frontier": False, "representation": "latent"}
        fixed = {**disabled, "method": "fixed", "axes": "none"}
        changes = {"native": {**disabled, "method": "raw"}, "centered": {**fixed, "eta": 0.}, "condition": {**fixed, "eta": 1.}, "reference": {**disabled, "method": "path", "axes": "path", "release": False}, "raw": {"method": "raw", "compensation": False, "frontier": False}, "full": {}}
        return {key: {**shared, **value} for key, value in changes.items()}
    if suite == "diagnosis":
        fixed = {"method": "fixed", "axes": "none", "prior": "off", "heldout": False, "representation": "latent", "compensation": False, "frontier": False}
        changes = {"raw": {"method": "raw", "prior": "off", "heldout": False, "control": False, "compensation": False, "frontier": False}, "target": {**fixed, "eta": 0.}, "middle": {**fixed, "eta": .5}, "source": {**fixed, "eta": 1.}, "mixture": {**fixed, "eta": .5, "representation": "mix"}, "rgb": {**fixed, "eta": .5, "representation": "rgb"}, "random": {**fixed, "eta": .5, "representation": "random"}, "full": {}}
        return {key: {**shared, **value} for key, value in changes.items()}
    core = {"raw": {"method": "raw", "prior": "off", "heldout": False, "control": False, "compensation": False, "frontier": False}, "target": {"axes": "none", "eta": 0.}, "source": {"axes": "none", "eta": 1.}, "eta": {"axes": "path"}, "field": {"axes": "field", "eta": 0.}, "full": {}}
    prior = {"prior": consistency, **({"flowconsistency": True} if consistency == "flow" else {})}
    extra = {"reference": {"rgb": {"representation": "rgb"}, "mixture": {"representation": "mix"}, "random": {"representation": "random"}, "conditional": {"representation": "cfg"}}, "prior": {"noprior": {"prior": "off"}, "boundedprior": {**prior, "priorcontrol": "bounded"}, "hardprior": {**prior, "priorcontrol": "hard"}, "softprior": {**prior, "priorcontrol": "soft"}}, "search": {"poll": {"algorithm": "poll"}, "randomsearch": {"algorithm": "random"}, "unguarded": {"monitor": False}}, "mechanism": {"decoder": {"cleanup": False}, "multistep": {"poststeps": 5}}, "guidance": {"cfgone": {"cfg": True, "guidance": 1., "referencepath": "prediction"}, "cfgtwo": {"cfg": True, "guidance": 2., "referencepath": "prediction"}, "cfgfour": {"cfg": True, "guidance": 4., "referencepath": "prediction"}}, "budget": {name: {"maxsamples": count, "maxcalls": count} for name, count in (("ten", 10), ("twelve", 12), ("fifteen", 15), ("twenty", 20), ("thirty", 30))}, "candidates": {"small": {"candidates": 8}, "large": {"candidates": 24}}, "core": {}}
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
        elif suite == "goals":
            extra["goals"] = {"targetonly": {"goals": ["target"]}, "globalonly": {"goals": ["global"]}}
        elif suite != "cost":
            raise ValueError("Unknown ablation suite.")
        else:
            extra["cost"] = {"light": {"candidates": 3, "axes": "path", "prior": "off"}, "short": {"candidates": 7}}
    return {key: {**shared, **value} for key, value in {**core, **extra[suite]}.items()}

def design(args, controls):
    # Verify the full arm before loading checkpoints.
    if args.suite == "temporal" and args.localization != "baseline":
        raise ValueError("Temporal trials require the shared baseline localization protocol.")
    validate(args)
    label = args.field + "full" if args.suite == "fields" else "controlled" if args.suite == "alignment" and args.fieldgeometry == "coupled" else "matched" if args.suite == "alignment" else "full"
    if label not in controls:
        raise ValueError("The ablation suite has no matching main reference.")
    reference = configuration(args)
    rows = {}
    for name, modification in controls.items():
        current = deepcopy(args)
        for key, value in modification.items():
            setattr(current, key, value)
        validate(current)
        identity = signature(current)
        rows[name] = {**identity, "changes": changes(reference, identity["settings"])}
    if rows[label]["changes"]:
        raise ValueError("The full ablation differs from the main configuration: " + ", ".join(rows[label]["changes"]))
    return {"main": label, "controlsignature": signature(args), "variants": rows}

def settings(args):
    excluded = {"output", "manifest", "split", "limit", "seeds", "seed", "dataset", "mapping", "images", "recipe", "suite", "dry"}
    result = {key: value for key, value in vars(args).items() if key not in excluded}
    result["seeds"] = sorted(set(getattr(args, "seeds", None) or [args.seed]))
    result["configuration"] = {key: jsonfile(getattr(args, key)) for key in ("constraints", "probabilities", "radiusprofile")}
    result["adapterhash"] = hashlib.sha256(locate(Path("."), clean(args.adapter, "adapter")).read_bytes()).hexdigest() if args.adapter else None
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for source in sorted([*root.glob("*.py"), *root.joinpath("engine").glob("*.py")]):
        digest.update(source.relative_to(root).as_posix().encode("utf-8") + b"\0" + source.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    result["codehash"] = digest.hexdigest()
    return result

def frozen(args, chosen):
    source = locate(Path("."), clean(args.recipe, "recipe"))
    payload = read(source)
    if not isinstance(payload, dict) or type(payload.get("schema")) is not int or payload.get("schema") != 1 or payload.get("split") != "val" or payload.get("suite") != args.suite or not payload.get("variants") or json.dumps(payload.get("settings"), sort_keys=True, allow_nan=False) != json.dumps(settings(args), sort_keys=True, allow_nan=False):
        raise ValueError("Frozen tests require unchanged validation control settings.")
    if not getattr(args, "manifest", None) or payload.get("manifestdigest") != digest(locate(Path("."), clean(args.manifest, "manifest"))):
        raise ValueError("Frozen tests require the unchanged split manifest.")
    if getattr(args, "dataset", None) is not None:
        mapping = locate(locate(Path("."), clean(args.dataset, "dataset")), clean(args.mapping, "mapping"))
        if payload.get("mappingdigest") != digest(mapping):
            raise ValueError("Frozen tests require the unchanged annotation mapping.")
    # Keep test sources disjoint from the validation recipe.
    bindings = payload.get("digests", {})
    if not isinstance(bindings, dict) or any(not isinstance(key, str) or not isinstance(value, str) or not value for key, value in bindings.items()):
        raise ValueError("Frozen source digests must be named nonempty strings.")
    if chosen and not bindings:
        raise ValueError("Frozen tests require nonempty validation source bindings.")
    current = set(identities(chosen).values())
    if current & set(bindings.values()):
        raise ValueError("Test images overlap validation sources.")
    known, paired = payload.get("pixels", {}), payload.get("paired", {})
    if not isinstance(known, dict) or not isinstance(paired, dict) or set(known) != set(bindings) or not set(paired) <= set(bindings) or any(not isinstance(value, str) or not value for value in [*known.values(), *paired.values()]):
        raise ValueError("Frozen recipes require complete source and paired pixel identities; regenerate validation.")
    currentpixels, currentpaired = set(pixels(chosen).values()), set(pixels(chosen, True).values())
    if currentpixels & (set(known.values()) | set(paired.values())) or currentpaired & set(known.values()):
        raise ValueError("Test pixels or paired edit chains overlap validation sources.")
    for field in ("family", "group"):
        known = payload.get(field, [])
        if not isinstance(known, list) or any(not isinstance(value, str) or not value for value in known):
            raise ValueError("Frozen source groups and families must be string lists.")
        if {getattr(item, field) for item in chosen if getattr(item, field)} & set(known):
            raise ValueError("Test families or source groups overlap validation.")
    controls = payload["variants"]
    declared = variants(args.suite, "flow" if getattr(args, "backbone", "vp") != "vp" else "ddpm", args)
    allowed = {key for value in declared.values() for key in value}
    if not isinstance(controls, dict) or any(not isinstance(key, str) or not key.isascii() or not key.isalpha() or key != key.casefold() or not isinstance(value, dict) or value.keys() - allowed for key, value in controls.items()):
        raise ValueError("Frozen variants contain unsupported controls.")
    if json.dumps(controls, sort_keys=True, allow_nan=False) != json.dumps(declared, sort_keys=True, allow_nan=False):
        raise ValueError("Frozen variants differ from the declared ablation design.")
    return controls

def main():
    parser = datasetargs(common(argparse.ArgumentParser(description="Run declared benchmark or validation/test ablations.")))
    parser.set_defaults(split="val", output="./trials")
    parser.add_argument("--suite", type=str.casefold, choices=("alignment", "temporal", "fields", "null", "core", "reference", "diagnosis", "leverage", "frontier", "attainment", "release", "geometry", "prior", "mechanism", "search", "guidance", "budget", "candidates", "cost", "solver", "goals"), default="core")
    parser.add_argument("--recipe", default=None)
    parser.add_argument("--dry", action="store_true", help="Print effective variants without loading data or checkpoints.")
    args = parser.parse_args()
    if args.method != "path" or args.split not in {"all", "val", "test"}:
        raise ValueError("Native trials require path mode and all, val, or test records.")
    if args.split == "all" and (args.manifest or args.recipe):
        raise ValueError("All-record benchmark trials require no manifest or validation recipe.")
    if args.split != "all" and not args.manifest:
        raise ValueError("Validation/test trials require a split manifest.")
    if args.adapter:
        raise ValueError("Frozen trials require verified native weights; use external adapters separately.")
    if (args.split == "test") != bool(args.recipe):
        raise ValueError("Test trials require a frozen validation recipe.")
    if args.suite == "prior" and args.backbone != "vp" and not args.flowconsistency:
        raise ValueError("Flow prior trials require --flowconsistency before loading the backbone.")
    controls = variants(args.suite, "flow" if args.backbone != "vp" else "ddpm", args)
    plan = design(args, controls)
    if args.dry:
        print(json.dumps({"schema": 1, "design": plan, "scope": "Declared controls only; no data or frozen recipe has been verified."}, indent=2, allow_nan=False))
        return
    paths = {"mapping": locate(locate(Path("."), clean(args.dataset, "dataset")), clean(args.mapping, "mapping"))}
    if args.manifest:
        paths["manifest"] = locate(Path("."), clean(args.manifest, "manifest"))
    inputs = {key: digest(path) for key, path in paths.items()}
    chosen = items(args)
    sources = identities(chosen)
    if args.recipe:
        controls = frozen(args, chosen)
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
        if any(digest(path) != inputs[key] for key, path in paths.items()):
            raise ValueError("Frozen ablation inputs changed during execution.")
        current = deepcopy(args)
        current.output = (output / label).as_posix()
        current.frozenrecipehash = hashlib.sha256(locate(Path("."), clean(args.recipe, "recipe")).read_bytes()).hexdigest() if args.recipe else None
        for key, value in modification.items():
            setattr(current, key, value)
        payload = execute(current, (editor, judge, evaluation, independent))
        if any(payload["inputs"][key]["sha256"] != value for key, value in inputs.items()):
            raise ValueError("An ablation arm used changed mapping or split inputs.")
        if any(sources.get(row["key"]) != row.get("inputsha256") for row in payload["rows"]):
            raise ValueError("An ablation arm used changed source image bytes.")
        if payload["controlsignature"]["sha256"] != plan["variants"][label]["sha256"]:
            raise RuntimeError("Executed control settings differ from the checked ablation plan.")
        results[label] = {"summary": payload["summary"], "cost": payload["cost"], "controller": payload["controller"], "controlsignature": payload["controlsignature"], "changes": plan["variants"][label]["changes"], "path": str(output / label / "summary.json"), "rows": len(payload["rows"])}
    if any(digest(path) != inputs[key] for key, path in paths.items()):
        raise ValueError("Ablation inputs changed before publication.")
    items(args)
    if args.split == "val":
        recipe = {"schema": 1, "split": "val", "suite": args.suite, "settings": settings(args), "checkpoints": checkpoints, "variants": controls, "digests": identities(chosen), "pixels": pixels(chosen), "paired": pixels(chosen, True), "group": sorted({item.group for item in chosen if item.group}), "family": sorted({item.family for item in chosen if item.family}), "manifestdigest": inputs["manifest"], "mappingdigest": inputs["mapping"], "note": "Configurations are frozen; no test-based winner is selected."}
        writejson(output / "recipe.json", recipe)
    writejson(output / "summary.json", {"schema": 1, "split": args.split, "settings": vars(args), "design": plan, "variants": results, "note": "Controller proxies and consistency remain diagnostics; use independent evidence."})

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
