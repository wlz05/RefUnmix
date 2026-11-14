# -*- coding: utf-8 -*-
# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from engine.reproduce import digest
from engine.serial import read, write
from engine.sources import ROLES, fetch, manifest, relative

METHODS = ("masactrl", "pnp", "flowedit", "infedit", "turboedit", "instantedit", "swiftedit", "chord", "rrls", "where", "refunmix")
NATIVE = ("sdturbo", "swiftbrush", "instaflow")
DEFAULTS = {"masactrl": "sd14", "pnp": "sd14", "flowedit": "sd3", "infedit": "lcm", "turboedit": "sdxlturbo", "instantedit": "perflow", "swiftedit": "swiftbrush", "chord": "sdturbo", "rrls": "sdturbo", "where": "sdturbo", "refunmix": "sdturbo"}
LEGACY = {"masactrl": "masa", "pnp": "pnp", "infedit": "infedit", "turboedit": "turbo", "instantedit": "instant", "swiftedit": "swift"}
WEIGHTS = {"masactrl": ("sd",), "pnp": ("sd",), "infedit": ("lcm",), "turboedit": ("xl",), "instantedit": ("perflow", "control"), "swiftedit": ("swift", "sd2", "ip")}
SEEDS = (42, 52, 3047, 3407)

def parser():
    result = argparse.ArgumentParser(description="Run, organize and score eleven editing methods with the same PIE-Bench protocol.")
    result.add_argument("--method", choices=METHODS, required=True)
    result.add_argument("--backbone", choices=(*NATIVE, "sd14", "sd3", "flux", "lcm", "sdxlturbo", "perflow"))
    result.add_argument("--action", choices=("plan", "fetch", "preflight", "generate", "organize", "score", "aggregate", "run"), default="plan")
    result.add_argument("--sources", default="baselines")
    result.add_argument("--weights", default="models/baselines")
    result.add_argument("--model")
    result.add_argument("--inverse-model", default="models", help="SD-Turbo base components used by the released SwiftEdit inversion stack.")
    result.add_argument("--dataset", default="data")
    result.add_argument("--mapping", default="mapping_file.json")
    result.add_argument("--images", default="annotation_images")
    result.add_argument("--output", default="outputs/comparison")
    result.add_argument("--input", help="Saved image tree, with a seed subdirectory when importing multiple seeds.")
    result.add_argument("--input-manifest", help="Explicit request, seed and relative image mapping for external exports.")
    result.add_argument("--clip", default="clip")
    result.add_argument("--cache", default="cache/metrics")
    result.add_argument("--protocol", default="protocol/metrics.py")
    result.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    result.add_argument("--batch", type=int, choices=(1, 2, 3, 4), default=1)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--python", help="Relative interpreter path for a separate compatible upstream environment.")
    result.add_argument("--network", action="store_true", help="Permit metric assets to be downloaded during scoring.")
    result.add_argument("--cpu-offload", action="store_true", help="Use model CPU offload for the official FlowEdit pipeline.")
    result.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return result

def compatible(method, backbone):
    if method not in METHODS:
        return False
    if method in {"chord", "rrls", "where", "refunmix"}:
        return backbone in NATIVE
    if method == "flowedit":
        return backbone in {"sd3", "flux"}
    return backbone == DEFAULTS[method]

def options(args):
    args.backbone = args.backbone or DEFAULTS[args.method]
    defaults = {"sdturbo": "models", "swiftbrush": "models/brush", "instaflow": "models/insta", "sd3": "models/sd3", "flux": "models/flux"}
    role = {"sd14": "sd", "lcm": "lcm", "sdxlturbo": "xl", "perflow": "perflow"}
    args.model = args.model or defaults.get(args.backbone, str(Path(args.weights) / role.get(args.backbone, args.backbone)))
    if not compatible(args.method, args.backbone):
        reason = " SwiftEdit requires trained components matched to the selected backbone." if args.method == "swiftedit" else ""
        raise ValueError("Unsupported released method-backbone combination." + reason)
    if args.seeds != [seed for seed in SEEDS if seed in args.seeds] or not args.seeds:
        raise ValueError("Use distinct seeds in the order 42, 52, 3047, 3407.")
    if args.method != "refunmix" and args.batch != 1:
        raise ValueError("Official external adapters process one benchmark request at a time.")
    if args.backbone == "instaflow" and args.batch != 1:
        raise ValueError("InstaFlow requires batch one.")
    if args.cpu_offload and args.method != "flowedit":
        raise ValueError("Model CPU offload is provided only for FlowEdit.")
    if args.method in LEGACY and args.device not in {"cuda", "cuda:0"} and args.action not in {"plan", "fetch", "organize", "score", "aggregate"}:
        raise ValueError("Released legacy generation requires the first visible CUDA device.")
    for name in ("sources", "weights", "model", "inverse_model", "dataset", "mapping", "images", "output", "clip", "cache", "protocol", "input", "input_manifest", "python"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, relative(value, name))
    if args.input is not None and args.input_manifest is not None:
        raise ValueError("Choose an image tree or an explicit import manifest.")
    protected = [args.dataset, args.sources, args.model, args.inverse_model, args.weights, args.clip, Path("engine"), Path("tests")]
    protected.extend(path for path in (args.input, args.input_manifest) if path is not None)
    for path in protected:
        out, source = args.output.resolve(), path.resolve()
        if out == Path.cwd().resolve() or out.is_relative_to(source) or source.is_relative_to(out):
            raise ValueError("Comparison output must not overlap code, inputs, sources or weights.")
    return args

def root(args):
    result = args.output / args.method / args.backbone
    return result / "main" if args.method == "refunmix" else result

def entry():
    return Path(__file__).resolve().parent.parent.joinpath("baselines.py").relative_to(Path.cwd().resolve()).as_posix()

def plan(args):
    role = ROLES.get(args.method)
    adapter = "official release" if role else "conduct.py main profile"
    return {"method": args.method, "backbone": args.backbone, "seeds": args.seeds, "requestsperseed": 700, "sources": str(args.sources / role) if role else None, "model": str(args.model), "weights": str(args.weights), "output": str(root(args)), "adapter": adapter, "sourcepresent": role is None or (args.sources / role).is_dir(), "action": "plan only", "commands": [command(args, args.seeds[0], False)]}

def command(args, seed, worker=True):
    seeds = [seed] if worker else args.seeds
    result = [str(args.python) if args.python else sys.executable, "-B", entry(), "--method", args.method, "--backbone", args.backbone, "--action", "generate" if worker else "run", "--seeds", *[str(value) for value in seeds], "--batch", str(args.batch), "--device", args.device]
    if worker:
        result.append("--worker")
    for name in ("sources", "weights", "model", "inverse_model", "dataset", "mapping", "images", "output", "clip", "cache", "protocol"):
        result.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
    if args.cpu_offload:
        result.append("--cpu-offload")
    return result

def meanpath(args):
    name = "mean.json" if args.seeds == list(SEEDS) else "mean" + "-".join(str(seed) for seed in args.seeds) + ".json"
    return root(args) / name

def idle(args):
    if args.device.startswith("cuda"):
        values = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        if values:
            raise RuntimeError("Wait for the active GPU experiment before starting another controller.")

@contextmanager
def guard(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if not path.stat().st_size:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)

def population(args):
    from data import records
    items = records(str(args.dataset), str(args.mapping), str(args.images))
    if len(items) != 700 or len({item.key for item in items}) != 700 or {item.category for item in items} != set("0123456789"):
        raise ValueError("Use all 700 official PIE-Bench requests across ten edit categories.")
    return items

def localword(mapping, item):
    value = mapping.get(item.key, {}).get("blended_word", "") if isinstance(mapping, dict) else ""
    if not isinstance(value, str):
        raise ValueError("PIE blended words must contain text.")
    words = value.split()
    return words[1] if len(words) > 1 else ""

def verify(summary, items, seed):
    from data import clean, locate
    from engine.format import read as imagefile
    value = read(summary)
    if value.get("recipe") is not None:
        recipe = locate(summary.parent, clean(value["recipe"], "recipe"))
        if value.get("recipehash") != digest(recipe):
            raise ValueError("The saved inference recipe has changed.")
    rows = value.get("rows", [])
    if value.get("seeds") != [seed] or len(rows) != len(items) or {r.get("key") for r in rows} != {i.key for i in items}:
        raise ValueError("The saved inference request-seed grid is incomplete or duplicated.")
    seen = set()
    for row in rows:
        path = locate(summary.parent, clean(row.get("image"), "prediction"))
        if row.get("seed") != seed or path in seen or not path.is_file() or row.get("resultsha256") != digest(path):
            raise ValueError("Saved predictions require distinct files, matching seeds and intact hashes.")
        seen.add(path)
        with imagefile(path) as image:
            if image.size != (512, 512):
                raise ValueError("Saved predictions must match the 512-pixel benchmark geometry.")
    return value

def normalize(value, method):
    import numpy as np
    from PIL import Image
    import torch
    if torch.is_tensor(value):
        if value.ndim == 4:
            value = value[-1]
        if value.ndim != 3 or value.shape[0] != 3 or not torch.isfinite(value).all():
            raise ValueError("The editor returned invalid RGB pixels.")
        value = Image.fromarray((value.detach().float().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8))
    if not isinstance(value, Image.Image):
        raise ValueError("The editor must return an RGB image or normalized RGB tensor.")
    if method in {"masactrl", "pnp"} and value.height == 512 and value.width > 512 and value.width % 512 == 0:
        value = value.crop((value.width - 512, 0, value.width, 512))
    if value.size != (512, 512):
        raise ValueError("The editor returned incompatible image geometry.")
    return value.convert("RGB")

def build(args):
    if args.method in {"chord", "rrls", "where"}:
        from engine.editors import Official
        return Official(args.method, args.backbone, args.sources / ROLES[args.method], args.model, args.clip, args.device)
    if args.method == "flowedit":
        from engine.flowedit import load
        return load(args.backbone, args.sources / "flow", args.model, args.device, args.cpu_offload)
    if args.device not in {"cuda", "cuda:0"}:
        raise ValueError("Released legacy adapters require the first visible CUDA device.")
    from engine import upstream
    roles = {name: args.weights / name for name in ("sd", "lcm", "xl", "perflow", "control", "swift", "sd2", "ip")}
    roles["turbo"] = args.inverse_model
    if args.method == "swiftedit":
        roles["brushunet"] = args.model / "unet"
    else:
        selected = {"masactrl": "sd", "pnp": "sd", "infedit": "lcm", "turboedit": "xl", "instantedit": "perflow"}[args.method]
        roles[selected] = args.model
    upstream.configure(args.sources, roles)
    return upstream.build(LEGACY[args.method])

def recipe(args, items):
    from engine.checkpoint import configuration, fingerprint
    upstream = manifest(args.sources, args.method) if args.method in ROLES else None
    roles = {name: args.weights / name for name in WEIGHTS.get(args.method, ())}
    selected = {"masactrl": "sd", "pnp": "sd", "infedit": "lcm", "turboedit": "xl", "instantedit": "perflow"}.get(args.method)
    if selected:
        roles[selected] = args.model
    if args.method == "swiftedit":
        roles["inversionbase"] = args.inverse_model
    if args.method in {"chord", "rrls", "where", "flowedit", "swiftedit"}:
        roles["native"] = args.model
    if args.method == "rrls":
        roles["clip"] = args.clip
    hashes = {}
    for name, path in roles.items():
        if not path.is_dir():
            raise FileNotFoundError("Missing model role " + name + ": " + str(path))
        checksum = fingerprint(path)
        if checksum is None:
            raise ValueError("The model role contains no checkpoint: " + name)
        hashes[name] = {"path": str(path), "sha256": checksum, "configuration": configuration(path)}
    scripts = Path(__file__).resolve().parent.parent
    paths = [scripts / "baselines.py", *sorted((scripts / "engine").glob("*.py"))]
    return {"method": args.method, "backbone": args.backbone, "seed": args.seeds[0], "mapping": digest(args.dataset / args.mapping), "sourceimages": {i.key: digest(i.image) for i in items}, "upstream": upstream, "weights": hashes, "code": {p.relative_to(scripts).as_posix(): digest(p) for p in paths}, "device": args.device, "offload": args.cpu_offload, "requests": len(items)}

def generate(args):
    from engine.format import read as imagefile
    from engine.reproduce import seed as randomseed
    import torch
    torch.set_num_threads(1)
    if args.method in {"masactrl", "pnp"}:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    items = population(args)
    if len(args.seeds) != 1:
        raise ValueError("Each isolated generation worker handles one seed.")
    seed = args.seeds[0]
    target = root(args) / str(seed)
    current = recipe(args, items)
    path = target / "recipe.json"
    if path.exists():
        if read(path) != current:
            raise ValueError("Existing outputs use different code, settings, inputs or weights.")
    else:
        if target.exists() and any(target.iterdir()):
            raise ValueError("Preserve the unrecognized output directory and choose a new one.")
        write(path, current)
    if (target / "summary.json").exists():
        return verify(target / "summary.json", items, seed)
    editor = build(args)
    mapping = read(args.dataset / args.mapping)
    rows = []
    for index, item in enumerate(items):
        output = target / "annotation_images" / item.relative.with_suffix(".png")
        record = target / "records" / (item.key + ".json")
        if output.exists() or record.exists():
            if not output.exists() or not record.exists():
                raise ValueError("A saved request pair is incomplete; preserve it for recovery.")
            row = read(record)
            if row.get("key") != item.key or row.get("seed") != seed or row.get("resultsha256") != digest(output) or row.get("image") != output.relative_to(target).as_posix():
                raise ValueError("A resumed request failed its integrity check.")
        else:
            randomseed(seed)
            with imagefile(item.image) as image:
                image = image.convert("RGB")
                if image.size != (512, 512):
                    raise ValueError("The PIE source image must be 512 by 512.")
                with torch.inference_mode():
                    result = editor(item, image, seed, localword(mapping, item))
            details = {}
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                result, details = result
            result = normalize(result, args.method)
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".tmp")
            result.save(temporary, format="PNG")
            temporary.replace(output)
            row = {"key": item.key, "seed": seed, "image": output.relative_to(target).as_posix(), "resultsha256": digest(output), "report": {"source": item.source, "target": item.target, "settings": {"method": args.method, "backbone": args.backbone}, "adapter": details}}
            write(record, row)
        rows.append(row)
        if (index + 1) % 25 == 0:
            print(args.method, args.backbone, seed, index + 1, len(items), flush=True)
    if args.method in ROLES and manifest(args.sources, args.method) != current["upstream"]:
        raise ValueError("The upstream implementation changed during generation.")
    scripts = Path(__file__).resolve().parent.parent
    if any(digest(scripts / name) != checksum for name, checksum in current["code"].items()):
        raise ValueError("The adapter source changed during generation.")
    write(target / "summary.json", {"seeds": [seed], "rows": rows, "recipe": "recipe.json", "recipehash": digest(path)})
    del editor
    gc.collect()
    torch.cuda.empty_cache()
    return verify(target / "summary.json", items, seed)

def imported(args, items, seed):
    from data import clean, locate
    from engine.format import read as imagefile
    explicit = {}
    if args.input_manifest:
        value = read(args.input_manifest)
        for row in value.get("rows", []):
            pair = (row.get("key"), row.get("seed"))
            if pair in explicit or pair[0] not in {i.key for i in items} or type(pair[1]) is not int or pair[1] not in args.seeds:
                raise ValueError("The external export manifest has unknown or duplicated request-seed pairs.")
            explicit[pair] = locate(args.input_manifest.parent, clean(row.get("image"), "external prediction"))
        if len(explicit) != len(items) * len(args.seeds):
            raise ValueError("The external export manifest is incomplete.")
        if len(set(explicit.values())) != len(explicit):
            raise ValueError("External request-seed pairs require separate prediction files.")
    elif args.input is None:
        raise ValueError("Output organization requires an image tree or explicit export manifest.")
    target = root(args) / str(seed)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("Organize external images into a new output directory.")
    paths = []
    for item in items:
        path = explicit.get((item.key, seed))
        if path is None:
            base = args.input / str(seed) if len(args.seeds) > 1 else args.input
            path = locate(base, item.relative)
            if not path.exists():
                path = path.with_suffix(".png")
        with imagefile(path) as image:
            normalize(image, args.method)
        paths.append(path)
    if len(set(path.resolve() for path in paths)) != len(paths):
        raise ValueError("Each external request requires its own prediction file.")
    rows = []
    for item, path in zip(items, paths):
        with imagefile(path) as image:
            result = normalize(image, args.method)
        output = target / "annotation_images" / item.relative.with_suffix(".png")
        output.parent.mkdir(parents=True, exist_ok=True)
        result.save(output, format="PNG")
        rows.append({"key": item.key, "seed": seed, "image": output.relative_to(target).as_posix(), "resultsha256": digest(output), "report": {"source": item.source, "target": item.target, "settings": {"method": args.method, "backbone": args.backbone}, "origin": "externally generated output", "input": path.as_posix(), "inputsha256": digest(path)}})
    write(target / "summary.json", {"seeds": [seed], "rows": rows})
    return verify(target / "summary.json", items, seed)

def assess(args, seed, items):
    target = root(args) / str(seed)
    verify(target / "summary.json", items, seed)
    metrics = target / "metrics.json"
    if not metrics.exists():
        script = Path(__file__).resolve().parent.parent / "assess.py"
        subprocess.run([sys.executable, "-B", str(script), "--input", str(target / "summary.json"), "--dataset", str(args.dataset), "--mapping", str(args.mapping), "--images", str(args.images), "--scope", "benchmark", "--seeds", str(seed), "--protocol", str(args.protocol), "--device", args.device, "--cache", str(args.cache), "--network" if args.network else "--offline", "--output", str(metrics)], check=True)
    from aggregate import combine
    return combine([metrics], [seed], str(args.dataset), str(args.mapping), str(args.images))

def main(argv=None):
    args = options(parser().parse_args(argv))
    if args.action == "plan":
        print(json.dumps(plan(args), indent=2))
        return
    if args.action == "fetch":
        if args.method == "refunmix":
            raise ValueError("This built-in method does not require an upstream editor checkout.")
        fetch(args.sources, [ROLES[args.method]])
        return
    if args.worker:
        if args.action != "generate" or args.method == "refunmix" or os.environ.get("REFUNMIX_BASELINE_OWNER") != str(os.getppid()):
            raise ValueError("Invalid isolated worker invocation.")
        generate(args)
        return
    items = population(args)
    if args.action == "preflight":
        if args.method == "refunmix":
            raise ValueError("Use the existing doctor.py entry to preflight RefUnmix.")
        source = manifest(args.sources, args.method)
        with guard(args.cache.parent / "inference.lock"):
            idle(args)
            editor = build(args)
            from engine.format import read as imagefile
            from engine.reproduce import seed as randomseed
            import torch
            mapping = read(args.dataset / args.mapping)
            for item in (items[0], items[-1]):
                randomseed(args.seeds[0])
                with imagefile(item.image) as image, torch.inference_mode():
                    result = editor(item, image.convert("RGB"), args.seeds[0], localword(mapping, item))
                normalize(result[0] if isinstance(result, tuple) else result, args.method)
            print(json.dumps({"passed": True, "requests": 2, "source": source}))
        return
    with guard(args.cache.parent / "inference.lock"):
        if args.action in {"generate", "run", "score"}:
            idle(args)
        for seed in args.seeds:
            target = root(args) / str(seed)
            if args.action in {"generate", "run"}:
                if args.method == "refunmix":
                    script = Path(__file__).resolve().parent.parent / "conduct.py"
                    subprocess.run([str(args.python) if args.python else sys.executable, "-B", str(script), "--action", "generate", "--seeds", str(seed), "--model", str(args.model), "--backbone", args.backbone, "--dataset", str(args.dataset), "--mapping", str(args.mapping), "--images", str(args.images), "--device", args.device, "--batch", str(args.batch), "--output", str(root(args).parent)], check=True)
                else:
                    subprocess.run(command(args, seed), check=True, env=dict(os.environ, REFUNMIX_BASELINE_OWNER=str(os.getpid())))
                verify(target / "summary.json", items, seed)
            if args.action == "organize":
                imported(args, items, seed)
            if args.action in {"score", "run"}:
                value = assess(args, seed, items)
                print(json.dumps({"method": args.method, "backbone": args.backbone, "seed": seed, "table": value["byseed"][str(seed)]["table"]}, allow_nan=False))
        if args.action in {"aggregate", "run"}:
            from aggregate import combine
            paths = [root(args) / str(seed) / "metrics.json" for seed in args.seeds]
            value = combine(paths, args.seeds, str(args.dataset), str(args.mapping), str(args.images))
            output = meanpath(args)
            if output.exists():
                if read(output) != value:
                    raise ValueError("The existing aggregate report differs from its input reports.")
            else:
                write(output, value)

# Author: Weiliang Zhu; Other main author: Yeming Dai, Yaming Lv
# Email: wlzchina05@gmail.com
