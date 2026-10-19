# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from data import clean, destination, locate
from engine.format import read as pixels
from engine.reproduce import digest, seed, snapshot
from engine.serial import read as readjson, write as publish

def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Value must be positive.")
    return number

def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Value must be nonnegative.")
    return number

def times(values):
    result = sorted(set(float(value) for value in values))
    if not result or any(not math.isfinite(value) or not 0 < value <= 1 for value in result):
        raise ValueError("Times must be finite and in (0, 1].")
    return result

def common(parser):
    parser.add_argument("--model", default="./models")
    parser.add_argument("--backbone", type=str.casefold, choices=("vp", "flux", "sd3"), default="vp")
    parser.add_argument("--flowguidance", type=float, default=3.5)
    parser.add_argument("--flowconsistency", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clip", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", type=str.casefold, choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--size", type=positive, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--noise", type=positive, default=1)
    parser.add_argument("--start", type=float, default=.90)
    parser.add_argument("--delta", type=float, default=.15)
    parser.add_argument("--field", type=str.casefold, choices=("pullback", "chord", "instant", "average"), default="pullback")
    parser.add_argument("--fieldridge", type=float, default=.05)
    parser.add_argument("--fieldprobe", type=float, default=.25)
    parser.add_argument("--fieldradius", type=float, default=1.)
    parser.add_argument("--fieldtolerance", type=float, default=1e-6)
    parser.add_argument("--fieldgeometry", type=str.casefold, choices=("coupled", "scalar"), default="coupled")
    parser.add_argument("--end", type=float, default=.30)
    parser.add_argument("--scale", type=float, default=1.)
    parser.add_argument("--cleanup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference", type=str.casefold, choices=("auto", "raw", "latent", "pixel"), default="auto")
    parser.add_argument("--guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--control", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--times", nargs="+", type=float, default=[.60, .75, .90])
    parser.add_argument("--proposal", type=str.casefold, choices=("clean", "chord"), default="clean")
    parser.add_argument("--method", type=str.casefold, choices=("path", "raw", "fixed", "projection", "joint", "thread", "base", "global"), default="path")
    parser.add_argument("--candidates", type=positive, default=12)
    parser.add_argument("--pollstep", type=float, default=.5)
    parser.add_argument("--minstep", type=float, default=1 / 16)
    parser.add_argument("--axes", type=str.casefold, choices=("all", "field", "path", "none"), default="all")
    parser.add_argument("--eta", type=float, default=0.)
    parser.add_argument("--representation", type=str.casefold, choices=("latent", "rgb", "mix", "random", "cfg"), default="latent")
    parser.add_argument("--poststeps", type=positive, default=1)
    parser.add_argument("--guidance", type=float, default=1.)
    parser.add_argument("--cfg", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--referencepath", choices=("auto", "prediction", "embedding", "discrete"), default="auto")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--prior", type=str.casefold, choices=("ddpm", "rec", "flow", "off"), default="ddpm")
    parser.add_argument("--priorprobes", type=positive, default=2)
    parser.add_argument("--priorslack", type=float, default=0.)
    parser.add_argument("--semanticscale", type=float, default=1.)
    parser.add_argument("--selection", type=str.casefold, choices=("attain", "protect"), default="attain")
    parser.add_argument("--gainloss", type=float, default=0.)
    parser.add_argument("--goals", nargs="+", default=None)
    parser.add_argument("--archive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--damagescale", type=float, default=1.)
    parser.add_argument("--priorscale", type=float, default=1.)
    parser.add_argument("--maxsamples", type=positive, default=64)
    parser.add_argument("--maxcalls", type=positive, default=None)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--heldout", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--attributes", nargs="+", default=None)
    parser.add_argument("--retain", nargs="+", default=None)
    parser.add_argument("--constraints", default=None)
    parser.add_argument("--probabilities", default=None)
    parser.add_argument("--variance", choices=("fixed_small", "fixed_large"), default=None)
    parser.add_argument("--algorithm", choices=("trust", "poll", "random"), default="trust")
    parser.add_argument("--priorcontrol", choices=("bounded", "hard", "soft"), default="bounded")
    parser.add_argument("--priorweight", type=float, default=.1)
    parser.add_argument("--secant", type=float, default=.0001)
    parser.add_argument("--condition", type=float, default=1000000.)
    parser.add_argument("--curvature", type=float, default=.05)
    parser.add_argument("--responseerror", type=float, default=.25)
    parser.add_argument("--hessian", type=float, default=1000000.)
    parser.add_argument("--damping", type=float, default=4.)
    parser.add_argument("--critical", type=float, default=1e-8)
    parser.add_argument("--monitor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compensation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--release", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--geometry", type=str.casefold, choices=("output", "latent", "attenuate"), default="output", help="Select compensation geometry without changing output protection checks.")
    parser.add_argument("--frontier", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--frontiersteps", type=int, default=150)
    parser.add_argument("--solversteps", type=int, default=250)
    parser.add_argument("--certificatestep", type=int, default=150)
    parser.add_argument("--certificate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--radiusprofile", default=None)
    parser.add_argument("--donors", nargs="+", type=float, default=[])
    parser.add_argument("--lock", type=str.casefold, choices=("cone", "semantic", "none"), default="cone")
    parser.add_argument("--basis", type=str.casefold, choices=("frequency", "channel"), default="frequency")
    parser.add_argument("--minreduce", type=float, default=0.)
    parser.add_argument("--ranktol", type=float, default=1e-6)
    parser.add_argument("--calibration", type=str.casefold, choices=("fixed", "shrink", "secant"), default="secant")
    parser.add_argument("--updates", type=nonnegative, default=2)
    parser.add_argument("--defect", type=float, default=.1)
    parser.add_argument("--factorial", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lowtime", type=float, default=None)
    parser.add_argument("--hightime", type=float, default=None)
    parser.add_argument("--strength", type=float, default=1.)
    parser.add_argument("--levels", type=positive, default=9)
    parser.add_argument("--checks", type=nonnegative, default=3)
    parser.add_argument("--tolerance", type=float, default=.02)
    parser.add_argument("--scoretol", type=float, default=.02)
    parser.add_argument("--mingain", type=float, default=0.)
    parser.add_argument("--slack", type=float, default=0.)
    parser.add_argument("--interaction", type=str.casefold, choices=("full", "additive"), default="full")
    parser.add_argument("--decision", type=str.casefold, choices=("joint", "scalar", "corners", "dense"), default="joint")
    parser.add_argument("--stages", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bands", type=str.casefold, choices=("both", "low", "high"), default="both")
    parser.add_argument("--coupling", type=str.casefold, choices=("full", "diagonal"), default="full")
    parser.add_argument("--grid", type=positive, default=2)
    parser.add_argument("--active", type=positive, default=4)
    parser.add_argument("--step", type=float, default=.25)
    parser.add_argument("--radius", type=float, default=.75)
    parser.add_argument("--budget", type=float, default=.0005)
    parser.add_argument("--sparse", type=float, default=.001)
    parser.add_argument("--ridge", type=float, default=.001)
    parser.add_argument("--backtrack", type=nonnegative, default=3)
    parser.add_argument("--globalstage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--localtext", default=None)
    parser.add_argument("--localization", type=str.casefold, choices=("baseline", "activity", "attention", "hybrid"), default="baseline")
    parser.add_argument("--attentionwords", default=None)
    parser.add_argument("--attentionqueries", type=positive, default=1024)
    parser.add_argument("--blend", type=float, default=.5)
    parser.add_argument("--threshold", type=float, default=.4)
    parser.add_argument("--signalfloor", type=float, default=1e-8)
    parser.add_argument("--components", type=positive, default=8)
    parser.add_argument("--minimum", type=positive, default=2)
    parser.add_argument("--small", type=nonnegative, default=2)
    parser.add_argument("--dilation", type=nonnegative, default=1)
    parser.add_argument("--partition", type=str.casefold, choices=("grid", "adaptive"), default="grid")
    parser.add_argument("--regions", type=positive, default=4)
    parser.add_argument("--leaf", type=positive, default=1)
    parser.add_argument("--partitionpenalty", type=float, default=.001)
    parser.add_argument("--probes", type=nonnegative, default=0)
    parser.add_argument("--trust", type=str.casefold, choices=("fixed", "curvature"), default="fixed")
    parser.add_argument("--maps", type=nonnegative, default=0)
    return parser

def localization(args):
    from engine.locality import Config
    config = Config(mode=args.localization, blend=args.blend, threshold=args.threshold, floor=args.signalfloor, components=args.components, minimum=args.minimum, small=args.small, dilation=args.dilation, partition=args.partition, regions=args.regions, leaf=args.leaf, penalty=args.partitionpenalty)
    config.validate()
    return config

def jointconfig(args):
    from engine.joint import Config
    config = Config(strength=args.strength, levels=args.levels, checks=args.checks, tolerance=args.tolerance, scoretol=args.scoretol, gain=args.mingain, slack=args.slack, budget=args.budget, interaction=args.interaction, decision=args.decision)
    config.validate()
    for time in (args.lowtime, args.hightime):
        if time is not None:
            times([time])
            if args.proposal == "chord" and time <= args.delta:
                raise ValueError("Band times must exceed the chord interval.")
    if args.method == "joint" and args.bands != "both":
        raise ValueError("Joint diagnostics always measure both bands independently.")
    if args.method == "joint" and args.partition != "grid":
        raise ValueError("Adaptive partitioning belongs to the legacy thread control.")
    return config

def projectionconfig(args):
    from engine.projection import Config
    config = Config(grid=args.grid, step=args.step, radius=args.radius, ridge=args.ridge, backtrack=args.backtrack, slack=args.slack, minreduce=args.minreduce, ranktol=args.ranktol, lock=args.lock, coupling=args.coupling, basis=args.basis, calibration=args.calibration, updates=args.updates, defect=args.defect, factorial=args.factorial)
    config.validate()
    args.donors = times(args.donors) if args.donors else []
    if args.size % 16 or args.grid > args.size // 16:
        raise ValueError("Projection requires a fitting grid on an even SD latent lattice.")
    if 2 * args.grid ** 2 * (1 + len(args.donors)) > 64:
        raise ValueError("The requested dictionary exceeds sixty-four raw directions.")
    if args.proposal == "chord" and any(value <= args.delta for value in args.donors):
        raise ValueError("Donor times must exceed the chord interval.")
    if args.method == "projection" and (args.partition != "grid" or args.bands != "both"):
        raise ValueError("Projection requires an aligned fixed grid and both band families.")
    if args.method == "projection" and (args.lowtime is not None or args.hightime is not None):
        raise ValueError("Use --donors for projection and band times for joint selection.")
    if args.method == "projection" and args.stages:
        raise ValueError("Stage diagnostics belong to the joint control.")
    return config

def pathconfig(args):
    from engine.poll import Config
    overrides = jsonfile(args.constraints, {})
    if not isinstance(overrides, dict) or overrides.keys() - {"slacks", "scales"}:
        raise ValueError("Constraints support only component slacks and scales.")
    config = Config(candidates=args.candidates, step=args.pollstep, minimum=args.minstep, slack=args.scoretol, priorslack=args.priorslack, semanticscale=args.semanticscale, damagescale=args.damagescale, priorscale=args.priorscale, axes=args.axes, eta=args.eta, slacks=overrides.get("slacks", {}), scales=overrides.get("scales", {}), algorithm=args.algorithm, priorcontrol=args.priorcontrol, priorweight=args.priorweight, secant=args.secant, ridge=args.ridge, condition=args.condition, curvature=args.curvature, responseerror=args.responseerror, hessian=args.hessian, damping=args.damping, critical=args.critical, monitor=args.monitor, compensation=args.compensation, release=args.release, geometry=args.geometry, ranktol=args.ranktol, frontier=args.frontier, frontiersteps=args.frontiersteps, solversteps=args.solversteps, certificatestep=args.certificatestep, certificate=args.certificate, radiusprofile=jsonfile(args.radiusprofile, {}), selection=args.selection, gainloss=args.gainloss, goals=tuple(args.goals or ()), archive=args.archive)
    config.validate()
    from engine.limit import Budget
    Budget(args.maxsamples, args.maxcalls, args.seconds)
    if args.size % 16:
        raise ValueError("Path control requires even latent geometry.")
    if args.reference != "auto" or args.donors or args.lowtime is not None or args.hightime is not None or args.stages:
        raise ValueError("Path control uses its own reference and two baseline bands.")
    if args.guidance < 0 or not args.cfg and args.guidance != 1:
        raise ValueError("Nonunit guidance requires an explicit CFG backend.")
    if args.adapter and (args.cfg or args.guidance != 1 or args.poststeps != 1 or not args.cleanup):
        raise ValueError("Single native generators cannot invent CFG or DDIM stages.")
    if args.adapter and args.prior != "off":
        raise ValueError("External generators need --prior off without native consistency.")
    if args.factorial and not args.control:
        raise ValueError("Decoded mechanism diagnostics require --control.")
    if args.method == "fixed" and (args.prior != "off" or args.heldout or args.axes != "none"):
        raise ValueError("Fixed interventions require --prior off --axes none without held-out priors.")
    if any(not isinstance(value, str) or not value.strip() for value in (args.attributes or []) + (args.retain or [])):
        raise ValueError("Semantic component descriptions must be nonempty.")
    return config

def jsonfile(value, default=None):
    if value is None:
        return default
    return readjson(locate(Path("."), clean(value, "configuration")))

def parser():
    result = common(argparse.ArgumentParser(description="Construct endpoint-aligned fields and control protected responses."))
    result.add_argument("--image", required=True)
    result.add_argument("--source", required=True)
    result.add_argument("--target", required=True)
    result.add_argument("--region", default=None)
    result.add_argument("--polarity", type=str.casefold, choices=("edit", "protect"), default="edit")
    result.add_argument("--output", default="./output.png")
    result.add_argument("--report", default="./report.json")
    return result

def readimage(value, mode="RGB"):
    path = locate(Path("."), clean(value, "image"))
    return pixels(path, mode)

def writejson(path, payload):
    path = clean(str(path), "report")
    publish(path, payload)

def saveartifacts(session, root):
    from engine.session import toimage
    images = getattr(session, "artifacts", {})
    if not images:
        return {}
    root = destination(str(root))
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    result = {}
    for name, pixels in sorted(images.items()):
        path = root / (name + ".png")
        toimage(pixels).save(path)
        result[name] = path.as_posix()
    return result

def validate(args):
    from engine.field import options
    options(args.field, args.fieldridge, args.fieldprobe, args.fieldradius, args.fieldtolerance, args.fieldgeometry)
    if args.field != "chord" and args.method not in {"path", "raw", "fixed"}:
        raise ValueError("Historical methods require --field chord; current fields use path/raw/fixed modes.")
    if args.compensation and args.method != "path":
        raise ValueError("Compensation probes belong only to adaptive path control.")
    if args.compensation and args.algorithm != "trust":
        raise ValueError("Compensation probes require trust response models.")
    if args.frontier and (args.method != "path" or args.algorithm != "trust"):
        raise ValueError("Frontier diagnostics require adaptive trust path control.")
    if any(isinstance(value, float) and not math.isfinite(value) for value in vars(args).values()):
        raise ValueError("Numeric options must be finite, including inactive report settings.")
    if args.flowguidance < 0:
        raise ValueError("Flow guidance embeddings must be nonnegative.")
    if args.backbone != "vp":
        if args.prior not in {"flow", "off"} or args.prior == "flow" and not args.flowconsistency or args.variance is not None:
            raise ValueError("Flow backbones require --prior off or declared --flowconsistency --prior flow.")
        if args.adapter or args.method not in {"path", "raw", "fixed"} or args.localization in {"attention", "hybrid"}:
            raise ValueError("Native flow control supports path/raw/fixed modes and frozen non-attention masks.")
        if args.cleanup and not 0 < args.end < 1:
            raise ValueError("Flow field editing requires an interior continuation sigma.")
    elif args.prior == "flow" or args.flowconsistency:
        raise ValueError("Flow consistency requires a native flow backbone.")
    args.times = times(args.times)
    if args.factorial and args.method not in {"projection", "path", "fixed"}:
        raise ValueError("Factorial diagnostics require projection or path control.")
    if args.size < 8 or args.size % 8 or not math.isfinite(args.start) or not 0 < args.start <= 1:
        raise ValueError("Size must be divisible by eight and start must be in (0, 1].")
    if not math.isfinite(args.delta) or not 0 <= args.delta < args.start:
        raise ValueError("Delta must be nonnegative and smaller than start.")
    if args.method in {"thread", "global"} and args.proposal == "chord" and min(args.times) <= args.delta:
        raise ValueError("Chord proposal times must exceed delta.")
    if args.method not in {"base", "raw", "fixed"} and not args.clip:
        raise ValueError("Adaptive selection requires a local --clip folder.")
    localization(args)
    if args.method in {"path", "raw", "fixed"}:
        config = pathconfig(args)
    elif args.method == "projection":
        config = projectionconfig(args)
    elif args.method == "joint":
        config = jointconfig(args)
    else:
        from engine.response import Config
        config = Config(grid=args.grid, step=args.step, radius=args.radius, sparse=args.sparse, ridge=args.ridge, budget=args.budget, active=args.active, backtrack=args.backtrack, coupling=args.coupling, local=args.method != "global", globalstage=args.globalstage, bands=args.bands, probes=args.probes, trust=args.trust, maps=args.maps)
        config.validate()
    if not math.isfinite(args.end) or not 0 <= args.end <= 1 or not math.isfinite(args.scale) or args.scale < 0:
        raise ValueError("Refinement time and edit scale are outside their supported ranges.")
    return config

def load(args):
    seed(args.seed, getattr(args, "deterministic", False))
    validate(args)
    root = str(locate(Path("."), clean(args.model, "model")))
    if args.backbone == "vp":
        from engine.session import Editor
        editor = Editor.load(root, args.device, args.precision, args.size)
    else:
        from engine.flow import Editor
        editor = Editor.load(root, args.device, args.precision, args.size, args.backbone, args.flowguidance, args.flowconsistency)
    judge = None
    if args.clip and args.method not in {"raw", "fixed"}:
        from engine.judge import ClipJudge
        judge = ClipJudge.load(str(locate(Path("."), clean(args.clip, "clip"))), args.device or "cpu")
    return editor, judge

def stopped(session, metadata, method, error, latent=None, judge=None, initial=None):
    # Preserve a paid raw output when calibration cannot continue.
    from engine.limit import Exhausted
    output, selected = session.origin, "source"
    if latent is not None:
        try:
            output, selected = session.decode(latent).clamp(0, 1), "raw"
        except Exhausted:
            pass
    counters = {key: value - (initial or {}).get(key, 0) for key, value in judge.stats().items()} if judge is not None else {}
    report = {**metadata, "method": method, "selected": selected, "fallback": True, "constraintmet": False, "budgetstopped": True, "reason": str(error), "stats": session.stats(), "judgestats": counters, "scope": "The stopped request retains its paid raw output or unchanged source in evaluation coverage."}
    return output, report, session

def run(editor, judge, image, source, target, args, mask=None, polarity="edit"):
    from dataclasses import replace
    from engine.response import select
    from engine.locality import locate as region, partition
    config = validate(args)
    seed(args.seed, getattr(args, "deterministic", False))
    initialstats = judge.stats() if judge is not None else {}
    from engine.limit import Budget, Exhausted
    limit = Budget(args.maxsamples, args.maxcalls, args.seconds) if args.method in {"path", "raw", "fixed"} else None
    session = editor.prepare(image, source, target, args.seed, args.noise, args.end, args.cleanup, args.scale, region=mask, polarity=polarity)
    if limit is not None:
        session.limit = limit
    pixels = session.origin.detach().float().cpu().contiguous().numpy()
    metadata = {"settings": dict(vars(args)), "source": source, "target": target, "imagesha256": hashlib.sha256(pixels.tobytes()).hexdigest(), "imageshape": list(pixels.shape), "imagehashformat": "fitted RGB float32 NCHW", "masksha256": hashlib.sha256(mask.convert("L").tobytes()).hexdigest() if mask is not None else None, "modelcheckpoint": getattr(editor, "metadata", None), "controlcheckpoint": getattr(judge, "metadata", None), "checkpointverification": "Weight hashes are recorded when local checkpoints are loaded."}
    metadata["reproducibility"] = snapshot()
    metadata["controlactive"] = judge is not None and args.method not in {"raw", "fixed", "base"}
    metadata["configuration"] = {key: jsonfile(getattr(args, key, None)) for key in ("constraints", "probabilities", "radiusprofile")}
    adapterfile = getattr(args, "adapter", None)
    metadata["adaptersha256"] = digest(locate(Path("."), clean(adapterfile, "adapter"))) if adapterfile is not None else None
    if source.strip().casefold() == target.strip().casefold():
        return session.origin, {**metadata, "method": args.method, "noop": True, "reason": "case-insensitive identical prompts", "stats": session.stats(), "judgestats": {}, "seed": args.seed, "proposal": args.proposal, "explicitregion": mask is not None}, session
    if args.method != "base" and session.mask is None and args.localization in {"attention", "hybrid"}:
        session.localization(source, target, args.attentionwords, args.attentionqueries)
    native, neutral, localizationbase = None, None, None
    if args.method in {"path", "raw", "fixed"}:
        from engine.field import build
        from engine.native import create
        if args.adapter:
            from engine.adapter import load as adapter
            native = adapter(args.adapter, session)
        else:
            native = create(session, args.poststeps, args.guidance, args.cfg, args.referencepath)
        try:
            localizationbase = session.difference(session.editor.index(args.start))
            constructed = build(session, native, args.field, args.start, args.delta, args.fieldridge, args.fieldprobe, args.fieldradius, args.fieldtolerance, args.fieldgeometry)
            base, neutral = constructed.value, constructed.neutral
            metadata["field"] = constructed.report
            limit.require(native.cost(0.), native.queries())
        except Exhausted as error:
            return stopped(session, metadata, args.method, error, judge=judge, initial=initialstats)
    else:
        base = session.field(args.start, args.delta)
        metadata["field"] = {"requested": "chord", "actual": "chord", "status": "historical"}
    if args.method in {"path", "raw", "fixed"}:
        from engine.path import Path as ReferencePath
        from engine.poll import select as search
        from engine.prior import Prior
        attention = session.attentions.get(session.editor.index(args.start)) if args.localization in {"attention", "hybrid"} else None
        editmask, localizationreport = region(localizationbase, {"base": localizationbase}, session.mask, attention, localization(args))
        localizationreport["fieldsource"] = "Shared instantaneous contrast at the declared start time."
        scoringmask = session.area if session.area is not None else editmask
        metadata["fieldsha256"] = hashlib.sha256(base.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
        metadata["fieldnorm"] = float(base.double().norm())
        metadata["protectionsha256"] = hashlib.sha256(scoringmask.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
        score = None
        if args.method == "path" and judge is not None:
            score = judge.prepare(session.origin, source, target, scoringmask, args.localtext, attributes=args.attributes, retain=args.retain) if args.attributes or args.retain else judge.prepare(session.origin, source, target, scoringmask, args.localtext)
        path = ReferencePath(native, base, args.representation, args.seed, neutral)
        if args.method == "raw":
            from engine.poll import protection
            from engine.access import describe
            try:
                latent, output = path.raw()
            except Exhausted as error:
                return stopped(session, metadata, args.method, error, path.fields.get((1., 1.)), judge, initialstats)
            constructed.observe(latent)
            damage, mass = protection(session.origin, 1 - scoringmask)
            scores = score(output) if score is not None else None
            report = {**metadata, "method": "raw", "renderer": "raw", "scores": scores, "damage": damage(output), "protectedvalid": mass > 0, "selected": "raw", "evaluations": int(score is not None), "backend": native.metadata(), "stats": session.stats(), "judgestats": {key: value - initialstats.get(key, 0) for key, value in judge.stats().items()} if judge is not None else {}, "explicitregion": mask is not None, "localization": localizationreport, "scope": "Native target continuation without reference subtraction."}
            report["access"] = describe(native, Prior(session, mode="off"))
            report["adoptedfieldnorm"] = metadata["fieldnorm"]
            return output, report, session
        if args.method == "fixed":
            # Export the intervention without feasibility-based reselection.
            from engine.poll import protection
            from engine.access import describe
            try:
                requested = (1., 1., args.eta)
                point = path.canonical(requested)
                _, output = path.candidate(point)
                point = path.canonical(point)
            except Exhausted as error:
                return stopped(session, metadata, args.method, error, path.fields.get((1., 1.)), judge, initialstats)
            constructed.observe(path.fields.get((1., 1.)))
            damage, mass = protection(session.origin, 1 - scoringmask)
            report = {**metadata, "method": "fixed", "renderer": args.representation, "requested": list(requested), "point": list(point), "selected": "fixed", "fallback": False, "selection": "unfiltered intervention", "scores": score(output) if score is not None else None, "damage": damage(output), "protectedvalid": mass > 0, "evaluations": int(score is not None), "backend": native.metadata(), "path": path.metadata(), "explicitregion": mask is not None, "localization": localizationreport, "scope": "Mechanism intervention; no semantic, protection, or quality guarantee."}
            if args.control:
                try:
                    report["mechanism"] = path.mechanism(args.factorial)
                except Exhausted as error:
                    report["mechanism"] = {"measured": False, "reason": str(error)}
            report.update({"access": describe(native, Prior(session, mode="off")), "stats": session.stats(), "judgestats": {key: value - initialstats.get(key, 0) for key, value in judge.stats().items()} if judge is not None else {}})
            report["adoptedfieldnorm"] = metadata["fieldnorm"]
            return output, report, session
        probabilities = jsonfile(args.probabilities)
        prior = Prior(session, args.priorprobes, args.seed ^ 0x317, args.prior, probabilities, args.variance)
        from engine.access import describe
        capabilities = describe(native, prior)
        try:
            result = search(path, score, 1 - scoringmask, prior, config)
        except Exhausted as error:
            return stopped(session, metadata, args.method, error, path.fields.get((1., 1.)), judge, initialstats)
        constructed.observe(path.fields.get((1., 1.)))
        report = result.report
        report["access"] = capabilities
        if args.control:
            try:
                report["mechanism"] = path.mechanism(args.factorial)
            except Exhausted as error:
                report["mechanism"] = {"measured": False, "reason": str(error)}
        if args.heldout and args.prior != "off":
            try:
                heldout = Prior(session, args.priorprobes, args.seed ^ 0x971, args.prior, probabilities, args.variance)
                report["heldout"] = {"value": heldout(result.latent, "heldout"), "metadata": heldout.metadata(), "scope": "Postselection diagnostic, never used for reselection."}
            except Exhausted as error:
                report["heldout"] = {"value": None, "reason": str(error)}
        judgestats = {key: value - initialstats.get(key, 0) for key, value in judge.stats().items()}
        report.update({**metadata, "method": "path", "renderer": args.representation, "seed": args.seed, "stats": session.stats(), "judgestats": judgestats, "explicitregion": mask is not None, "localization": localizationreport, "mask": "explicit" if session.mask is not None else "inferred", "evaluationnote": "Controller proxies and consistency are not independent quality evidence."})
        amplitudes = report.get("point") or (1., 1., 0.)
        report["adoptedfieldnorm"] = float((amplitudes[0] * path.low + amplitudes[1] * path.high).double().norm())
        return result.image, report, session
    reference = ("pixel" if args.method == "projection" else "raw") if args.reference == "auto" else args.reference
    render = session.render if reference == "raw" else lambda field: session.render(field, reference=reference)
    metadata["renderer"] = reference
    if args.method == "base":
        output = render(base)
        report = {"method": "base", "start": args.start, "delta": args.delta}
    elif args.method == "projection":
        from engine.projection import select as project
        pool = {time: session.field(time, args.delta if args.proposal == "chord" else 0.) for time in args.donors}
        attention = session.attention() if args.localization in {"attention", "hybrid"} else None
        editmask, localizationreport = region(base, {"base": base, **pool}, session.mask, attention, localization(args))
        score = judge.prepare(session.origin, source, target, editmask, args.localtext)
        raw, rawscores, admit = None, None, None
        if reference != "raw" and args.guard:
            from engine.reference import safeguard
            raw = session.render(base)
            rawscores = score(raw)
            admit = lambda pixels, scores: safeguard(pixels, raw, session.origin, editmask, scores, rawscores, args.slack, args.minreduce)[1]["accepted"]
        transport = (lambda field: session.render(field, cleanup=False, reference=reference)) if args.factorial and args.cleanup else None
        result = project(base, render, score, session.origin, editmask, projectionconfig(args), pool, transport, admit)
        output, report = result.image, result.report
        if reference != "raw" and args.guard:
            output, guard = safeguard(output, raw, session.origin, editmask, report["scores"], rawscores, args.slack, args.minreduce)
            report["referenceguard"] = guard
            report["controller"] = {key: report[key] for key in ("selected", "fallback", "constraintmet", "scores", "audit", "damagechange", "gain", "weights", "evaluations")}
            report["evaluations"] += 1
            report.update({"scores": guard["scores"], "audit": guard["audit"], "damagechange": guard["damagechange"] if guard["accepted"] else 0., "gain": guard["scores"]["score"] - guard["rawscores"]["score"], "constraintmet": guard["accepted"]})
            if not guard["accepted"]:
                report.update({"selected": "raw", "fallback": True, "weights": [], "reason": "Centered editing failed actual raw-baseline checks."})
            else:
                report.update({"selected": "centered" if report["selected"] == "base" else report["selected"], "fallback": False})
        report.update({"localization": localizationreport, "partition": {"mode": "aligned", "grid": args.grid}, "scope": "Reference-centered output control with raw semantic and protection safeguards." if reference != "raw" else "Joint-response calibrated raw field control.", "donors": args.donors, "timesnote": "Only --donors adds times; the legacy --times pool is inactive."})
    elif args.method == "joint":
        from engine.joint import select as jointselect
        from engine.wavelet import split
        lowfield = base if args.lowtime is None else session.field(args.lowtime, args.delta if args.proposal == "chord" else 0.)
        highfield = base if args.hightime is None else session.field(args.hightime, args.delta if args.proposal == "chord" else 0.)
        pool = {"low": lowfield, "high": highfield}
        attention = session.attention() if args.localization in {"attention", "hybrid"} else None
        editmask, localizationreport = region(base, pool, session.mask, attention, localization(args))
        score = judge.prepare(session.origin, source, target, editmask, args.localtext)
        transport = (lambda field: session.render(field, cleanup=False, reference=reference)) if args.stages else None
        result = jointselect(base, split(lowfield)[0], split(highfield)[1], render, score, session.origin, editmask, jointconfig(args), transport)
        output, report = result.image, result.report
        report.update({"localization": localizationreport, "bandtimes": {"low": args.lowtime, "high": args.hightime, "default": "baseline chord"}, "partition": {"mode": "none"}, "scope": "Global band interventions with frozen spatial scoring support.", "timesnote": "The legacy --times pool is inactive; only explicit band times are used."})
    else:
        pool = {time: session.field(time, args.delta if args.proposal == "chord" else 0.) for time in args.times}
        attention = session.attention() if args.localization in {"attention", "hybrid"} else None
        editmask, localizationreport = region(base, pool, session.mask, attention, localization(args))
        score = judge.prepare(session.origin, source, target, editmask, args.localtext)
        windows, partitionreport = partition(base, pool, editmask, args.grid, localization(args)) if config.local and not localizationreport["uncertain"] else ([], {"mode": "disabled", "reason": "Global-only mode or uncertain localization."})
        if not windows:
            config = replace(config, local=False)
        result = select(session.anchor, base, pool, render, score, session.origin, editmask, config, windows)
        output, report = result.image, result.report
        report.update({"localization": localizationreport, "partition": partitionreport})
    if args.control:
        report["control"] = session.diagnostic(base, session.mask if args.method == "base" else editmask)
        if judge is not None:
            referenceimage = session.neutral(session.cleanup, pixels=True)["pixels"].clamp(0, 1)
            report["control"]["zeroscores"] = score(referenceimage) if args.method != "base" else judge.prepare(session.origin, source, target, session.mask, args.localtext)(referenceimage)
    judgestats = {key: value - initialstats.get(key, 0) for key, value in judge.stats().items()} if judge is not None else {}
    report.update({**metadata, "method": args.method, "times": args.times, "proposal": args.proposal, "seed": args.seed, "stats": session.stats(), "judgestats": judgestats, "explicitregion": mask is not None, "mask": "explicit" if session.mask is not None else "inferred"})
    return output, report, session

def main():
    args = parser().parse_args()
    image = readimage(args.image)
    mask = readimage(args.region, "L") if args.region else None
    output, report = destination(args.output, "output"), destination(args.report, "report")
    if output.suffix != ".png" or report.suffix != ".json":
        raise ValueError("Use lossless .png outputs and .json reports.")
    inputs = {locate(Path("."), clean(value, "input")).resolve() for value in (args.image, args.region) if value}
    if output.resolve() in inputs or report.resolve() in inputs or report.resolve() == output.resolve():
        raise ValueError("Inputs, output, and report must have different paths.")
    if output.exists() or report.exists():
        raise FileExistsError("Output or report already exists.")
    proof = output.parent / (output.stem + "proof")
    if args.factorial and proof.exists():
        raise FileExistsError(proof)
    editor, judge = load(args)
    image, payload, session = run(editor, judge, image, args.source, args.target, args, mask, args.polarity)
    from engine.session import toimage
    output.parent.mkdir(parents=True, exist_ok=True)
    toimage(image).save(output)
    payload["artifacts"] = saveartifacts(session, proof)
    payload["artifactsha256"] = {key: digest(Path(value)) for key, value in payload["artifacts"].items()}
    writejson(report, payload)
    print(json.dumps({"image": str(output), "report": str(report), "stats": session.stats()}))

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
