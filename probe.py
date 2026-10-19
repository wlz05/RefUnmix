# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import argparse
import json
import torch
from data import destination
from engine.adapter import load as adapter
from engine.limit import Budget
from engine.native import create
from engine.field import build
from engine.session import toimage
from run import common, load, readimage, writejson

def main():
    parser = common(argparse.ArgumentParser(description="Measure matched conditional branches without claiming editing gains."))
    parser.set_defaults(method="raw", prior="off")
    parser.add_argument("--image", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", default="./probes")
    parser.add_argument("--withfield", action="store_true")
    parser.add_argument("--etas", nargs="+", type=float, default=[0., .5, 1.])
    args = parser.parse_args()
    output = destination(args.output)
    if output.exists():
        raise FileExistsError(output)
    if any(not 0 <= eta <= 1 for eta in args.etas):
        raise ValueError("Probe reference coefficients must be in [0, 1].")
    editor, _ = load(args)
    limit = Budget(args.maxsamples, args.maxcalls, args.seconds)
    session = editor.prepare(readimage(args.image), args.source, args.target, args.seed, args.noise, args.end, args.cleanup, args.scale)
    session.limit = limit
    native = adapter(args.adapter, session) if args.adapter else create(session, args.poststeps, args.guidance, args.cfg, args.referencepath)
    zero = torch.zeros_like(session.anchor)
    if args.withfield and not getattr(native, "fields", True):
        raise ValueError("This native generator lacks a legitimate field adapter.")
    constructed = build(session, native, args.field, args.start, args.delta, args.fieldridge, args.fieldprobe, args.fieldradius, args.fieldtolerance, args.fieldgeometry) if args.withfield else None
    limit.phase = "diagnostic"
    endpoints = {"sourcezero": native.endpoint(zero, 1.), "targetzero": constructed.neutral if constructed is not None and constructed.neutral is not None else native.endpoint(zero)}
    field = constructed.value if constructed is not None else None
    if field is not None:
        if not getattr(native, "fields", True):
            raise ValueError("This native generator lacks a legitimate field adapter.")
        endpoints.update({"sourcefield": native.endpoint(field, 1.), "targetfield": native.endpoint(field)})
    condition = endpoints["targetzero"] - endpoints["sourcezero"]
    bias = endpoints["sourcezero"] - session.anchor
    values = {"bias": float(bias.double().square().mean().sqrt()), "condition": float(condition.double().square().mean().sqrt())}
    if field is not None:
        effect = endpoints["sourcefield"] - endpoints["sourcezero"]
        interaction = endpoints["targetfield"] - endpoints["sourcefield"] - condition
        values.update({"field": float(effect.double().square().mean().sqrt()), "interaction": float(interaction.double().square().mean().sqrt()), "identityerror": float((endpoints["targetfield"] - session.anchor - bias - condition - effect - interaction).abs().max())})
        first = session.difference(session.editor.index(args.start))
        second = first if args.delta == 0 else session.difference(session.editor.index(args.start - args.delta))
        request = session.scale * (first.double() + second.double()) / 2
        displacement = endpoints["targetfield"].double() - endpoints["targetzero"].double()
        values["endpointfit"] = float((displacement - request).square().mean())
        values["requestnorm"] = float(request.norm())
    references = []
    if native.continuous:
        for eta in sorted(set(args.etas)):
            latent = endpoints["targetzero"] if eta == 0 else endpoints["sourcezero"] if eta == 1 else native.endpoint(zero, eta)
            affine = (1 - eta) * endpoints["targetzero"] + eta * endpoints["sourcezero"]
            references.append({"eta": eta, "nonaffinity": float((latent - affine).double().square().mean().sqrt())})
    images = {name: toimage(session.decode(value)) for name, value in endpoints.items()}
    output.mkdir(parents=True)
    for name, image in images.items():
        image.save(output / (name + ".png"))
    payload = {"schema": 1, "field": constructed.report if constructed is not None else None, "mechanism": values, "references": references, "backend": native.metadata(), "stats": session.stats(), "scope": "Matched native-condition diagnostics; no deletion or editing-quality claim without the original computation graph."}
    writejson(output / "summary.json", payload)
    print(json.dumps(payload, allow_nan=False))

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
