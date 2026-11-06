# -*- coding: utf-8 -*-
# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

import torch
import torch.nn.functional as functional
from engine.response import region
from engine.responsibility import resolve

def quantize(pixels):
    return (pixels.double().clamp(0, 1) * 255).round().float() / 255

def edit(session, args, transported, metadata, times, calibration):
    mode = args.ablation
    if mode not in {"assignment", "reference", "gating", "target"}:
        raise ValueError("Unknown structural RefUnmix ablation.")
    if mode == "target":
        anchors, embeddings = transported, session.target
    elif mode == "reference":
        anchors = torch.cat((transported, transported))
        embeddings = torch.cat((session.source, session.target))
    else:
        anchors = torch.cat((session.anchor, transported, transported))
        embeddings = torch.cat((session.source, session.source, session.target))
    count = anchors.shape[0]
    predictions = session.predict(anchors, session.noises[:1].expand(count, -1, -1, -1), session.editor.index(session.end), embeddings)
    response = {"ablation": mode, "effectiveinnovationgain": args.innovationgain if mode == "reference" else 1}
    if mode == "target":
        pixels = quantize(session.decode(predictions))
    else:
        weight = region(predictions[-1:] - predictions[-2:-1]).pow(args.responsepower)
        positive, negative = weight.pow(args.responsesharp), (1 - weight).pow(args.responsesharp)
        weight = functional.interpolate(positive / (positive + negative).clamp_min(1e-12), session.origin.shape[-2:], mode="bilinear", align_corners=False)
        decoded = [session.decode(predictions[index:index + 1]) for index in range(count)]
        reference, shared, target = (session.origin, decoded[0], decoded[1]) if mode == "reference" else decoded
        conditional, advection = target - shared, shared - reference
        if mode == "assignment":
            pixels = quantize(session.origin + conditional + weight * advection)
        elif mode == "gating":
            pixels = quantize(session.origin + weight * (conditional + advection))
        else:
            pixels, details = resolve(session.origin, conditional, advection, weight, args.responsepenalty, args.innovationgain)
            response.update(details)
    report = {**metadata, "method": "single", "renderer": "ablation", "ablation": mode, "field": {"requested": args.singlemode, "actual": args.singlemode, "times": times, "gain": args.singlegain, "calibration": calibration, "response": response}, "stats": session.stats(), "selected": "single", "controller": {"enabled": False}, "candidates": [], "explicitregion": False, "scope": "Target-only is a constructed truncation; other modes retain their declared endpoint and allocation mechanisms."}
    return pixels, report, session

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
