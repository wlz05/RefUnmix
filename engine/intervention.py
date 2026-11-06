# -*- coding: utf-8 -*-
# RefUnmix: Separating Prompt Response from Shared Transport in One-Step Image Editing

import torch
import torch.nn.functional as functional
from engine.response import region
from engine.responsibility import resolve
from engine.grounding import reference as grounded, restrict, supply, support

MODES = {"full", "assignment", "reference", "gating", "target", "grounding", "attraction"}

def quantize(pixels):
    return (pixels.double().clamp(0, 1) * 255).round().float() / 255

def edit(session, args, transported, metadata, times, calibration):
    mode = args.ablation
    if mode not in MODES or not args.referencelocal:
        raise ValueError("Unknown grounded RefUnmix intervention.")
    response = {"ablation": mode, "effectiveinnovationgain": args.innovationgain}
    if mode == "target":
        predictions = session.predict(transported, session.noises[:1], session.editor.index(session.end), session.target)
        pixels = quantize(session.decode(predictions))
        response["effectiveinnovationgain"] = 1.
    else:
        anchors = torch.cat((session.anchor, transported, transported))
        embeddings = torch.cat((session.source, session.source, session.target))
        with session.collector:
            predictions = session.predict(anchors, session.noises[:1].expand(3, -1, -1, -1), session.editor.index(session.end), embeddings, 0., 0., 3)
        captured = session.collector.result()
        if captured is None and any(session.collector.indices):
            raise RuntimeError("Matched endpoint attention was not captured by the backend.")
        tick = session.editor.index(session.end)
        session.attentions[tick] = captured
        session.attentionlog[tick] = session.collector.report()
        weight = region(predictions[2:3] - predictions[1:2]).pow(args.responsepower)
        positive, negative = weight.pow(args.responsesharp), (1 - weight).pow(args.responsesharp)
        weight = functional.interpolate(positive / (positive + negative).clamp_min(1e-12), session.origin.shape[-2:], mode="bilinear", align_corners=False)
        if mode == "reference":
            reference = session.origin
            shared, target = [session.decode(predictions[index:index + 1]) for index in (1, 2)]
        else:
            reference, shared, target = [session.decode(predictions[index:index + 1]) for index in range(3)]
        attention = session.attention()
        if attention is None:
            attention = torch.zeros_like(predictions[:1, :1])
        scope = support(attention, session.origin.shape[-2:])
        if mode != "grounding":
            if mode != "reference":
                reference, scope = grounded(session.origin, reference, attention)
            weight = supply(weight, scope)
        session.artifacts["scope"] = scope.expand_as(session.origin)
        conditional, advection = target - shared, shared - reference
        if mode == "assignment":
            from engine.fixed import resolve as fixed
            pixels, details = fixed(session.origin, conditional, advection, weight, args.innovationgain)
            response.update(details)
        elif mode == "gating":
            pixels = quantize(session.origin + weight * (conditional + advection))
            response["effectiveinnovationgain"] = 1.
        else:
            pixels, details = resolve(session.origin, conditional, advection, weight, args.responsepenalty, args.innovationgain)
            response.update(details)
        if mode != "grounding":
            pixels, response["support"] = restrict(session.origin, pixels, scope, args.retainedcontext)
            response["digitalerrorafter"] = response["support"]["digitalerrorafter"]
        metadata["grounding"] = {"enabled": mode != "grounding", "mean": float(scope.mean()), "tokens": session.collector.indices, "reference": "source RGB" if mode == "reference" else "matched endpoint with source transition", "attention": "Same source-zero and target-field endpoint queries"}
    report = {**metadata, "method": "single", "renderer": "intervention", "ablation": mode, "field": {"requested": args.singlemode, "actual": args.singlemode, "times": times, "gain": args.singlegain, "calibration": calibration, "response": response}, "stats": session.stats(), "selected": "single", "controller": {"enabled": False}, "candidates": [], "explicitregion": False, "scope": "Paired structural control; target-only and unified gating are constructed mechanism comparisons."}
    return pixels, report, session

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
