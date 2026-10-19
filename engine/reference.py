# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
from math import isfinite
import torch
from engine.response import Protection

def aligned(origin, edited, neutral):
    for value in (origin, edited, neutral):
        if not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != 3 or not value.is_floating_point() or not torch.isfinite(value).all().item():
            raise ValueError("Reference images must be finite floating point RGB tensors.")
    if edited.shape != origin.shape or neutral.shape != origin.shape or edited.device != origin.device or neutral.device != origin.device:
        raise ValueError("Reference images must share shape and device.")
    if not min(origin.shape) or torch.any((origin < 0) | (origin > 1)).item():
        raise ValueError("The original RGB image must be nonempty and in [0, 1].")

def center(origin, edited, neutral):
    # Subtract unclipped outputs before clamping.
    aligned(origin, edited, neutral)
    result = origin.float() + (edited.float() - neutral.float())
    if not torch.isfinite(result).all().item():
        raise RuntimeError("Centered output exceeds floating point range.")
    return result.clamp(0, 1)

def decomposition(origin, edited, neutral, mask=None):
    # Separate zero-field drift from editing response.
    aligned(origin, edited, neutral)
    mask = torch.zeros_like(origin[:, :1]) if mask is None else mask
    protection = Protection(origin, mask)
    intercept = neutral.double() - origin.double()
    response = edited.double() - neutral.double()
    total = edited.double() - origin.double()

    def energy(value):
        return float(value.square().mean())

    def parts(first, second, actual):
        left, right = float(first.square().sum()), float(second.square().sum())
        cross, whole = float(2 * (first * second).sum()), float(actual.square().sum())
        return {"intercept": left, "response": right, "cross": cross, "total": whole, "identityerror": abs(whole - left - right - cross)}

    first = (intercept.detach().cpu() * protection.factor.double()).flatten()
    second = (response.detach().cpu() * protection.factor.double()).flatten()
    report = {"unclipped": {"intercept": energy(intercept), "response": energy(response), "cross": float(2 * (intercept * response).mean()), "total": energy(total)}, "protected": parts(first, second, first + second) if protection.valid else {}, "protectedvalid": protection.valid, "coverage": protection.coverage, "rawdamage": protection.energy(edited.clamp(0, 1)), "centerdamage": protection.energy(center(origin, edited, neutral)), "neutraldamage": protection.energy(neutral.clamp(0, 1)), "saturation": float(((origin.double() + response < 0) | (origin.double() + response > 1)).double().mean()), "note": "A zero-field target branch can contain useful semantics; its drift is not automatically error."}
    report["unclipped"]["identityerror"] = abs(report["unclipped"]["total"] - report["unclipped"]["intercept"] - report["unclipped"]["response"] - report["unclipped"]["cross"])
    return report

def safeguard(candidate, fallback, origin, mask, scores, reference, slack=0., minreduce=0.):
    # Retain semantics while improving raw protection.
    from engine.projection import measured
    aligned(origin, candidate, fallback)
    if torch.any((candidate < 0) | (candidate > 1)).item() or torch.any((fallback < 0) | (fallback > 1)).item():
        raise ValueError("Guard candidates must be final RGB outputs in [0, 1].")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0 for value in (slack, minreduce)):
        raise ValueError("Guard tolerances must be finite and nonnegative.")
    scores, reference = measured(scores), measured(reference)
    if set(scores) != set(reference):
        raise ValueError("Guard score keys must match the raw reference.")
    protection = Protection(origin, mask)
    original, changed = protection.energy(fallback), protection.energy(candidate)
    semantic = all(scores[key] >= reference[key] - slack - 1e-10 for key in reference)
    improves = protection.valid and changed < original - max(minreduce, 1e-10)
    accepted = semantic and improves
    selected = candidate if accepted else fallback
    return selected, {"accepted": accepted, "semanticvalid": semantic, "improves": improves, "protectedvalid": protection.valid, "rawscores": reference, "candidatescores": scores, "rawdamage": original, "candidatedamage": changed, "damagechange": changed - original if protection.valid else None, "scores": scores if accepted else reference, "audit": protection.audit(selected), "rule": "Every recorded semantic score must retain the raw baseline while protected damage decreases."}

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
