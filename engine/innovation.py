# -*- coding: utf-8 -*-
# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

import math
import torch
from engine.session import tensor

@torch.no_grad()
def amplify(origin, conditional, shared, responsibility, coefficient, baseline, gain=2., penalty=1.):
    # Maximize eta under ||eta*d + s*a||^2 <= ||d + q*a||^2.
    for value in (origin, conditional, shared, baseline):
        tensor(value, 3)
    if not math.isfinite(gain) or not 1 <= gain <= 4:
        raise ValueError("Innovation gain must be finite and in [1, 4].")
    for value in (responsibility, coefficient):
        tensor(value, 1)
    if any(value.shape != origin.shape or value.device != origin.device for value in (conditional, shared, baseline)) or any(value.shape[-2:] != origin.shape[-2:] or value.device != origin.device or torch.any((value < 0) | (value > 1)) for value in (responsibility, coefficient)) or not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("Innovation endpoint geometry is invalid.")
    if torch.any((origin < 0) | (origin > 1)) or torch.any((baseline < 0) | (baseline > 1)):
        raise ValueError("Source and baseline pixels must be in [0, 1].")
    if gain == 1:
        return baseline, {"requested": gain, "effective": 1., "acceptedfraction": 0.}
    x, d, a, w, q = (value.double() for value in (origin, conditional, shared, responsibility, coefficient))
    square = a.square().sum(1, keepdim=True)
    signal = d.square().sum(1, keepdim=True)
    cross = (d * a).sum(1, keepdim=True)
    budget = (d + q * a).square().sum(1, keepdim=True)
    safe = square.clamp_min(1e-20)
    schur = (signal - cross.square() / safe).clamp_min(0)
    positivecap = (budget / signal.clamp_min(1e-20)).clamp_min(0).sqrt()
    switch = square / (-cross).clamp_min(1e-20)
    freecap = (budget / schur.clamp_min(1e-20)).clamp_min(0).sqrt()
    discriminant = (cross.square() - signal * (square - budget)).clamp_min(0)
    boundarycap = (-cross + discriminant.sqrt()) / signal.clamp_min(1e-20)
    negativecap = torch.where(freecap <= switch, freecap, boundarycap)
    maximum = torch.where(cross >= 0, positivecap, negativecap)
    maximum = torch.where((square > 1e-20) & (signal > 1e-20), maximum, 1.)
    effective = maximum.clamp(1., gain)
    gram = (square * signal - cross.square()).clamp_min(0)
    determinant = (square * budget - effective.square() * gram).clamp_min(0)
    lower = ((-effective * cross - determinant.sqrt()) / safe).clamp(0, 1)
    upper = ((-effective * cross + determinant.sqrt()) / safe).clamp(0, 1)
    desired = (penalty * w * square - (1 - w) * effective * cross) / ((penalty + 1 - w) * square).clamp_min(1e-20)
    selected = torch.minimum(torch.maximum(desired, lower), upper)
    valid = (square > 1e-20) & (signal > 1e-20) & (lower <= upper)
    effective = torch.where(valid, effective, 1.)
    selected = torch.where(valid, selected, q)
    proposal = (x + effective * d + selected * a).clamp(0, 1)
    sourcebytes = (x.clamp(0, 1) * 255).round()
    candidatebytes = (proposal * 255).round()
    baselinebytes = (baseline.double() * 255).round()
    oldcost = (baselinebytes - sourcebytes).square().sum(1, keepdim=True)
    newcost = (candidatebytes - sourcebytes).square().sum(1, keepdim=True)
    # Round inward on the byte lattice when quantization exceeds the budget.
    contraction = (oldcost / newcost.clamp_min(1)).sqrt().clamp(max=1)
    delta = candidatebytes - sourcebytes
    inward = sourcebytes + delta.sign() * (delta.abs() * contraction).floor()
    quantized = torch.where(newcost <= oldcost, candidatebytes, inward)
    pixels = quantized.float() / 255
    actualcost = ((pixels.double() * 255).round() - sourcebytes).square().sum(1, keepdim=True)
    if not torch.isfinite(pixels).all() or torch.any(actualcost > oldcost):
        raise RuntimeError("Innovation amplification exceeded its digital endpoint budget.")
    report = {"requested": gain, "effective": float(effective.mean()), "acceptedfraction": float(valid.double().mean()), "quantizedfraction": float((newcost > oldcost).double().mean()), "digitalerrorbefore": float(oldcost.sum()), "digitalerrorafter": float(actualcost.sum()), "scope": "Per-pixel RGB source-error budget relative to the unamplified field."}
    return pixels, report

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
