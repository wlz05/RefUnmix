# -*- coding: utf-8 -*-
# RefUnmix: Separating Prompt Response from Shared Transport in One-Step Image Editing

import math
import torch
from engine.session import tensor

@torch.no_grad()
def resolve(origin, conditional, shared, responsibility, penalty=1., gain=1.):
    # Solve shared adoption from the coupled RGB responses.
    for value in (origin, conditional, shared):
        tensor(value, 3)
    tensor(responsibility, 1)
    if any(value.shape != origin.shape or value.device != origin.device for value in (conditional, shared)) or responsibility.shape[-2:] != origin.shape[-2:] or responsibility.device != origin.device or torch.any((responsibility < 0) | (responsibility > 1)) or not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("Endpoint responsibility geometry is invalid.")
    if torch.any((origin < 0) | (origin > 1)):
        raise ValueError("Source pixels must be in [0, 1].")
    x, d, a, w = (value.double() for value in (origin, conditional, shared, responsibility))
    square = a.square().sum(1, keepdim=True)
    interaction = (d * a).sum(1, keepdim=True)
    numerator = penalty * w * square - (1 - w) * interaction
    denominator = (penalty + 1 - w) * square
    q = torch.where(square > 1e-20, numerator / denominator.clamp_min(1e-20), w).clamp(0, 1)
    prior = (x + d + w * a).clamp(0, 1)
    corrected = (x + d + q * a).clamp(0, 1)
    sourcebytes = (x.clamp(0, 1) * 255).round()
    priorbytes, correctedbytes = ((value * 255).round() for value in (prior, corrected))
    priorcost = (priorbytes - sourcebytes).square().sum(1, keepdim=True)
    correctedcost = (correctedbytes - sourcebytes).square().sum(1, keepdim=True)
    keep = correctedcost <= priorcost
    q = torch.where(keep, q, w)
    pixels = torch.where(keep, correctedbytes, priorbytes).float() / 255
    actualcost = ((pixels.double() * 255).round() - sourcebytes).square().sum(1, keepdim=True)
    if torch.any(actualcost > priorcost):
        raise RuntimeError("The digital endpoint envelope exceeded its source-error budget.")
    report = {"penalty": penalty, "responsibilitymean": float(w.mean()), "responsibilityquantiles": torch.quantile(w.flatten(), torch.tensor([.1, .5, .9], device=w.device, dtype=w.dtype)).cpu().tolist(), "adoptionmean": float(q.mean()), "conditionalnorm": float(d.norm()), "sharednorm": float(a.norm()), "interactionmean": float(interaction.mean()), "digitalerrorbefore": float(priorcost.sum()), "digitalerrorafter": float(actualcost.sum()), "clipcorrections": int((~keep).sum()), "normspace": "decoded RGB endpoint response", "scope": "Source-error contraction relative to the responsibility prior; no annotation mask or semantic-score selection."}
    if not math.isfinite(gain) or not 1 <= gain <= 4:
        raise ValueError("Innovation gain must be finite and in [1, 4].")
    if gain > 1:
        from engine.innovation import amplify
        pixels, report["innovation"] = amplify(origin, conditional, shared, responsibility, q, pixels, gain, penalty)
        report["digitalerrorafter"] = report["innovation"]["digitalerrorafter"]
    return pixels, report

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
