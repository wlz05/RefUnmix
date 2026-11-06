# -*- coding: utf-8 -*-
# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

import math
import torch
from engine.session import tensor

@torch.no_grad()
def resolve(origin, conditional, shared, responsibility, gain=1.):
    # Keep shared adoption fixed during prompt enhancement.
    for value in (origin, conditional, shared):
        tensor(value, 3)
    tensor(responsibility, 1)
    if any(value.shape != origin.shape or value.device != origin.device for value in (conditional, shared)):
        raise ValueError("Response tensors must share image geometry and device.")
    if responsibility.shape != origin[:, :1].shape or responsibility.device != origin.device:
        raise ValueError("Responsibility must match the image geometry and device.")
    if torch.any((origin < 0) | (origin > 1)) or torch.any((responsibility < 0) | (responsibility > 1)):
        raise ValueError("Source and responsibility values must be in [0, 1].")
    if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or not 1 <= gain <= 4:
        raise ValueError("Prompt gain must be finite and in [1, 4].")
    x, d, a, q = (value.double() for value in (origin, conditional, shared, responsibility))
    signal = d.square().sum(1, keepdim=True)
    cross = (d * a).sum(1, keepdim=True)
    limit = -1 - 2 * q * cross / signal.clamp_min(1e-20)
    effective = torch.where(signal > 1e-20, limit.clamp(1., gain), 1.)
    baseline = ((x + d + q * a).clamp(0, 1) * 255).round()
    proposal = ((x + effective * d + q * a).clamp(0, 1) * 255).round()
    source = (x * 255).round()
    oldcost = (baseline - source).square().sum(1, keepdim=True)
    newcost = (proposal - source).square().sum(1, keepdim=True)
    factor = (oldcost / newcost.clamp_min(1)).sqrt().clamp(max=1)
    delta = proposal - source
    inward = source + delta.sign() * (delta.abs() * factor).floor()
    result = torch.where(newcost <= oldcost, proposal, inward).float() / 255
    actual = ((result.double() * 255).round() - source).square().sum(1, keepdim=True)
    if not torch.isfinite(result).all() or torch.any(actual > oldcost):
        raise RuntimeError("Fixed-adoption enhancement exceeded the digital budget.")
    return result, {"assignment": "fixed responsibility prior in both stages", "requested": gain, "effective": float(effective.mean()), "digitalerrorbefore": float(oldcost.sum()), "digitalerrorafter": float(actual.sum())}

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
