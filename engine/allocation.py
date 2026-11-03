# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import numpy as np
import torch
import torch.nn.functional as functional

def coefficients(benefit, energy, lower=.5, upper=2.):
    # Allocate residual energy with one constrained quadratic solve.
    benefit, energy = (np.asarray(value, dtype=np.float64) for value in (benefit, energy))
    if benefit.shape != energy.shape or not np.isfinite(benefit).all() or not np.isfinite(energy).all() or np.any(energy < 0):
        raise ValueError("Allocation coordinates must be finite and compatible.")
    if not 0 <= lower <= 1 <= upper:
        raise ValueError("Allocation bounds must include the original field.")
    budget = float(energy.sum())
    if budget <= 1e-20:
        return np.ones_like(energy)
    active = energy > max(float(energy.max()) * 1e-12, 1e-20)
    ratio = np.divide(benefit, energy, out=np.zeros_like(benefit), where=active)
    if not np.any(ratio > 0):
        return np.full_like(energy, lower)
    def weights(multiplier):
        result = np.clip(ratio / max(multiplier, np.finfo(float).tiny), lower, upper)
        result[~active] = 1.
        return result
    high = max(float(ratio.max()), 1e-12) * max(2., 2. / max(lower, 1e-6))
    low = 0.
    for _ in range(64):
        middle = (low + high) * .5
        if float((energy * weights(middle) ** 2).sum()) > budget:
            low = middle
        else:
            high = middle
    return weights(high)

def balanced(benefit, protection, energy, ridge=.1, lower=.5, upper=2.):
    # Balance semantic ascent against the complete quadratic protection cost.
    benefit, protection, energy = (np.asarray(value, dtype=np.float64) for value in (benefit, protection, energy))
    if benefit.shape != protection.shape or benefit.shape != energy.shape or any(not np.isfinite(value).all() for value in (benefit, protection, energy)) or np.any(protection < 0) or np.any(energy < 0) or not np.isfinite(ridge) or ridge <= 0 or not 0 <= lower <= 1 <= upper:
        raise ValueError("Balanced response coordinates must be finite and compatible.")
    metric = np.maximum(protection + ridge * energy, 1e-20)
    semantic = float((benefit ** 2 / metric).sum())
    repair = float((protection ** 2 / metric).sum())
    if semantic <= 1e-20 or repair <= 1e-20:
        return np.ones_like(energy)
    change = (np.sqrt(repair / semantic) * benefit - protection) / metric
    step = 1.
    negative, positive = change < 0, change > 0
    if np.any(negative):
        step = min(step, float(((1 - lower) / -change[negative]).min()))
    if np.any(positive):
        step = min(step, float(((upper - 1) / change[positive]).min()))
    result = 1 + step * change
    if float((protection * result ** 2).sum()) > float(protection.sum()) * (1 + 1e-9) + 1e-12 or float((benefit * (result - 1)).sum()) < -1e-10:
        raise RuntimeError("Balanced response constraints failed.")
    return result

def redistribute(origin, candidate, judge, source, target, patch=32, lower=.5, upper=2., support=None, policy="energy", ridge=.1, additional=None):
    # Reweight a fixed field using a separate directional semantic gradient.
    if patch < 1 or origin.shape != candidate.shape or origin.shape[-1] % patch or origin.shape[-2] % patch:
        raise ValueError("Allocation patches must tile both image dimensions.")
    from engine.semantic import feature
    judges = [judge] + ([additional] if additional is not None else [])
    gradients = []
    for current in judges:
        with torch.no_grad():
            text = current.texts([source, target])
            direction = functional.normalize(text[1:2] - text[:1], dim=-1)
        with torch.enable_grad():
            pixels = candidate.detach().clone().requires_grad_(True)
            region = support if support is not None else torch.ones_like(pixels[:, :1])
            processed = current.pixels(pixels * region)
            features = functional.normalize(feature(current, processed), dim=-1)
            score = (features * (text[1:2] if support is not None else direction)).sum()
            gradients.append(torch.autograd.grad(score, pixels, only_inputs=True)[0].detach())
    gradient = gradients[0]
    delta = candidate - origin
    benefits = [functional.avg_pool2d((value * delta).sum(1, keepdim=True), patch) * patch ** 2 for value in gradients]
    benefit = benefits[0]
    energy = functional.avg_pool2d(delta.square().sum(1, keepdim=True), patch) * patch ** 2
    protection = 1 - support if support is not None else torch.ones_like(delta[:, :1])
    protected = functional.avg_pool2d((protection * delta.square()).sum(1, keepdim=True), patch) * patch ** 2
    matrixreport = None
    if policy == "matrix":
        from engine.matrix import propose
        change, matrixreport = propose(delta, gradients, protection, patch, ridge)
        consensus, weight = None, None
    else:
        consensus = None
        if policy == "consensus":
            from engine.consensus import allocate
            allocation, consensus = allocate(np.stack([value.cpu().numpy() for value in benefits]), protected.cpu().numpy(), energy.cpu().numpy(), ridge, lower, upper)
        elif policy == "balanced":
            allocation = balanced(benefit.cpu().numpy(), protected.cpu().numpy(), energy.cpu().numpy(), ridge, lower, upper)
        else:
            allocation = coefficients(benefit.cpu().numpy(), energy.cpu().numpy(), lower, upper)
        weight = torch.as_tensor(allocation, device=origin.device, dtype=origin.dtype)
        weight = functional.interpolate(weight, origin.shape[-2:], mode="bilinear", align_corners=False)
        change = weight * delta
    before, after = delta.double().square().sum(), change.double().square().sum()
    if policy in {"balanced", "consensus", "matrix"}:
        increment = change - delta
        linear = float((protection.double() * delta.double() * increment.double()).sum())
        quadratic = float((protection.double() * increment.double().square()).sum())
        fraction = min(1., max(0., -2 * linear / quadratic)) if quadratic > 1e-20 else 1.
        change = delta + fraction * increment
    else:
        factor = min(1., float((before / after.clamp_min(1e-20)).sqrt()))
        change = change * factor
    gains = [float((value.double() * (change - delta).double()).sum()) for value in gradients]
    gain = min(gains)
    accepted = gain >= -1e-12
    if not accepted:
        change = delta
    pixels = (origin + change).clamp(0, 1)
    report = {"policy": policy, "ridge": ridge, "protectionbefore": float((protection.double() * delta.double().square()).sum()), "protectionafter": float((protection.double() * change.double().square()).sum()), "patch": patch, "lower": lower, "upper": upper, "linearsemanticgain": gain, "accepted": accepted, "energybefore": float(before), "energyafter": float(change.double().square().sum()), "matrix": matrixreport, "weightmean": float(weight.mean()) if weight is not None else None, "weightmaximum": float(weight.max()) if weight is not None else None, "imagebackwards": len(judges), "gradientmodel": [value.metadata["family"] for value in judges], "headgains": gains, "consensus": consensus, "evaluationmodel": "CLIP ViT-L/14", "support": "conditional response" if support is not None else "global", "candidates": 0}
    if not torch.isfinite(pixels).all() or (policy == "energy" and report["energyafter"] > report["energybefore"] * (1 + 1e-6) + 1e-12) or (policy in {"balanced", "consensus", "matrix"} and report["protectionafter"] > report["protectionbefore"] * (1 + 1e-6) + 1e-12):
        raise RuntimeError("The allocated field exceeds its finite energy budget.")
    return pixels, report

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
