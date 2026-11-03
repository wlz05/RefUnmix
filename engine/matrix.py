# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import math
import numpy as np
import torch
import torch.nn.functional as functional
from engine.session import tensor

def propose(delta, gradients, protection, patch=32, ridge=.1, radius=.5):
    # Rotate local residual responses inside a regularized protection ellipsoid.
    if not math.isfinite(ridge) or ridge <= 0 or not 0 < radius <= 1:
        raise ValueError("Matrix response regularization and radius are invalid.")
    tensor(delta, 3)
    tensor(protection, 1)
    if len(gradients) not in (1, 2) or any(tensor(value, 3).shape != delta.shape or value.device != delta.device for value in gradients):
        raise ValueError("Matrix semantic gradients must match the residual field.")
    height, width = delta.shape[-2:]
    if type(patch) is not int or patch < 1 or height % patch or width % patch or protection.shape[-2:] != (height, width) or torch.any((protection < 0) | (protection > 1)):
        raise ValueError("Matrix response patches and protection weights are invalid.")
    def blocks(value):
        pooled = functional.avg_pool2d(value.flatten(1, 2), patch) * patch ** 2
        return pooled.permute(0, 2, 3, 1).reshape(-1, 3, 3).double().cpu().numpy()
    outer = delta[:, :, None] * delta[:, None, :]
    gram = blocks(outer * protection[:, None])
    full = blocks(outer)
    scale = np.maximum(np.trace(full, axis1=-2, axis2=-1) / 3, 1e-12)
    metric = gram + ridge * scale[:, None, None] * np.eye(3)
    inverse = np.linalg.inv(metric)
    heads = [blocks(value[:, :, None] * delta[:, None, :]) for value in gradients]
    def inner(left, right):
        return float(((left @ inverse) * right).sum())
    norms = [math.sqrt(max(0., inner(value, value))) for value in heads]
    if any(value <= 1e-12 for value in norms):
        return delta, {"status": "unresolved semantic head"}
    normalized = [value / norm for value, norm in zip(heads, norms)]
    repair = inner(gram, gram)
    if repair <= 1e-20:
        return delta, {"status": "zero measured protection error"}
    mixture = 0.
    if len(normalized) == 2:
        difference = normalized[1] - normalized[0]
        def derivative(alpha):
            value = normalized[0] + alpha * difference
            square = inner(value, value)
            return math.sqrt(repair / square) * inner(value, difference) - inner(gram, difference) if square > 1e-20 else 0.
        if derivative(0.) >= 0:
            mixture = 0.
        elif derivative(1.) <= 0:
            mixture = 1.
        else:
            left, right = 0., 1.
            for _ in range(64):
                middle = (left + right) * .5
                if derivative(middle) < 0:
                    left = middle
                else:
                    right = middle
            mixture = (left + right) * .5
        semantic = normalized[0] + mixture * difference
    else:
        semantic = normalized[0]
    square = inner(semantic, semantic)
    if square <= 1e-20:
        return delta, {"status": "conflicting semantic heads"}
    coefficient = math.sqrt(repair / square)
    matrices = (coefficient * semantic - gram) @ inverse
    nh, nw = height // patch, width // patch
    values = torch.as_tensor(matrices.reshape(1, nh, nw, 9).transpose(0, 3, 1, 2).copy(), device=delta.device, dtype=delta.dtype)
    values = functional.interpolate(values, (height, width), mode="bilinear", align_corners=False).reshape(1, 3, 3, height, width)
    increment = torch.einsum("boihw,bihw->bohw", values, delta)
    linear = float((protection.double() * delta.double() * increment.double()).sum())
    quadratic = float((protection.double() * increment.double().square()).sum())
    length = float(increment.double().norm())
    step = min(1., radius * float(delta.double().norm()) / max(length, 1e-20))
    if quadratic > 1e-20:
        step = min(step, max(0., -2 * linear / quadratic))
    gains = [float((value.double() * increment.double()).sum()) for value in gradients]
    if min(gains) < -1e-12:
        step = 0.
    result = delta + step * increment
    eigenvalues = np.linalg.eigvalsh(full)
    ranks = (eigenvalues > np.maximum(eigenvalues.max(-1, keepdims=True) * 1e-6, 1e-12)).sum(-1)
    report = {"dictionaryrank": int(3 * ranks.sum()), "status": "resolved" if step > 0 else "no common interpolated ascent", "mixture": mixture, "step": step, "ridge": ridge, "radius": radius, "couplingcosine": inner(gram, semantic) / math.sqrt(repair * square), "predictedheadgains": [step * value for value in gains], "dimensions": 9 * nh * nw}
    return result, report

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
