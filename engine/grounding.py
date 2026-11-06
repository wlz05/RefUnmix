# -*- coding: utf-8 -*-
# RefUnmix: Separating Prompt Response from Shared Transport in One-Step Image Editing

import math
import torch
import torch.nn.functional as functional

def support(attention, shape):
    # Normalize only the model's changed-word attention.
    if attention.ndim != 4 or attention.shape[:2] != (1, 1) or not attention.numel() or not torch.isfinite(attention).all() or torch.any(attention < 0):
        raise ValueError("Word attention must be a finite nonnegative map.")
    low, high = torch.quantile(attention.float().flatten(), torch.tensor([.1, .9], device=attention.device))
    span = high - low
    if span <= 1e-8:
        return torch.zeros((1, 1, *shape), device=attention.device)
    value = ((attention.float() - low) / span).clamp(0, 1)
    value = ((value - .25) / .5).clamp(0, 1)
    value = value.square() * (3 - 2 * value)
    return functional.interpolate(value, shape, mode="bilinear", align_corners=False)

def reference(origin, decoded, attention):
    if origin.shape != decoded.shape or origin.ndim != 4 or origin.shape[:2] != (1, 3):
        raise ValueError("Reference images must share single-image RGB geometry.")
    if origin.device != decoded.device or origin.device != attention.device:
        raise ValueError("Reference tensors must share a device.")
    if not torch.isfinite(decoded).all() or not torch.isfinite(origin).all() or torch.any((origin < 0) | (origin > 1)):
        raise ValueError("Source pixels must be in [0, 1] and decoded responses must be finite.")
    weight = support(attention, origin.shape[-2:])
    result = decoded + weight * (origin - decoded)
    return result, weight

def supply(weight, scope):
    # Permit native transport inside the semantic support.
    if weight.shape != scope.shape or weight.ndim != 4 or weight.shape[:2] != (1, 1) or weight.device != scope.device:
        raise ValueError("Responsibility and semantic support must share geometry.")
    if any(not torch.isfinite(value).all() or torch.any((value < 0) | (value > 1)) for value in (weight, scope)):
        raise ValueError("Responsibilities must be finite and in [0, 1].")
    return weight + scope * (1 - weight)

def restrict(origin, pixels, scope, retained=0.):
    # Contract the final byte displacement outside semantic support.
    if origin.shape != pixels.shape or origin.ndim != 4 or origin.shape[:2] != (1, 3) or scope.shape != origin[:, :1].shape:
        raise ValueError("Output support must match RGB image geometry.")
    if any(value.device != origin.device or not torch.isfinite(value).all() or torch.any((value < 0) | (value > 1)) for value in (pixels, origin, scope)):
        raise ValueError("Supported outputs must be finite and in [0, 1].")
    if isinstance(retained, bool) or not isinstance(retained, (int, float)) or not math.isfinite(retained) or not 0 <= retained <= 1:
        raise ValueError("Retained context must be finite and in [0, 1].")
    scope = retained + (1 - retained) * scope
    source = (origin.double() * 255).round()
    delta = (pixels.double() * 255).round() - source
    restricted = delta.sign() * (delta.abs() * scope.double()).floor()
    result = (source + restricted).float() / 255
    before = delta.square().sum(1, keepdim=True)
    after = restricted.square().sum(1, keepdim=True)
    if torch.any(after > before):
        raise RuntimeError("Semantic support increased source displacement.")
    return result, {"retainedcontext": retained, "digitalerrorbefore": float(before.sum()), "digitalerrorafter": float(after.sum()), "scope": "Fixed inferred support; no annotation or quality-score selection."}

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
