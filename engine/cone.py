# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import torch

def project(first, second, advection, normalize=True, geometry="local"):
    dimensions = (1, 2, 3) if geometry == "global" else (1,)
    # Allocate transport energy inside the observed conditional response cone.
    first, second, advection = first.double(), second.double(), advection.double()
    norm = second.square().sum(dimensions, keepdim=True).sqrt()
    fallback = first.square().sum(dimensions, keepdim=True).sqrt()
    primary = torch.where(norm > 1e-8, second / norm.clamp_min(1e-12), first / fallback.clamp_min(1e-12))
    remaining = first - (first * primary).sum(dimensions, keepdim=True) * primary
    length = remaining.square().sum(dimensions, keepdim=True).sqrt()
    secondary = torch.where(length > 1e-4 * fallback, remaining / length.clamp_min(1e-12), torch.zeros_like(remaining))
    original = second + advection
    positive = (original * primary).sum(dimensions, keepdim=True).clamp_min(0)
    lateral = (original * secondary).sum(dimensions, keepdim=True)
    value = positive * primary + lateral * secondary
    before, after = float(original.norm()), float(value.norm())
    scale = before / after if normalize and after > 1e-10 else 1.
    value = value * scale
    if not torch.isfinite(value).all():
        raise ValueError('Conditional response projection must be finite.')
    return value.float(), {'rawnorm': before, 'projectednorm': after, 'scale': scale, 'fieldnorm': float(value.norm()), 'retainedenergy': after ** 2 / before ** 2 if before else 1.}

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
