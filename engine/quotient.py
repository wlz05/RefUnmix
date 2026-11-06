# -*- coding: utf-8 -*-
# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

import torch
import torch.nn.functional as functional

def local(matrix, linear):
    # Resolve two reference coefficients with exact box active sets.
    a, b, c = matrix[0, 0], matrix[0, 1], matrix[1, 1]
    x, y = linear[0], linear[1]
    determinant = a * c - b.square()
    safe = determinant.clamp_min(1e-16)
    free = torch.stack(((c * x - b * y) / safe, (a * y - b * x) / safe))
    valid = (determinant > 1e-16) & (free >= 0).all(0) & (free <= 1).all(0)
    zero = torch.zeros_like(x)
    one = torch.ones_like(x)
    cases = [torch.stack((zero, zero)), free, torch.stack((zero, (y / c.clamp_min(1e-16)).clamp(0, 1))), torch.stack((one, ((y - b) / c.clamp_min(1e-16)).clamp(0, 1))), torch.stack(((x / a.clamp_min(1e-16)).clamp(0, 1), zero)), torch.stack((((x - b) / a.clamp_min(1e-16)).clamp(0, 1), one))]
    values = torch.stack(cases)
    cost = a * values[:, 0].square() + 2 * b * values[:, 0] * values[:, 1] + c * values[:, 1].square() - 2 * x * values[:, 0] - 2 * y * values[:, 1]
    cost[1] = torch.where(valid, cost[1], torch.full_like(cost[1], float('inf')))
    index = cost.argmin(0)
    return values.gather(0, index[None, None].expand(1, 2, *index.shape))[0]

@torch.no_grad()
def field(origin, image, columns, weight, mode='local', direction=None, patch=16):
    # Preserve conditional innovation while quotienting source-shared drift.
    columns = columns.double()
    residual = image.double() - origin.double()
    if mode in {'orthogonal', 'conditional'}:
        axis = direction.double()[None]
        square = axis.square().sum(2, keepdim=True)
        uncertainty = columns.square().sum(2, keepdim=True).mean(0, keepdim=True) if mode == 'conditional' else 0.
        projection = (columns * axis).sum(2, keepdim=True) / (square + uncertainty).clamp_min(1e-12)
        columns = columns - projection * axis
    h, w = origin.shape[-2:]
    shape = (max(1, h // patch), max(1, w // patch))
    scalar = weight.double()[0, 0]
    gram = (columns[:, None] * columns[None, :]).sum((2, 3)) * scalar
    linear = (columns * residual * weight).sum((1, 2))
    pooled = functional.adaptive_avg_pool2d(gram.flatten(0, 1)[None], shape)[0].reshape(2, 2, *shape)
    target = functional.adaptive_avg_pool2d(linear[None], shape)[0]
    coefficients = local(pooled, target)
    coefficients = functional.interpolate(coefficients[None], (h, w), mode='bilinear', align_corners=False)[0]
    correction = (columns * coefficients[:, None, None]).sum(0)
    numerator = float((correction * residual * weight).sum())
    denominator = float((correction.square() * weight).sum())
    scale = min(1., max(0., numerator / denominator)) if denominator > 1e-16 else 0.
    result = (image.double() - scale * correction).float().clamp(0, 1)
    before = float((residual.square() * weight).sum())
    after = float(((result.double() - origin.double()).square() * weight).sum())
    if after > before + 1e-8:
        raise RuntimeError('Reference quotient increased its measured preservation cost.')
    return result, {'mode': mode, 'patch': patch, 'scale': scale, 'coefficients': coefficients.mean((1, 2)).tolist(), 'before': before, 'after': after}

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
