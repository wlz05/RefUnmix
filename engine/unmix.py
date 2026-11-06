# -*- coding: utf-8 -*-
# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

import math
import torch
import torch.nn.functional as functional
from engine.session import tensor

def project(value, reference, window=5, ridge=.1):
    # Average reference quotients with normalized overlapping patch coverage.
    tensor(value)
    tensor(reference)
    if value.shape != reference.shape or value.device != reference.device or type(window) is not int or not 1 <= window <= 31 or window % 2 != 1 or not math.isfinite(ridge) or ridge < 0:
        raise ValueError("Local reference quotient geometry is invalid.")
    value, reference = value.double(), reference.double()
    def box(operand):
        return functional.avg_pool2d(operand, window, 1, window // 2, count_include_pad=True) * window ** 2
    coverage = box(torch.ones_like(reference[:, :1]))
    numerator = box((reference * value).sum(1, keepdim=True) / coverage)
    denominator = box(reference.square().sum(1, keepdim=True) / coverage)
    coefficients = torch.where(denominator > 1e-20, numerator / ((1 + ridge) * denominator.clamp_min(1e-20)), 0.)
    combined = box(coefficients) / coverage
    result = value - combined * reference
    before, after = float(value.square().sum()), float(result.square().sum())
    if not torch.isfinite(result).all() or after > before * (1 + 1e-10) + 1e-12:
        raise RuntimeError("The normalized local quotient increased field energy.")
    report = {"window": window, "ridge": ridge, "coefficientmean": float(combined.mean()), "coefficientrms": float(combined.square().mean().sqrt()), "energybefore": before, "energyafter": after, "normspace": "input field"}
    return result, report

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
