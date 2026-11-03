# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import math
import numpy as np

def allocate(benefits, protection, energy, ridge=.1, lower=.5, upper=2.):
    # Maximize the common semantic gain inside one protection ellipsoid.
    from engine.allocation import balanced
    benefits, protection, energy = (np.asarray(value, dtype=np.float64) for value in (benefits, protection, energy))
    if benefits.shape[0] != 2 or benefits.shape[1:] != protection.shape or protection.shape != energy.shape or not np.isfinite(benefits).all() or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("Consensus requires two finite compatible semantic heads.")
    metric = np.maximum(protection + ridge * energy, 1e-20)
    norms = np.sqrt((benefits ** 2 / metric).reshape(2, -1).sum(1))
    if np.any(norms <= 1e-12):
        return np.ones_like(energy), {"status": "unresolved semantic head"}
    heads = benefits / norms.reshape((2,) + (1,) * protection.ndim)
    difference = heads[1] - heads[0]
    repair = float((protection ** 2 / metric).sum())
    if repair <= 1e-20:
        return np.ones_like(energy), {"status": "zero measured protection error"}
    def derivative(alpha):
        mixed = heads[0] + alpha * difference
        square = float((mixed ** 2 / metric).sum())
        if square <= 1e-20:
            return 0.
        return math.sqrt(repair / square) * float((mixed * difference / metric).sum()) - float((protection * difference / metric).sum())
    if derivative(0.) >= 0:
        alpha = 0.
    elif derivative(1.) <= 0:
        alpha = 1.
    else:
        left, right = 0., 1.
        for _ in range(64):
            alpha = (left + right) * .5
            slope = derivative(alpha)
            if slope < 0:
                left = alpha
            else:
                right = alpha
        alpha = (left + right) * .5
    mixed = heads[0] + alpha * difference
    result = balanced(mixed, protection, energy, ridge, lower, upper)
    gains = [float((head * (result - 1)).sum()) for head in heads]
    if min(gains) < -1e-7:
        return np.ones_like(energy), {"status": "no common semantic ascent", "gains": gains, "mixture": alpha}
    return result, {"status": "resolved", "gains": gains, "mixture": alpha}

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
