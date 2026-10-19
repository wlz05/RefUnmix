# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import math
import numpy as np

def complete(initial, directions, observed, metric, tolerance=1e-6):
    # Fit independent chords with minimal response changes.
    initial, directions, observed, metric = (np.asarray(value, dtype=np.float64) for value in (initial, directions, observed, metric))
    if initial.ndim != 2 or directions.ndim != 2 or observed.ndim != 2:
        raise ValueError("Secant arrays must be matrices.")
    count, samples = directions.shape
    if initial.shape[1] != count or observed.shape != (initial.shape[0], samples) or metric.shape != (count, count):
        raise ValueError("Secant dimensions disagree.")
    if any(not np.isfinite(value).all() for value in (initial, directions, observed, metric)):
        raise ValueError("Secant arrays must be finite.")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("Secant tolerance must be in (0, 1).")
    if not np.allclose(metric, metric.T, atol=1e-10, rtol=1e-8):
        raise ValueError("The secant metric must be symmetric.")
    try:
        root = np.linalg.cholesky(metric).T
    except np.linalg.LinAlgError as error:
        raise ValueError("The secant metric must be positive definite.") from error
    whitened = root @ directions
    lengths = np.linalg.norm(whitened, axis=0)
    if not np.isfinite(whitened).all() or not np.isfinite(lengths).all():
        raise ValueError("Secant directions exceeded numerical range.")
    kept, units = [], []
    for index in reversed(range(samples)):
        if lengths[index] <= 1e-12:
            continue
        vector = whitened[:, index] / lengths[index]
        for _ in range(2):
            for previous in units:
                vector -= float(vector @ previous) * previous
        length = float(np.linalg.norm(vector))
        if length <= tolerance:
            continue
        kept.append(index)
        units.append(vector / length)
    fitted = initial.copy()
    condition, error = None, 0.
    if kept:
        selected = directions[:, kept] / lengths[kept]
        measured = observed[:, kept] / lengths[kept]
        gram = selected.T @ metric @ selected
        condition = float(np.linalg.cond(gram))
        correction = (measured - initial @ selected) @ np.linalg.solve(gram, selected.T @ metric)
        fitted += correction
        error = float(np.linalg.norm(fitted @ directions[:, kept] - observed[:, kept]))
    if not np.isfinite(fitted).all() or not math.isfinite(error) or condition is not None and not math.isfinite(condition):
        raise RuntimeError("Secant completion exceeded numerical range.")
    change = float(np.linalg.norm((fitted - initial) @ np.linalg.solve(root, np.eye(count))))
    if not math.isfinite(change):
        raise RuntimeError("The physical response change exceeded numerical range.")
    return fitted, {"kept": kept, "pruned": [index for index in range(samples) if index not in kept], "rank": len(kept), "fiterror": error, "condition": condition, "change": change}

def defect(actual, predicted):
    # Normalize disagreement without trusting zero signals.
    actual, predicted = (np.asarray(value, dtype=np.float64) for value in (actual, predicted))
    if actual.shape != predicted.shape or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Response comparisons require matching finite arrays.")
    error = float(np.linalg.norm(actual - predicted))
    denominator = max(float(np.linalg.norm(actual)), float(np.linalg.norm(predicted)), 1e-8)
    if not math.isfinite(error) or not math.isfinite(denominator):
        raise ValueError("Response disagreement exceeded numerical range.")
    return {"absolute": error, "relative": error / denominator}

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
