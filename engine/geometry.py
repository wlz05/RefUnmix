# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import math
import numpy as np
from engine.theory import fit

def tangent(edit, protect, target, anchor, metric, penalty=1., ridge=.05, radius=1., tolerance=1e-6, residual=None):
    # Allocate endpoint response inside the baseline tangent space.
    edit, protect, target, anchor, metric = (np.asarray(value, dtype=float) for value in (edit, protect, target, anchor, metric))
    count = len(anchor) if anchor.ndim == 1 else 0
    if not count or target.ndim != 1 or not len(target) or edit.ndim != 2 or edit.shape != (len(target), count) or protect.ndim != 2 or protect.shape[1] != count or metric.shape != (count, count) or any(not np.isfinite(value).all() for value in (edit, protect, target, anchor, metric)) or not np.allclose(metric, metric.T, rtol=1e-12, atol=1e-14):
        raise ValueError("Tangent responses require finite matching coefficient geometry.")
    for value in (penalty, ridge, radius, tolerance):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Tangent controls must be finite and nonnegative.")
    if ridge <= 0 or not 0 < tolerance < 1:
        raise ValueError("Tangent regularization and rank tolerance must be positive.")
    residual = np.zeros(protect.shape[0]) if residual is None else np.asarray(residual, dtype=float)
    if residual.shape != (protect.shape[0],) or not np.isfinite(residual).all():
        raise ValueError("Tangent protection residuals must match the measured response.")
    try:
        root = np.linalg.cholesky(metric).T
    except np.linalg.LinAlgError as error:
        raise ValueError("Tangent metrics must be positive definite.") from error
    normal = root @ anchor
    length = float(np.linalg.norm(normal))
    if not math.isfinite(length):
        raise ValueError("Tangent anchors exceed numerical range.")
    report = {"rank": 0, "anchorzero": length == 0, "tangenterror": 0., "norm": 0., "status": "zero anchor", "scope": "Regularized endpoint response allocation under one metric tangent constraint."}
    if length == 0 or radius == 0:
        return np.zeros(count), report
    _, _, vectors = np.linalg.svd((normal / length)[None], full_matrices=True)
    basis = np.linalg.solve(root, vectors[1:].T)
    report["rank"] = basis.shape[1]
    if not basis.shape[1]:
        report["status"] = "no tangent direction"
        return np.zeros(count), report
    response = np.vstack((edit @ basis, math.sqrt(penalty) * protect @ basis))
    requested = np.r_[target, -math.sqrt(penalty) * residual]
    from engine.field import solve
    coefficients, fitted = solve(response, requested, np.zeros(basis.shape[1]), ridge, bound=radius)
    value = basis @ coefficients
    norm = float(value @ metric @ value)
    defect = abs(float(anchor @ metric @ value))
    if not np.isfinite(value).all() or not math.isfinite(norm) or norm > radius * radius * (1 + 1e-9) or defect > tolerance * max(length * math.sqrt(max(norm, 0.)), np.finfo(float).tiny):
        raise ValueError("Tangent allocation failed its original constraints.")
    report.update(status="solved", norm=math.sqrt(max(norm, 0.)), tangenterror=defect, model=fitted, predicted=(edit @ value).tolist(), protectednorm=float(np.linalg.norm(protect @ value)), residualnorm=float(np.linalg.norm(residual + protect @ value)))
    return value, report

def adoption(residual, effect, maximum=1.):
    # Bound positive adoption by the unchanged protection budget.
    residual, effect = np.asarray(residual, dtype=float), np.asarray(effect, dtype=float)
    if residual.ndim != 1 or effect.shape != residual.shape or any(not np.isfinite(value).all() for value in (residual, effect)) or isinstance(maximum, bool) or not isinstance(maximum, (int, float)) or not math.isfinite(maximum) or maximum < 0:
        raise ValueError("Adoption requires finite matching protected responses.")
    square, cross = float(effect @ effect), float(2 * residual @ effect)
    if not math.isfinite(square) or not math.isfinite(cross):
        raise ValueError("Adoption coefficients exceed numerical range.")
    bound = maximum if square == 0 else min(maximum, -cross / square) if cross < 0 else 0.
    return max(0., bound), {"cross": cross, "quadratic": square, "maximum": maximum}

def matching(displacement, effect, maximum=1., tolerance=.01):
    # Intersect the full norm tolerance band with positive adoption.
    displacement, effect = np.asarray(displacement, dtype=float), np.asarray(effect, dtype=float)
    if displacement.ndim != 1 or not len(displacement) or effect.shape != displacement.shape or any(not np.isfinite(value).all() for value in (displacement, effect)) or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in (maximum, tolerance)) or maximum < 0 or not 0 < tolerance < 1:
        raise ValueError("Norm matching requires finite aligned responses and valid bounds.")
    scale = max(float(np.max(np.abs(displacement))), float(np.max(np.abs(effect))))
    if scale == 0 or not np.any(displacement != 0) or maximum == 0:
        return []
    first, second = displacement / scale, effect / scale
    square, cross, level = float(second @ second), float(2 * first @ second), float(first @ first)
    if level == 0:
        return []
    if square == 0:
        return [(0., maximum)]
    upper = ((1 + tolerance) ** 2 - 1) * level
    lower = ((1 - tolerance) ** 2 - 1) * level
    root = math.sqrt(cross * cross + 4 * square * upper)
    end = 2 * upper / (root + cross) if cross >= 0 else (root - cross) / (2 * square)
    end = min(maximum, end)
    discriminant = cross * cross + 4 * square * lower
    if cross >= 0 or discriminant <= 0:
        return [(0., end)] if end > 0 else []
    root = math.sqrt(discriminant)
    left, right = -2 * lower / (-cross + root), (-cross + root) / (2 * square)
    result = [(0., min(left, end))] if min(left, end) > 0 else []
    if right <= end:
        result.append((right, end))
    return result

def response(model, points, values, regularization=.0001, initial=None):
    # Reuse the same measured chords in clean latent space.
    points, values = np.asarray(points, dtype=float), np.asarray(values, dtype=float)
    active = np.asarray(model.active)
    anchor = np.asarray(model.anchor, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or values.ndim != 2 or values.shape[0] != len(points) or values.shape[1] < 1 or active.ndim != 1 or active.dtype.kind not in "iu" or len(set(active.tolist())) != len(active) or any(axis not in (0, 1, 2) for axis in active) or not len(active):
        raise ValueError("Latent responses require matching measured control rows.")
    if anchor.shape != (3,) or any(not np.isfinite(value).all() for value in (points, values, anchor)) or isinstance(regularization, bool) or not isinstance(regularization, (int, float)) or not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("Latent response inputs must be finite.")
    offsets = points[:, active] - anchor[active]
    centers = np.flatnonzero(np.linalg.norm(offsets, axis=1) < 1e-12)
    indices = np.asarray(model.indices)
    if len(centers) != 1 or indices.ndim != 1 or indices.dtype.kind not in "iu" or len(set(indices.tolist())) != len(indices) or np.any(indices < 0) or np.any(indices >= len(points)):
        raise ValueError("Latent fits must reuse one visited anchor and its neighbors.")
    center = int(centers[0])
    chosen = [int(index) for index in indices if index != center]
    directions = offsets[chosen].T
    if directions.shape != model.directions.shape or not np.allclose(directions, model.directions, atol=1e-12, rtol=1e-12):
        raise ValueError("Latent and output fits must share identical chords.")
    observed = (values[chosen] - values[center]).T
    previous = np.zeros((values.shape[1], len(active))) if initial is None else np.asarray(initial, dtype=float)
    fitted = fit(previous, directions, observed, regularization)
    error = float(np.linalg.norm(observed - fitted @ directions))
    if not np.isfinite(fitted).all() or not math.isfinite(error):
        raise ValueError("Latent response fits exceed numerical range.")
    return fitted, {"neighbors": indices.tolist(), "fiterror": error, "dimension": values.shape[1], "queries": 0, "scope": "Unweighted clean-latent endpoint geometry; not an external editor reproduction."}

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
