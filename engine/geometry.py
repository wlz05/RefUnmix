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
    left, singular, _ = np.linalg.svd(response, full_matrices=False)
    damping = fitted["damping"] + fitted["multiplier"]
    gates = singular ** 2 / (singular ** 2 + damping)
    modal = singular * (left.T @ requested) / (singular ** 2 + damping)
    energy = modal ** 2
    attention = energy / energy.sum() if energy.sum() > 0 else np.zeros_like(energy)
    value = basis @ coefficients
    norm = float(value @ metric @ value)
    defect = abs(float(anchor @ metric @ value))
    if not np.isfinite(value).all() or not math.isfinite(norm) or norm > radius * radius * (1 + 1e-9) or defect > tolerance * max(length * math.sqrt(max(norm, 0.)), np.finfo(float).tiny):
        raise ValueError("Tangent allocation failed its original constraints.")
    report.update(status="solved", norm=math.sqrt(max(norm, 0.)), tangenterror=defect, model=fitted, predicted=(edit @ value).tolist(), protectednorm=float(np.linalg.norm(protect @ value)), residualnorm=float(np.linalg.norm(residual + protect @ value)), attention={"gates": gates.tolist(), "energy": attention.tolist(), "singular": singular.tolist(), "unresolved": float(np.linalg.norm(requested - left @ (left.T @ requested))), "scope": "Response-mode diagnostics of the same regularized inverse; individual modes are not unique at repeated singular values."})
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

def equivalent(edit, protect, residual, value, anchor, metric, ridge=.05, radius=1., tolerance=1e-6):
    # Repair protection inside a fixed semantic response class.
    edit, protect, residual, value, anchor, metric = (np.asarray(item, dtype=float) for item in (edit, protect, residual, value, anchor, metric))
    count = len(value) if value.ndim == 1 else 0
    if not count or anchor.shape != value.shape or edit.ndim != 2 or edit.shape[1] != count or protect.ndim != 2 or protect.shape[1] != count or residual.shape != (protect.shape[0],) or metric.shape != (count, count) or not np.allclose(metric, metric.T, rtol=1e-12, atol=1e-14) or any(not np.isfinite(item).all() for item in (edit, protect, residual, value, anchor, metric)):
        raise ValueError("Equivalent responses require finite matching coefficient geometry.")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in (ridge, radius, tolerance)) or ridge <= 0 or radius < 0 or not 0 < tolerance < 1:
        raise ValueError("Equivalent response controls are invalid.")
    try:
        root = np.linalg.cholesky(metric).T
    except np.linalg.LinAlgError as error:
        raise ValueError("Equivalent response metrics must be positive definite.") from error
    normal = root @ anchor
    length = float(np.linalg.norm(normal))
    if length == 0 or not math.isfinite(length):
        raise ValueError("Equivalent response repair requires a nonzero anchor.")
    _, _, vectors = np.linalg.svd((normal / length)[None], full_matrices=True)
    tangent = np.linalg.solve(root, vectors[1:].T)
    initial = tangent.T @ metric @ value
    norm = float(np.linalg.norm(initial))
    if norm > radius * (1 + 1e-9) or np.linalg.norm(root @ (tangent @ initial - value)) > tolerance * max(norm, np.finfo(float).tiny):
        raise ValueError("The semantic anchor must already satisfy the tangent ball.")
    initialresidual = residual + protect @ value
    before = float(initialresidual @ initialresidual)
    if not math.isfinite(before):
        raise ValueError("Equivalent response residuals exceed numerical range.")
    response = edit @ tangent
    _, singular, right = np.linalg.svd(response, full_matrices=True)
    cutoff = tolerance * float(singular[0]) if len(singular) else 0.
    rank = int(np.sum(singular > max(cutoff, np.finfo(float).tiny)))
    null = right[rank:].T
    report = {"status": "no equivalent direction", "rank": rank, "nullity": null.shape[1], "before": before, "after": before, "movement": 0., "semanticerror": 0., "norm": norm, "queries": 0, "scope": "Protection repair at a fixed measured semantic response, verified again on actual output."}
    if not null.shape[1] or radius == 0:
        return value.copy(), report
    # Separate fixed semantics from the shifted protection coordinates.
    center = null.T @ initial
    fixed = initial - null @ center
    bound = math.sqrt(max(0., radius * radius - float(fixed @ fixed)))
    matrix = protect @ tangent @ null
    base = residual + protect @ tangent @ fixed
    gram = matrix.T @ matrix
    damping = ridge * max(float(np.trace(gram)) / len(center), 1e-12)
    hessian = gram + damping * np.eye(len(center))
    target = -matrix.T @ base + damping * center
    eigenvalues, rotation = np.linalg.eigh(hessian)
    if eigenvalues[0] <= 0 or not np.isfinite(target).all():
        raise ValueError("Equivalent repair equations exceed numerical range.")
    rotated = rotation.T @ target
    multiplier = 0.
    if bound == 0:
        coordinates = np.zeros_like(center)
    else:
        coordinates = rotation @ (rotated / eigenvalues)
        if np.linalg.norm(coordinates) > bound:
            lower, upper = 0., max(float(np.linalg.norm(target)) / bound, float(eigenvalues[-1]))
            if not math.isfinite(upper):
                raise ValueError("Equivalent repair multipliers exceed numerical range.")
            for _ in range(100):
                middle = lower + (upper - lower) / 2
                if np.linalg.norm(rotated / (eigenvalues + middle)) > bound:
                    lower = middle
                else:
                    upper = middle
            multiplier = upper
            coordinates = rotation @ (rotated / (eigenvalues + multiplier))
    result = tangent @ (fixed + null @ coordinates)
    move = result - value
    finalresidual = residual + protect @ result
    after = float(finalresidual @ finalresidual)
    error = float(np.linalg.norm(edit @ move))
    whitened = root @ move
    movement = float(whitened @ whitened)
    objective = after + damping * movement
    allowed = tolerance * max(float(np.linalg.norm(response)) * math.sqrt(max(movement, 0.)), np.finfo(float).tiny)
    report.update(status="repaired", after=after, movement=math.sqrt(max(movement, 0.)), semanticerror=error, norm=float(np.linalg.norm(root @ result)), damping=damping, multiplier=multiplier, objective=objective)
    if not np.isfinite(result).all() or not all(math.isfinite(item) for item in (after, error, movement, objective)):
        raise ValueError("Equivalent repair exceeds numerical range.")
    tangenterror = abs(float(normal @ (root @ result)))
    if report["norm"] > radius * (1 + 1e-9) or objective > before + 1e-10 * max(before, np.finfo(float).tiny) or error > allowed or tangenterror > tolerance * max(length * report["norm"], np.finfo(float).tiny):
        report.update(status="unchanged after constraint check", attemptedafter=after, attemptedsemanticerror=error, after=before, semanticerror=0., movement=0., norm=norm)
        return value.copy(), report
    return result, report

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
