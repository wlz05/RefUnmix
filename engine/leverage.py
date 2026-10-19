# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import math
import numpy as np

def factor(matrix, tolerance):
    # Retain only numerically resolved response directions.
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    cutoff = max(1e-12, tolerance * singular[0]) if len(singular) else 1e-12
    kept = singular > cutoff
    return left[:, kept], singular[kept], right[kept], cutoff

def nullgain(matrix, gradient, tolerance):
    _, _, right, _ = factor(matrix, tolerance)
    return float(np.linalg.norm(gradient - right.T @ (right @ gradient)))

def analyze(response, semantic, residual, metric, active, names=None, scales=None, damagescale=1., tolerance=1e-6):
    # Diagnose reference value in joint output-response space.
    response, semantic, residual, metric = (np.asarray(value, dtype=float) for value in (response, semantic, residual, metric))
    active = tuple(active)
    if not active or len(set(active)) != len(active) or any(type(axis) not in (int, np.int32, np.int64) or axis not in (0, 1, 2) for axis in active):
        raise ValueError("Response axes must be unique supported coordinates.")
    active = tuple(int(axis) for axis in active)
    size = len(active)
    if response.ndim != 2 or semantic.ndim != 2 or residual.ndim != 1 or response.shape != (len(residual), size) or semantic.shape[1] != size or not len(semantic) or metric.shape != (size, size):
        raise ValueError("Leverage response dimensions disagree.")
    if any(not np.isfinite(value).all() for value in (response, semantic, residual, metric)) or not np.allclose(metric, metric.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Leverage arrays must be finite with a symmetric metric.")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or not 0 < tolerance < 1 or isinstance(damagescale, bool) or not isinstance(damagescale, (int, float)) or not math.isfinite(damagescale) or damagescale <= 0:
        raise ValueError("Leverage scales and rank tolerance are invalid.")
    names = list(names) if names is not None else ["component" + str(index) for index in range(len(semantic))]
    scales = np.ones(len(semantic)) if scales is None else np.asarray(scales, dtype=float)
    if len(names) != len(semantic) or any(not isinstance(name, str) or not name.isascii() or not name for name in names) or len(set(names)) != len(names) or scales.shape != (len(semantic),) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Named semantic scales must match response rows.")
    try:
        inverse = np.linalg.solve(np.linalg.cholesky(metric).T, np.eye(size))
    except np.linalg.LinAlgError as error:
        raise ValueError("Leverage metrics must be positive definite.") from error
    result = {"active": 2 in active, "axes": list(active), "tolerance": tolerance, "semanticscales": dict(zip(names, scales.tolist())), "damagescale": damagescale, "scope": "Local fitted responses only; no semantic purity, nonlinear feasibility, or causal utility certificate."}
    if 2 not in active:
        return result
    reference = active.index(2)
    fields = [index for index, axis in enumerate(active) if axis != 2]
    if np.any(np.abs(metric[reference, fields]) > 1e-12):
        raise ValueError("Reference diagnostics require a block-separable control metric.")
    fieldmetric = metric[np.ix_(fields, fields)]
    fieldinverse = np.linalg.solve(np.linalg.cholesky(fieldmetric).T, np.eye(len(fields))) if fields else np.empty((0, 0))
    basis = response[:, fields] @ fieldinverse
    left, singular, right, cutoff = factor(basis, tolerance)
    effect = response[:, reference]
    coordinates = right.T @ ((left.T @ effect) / singular) if len(singular) else np.zeros(len(fields))
    fieldmove = -fieldinverse @ coordinates
    direction = np.zeros(size)
    direction[reference], direction[fields] = 1., fieldmove
    remaining = effect - basis @ coordinates
    compensation = semantic[:, reference] + semantic[:, fields] @ fieldmove
    jointfields = np.vstack((basis / math.sqrt(damagescale), (semantic[:, fields] / scales[:, None]) @ fieldinverse))
    jointreference = np.r_[effect / math.sqrt(damagescale), semantic[:, reference] / scales]
    jointleft, _, _, jointcutoff = factor(jointfields, tolerance)
    jointremaining = jointreference - jointleft @ (jointleft.T @ jointreference)
    jointnorm = float(np.linalg.norm(jointreference))
    effectnorm = float(np.linalg.norm(effect))
    neutral = {}
    fullresponse = response @ inverse
    for index, name in enumerate(names):
        full = nullgain(fullresponse, (semantic[index] / scales[index]) @ inverse, tolerance)
        field = nullgain(basis, (semantic[index, fields] / scales[index]) @ fieldinverse, tolerance)
        neutral[name] = {"full": full, "field": field, "increment": full - field}
    result.update({"jointnovelty": float(np.linalg.norm(jointremaining)) / jointnorm if jointnorm > 1e-12 else 0., "jointsignal": jointnorm, "protectionuncancellable": float(np.linalg.norm(remaining)) / effectnorm if effectnorm > 1e-12 else 0., "protectionresidual": float(np.linalg.norm(remaining)), "fieldrank": len(singular), "protectioncutoff": cutoff, "jointcutoff": jointcutoff, "direction": direction.tolist(), "directionnorm": float(np.sqrt(direction @ metric @ direction)), "compensatedsemantic": dict(zip(names, compensation.tolist())), "damageslope": float(2 * residual @ remaining), "damagecurvature": float(remaining @ remaining), "neutralgain": neutral, "neutralnote": "Symmetric interior, zero incremental protected response; not unchanged total distortion under finite nonlinear edits."})
    if not np.isfinite(compensation).all() or any(not math.isfinite(value) for observed in neutral.values() for value in observed.values()) or any(not math.isfinite(result[key]) for key in ("jointnovelty", "jointsignal", "protectionuncancellable", "protectionresidual", "directionnorm", "damageslope", "damagecurvature")):
        raise FloatingPointError("Leverage diagnostics exceeded numerical range.")
    return result

def probe(model, diagnostics, points, radius, lower, scales, damage, damagescale):
    # Test a compensated direction within the existing query budget.
    if not diagnostics.get("active") or not diagnostics.get("geometryresolved") or diagnostics.get("fittrusted") is not True:
        return None
    direction = np.asarray(diagnostics["direction"], dtype=float)
    lower, scales = np.asarray(lower, dtype=float), np.asarray(scales, dtype=float)
    if direction.shape != model.root.shape or lower.shape != model.scores.shape or scales.shape != lower.shape or not all(np.isfinite(value).all() for value in (direction, lower, scales)) or np.any(scales <= 0) or not all(math.isfinite(value) and value > 0 for value in (radius, damagescale)) or not math.isfinite(damage) or damage < 0:
        raise ValueError("Compensation probe controls are invalid.")
    length = float(np.sqrt(direction @ model.metric @ direction))
    if length <= 1e-12:
        return None
    existing = [np.asarray(point, dtype=float) for point in points]
    if any(point.shape != model.anchor.shape or not np.isfinite(point).all() for point in existing):
        raise ValueError("Visited compensation points must match control geometry.")
    limits = np.array([2., 2., 1.])[model.active]
    before = model.restore(np.zeros(len(direction)), lower, scales, damage, damagescale)
    options = []
    for sign in (-1., 1.):
        unit = sign * direction
        maximum = radius / length
        for index, change in enumerate(unit):
            if abs(change) > 1e-14:
                position = model.anchor[model.active[index]]
                boundary = (limits[index] - position) / change if change > 0 else -position / change
                maximum = min(maximum, max(0., boundary))
        amplitudes = {maximum * fraction for fraction in (1., .5, .25)}
        curvature, slope = float(unit @ model.gram @ unit), float(2 * model.cross @ unit)
        if curvature > 0:
            amplitudes.add(float(np.clip(-slope / (2 * curvature), 0, maximum)))
        for value, change in zip(lower - model.scores, model.semantic @ unit):
            if abs(change) > 1e-14:
                boundary = float(value / change)
                if 0 <= boundary <= maximum:
                    amplitudes.add(boundary)
        for amplitude in sorted(amplitudes):
            move = amplitude * unit
            if np.linalg.norm(move * model.root) <= 1e-10:
                continue
            point = model.anchor.copy()
            point[model.active] += move
            if any(np.linalg.norm((point[model.active] - old[model.active]) * model.root) <= 1e-10 for old in existing):
                continue
            predicted, semantic, _ = model.prediction(move, 0.)
            violation = model.restore(move, lower, scales, damage, damagescale)
            if before > 1e-12:
                improves = violation < before - 1e-12
            else:
                improves = violation <= 1e-10 and predicted < model.damage - 1e-12
            if improves:
                options.append((violation, predicted, -float(np.sum((semantic - model.scores) / scales)), tuple(point)))
    return min(options)[-1] if options else None

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
