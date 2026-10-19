# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import math
import numpy as np
from engine.attainment import gain

def directions(model, damping, tolerance=1e-6, geometry="output"):
    # Cancel protected responses before measuring retained edits.
    active = np.asarray(model.active)
    count = len(active)
    response, semantic, metric = (np.asarray(value, dtype=float) for value in (model.response, model.semantic, model.metric))
    if active.ndim != 1 or active.dtype.kind not in "iu" or len(set(active.tolist())) != count or any(axis not in (0, 1, 2) for axis in active) or response.ndim != 2 or response.shape[1] != count or semantic.ndim != 2 or semantic.shape[1] != count or metric.shape != (count, count) or any(not np.isfinite(value).all() for value in (response, semantic, metric)):
        raise ValueError("Release responses require finite matching control geometry.")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in (damping, tolerance)) or damping <= 0 or not 0 < tolerance < 1 or not np.allclose(metric, metric.T, rtol=1e-12, atol=1e-14):
        raise ValueError("Release damping, tolerance, or metric is invalid.")
    if not math.isfinite(4 * damping):
        raise ValueError("Release damping expansion exceeds numerical range.")
    if geometry not in {"output", "latent", "attenuate"}:
        raise ValueError("Unknown compensation geometry.")
    if not count:
        return []
    np.linalg.cholesky(metric)
    if 2 not in active:
        return []
    reference = int(np.flatnonzero(active == 2)[0])
    fields = np.flatnonzero(active != 2)
    basis = response if geometry != "latent" else np.asarray(getattr(model, "latentresponse", None), dtype=float)
    if basis.ndim != 2 or basis.shape[1] != count or not np.isfinite(basis).all():
        raise ValueError("Latent compensation requires matching measured endpoint responses.")
    if np.any(np.abs(metric[fields, reference]) > 1e-12) or np.any(np.abs(metric[reference, fields]) > 1e-12):
        raise ValueError("Release compensation requires a block-separable metric.")
    root = np.linalg.cholesky(metric[np.ix_(fields, fields)]).T
    inverse = np.linalg.solve(root, np.eye(len(fields)))
    whitened = basis[:, fields] @ inverse
    left, singular, right = np.linalg.svd(whitened, full_matrices=False)
    cutoff = max(np.finfo(float).tiny, tolerance * float(singular[0])) if len(singular) else np.finfo(float).tiny
    resolved = singular > cutoff
    source = basis[:, reference]
    rows = []
    for value in (0., damping, 4 * damping):
        weights = np.zeros_like(singular)
        if value == 0:
            weights[resolved] = 1 / singular[resolved]
        else:
            weights = singular / (singular * singular + value)
        fieldmove = -inverse @ (right.T @ (weights * (left.T @ source)))
        if geometry == "attenuate":
            fieldmove = np.zeros(len(fields))
        direction = np.zeros(count)
        direction[fields], direction[reference] = fieldmove, 1.
        protected, retained = response @ direction, semantic @ direction
        norm = float(direction @ metric @ direction)
        if not np.isfinite(direction).all() or not np.isfinite(protected).all() or not np.isfinite(retained).all() or not math.isfinite(norm) or norm <= 0:
            raise ValueError("Release compensation exceeds numerical range.")
        if any(np.allclose(direction, row["direction"], rtol=1e-12, atol=1e-14) for row in rows):
            continue
        diagnostics = {"basisnorm": float(np.linalg.norm(basis @ direction)), "protectednorm": float(np.linalg.norm(protected)), "uncompensatednorm": float(np.linalg.norm(response[:, reference]))}
        if any(not math.isfinite(value) for value in diagnostics.values()):
            raise ValueError("Compensation diagnostics exceed numerical range.")
        rows.append({"direction": direction, "protected": protected, "retained": retained, "norm": norm, "damping": value, "rank": int(resolved.sum()), "cutoff": cutoff, "geometry": geometry, **diagnostics})
    return rows

def interval(model, direction, radius, lower, damage):
    # Intersect every protection and semantic bound exactly.
    direction, lower = np.asarray(direction, dtype=float), np.asarray(lower, dtype=float)
    limits = np.array([2., 2., 1.])
    arrays = (model.anchor, model.residual, model.scores, direction, lower)
    if model.anchor.shape != (3,) or model.residual.shape != (model.response.shape[0],) or model.scores.shape != (model.semantic.shape[0],) or direction.shape != (len(model.active),) or lower.shape != model.scores.shape or any(not np.isfinite(value).all() for value in arrays) or np.any(model.anchor < 0) or np.any(model.anchor > limits):
        raise ValueError("Release anchors and floors must match finite responses.")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in (radius, damage)) or radius <= 0 or damage < 0:
        raise ValueError("Release budgets must be finite and nonnegative.")
    if not math.isfinite(radius * radius) or radius * radius == 0:
        raise ValueError("Release squared radii exceed numerical range.")
    norm = float(direction @ model.metric @ direction)
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Release directions need a positive metric norm.")
    extent = radius / math.sqrt(norm)
    if not math.isfinite(extent) or extent == 0:
        raise ValueError("Release intervals exceed numerical range.")
    lo, hi = -extent, extent

    def linear(slope, offset):
        nonlocal lo, hi
        if slope > 0:
            lo = max(lo, -offset / slope)
        elif slope < 0:
            hi = min(hi, -offset / slope)
        elif offset < 0:
            lo, hi = 1., 0.

    for axis, slope in zip(model.active, direction):
        linear(float(slope), float(model.anchor[axis]))
        linear(float(-slope), float(limits[axis] - model.anchor[axis]))
    for slope, offset in zip(model.semantic @ direction, model.scores - lower):
        linear(float(slope), float(offset))
    protected = model.response @ direction
    a, b, c = float(protected @ protected), float(2 * model.residual @ protected), float(model.residual @ model.residual - damage)
    if not all(math.isfinite(value) for value in (a, b, c)):
        raise ValueError("Release protection polynomial exceeds numerical range.")
    if a == 0:
        linear(-b, -c)
    else:
        scale = max(a, abs(b), abs(c))
        aa, bb, cc = a / scale, b / scale, c / scale
        if aa == 0:
            raise ValueError("Release protection curvature underflows normalization.")
        discriminant = bb * bb - 4 * aa * cc
        if discriminant < 0:
            return None
        root = math.sqrt(discriminant)
        q = -.5 * (bb + math.copysign(root, bb))
        first, second = (-bb / (2 * aa),) * 2 if q == 0 else (q / aa, cc / q)
        lo, hi = max(lo, min(first, second)), min(hi, max(first, second))
    return (lo, hi) if lo <= hi and math.isfinite(lo) and math.isfinite(hi) else None

def propose(model, radius, lower, damage, reference, scales, indices, damping, loss=0., tolerance=1e-6, checkpoint=None, geometry="output"):
    # Release reference effects along compensated feasible paths.
    reference, scales = np.asarray(reference, dtype=float), np.asarray(scales, dtype=float)
    current = gain(model.scores, reference, scales, indices)
    indices = np.asarray(indices, dtype=int)
    if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0:
        raise ValueError("Release loss must be finite and nonnegative.")
    offsets = (model.scores[indices] - reference[indices]) / scales[indices]
    options = []
    for row in directions(model, damping, tolerance, geometry):
        if checkpoint is not None:
            checkpoint()
        bounds = interval(model, row["direction"], radius, lower, damage)
        if bounds is None:
            continue
        lo, hi = bounds
        slopes = row["retained"][indices] / scales[indices]
        candidates = [lo, hi, float(np.clip(0., lo, hi))]
        for index, slope in enumerate(slopes):
            candidates.append(float(np.clip(slope / (2 * damping * row["norm"]), lo, hi)))
            for other in range(index):
                difference = slope - slopes[other]
                if difference != 0:
                    crossing = (offsets[other] - offsets[index]) / difference
                    if lo <= crossing <= hi:
                        candidates.append(float(crossing))
        for amplitude in set(candidates):
            move = amplitude * row["direction"]
            point = model.anchor.copy()
            point[model.active] += move
            predicted = float(np.min(offsets + amplitude * slopes))
            norm = float(move @ model.metric @ move)
            residual = model.residual + model.response @ move
            cost = float(residual @ residual)
            scores = model.scores + model.semantic @ move
            limits = np.array([2., 2., 1.])
            tiny = np.finfo(float).tiny
            valid = np.isfinite(point).all() and math.isfinite(norm) and math.isfinite(cost) and math.isfinite(predicted) and np.isfinite(scores).all()
            valid = bool(valid and np.all(point >= 0) and np.all(point <= limits) and norm <= radius * radius * (1 + 1e-10) and cost <= damage + 1e-10 * max(damage, tiny) and np.all(scores >= np.asarray(lower) - 1e-10 * np.maximum(np.maximum(np.abs(scores), np.abs(lower)), tiny)))
            if not valid:
                continue
            objective = -predicted + damping * norm
            options.append({"point": tuple(point), "move": move, "predicted": -predicted, "start": -current, "objective": objective, "converged": True, "piece": "protected-response release", "phase": "attainment", "solver": {"operator": "release", "amplitude": amplitude, "interval": list(bounds), "direction": row["direction"].tolist(), "retained": row["retained"].tolist(), "protected": row["protected"].tolist(), "compensationdamping": row["damping"], "proposaldamping": damping, "predicteddamage": cost, "gain": predicted, "rank": row["rank"], "cutoff": row["cutoff"], "geometry": geometry, "basisnorm": row["basisnorm"], "protectednorm": row["protectednorm"], "uncompensatednorm": row["uncompensatednorm"], "iterations": 0, "scope": "Measured compensation geometry; feasibility always uses protected decoded outputs."}})
    selected = min(options, key=lambda row: (row["objective"], row["point"])) if options else None
    return selected if selected is not None and selected["solver"]["gain"] > current + loss + 1e-12 else None

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
