# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import math
import numpy as np
from engine.attainment import gain

def directions(model, damping, tolerance=1e-6, geometry="output", policy="increment"):
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
    if policy not in {"increment", "residual"}:
        raise ValueError("Unknown protected-response compensation policy.")
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
    residual = model.residual if geometry != "latent" else getattr(model, "latentresidual", None) if policy == "residual" else np.zeros(basis.shape[0])
    residual = np.asarray(residual, dtype=float)
    if residual.shape != (basis.shape[0],) or not np.isfinite(residual).all():
        raise ValueError("Compensation residuals must match the measured response space.")
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
        diagnostics = {"basisnorm": float(np.linalg.norm(basis @ direction)), "protectednorm": float(np.linalg.norm(protected)), "uncompensatednorm": float(np.linalg.norm(response[:, reference]))}
        if any(not math.isfinite(value) for value in diagnostics.values()):
            raise ValueError("Compensation diagnostics exceed numerical range.")
        offsets = [("increment", np.zeros(count))]
        if policy == "residual" and geometry != "attenuate":
            offset = np.zeros(count)
            offset[fields] = -inverse @ (right.T @ (weights * (left.T @ residual)))
            offsets.append(("residual", offset))
        for kind, offset in offsets:
            if not np.isfinite(offset).all():
                if kind == "residual":
                    continue
                raise ValueError("Residual compensation exceeds numerical range.")
            if any(np.allclose(direction, row["direction"], rtol=1e-12, atol=1e-14) and np.allclose(offset, row["offset"], rtol=1e-12, atol=1e-14) for row in rows):
                continue
            rows.append({"direction": direction, "offset": offset, "policy": kind, "protected": protected, "retained": retained, "norm": norm, "damping": value, "rank": int(resolved.sum()), "cutoff": cutoff, "geometry": geometry, **diagnostics})
    return rows

def interval(model, direction, radius, lower, damage, offset=None):
    # Intersect every protection and semantic bound exactly.
    direction, lower = np.asarray(direction, dtype=float), np.asarray(lower, dtype=float)
    offset = np.zeros_like(direction) if offset is None else np.asarray(offset, dtype=float)
    limits = np.array([2., 2., 1.])
    arrays = (model.anchor, model.residual, model.scores, direction, lower, offset)
    if model.anchor.shape != (3,) or model.residual.shape != (model.response.shape[0],) or model.scores.shape != (model.semantic.shape[0],) or direction.shape != (len(model.active),) or offset.shape != direction.shape or lower.shape != model.scores.shape or any(not np.isfinite(value).all() for value in arrays) or np.any(model.anchor < 0) or np.any(model.anchor > limits):
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
    lo, hi = -math.inf, math.inf

    def linear(slope, offset):
        nonlocal lo, hi
        if slope > 0:
            lo = max(lo, -offset / slope)
        elif slope < 0:
            hi = min(hi, -offset / slope)
        elif offset < 0:
            lo, hi = 1., 0.

    def quadratic(a, b, c):
        nonlocal lo, hi
        if not all(math.isfinite(value) for value in (a, b, c)) or a < 0:
            raise ValueError("Release constraint polynomials exceed numerical range.")
        if a == 0:
            linear(-b, -c)
            return
        scale = max(a, abs(b), abs(c))
        aa, bb, cc = a / scale, b / scale, c / scale
        if aa == 0:
            raise ValueError("Release protection curvature underflows normalization.")
        discriminant = bb * bb - 4 * aa * cc
        if discriminant < 0:
            lo, hi = 1., 0.
            return
        root = math.sqrt(discriminant)
        q = -.5 * (bb + math.copysign(root, bb))
        first, second = (-bb / (2 * aa),) * 2 if q == 0 else (q / aa, cc / q)
        lo, hi = max(lo, min(first, second)), min(hi, max(first, second))

    # Intersect the shifted trust ball before testing decoded damage.
    quadratic(norm, float(2 * offset @ model.metric @ direction), float(offset @ model.metric @ offset - radius * radius))
    center = model.anchor.copy()
    center[model.active] += offset
    for axis, slope in zip(model.active, direction):
        linear(float(slope), float(center[axis]))
        linear(float(-slope), float(limits[axis] - center[axis]))
    for slope, margin in zip(model.semantic @ direction, model.scores + model.semantic @ offset - lower):
        linear(float(slope), float(margin))
    protected = model.response @ direction
    residual = model.residual + model.response @ offset
    quadratic(float(protected @ protected), float(2 * residual @ protected), float(residual @ residual - damage))
    return (lo, hi) if lo <= hi and math.isfinite(lo) and math.isfinite(hi) else None

def propose(model, radius, lower, damage, reference, scales, indices, damping, loss=0., tolerance=1e-6, checkpoint=None, geometry="output", policy="increment"):
    # Release reference effects along compensated feasible paths.
    reference, scales = np.asarray(reference, dtype=float), np.asarray(scales, dtype=float)
    current = gain(model.scores, reference, scales, indices)
    indices = np.asarray(indices, dtype=int)
    if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0:
        raise ValueError("Release loss must be finite and nonnegative.")
    offsets = (model.scores[indices] - reference[indices]) / scales[indices]
    options = []
    skipped = []
    paths = directions(model, damping, tolerance, geometry, policy)
    for row in paths:
        if checkpoint is not None:
            checkpoint()
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                bounds = interval(model, row["direction"], radius, lower, damage, row["offset"])
        except (ValueError, FloatingPointError, OverflowError) as error:
            if row["policy"] != "residual":
                raise
            skipped.append({"damping": row["damping"], "reason": str(error)})
            continue
        if bounds is None:
            continue
        lo, hi = bounds
        slopes = row["retained"][indices] / scales[indices]
        intercepts = offsets + (model.semantic @ row["offset"])[indices] / scales[indices]
        metriccross = float(row["offset"] @ model.metric @ row["direction"])
        candidates = [lo, hi, float(np.clip(0., lo, hi))]
        for index, slope in enumerate(slopes):
            candidates.append(float(np.clip((slope / (2 * damping) - metriccross) / row["norm"], lo, hi)))
            for other in range(index):
                difference = slope - slopes[other]
                if difference != 0:
                    crossing = (intercepts[other] - intercepts[index]) / difference
                    if lo <= crossing <= hi:
                        candidates.append(float(crossing))
        for amplitude in set(candidates):
            move = row["offset"] + amplitude * row["direction"]
            point = model.anchor.copy()
            point[model.active] += move
            predicted = float(np.min(intercepts + amplitude * slopes))
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
            # Account for repair of the current protected residual.
            effect = model.response @ move
            cross = float(2 * model.residual @ effect)
            quadratic = float(effect @ effect)
            options.append({"point": tuple(point), "move": move, "predicted": -predicted, "start": -current, "objective": objective, "converged": True, "piece": "protected-response release", "phase": "attainment", "solver": {"operator": "release", "policy": row["policy"], "requestedpolicy": policy, "offset": row["offset"].tolist(), "repairnorm": float(np.linalg.norm(model.response @ row["offset"])), "repairtarget": (model.semantic @ row["offset"]).tolist(), "amplitude": amplitude, "interval": list(bounds), "direction": row["direction"].tolist(), "retained": row["retained"].tolist(), "protected": row["protected"].tolist(), "compensationdamping": row["damping"], "proposaldamping": damping, "predicteddamage": cost, "damagecross": cross, "damagequadratic": quadratic, "repaircredit": max(0., -cross), "gain": predicted, "rank": row["rank"], "cutoff": row["cutoff"], "geometry": geometry, "basisnorm": row["basisnorm"], "protectednorm": row["protectednorm"], "uncompensatednorm": row["uncompensatednorm"], "iterations": 0, "scope": "Measured compensation geometry; feasibility always uses protected decoded outputs."}})
    selected = min(options, key=lambda row: (row["objective"], row["point"])) if options else None
    if selected is not None:
        previous = [row["objective"] for row in options if row["solver"]["policy"] == "increment"]
        shifted = [row["objective"] for row in options if row["solver"]["policy"] == "residual"]
        old = min(previous) if previous else None
        new = min(shifted) if shifted else None
        selected["solver"]["comparison"] = {"increment": old, "residual": new, "improvement": max(0., old - selected["objective"]) if old is not None else None, "paths": len(paths), "skipped": skipped, "queries": 0, "scope": "Same frozen response model and bounds; not a comparison of generated images."}
    return selected if selected is not None and selected["solver"]["gain"] > current + loss + 1e-12 else None

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
