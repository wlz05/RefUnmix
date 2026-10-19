# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import math
import numpy as np
from engine.limit import Exhausted
from engine.quadratic import solve

def gain(scores, reference, scales, indices=None):
    # Measure the least improved editing requirement.
    scores, reference, scales = (np.asarray(value, dtype=float) for value in (scores, reference, scales))
    if scores.ndim != 1 or not len(scores) or reference.shape != scores.shape or scales.shape != scores.shape or any(not np.isfinite(value).all() for value in (scores, reference, scales)) or np.any(scales <= 0):
        raise ValueError("Attainment scores require finite matching positive scales.")
    selected = np.arange(len(scores)) if indices is None else np.asarray(indices)
    if selected.ndim != 1 or not len(selected) or selected.dtype.kind not in "iu" or len(set(selected.tolist())) != len(selected) or np.any(selected < 0) or np.any(selected >= len(scores)):
        raise ValueError("Attainment goals must be unique valid indices.")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        changes = (scores[selected] - reference[selected]) / scales[selected]
    if not np.isfinite(changes).all():
        raise ValueError("Attainment normalization exceeds numerical range.")
    return float(changes.min())

def choose(rows, loss=0., mode="attain"):
    # Preserve attained gain before minimizing bounded cost.
    if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0 or mode not in {"attain", "protect"}:
        raise ValueError("Selection policy or attainment tolerance is invalid.")
    eligible = [row for row in rows if row.feasible and row.complete]
    if not eligible or any(row.objective is None or not math.isfinite(row.objective) or not math.isfinite(row.taskgain) for row in eligible):
        raise ValueError("Selection needs finite completed feasible candidates.")
    best = max(row.taskgain for row in eligible)
    threshold = max(0., best - loss) if mode == "attain" else None
    band = [row for row in eligible if threshold is None or row.taskgain >= threshold]
    if not band:
        raise ValueError("The attainment band requires a retained raw candidate.")
    selected = min(band, key=lambda row: (row.objective, row.key != "raw", row.key))
    return selected, {"policy": mode, "bestgain": best, "gainfloor": threshold, "gainloss": loss, "selectedgain": selected.taskgain, "bandcount": len(band), "eligible": len(eligible), "scope": "Measured control goals, not independent task correctness."}

def maximize(model, radius, lower, damage, reference, scales, indices, ridge=1e-6, iterations=250, checkpoint=None, columns=None):
    # Enumerate bottleneck faces on one protected feasible set.
    lower, reference, scales = (np.asarray(value, dtype=float) for value in (lower, reference, scales))
    gain(model.scores, reference, scales, indices)
    indices = np.asarray(indices, dtype=int)
    count = len(model.active)
    columns = np.arange(count) if columns is None else np.asarray(columns)
    if columns.ndim != 1 or columns.dtype.kind not in "iu" or len(set(columns.tolist())) != len(columns) or np.any(columns < 0) or np.any(columns >= count):
        raise ValueError("Attainment columns must be unique valid indices.")
    if lower.shape != model.scores.shape or not np.isfinite(lower).all() or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in (radius, ridge)) or isinstance(damage, bool) or not isinstance(damage, (int, float)) or not math.isfinite(damage) or damage < 0 or type(iterations) is not int or not 1 <= iterations <= 10000:
        raise ValueError("Attainment budgets or semantic floors are invalid.")
    squaredradius = radius * radius
    if not math.isfinite(squaredradius) or squaredradius == 0 or not math.isfinite(ridge * squaredradius):
        raise ValueError("Attainment squared budgets exceed numerical range.")
    arrays = (model.anchor, model.scores, model.metric, model.gram, model.cross, model.semantic)
    if model.anchor.shape != (3,) or model.metric.shape != (count, count) or model.gram.shape != (count, count) or model.cross.shape != (count,) or model.semantic.shape != (len(lower), count) or any(not np.isfinite(value).all() for value in arrays) or not math.isfinite(model.damage) or model.damage < 0 or model.active.ndim != 1 or model.active.dtype.kind not in "iu" or len(set(model.active.tolist())) != count or any(axis not in (0, 1, 2) for axis in model.active):
        raise ValueError("Attainment model geometry is invalid.")
    limits = np.array([2., 2., 1.])
    if np.any(model.anchor < 0) or np.any(model.anchor > limits):
        raise ValueError("Attainment anchors must satisfy control bounds.")
    if count:
        if not np.allclose(model.metric, model.metric.T) or not np.allclose(model.gram, model.gram.T):
            raise ValueError("Attainment quadratic geometry must be symmetric.")
        try:
            np.linalg.cholesky(model.metric)
        except np.linalg.LinAlgError as error:
            raise ValueError("Attainment control metrics must be positive definite.") from error
        if np.linalg.eigvalsh(model.gram)[0] < -1e-12 * max(1., float(np.linalg.norm(model.gram))):
            raise ValueError("Attainment protection curvature must be positive semidefinite.")
    if not len(columns):
        feasible = bool(model.damage <= damage and np.all(model.scores >= lower))
        value = gain(model.scores, reference, scales, indices)
        return {"point": tuple(model.anchor) if feasible else None, "move": np.zeros(count), "gain": value if feasible else None, "lower": value if feasible else None, "upper": value if feasible else None, "verified": feasible, "pieces": [], "iterations": 0, "scope": "Fixed local control, not nonlinear task capacity."}
    metric = model.metric[np.ix_(columns, columns)]
    gram, cross = model.gram[np.ix_(columns, columns)], model.cross[columns]
    semantic = model.semantic[:, columns]
    bounds = list(zip(-model.anchor[model.active[columns]], limits[model.active[columns]] - model.anchor[model.active[columns]]))
    zero, empty = np.zeros_like(metric), np.zeros(len(columns))
    base = [(metric, empty, -squaredradius), (gram, 2 * cross, model.damage - damage)]
    base.extend((zero, -row, requirement - observed) for row, requirement, observed in zip(semantic, lower, model.scores))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        offsets = (model.scores[indices] - reference[indices]) / scales[indices]
        slopes = semantic[indices] / scales[indices, None]
    options, pieces = [], []
    for head in range(len(indices)):
        constraints = [*base, *((zero, slopes[head] - slopes[other], offsets[head] - offsets[other]) for other in range(len(indices)) if other != head)]
        try:
            result = solve(ridge * metric, -slopes[head], -float(offsets[head]), constraints, bounds, iterations, checkpoint)
        except Exhausted:
            raise
        except (ValueError, np.linalg.LinAlgError, RuntimeError, FloatingPointError, OverflowError) as error:
            pieces.append({"head": int(indices[head]), "dualbound": None, "reason": str(error), "iterations": 0})
            continue
        details = {key: value for key, value in result.items() if key != "point"}
        details["head"] = int(indices[head])
        pieces.append(details)
        if result["point"] is None:
            continue
        move = np.asarray(result["point"], dtype=float)
        if move.shape != (len(columns),) or not np.isfinite(move).all():
            details["reason"] = "Solver controls have invalid shape or values."
            continue
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            norm = float(move @ metric @ move)
            predicted = float(model.damage + 2 * cross @ move + move @ gram @ move)
            scores = model.scores + semantic @ move
        tolerance, tiny = 1e-8, np.finfo(float).tiny
        valid = move.shape == (len(columns),) and np.isfinite(move).all() and math.isfinite(norm) and math.isfinite(predicted) and np.isfinite(scores).all()
        valid = bool(valid and norm <= squaredradius * (1 + tolerance) and predicted <= damage + tolerance * max(damage, model.damage, tiny) and np.all(scores >= lower - tolerance * np.maximum(np.maximum(np.abs(lower), np.abs(model.scores)), tiny)) and all(a - tolerance <= value <= b + tolerance for value, (a, b) in zip(move, bounds)))
        if not valid:
            details["reason"] = "Original local budgets failed numerical rechecks."
            continue
        value = gain(scores, reference, scales, indices)
        objective = ridge * norm - value
        fullmove = np.zeros(count)
        fullmove[columns] = move
        point = model.anchor.copy()
        point[model.active] += fullmove
        options.append({"point": tuple(point), "move": fullmove, "gain": value, "objective": objective, "converged": result["converged"], "head": int(indices[head])})
    if not options:
        return {"point": None, "move": None, "gain": None, "lower": None, "upper": None, "verified": False, "pieces": pieces, "iterations": sum(row["iterations"] for row in pieces), "scope": "Unresolved local attainment; missing is not zero."}
    chosen = min(options, key=lambda row: (row["objective"], row["point"], row["head"]))
    duals = [row.get("dualbound") for row in pieces]
    tolerance = max([row.get("tolerance", 0.) for row in pieces] + [1e-12])
    upper = ridge * squaredradius - min(duals) + tolerance if all(value is not None and math.isfinite(value) for value in duals) else None
    consistent = upper is None or math.isfinite(upper) and upper >= chosen["gain"] - tolerance
    return {**chosen, "lower": chosen["gain"] if consistent else None, "upper": max(chosen["gain"], upper) if upper is not None and consistent else None, "verified": bool(consistent), "pieces": pieces, "iterations": sum(row["iterations"] for row in pieces), "regularizationbias": ridge * squaredradius, "tolerance": tolerance, "scope": "Numerical protected local attainment, not generator or task optimality."}

def propose(model, radius, lower, scales, damage, damagescale, cap, ridge, reference, indices, loss=0., iterations=250, checkpoint=None):
    # Advance the bottleneck before reducing protected cost.
    from engine.trust import propose as protected
    if isinstance(loss, bool) or not math.isfinite(loss) or loss < 0:
        raise ValueError("Attainment loss must be finite and nonnegative.")
    solved = maximize(model, radius, lower, damage, reference, scales, indices, ridge, iterations, checkpoint)
    if solved["point"] is None:
        return None
    current = gain(model.scores, reference, scales, indices)
    if solved["gain"] > current + loss + 1e-12:
        return {"point": solved["point"], "move": solved["move"], "predicted": -solved["gain"], "start": -current, "objective": solved["objective"], "converged": solved["converged"], "piece": "attainment", "phase": "attainment", "solver": {key: value for key, value in solved.items() if key not in {"point", "move"}}}
    floors = np.asarray(lower, dtype=float).copy()
    threshold = max(0., current - loss)
    indices = np.asarray(indices, dtype=int)
    floors[indices] = np.maximum(floors[indices], np.asarray(reference)[indices] + np.asarray(scales)[indices] * threshold)
    result = protected(model, radius, floors, scales, damage, damagescale, cap, ridge, iterations=iterations, checkpoint=checkpoint)
    if result is not None:
        result.update({"phase": "protection", "gainfloor": threshold})
        result["solver"]["attainment"] = {key: value for key, value in solved.items() if key not in {"point", "move"}}
        result["solver"]["iterations"] += solved["iterations"]
    return result

def capacity(model, radius, lower, damage, reference, scales, indices, ridge=1e-6, iterations=150, checkpoint=None):
    # Compare simultaneous attainment without independent optima.
    full = maximize(model, radius, lower, damage, reference, scales, indices, ridge, iterations, checkpoint)
    field = maximize(model, radius, lower, damage, reference, scales, indices, ridge, iterations, checkpoint, np.flatnonzero(model.active != 2))
    valid = full["verified"] and field["verified"] and all(row["lower"] is not None and row["upper"] is not None for row in (full, field))
    tolerance = max(full.get("tolerance", 0.), field.get("tolerance", 0.), 1e-10)
    low = max(0., full["lower"] - field["upper"]) if valid else None
    high = full["upper"] - field["lower"] if valid else None
    consistent = bool(valid and math.isfinite(low) and math.isfinite(high) and high >= low - tolerance)
    return {"full": {key: value for key, value in full.items() if key != "move"}, "field": {key: value for key, value in field.items() if key != "move"}, "verified": consistent, "incrementlower": low if consistent else None, "incrementupper": max(low, high) if consistent else None, "positiveidentified": bool(consistent and low > tolerance), "iterations": full["iterations"] + field["iterations"], "tolerance": tolerance, "scope": "Same frozen protected model and all floors; missing is not zero."}

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
