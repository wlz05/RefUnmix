# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import math
import numpy as np
from engine.limit import Exhausted
from engine.quadratic import solve

def support(model, indices, gradient, lower, damage, radius, regularization, iterations, checkpoint):
    # Bound semantic support on the same protected feasible set.
    indices = np.asarray(indices, dtype=int)
    if not len(indices) or (np.all(gradient[indices] == 0) and model.damage <= damage and np.all(model.scores >= lower)):
        feasible = bool(model.damage <= damage and np.all(model.scores >= lower))
        return {"verified": feasible, "lower": 0. if feasible else None, "upper": 0. if feasible else None, "iterations": 0, "point": model.anchor.tolist() if feasible else None, "reason": "Fixed or zero-objective feasible point checked directly.", "tolerance": 0.}
    metric, gram = model.metric[np.ix_(indices, indices)], model.gram[np.ix_(indices, indices)]
    semantic, cross = model.semantic[:, indices], model.cross[indices]
    zero, empty = np.zeros_like(metric), np.zeros(len(indices))
    constraints = [(metric, empty, -radius ** 2), (gram, 2 * cross, model.damage - damage)]
    constraints.extend((zero, -row, requirement - observed) for row, requirement, observed in zip(semantic, lower, model.scores))
    limits = np.array([2., 2., 1.])[model.active[indices]]
    center = model.anchor[model.active[indices]]
    bounds = list(zip(-center, limits - center))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = solve(regularization * metric, -gradient[indices], 0., constraints, bounds, iterations, checkpoint)
    except Exhausted:
        raise
    except (ValueError, np.linalg.LinAlgError, RuntimeError, FloatingPointError, OverflowError) as error:
        return {"verified": False, "lower": None, "upper": None, "iterations": 0, "reason": str(error)}
    move = result.get("point")
    dual = result.get("dualbound")
    if move is None or dual is None or not math.isfinite(dual):
        return {"verified": False, "lower": None, "upper": None, "iterations": result["iterations"], "reason": "No feasible point and finite dual bound were verified.", "solver": {key: value for key, value in result.items() if key != "point"}}
    move = np.asarray(move, dtype=float)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            norm = float(move @ metric @ move)
            predicted = float(model.damage + 2 * cross @ move + move @ gram @ move)
            scores = model.scores + semantic @ move
        tiny, feasibility = np.finfo(float).tiny, 1e-8
        valid = move.shape == (len(indices),) and np.isfinite(move).all() and math.isfinite(norm) and math.isfinite(predicted) and np.isfinite(scores).all()
        valid = bool(valid and norm <= radius ** 2 * (1 + feasibility) and predicted <= damage + feasibility * max(damage, model.damage, tiny) and np.all(scores >= lower - feasibility * np.maximum(np.maximum(np.abs(lower), np.abs(model.scores)), tiny)) and all(first - feasibility <= value <= last + feasibility for value, (first, last) in zip(move, bounds)))
    except (ValueError, FloatingPointError, OverflowError):
        valid = False
    if not valid:
        return {"verified": False, "lower": None, "upper": None, "iterations": result["iterations"], "reason": "Support point violates model budgets beyond the numerical feasibility tolerance."}
    value = float(gradient[indices] @ move)
    tolerance = float(result.get("tolerance", 1e-7))
    upper = regularization * radius ** 2 - dual + tolerance
    consistent = bool(math.isfinite(value) and math.isfinite(upper) and upper >= value - tolerance)
    point = model.anchor.copy()
    point[model.active[indices]] += move
    return {"verified": consistent, "lower": value if consistent else None, "upper": max(value, upper) if consistent else None, "iterations": result["iterations"], "point": point.tolist(), "tolerance": tolerance, "feasibilitytolerance": feasibility, "regularizationbias": regularization * radius ** 2, "reason": "Numerical primal-dual support interval." if consistent else "Inconsistent numerical support bounds.", "solver": {key: value for key, value in result.items() if key != "point"}}

def analyze(model, lower, scales, damage, radius, names=None, regularization=1e-6, iterations=150, checkpoint=None):
    # Compare finite-budget capacity without generating more images.
    lower, scales = np.asarray(lower, dtype=float), np.asarray(scales, dtype=float)
    count = len(model.active)
    if lower.shape != model.scores.shape or scales.shape != lower.shape or not np.isfinite(lower).all() or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Frontier semantic controls must match finite response rows.")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in (radius, regularization)) or isinstance(damage, bool) or not isinstance(damage, (int, float)) or not math.isfinite(damage) or damage < 0 or type(iterations) is not int or not 1 <= iterations <= 10000:
        raise ValueError("Frontier budgets and numerical controls are invalid.")
    squaredradius = radius * radius
    if not math.isfinite(squaredradius) or squaredradius <= 0 or not math.isfinite(regularization * squaredradius):
        raise ValueError("Frontier squared budgets exceed the numerical range.")
    arrays = (model.anchor, model.scores, model.semantic, model.metric, model.gram, model.cross)
    if count < 1 or len(set(model.active)) != count or any(axis not in (0, 1, 2) for axis in model.active) or model.anchor.shape != (3,) or model.semantic.shape != (len(lower), count) or model.metric.shape != (count, count) or model.gram.shape != (count, count) or model.cross.shape != (count,) or any(not np.isfinite(value).all() for value in arrays) or not math.isfinite(model.damage) or model.damage < 0:
        raise ValueError("Frontier model geometry is invalid.")
    if not np.allclose(model.metric, model.metric.T, rtol=1e-10, atol=1e-12) or np.linalg.eigvalsh(model.metric)[0] <= 0 or not np.allclose(model.gram, model.gram.T, rtol=1e-10, atol=1e-12) or np.linalg.eigvalsh(model.gram)[0] < -1e-12 * max(1., np.linalg.norm(model.gram)):
        raise ValueError("Frontier model quadratics must be convex.")
    limits = np.array([2., 2., 1.])
    if np.any(model.anchor < 0) or np.any(model.anchor > limits):
        raise ValueError("Frontier anchors must satisfy control bounds.")
    if names is not None and not isinstance(names, (list, tuple)):
        raise ValueError("Frontier component names must be a list or tuple.")
    names = list(names) if names is not None else ["component" + str(index) for index in range(len(lower))]
    if not names or len(names) != len(lower) or any(not isinstance(name, str) or not name or not name.isascii() for name in names) or len(set(names)) != len(names):
        raise ValueError("Frontier component names must be unique English labels.")
    report = {"active": 2 in model.active, "radius": radius, "damagebudget": damage, "regularization": regularization, "components": {}, "iterations": 0, "scope": "Frozen local proxy with the same box, radius, damage budget, and all semantic floors; no nonlinear or causal certificate."}
    if not report["active"]:
        report["status"] = "Reference coordinate is not active."
        return report
    full = np.arange(count)
    field = np.flatnonzero(model.active != 2)
    for index, name in enumerate(names):
        if checkpoint is not None:
            checkpoint()
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                gradient = model.semantic[index] / scales[index]
        except FloatingPointError:
            report["components"][name] = {"verified": False, "incrementlower": None, "incrementupper": None, "positiveidentified": False, "reason": "Normalized semantic response exceeds the numerical range."}
            continue
        first = support(model, full, gradient, lower, damage, radius, regularization, iterations, checkpoint)
        second = support(model, field, gradient, lower, damage, radius, regularization, iterations, checkpoint)
        valid = first["verified"] and second["verified"]
        minimum = max(0., first["lower"] - second["upper"]) if valid else None
        maximum = first["upper"] - second["lower"] if valid else None
        tolerance = max(first.get("tolerance", 0.), second.get("tolerance", 0.), 1e-10)
        consistent = bool(valid and maximum >= minimum - tolerance)
        report["components"][name] = {"full": first, "field": second, "verified": consistent, "incrementlower": minimum if consistent else None, "incrementupper": max(minimum, maximum) if consistent else None, "positiveidentified": bool(consistent and minimum > tolerance), "tolerance": tolerance, "note": "Field comparison freezes eta at this anchor, not necessarily at the target-reference endpoint."}
        report["iterations"] += first["iterations"] + second["iterations"]
    report["status"] = "Numerical intervals recorded; unverified components remain missing."
    return report

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
