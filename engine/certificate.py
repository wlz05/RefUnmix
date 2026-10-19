# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import math
import numpy as np
from scipy.optimize import minimize
from engine.theory import arrays, support
from engine.limit import Exhausted

def normgradient(value):
    length = float(np.linalg.norm(value))
    return length, value / length if length > 1e-15 else np.zeros_like(value)

def distancebounds(lower, upper, scale):
    # Bound only the unboxed local joint deficit.
    if not all(math.isfinite(value) and value >= 0 for value in (lower, upper, scale)) or scale <= 0:
        raise ValueError("Local distance bounds exceed the numerical range.")
    consistent = lower <= upper + 1e-9 * scale
    return {"lower": lower if consistent else None, "upper": max(lower, upper) if consistent else None, "gap": max(0., upper - lower) if consistent else None, "consistent": bool(consistent), "units": "Fixed semantic and protected-response units, not independently scale invariant.", "scope": "Unboxed affine proxy distance; the upper bound uses the zero step, not a boxed or nonlinear feasibility certificate."}

def identify(response, residual, semantics, metric, beta, radius, damage, iterations=150, checkpoint=None):
    # Verify normalized witnesses without certifying nonlinear images.
    response, residual, semantics, metric = arrays(response, residual, semantics, metric)
    beta = np.asarray(beta, dtype=float)
    if beta.shape != (semantics.shape[0],) or not np.isfinite(beta).all() or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in (radius, damage)) or type(iterations) is not int or not 1 <= iterations <= 10000:
        raise ValueError("Normalized certificate controls are invalid.")
    if checkpoint is not None and not callable(checkpoint):
        raise ValueError("Certificate checkpoints must be callable.")
    if checkpoint is not None:
        checkpoint()
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            scale = max(float(np.linalg.norm(response)), float(np.linalg.norm(residual)), float(np.linalg.norm(semantics)), float(np.linalg.norm(beta)), math.sqrt(damage), 1e-30)
            upper = math.hypot(float(np.linalg.norm(np.maximum(beta, 0.))), max(0., float(np.linalg.norm(residual)) - math.sqrt(damage)))
        except FloatingPointError as error:
            raise ValueError("Certificate data norms exceed the numerical range.") from error
    distancebounds(0., upper, scale)
    if np.all(beta <= 0) and float(residual @ residual) <= damage:
        return {"margin": 0., "normalizedmargin": 0., "threshold": 0., "infeasible": False, "status": "no certificate found", "multipliers": np.zeros(len(beta)).tolist(), "dualweights": np.zeros(response.shape[1] + 1).tolist(), "norm": 0., "bound": 0., "requirement": 0., "converged": True, "iterations": 0, "dimension": len(beta), "optimalitycertified": True, "primalwitness": np.zeros(response.shape[1]).tolist(), "distance": distancebounds(0., upper, scale), "scope": "The verified zero step certifies only local proxy feasibility."}
    joined = np.column_stack((response, residual))
    left, singular, right = np.linalg.svd(joined, full_matrices=False)
    rank = int(np.sum(singular > (singular[0] * 1e-12 if len(singular) else 0)))
    basis = left[:, :rank]
    projected, center = basis.T @ response, basis.T @ residual
    root = np.linalg.cholesky(metric)
    first = np.linalg.solve(root, semantics.T / scale)
    second = np.linalg.solve(root, projected.T / scale)
    target, origin, protected = beta / scale, center / scale, math.sqrt(damage) / scale
    count, dimensions = len(beta), len(beta) + rank

    def unpack(value):
        return value[:count], value[count:dimensions]

    def objective(value):
        multipliers, dual = unpack(value)
        return float(-target @ multipliers - origin @ dual + radius * value[-2] + protected * value[-1])

    linear = np.r_[-target, -origin, radius, protected]

    def firstconstraint(value):
        multipliers, dual = unpack(value)
        return value[-2] - np.linalg.norm(first @ multipliers - second @ dual)

    def firstgradient(value):
        multipliers, dual = unpack(value)
        _, direction = normgradient(first @ multipliers - second @ dual)
        return np.r_[-first.T @ direction, second.T @ direction, 1., 0.]

    def secondconstraint(value):
        return value[-1] - np.linalg.norm(value[count:dimensions])

    def secondgradient(value):
        _, direction = normgradient(value[count:dimensions])
        return np.r_[np.zeros(count), -direction, 0., 1.]

    constraints = [{"type": "ineq", "fun": firstconstraint, "jac": firstgradient}, {"type": "ineq", "fun": secondconstraint, "jac": secondgradient}, {"type": "ineq", "fun": lambda value: 1 - value[:dimensions] @ value[:dimensions], "jac": lambda value: np.r_[-2 * value[:dimensions], 0., 0.]}]
    seeds = [np.zeros(dimensions)]
    for index in range(count):
        seed = np.zeros(dimensions)
        seed[index] = 1.
        seeds.append(seed)
    if rank and np.linalg.norm(center) > 0:
        seed = np.r_[np.zeros(count), center / np.linalg.norm(center)]
        seeds.append(seed)
    best, bestvalue, used, converged = np.zeros(dimensions), 0., 0, False

    def consider(value):
        nonlocal best, bestvalue
        if value.shape != (dimensions,) or not np.isfinite(value).all():
            return
        value = value.copy()
        value[:count] = np.maximum(value[:count], 0)
        value /= max(1., float(np.linalg.norm(value)))
        multipliers, dual = unpack(value)
        estimate = float(target @ multipliers + origin @ dual - radius * np.linalg.norm(first @ multipliers - second @ dual) - protected * np.linalg.norm(dual))
        if math.isfinite(estimate) and estimate > bestvalue:
            best, bestvalue = value, estimate

    for seed in seeds:
        consider(seed)
    start = np.r_[best, np.linalg.norm(first @ best[:count] - second @ best[count:]), np.linalg.norm(best[count:])]
    if checkpoint is not None:
        checkpoint()
    failure = None
    try:
        solved = minimize(objective, start, jac=lambda value: linear, bounds=[(0., None)] * count + [(None, None)] * rank + [(0., None)] * 2, constraints=constraints, method="SLSQP", callback=None if checkpoint is None else lambda value: checkpoint(), options={"ftol": 1e-11, "maxiter": iterations})
        used += int(solved.nit)
        converged = bool(solved.success)
        consider(np.asarray(solved.x[:dimensions], dtype=float))
    except Exhausted:
        raise
    except (np.linalg.LinAlgError, RuntimeError, FloatingPointError, OverflowError) as error:
        failure = str(error)
    multipliers, reduced = unpack(best)
    weights = right[:rank].T @ (reduced / singular[:rank]) if rank else np.zeros(joined.shape[1])
    dual = joined @ weights
    length = math.sqrt(float(multipliers @ multipliers + dual @ dual))
    if length > 1:
        multipliers, weights, dual = multipliers / length, weights / length, dual / length
    checked = support(response, residual, semantics, metric, beta, multipliers, dual, radius, damage)
    if not all(math.isfinite(checked[key]) for key in ("margin", "bound", "requirement")):
        raise ValueError("Checked certificate values exceed the numerical range.")
    tolerance = 1e-9 * scale
    distance = distancebounds(max(0., checked["margin"]), upper, scale)
    margin = max(0., checked["margin"]) if distance["consistent"] else 0.
    verified = bool(distance["consistent"] and margin > tolerance)
    status = "unverified local distance interval" if not distance["consistent"] else "certified local infeasible" if verified else "no certificate found"
    return {"margin": margin, "normalizedmargin": margin / scale, "threshold": tolerance, "infeasible": verified, "status": status, "multipliers": multipliers.tolist(), "dualweights": weights.tolist(), "norm": math.sqrt(float(multipliers @ multipliers + dual @ dual)), "bound": checked["bound"], "requirement": checked["requirement"], "converged": converged, "iterations": used, "dimension": dimensions, "failure": failure, "optimalitycertified": False, "distance": distance, "scope": "Verified witness for the fixed local proxy; a zero lower bound never certifies feasibility."}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
