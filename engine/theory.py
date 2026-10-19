# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import math
import numpy as np

def arrays(response, residual, semantics, metric):
    response, residual, semantics, metric = (np.asarray(value, dtype=np.float64) for value in (response, residual, semantics, metric))
    if response.ndim != 2 or min(response.shape) < 1 or residual.shape != (response.shape[0],) or semantics.ndim != 2 or semantics.shape[1] != response.shape[1] or metric.shape != (response.shape[1], response.shape[1]):
        raise ValueError("Local response dimensions disagree.")
    if any(not np.isfinite(value).all() for value in (response, residual, semantics, metric)) or not np.allclose(metric, metric.T):
        raise ValueError("Local arrays must be finite and symmetric.")
    try:
        np.linalg.cholesky(metric)
    except np.linalg.LinAlgError as error:
        raise ValueError("The control metric must be positive definite.") from error
    return response, residual, semantics, metric

def support(response, residual, semantics, metric, beta, multipliers, dual, radius, damage):
    # Certify infeasibility only for the local proxy.
    response, residual, semantics, metric = arrays(response, residual, semantics, metric)
    beta, multipliers, dual = (np.asarray(value, dtype=np.float64) for value in (beta, multipliers, dual))
    if beta.shape != (semantics.shape[0],) or multipliers.shape != beta.shape or dual.shape != residual.shape:
        raise ValueError("Dual certificate dimensions disagree.")
    if any(not np.isfinite(value).all() for value in (beta, multipliers, dual)) or np.any(multipliers < 0) or not all(math.isfinite(value) and value >= 0 for value in (radius, damage)):
        raise ValueError("Dual values and local bounds are invalid.")
    vector = semantics.T @ multipliers - response.T @ dual
    bound = radius * math.sqrt(max(0., float(vector @ np.linalg.solve(metric, vector)))) + math.sqrt(damage) * float(np.linalg.norm(dual)) - float(dual @ residual)
    requirement = float(multipliers @ beta)
    return {"bound": bound, "requirement": requirement, "margin": requirement - bound, "infeasible": requirement > bound + 1e-12, "scope": "Fixed local proxy; not a global image certificate."}

def fit(initial, directions, observed, regularization):
    # Fit regularized secants without creating control directions.
    initial, directions, observed = (np.asarray(value, dtype=np.float64) for value in (initial, directions, observed))
    if initial.ndim != 2 or directions.ndim != 2 or observed.shape != (initial.shape[0], directions.shape[1]) or directions.shape[0] != initial.shape[1]:
        raise ValueError("Regularized secant dimensions disagree.")
    if any(not np.isfinite(value).all() for value in (initial, directions, observed)) or not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("Secant inputs and regularization are invalid.")
    gram = directions.T @ directions + regularization * np.eye(directions.shape[1])
    return initial + (observed - initial @ directions) @ np.linalg.solve(gram, directions.T)

def solve(response, residual, semantics, metric, beta, radius, damage, regularization=.001):
    # Solve a convex local diagnostic with verified constraints.
    from scipy.optimize import minimize
    response, residual, semantics, metric = arrays(response, residual, semantics, metric)
    beta = np.asarray(beta, dtype=np.float64)
    if beta.shape != (semantics.shape[0],) or not np.isfinite(beta).all() or not all(math.isfinite(value) and value >= 0 for value in (radius, damage)) or not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("Local diagnostic constraints are invalid.")

    def objective(value):
        shifted = residual + response @ value
        return float(shifted @ shifted + regularization * (value @ metric @ value))

    def gradient(value):
        return 2 * response.T @ (residual + response @ value) + 2 * regularization * metric @ value

    constraints = [{"type": "ineq", "fun": lambda value: radius ** 2 - value @ metric @ value, "jac": lambda value: -2 * metric @ value}, {"type": "ineq", "fun": lambda value: damage - np.square(residual + response @ value).sum(), "jac": lambda value: -2 * response.T @ (residual + response @ value)}]
    if len(beta):
        constraints.append({"type": "ineq", "fun": lambda value: semantics @ value - beta, "jac": lambda value: semantics})
    fitted = minimize(objective, np.zeros(response.shape[1]), jac=gradient, constraints=constraints, method="SLSQP", options={"ftol": 1e-12, "maxiter": 2000})
    point = fitted.x
    feasible = np.isfinite(point).all() and point @ metric @ point <= radius ** 2 + 1e-8 and np.square(residual + response @ point).sum() <= damage + 1e-8 and np.all(semantics @ point >= beta - 1e-8)
    return {"point": point.tolist() if feasible else None, "feasible": bool(feasible), "converged": bool(fitted.success), "objective": objective(point) if feasible else None, "reason": str(fitted.message), "scope": "Local QCQP only; never used by native search."}

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
