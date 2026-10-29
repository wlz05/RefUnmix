# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import math
import numpy as np
from scipy.optimize import minimize, nnls

def solve(hessian, linear, constant, constraints, bounds, iterations=250, checkpoint=None):
    # Verify convex pieces with primal checks and dual bounds.
    hessian, linear = np.asarray(hessian, dtype=float), np.asarray(linear, dtype=float)
    size = len(linear)
    if size < 1 or hessian.shape != (size, size) or not np.isfinite(hessian).all() or not np.isfinite(linear).all() or not math.isfinite(constant) or not np.allclose(hessian, hessian.T) or type(iterations) is not int or not 1 <= iterations <= 10000:
        raise ValueError("Quadratic objectives or iteration limits are invalid.")
    try:
        np.linalg.cholesky(hessian)
    except np.linalg.LinAlgError as error:
        raise ValueError("Quadratic objectives must be positive definite.") from error
    if len(bounds) != size or any(not all(math.isfinite(value) for value in pair) or pair[0] > pair[1] for pair in bounds):
        raise ValueError("Quadratic parameter bounds are invalid.")
    terms = []
    for matrix, vector, intercept in constraints:
        matrix, vector = np.asarray(matrix, dtype=float), np.asarray(vector, dtype=float)
        if matrix.shape != hessian.shape or vector.shape != linear.shape or not np.isfinite(matrix).all() or not np.isfinite(vector).all() or not math.isfinite(intercept) or not np.allclose(matrix, matrix.T) or np.linalg.eigvalsh(matrix)[0] < -1e-12 * max(1., np.linalg.norm(matrix)):
            raise ValueError("Quadratic constraints must be finite and convex.")
        terms.append((matrix, vector, float(intercept)))
    for index, (lower, upper) in enumerate(bounds):
        direction = np.eye(size)[index]
        terms.extend([(np.zeros_like(hessian), -direction, lower), (np.zeros_like(hessian), direction, -upper)])
    objectiveweight = max(float(np.linalg.norm(hessian)), float(np.linalg.norm(linear)), abs(constant), 1e-30)
    hessian, linear, constant = hessian / objectiveweight, linear / objectiveweight, constant / objectiveweight
    normalized = [(matrix / weight, vector / weight, intercept / weight) for matrix, vector, intercept in terms for weight in [max(float(np.linalg.norm(matrix)), float(np.linalg.norm(vector)), abs(intercept), 1e-30)]]

    def objective(value):
        return float(value @ hessian @ value + linear @ value + constant)

    def gradient(value):
        return 2 * hessian @ value + linear

    def values(value):
        return np.array([value @ matrix @ value + vector @ value + intercept for matrix, vector, intercept in normalized])

    def jacobian(value):
        return np.array([2 * matrix @ value + vector for matrix, vector, _ in normalized])

    inequalities = [{"type": "ineq", "fun": lambda value: -values(value), "jac": lambda value: -jacobian(value)}]
    if checkpoint is not None:
        checkpoint()
    solved = minimize(objective, np.zeros(size), jac=gradient, bounds=bounds, constraints=inequalities, method="SLSQP", callback=None if checkpoint is None else lambda value: checkpoint(), options={"ftol": 1e-12, "maxiter": iterations})
    point = np.asarray(solved.x, dtype=float)
    if point.shape != (size,) or not np.isfinite(point).all():
        return {"point": None, "feasible": False, "converged": False, "solversuccess": bool(solved.success), "iterations": int(solved.nit), "reason": str(solved.message), "dualbound": None, "gap": None, "optimalitycertified": False}
    residuals, gradients = values(point), jacobian(point)
    feasible = bool(np.all(residuals <= 1e-8))
    active = np.flatnonzero(residuals >= -1e-7)
    multipliers = np.zeros(len(normalized))
    try:
        if len(active):
            multipliers[active] = nnls(gradients[active].T, -gradient(point), maxiter=1000)[0]
    except (RuntimeError, np.linalg.LinAlgError):
        pass
    lagrangian = hessian + sum((weight * row[0] for weight, row in zip(multipliers, normalized)), np.zeros_like(hessian))
    vector = linear + sum((weight * row[1] for weight, row in zip(multipliers, normalized)), np.zeros_like(linear))
    intercept = constant + sum(weight * row[2] for weight, row in zip(multipliers, normalized))
    dualbound = float(intercept - .25 * vector @ np.linalg.solve(lagrangian, vector))
    current, gap = objective(point), objective(point) - dualbound
    certified = bool(feasible and math.isfinite(gap) and -1e-8 <= gap <= 1e-7)
    return {"point": point if feasible else None, "feasible": feasible, "converged": bool(feasible and (solved.success or certified)), "solversuccess": bool(solved.success), "iterations": int(solved.nit), "reason": str(solved.message), "objective": current * objectiveweight if feasible else None, "dualbound": dualbound * objectiveweight if math.isfinite(dualbound) else None, "gap": max(0., gap) * objectiveweight if math.isfinite(gap) else None, "optimalitycertified": certified, "tolerance": 1e-7 * objectiveweight, "scope": "Numerical convex-piece certificate, not exact arithmetic or nonlinear image optimality."}

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
