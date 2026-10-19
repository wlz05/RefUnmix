# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import numpy as np
from scipy.optimize import minimize
from engine.theory import fit
from engine.limit import Exhausted

class Model:
    def __init__(self, anchor, points, residuals, scores, penalties, active, regularization=.0001, initial=None, neighbors=7):
        self.anchor = np.asarray(anchor, dtype=float)
        self.active = np.asarray(active, dtype=int)
        self.metric = np.diag(np.array([.25, .25, 1.])[self.active])
        self.root = np.sqrt(np.diag(self.metric))
        points = np.asarray(points, dtype=float)
        offsets = points[:, self.active] - self.anchor[self.active]
        distance = np.linalg.norm(offsets * self.root, axis=1)
        self.indices = np.argsort(distance, kind="stable")[:neighbors]
        zero = np.flatnonzero(distance < 1e-12)
        if len(zero) != 1 or len(self.active) == 0:
            raise ValueError("Response models require one active visited anchor.")
        center = int(zero[0])
        self.residual = np.asarray(residuals[center], dtype=float)
        self.scores = np.asarray(scores[center], dtype=float)
        self.penalty = penalties[center]
        chosen = [int(index) for index in self.indices if index != center]
        self.directions = offsets[chosen].T
        rows = len(self.residual) + len(self.scores)
        previous = np.zeros((rows, len(self.active))) if initial is None else np.asarray(initial, dtype=float)
        observed = np.stack([np.r_[np.asarray(residuals[index]) - self.residual, np.asarray(scores[index]) - self.scores] for index in chosen], axis=1) if chosen else np.empty((rows, 0))
        self.observed = observed
        self.oldprediction = previous @ self.directions
        self.innovation = observed - self.oldprediction
        self.fitted = fit(previous, self.directions, observed, regularization)
        self.fiterror = observed - self.fitted @ self.directions
        self.response = self.fitted[:len(self.residual)]
        self.semantic = self.fitted[len(self.residual):]
        self.gram = self.response.T @ self.response
        self.cross = self.response.T @ self.residual
        self.damage = float(self.residual @ self.residual)
        known = [index for index in chosen if penalties[index] is not None]
        self.slope = np.zeros(len(self.active))
        if self.penalty is not None and known:
            self.slope = fit(np.zeros((1, len(self.active))), offsets[known].T, np.asarray([[penalties[index] - self.penalty for index in known]]), regularization)[0]
        normalized = offsets[chosen] * self.root
        if len(normalized):
            _, singular, vectors = np.linalg.svd(normalized, full_matrices=True)
            tolerance = max(1e-12, singular[0] * 1e-10)
            self.rank = int(np.sum(singular > tolerance))
            self.condition = float(singular[0] / singular[-1]) if self.rank == len(self.active) else None
            self.weak = vectors[-1]
        else:
            self.rank, self.condition, self.weak = 0, None, np.eye(len(self.active))[0]

    def prediction(self, value, cap):
        damage = self.damage + 2 * self.cross @ value + value @ self.gram @ value
        penalty = None if self.penalty is None else float(np.clip(self.penalty + self.slope @ value, 0, np.inf if cap is None else cap))
        return float(max(0., damage)), self.scores + self.semantic @ value, penalty

    def restore(self, value, lower, scales, damage, damagescale):
        predicted, semantic, _ = self.prediction(value, 0)
        return float(np.maximum(lower - semantic, 0) @ (1 / scales) + max(0., predicted - damage) / damagescale)

    def metadata(self):
        return {"rank": self.rank, "dimension": len(self.active), "condition": self.condition, "neighbors": self.indices.tolist(), "priorobserved": self.penalty is not None, "fiterror": float(np.linalg.norm(self.fiterror)), "innovation": float(np.linalg.norm(self.innovation)), "scope": "Paid finite responses; local proposal only."}

def geometry(model, points, radius):
    # Repair geometry along the weakest normalized direction.
    limits = np.array([2., 2., 1.])
    choices = []
    for sign in (-1, 1):
        direction = sign * model.weak / model.root
        length = radius
        for axis, change in zip(model.active, direction):
            if abs(change) > 1e-14:
                bound = (limits[axis] - model.anchor[axis]) / change if change > 0 else -model.anchor[axis] / change
                length = min(length, max(0., bound))
        if length <= 1e-12:
            continue
        point = model.anchor.copy()
        point[model.active] += length * direction
        distance = min(float(np.linalg.norm((point[model.active] - np.asarray(value)[model.active]) * model.root)) for value in points)
        if distance > 1e-10:
            choices.append((length, distance, tuple(point)))
    return max(choices, key=lambda row: (row[0], row[1], row[2]))[2] if choices else None

def propose(model, radius, lower, scales, damage, damagescale, cap, ridge, recovery=False, iterations=250, checkpoint=None):
    # Solve clipped-prior pieces before real candidate verification.
    count = len(model.active)
    limits = np.array([2., 2., 1.])[model.active]
    bounds = list(zip(-model.anchor[model.active], limits - model.anchor[model.active]))
    lower, scales = np.asarray(lower, dtype=float), np.asarray(scales, dtype=float)
    if lower.shape != model.scores.shape or scales.shape != lower.shape or not np.isfinite(lower).all() or not np.isfinite(scales).all() or np.any(scales <= 0) or not all(np.isfinite(value) and value > 0 for value in (radius, damagescale, ridge)) or not np.isfinite(damage) or damage < 0 or cap is not None and (not np.isfinite(cap) or cap < 0) or type(iterations) is not int or not 1 <= iterations <= 10000:
        raise ValueError("Local proposal controls are invalid.")

    def predicted(value):
        return model.damage + 2 * model.cross @ value + value @ model.gram @ value

    def derivative(value):
        return 2 * model.cross + 2 * model.gram @ value

    trust = {"type": "ineq", "fun": lambda value: radius ** 2 - value @ model.metric @ value, "jac": lambda value: -2 * model.metric @ value}

    def feasible(value, constraints):
        return np.isfinite(value).all() and all(float(np.min(np.atleast_1d(item["fun"](value)))) >= -1e-8 for item in constraints) and all(a - 1e-8 <= x <= b + 1e-8 for x, (a, b) in zip(value, bounds))

    if recovery:
        def objective(value):
            return model.restore(value, lower, scales, damage, damagescale) + ridge * (value @ model.metric @ value)

        def gradient(value):
            active = lower > model.scores + model.semantic @ value
            result = -(model.semantic[active] / scales[active, None]).sum(0) + 2 * ridge * model.metric @ value
            return result + derivative(value) / damagescale if predicted(value) > damage else result

        if checkpoint is not None:
            checkpoint()
        solved = minimize(objective, np.zeros(count), jac=gradient, bounds=bounds, constraints=[trust], method="SLSQP", callback=None if checkpoint is None else lambda value: checkpoint(), options={"ftol": 1e-12, "maxiter": iterations})
        if not feasible(solved.x, [trust]):
            return None
        point = model.anchor.copy()
        point[model.active] += solved.x
        return {"point": tuple(point), "move": solved.x, "predicted": model.restore(solved.x, lower, scales, damage, damagescale), "start": model.restore(np.zeros(count), lower, scales, damage, damagescale), "converged": bool(solved.success), "piece": "recovery", "solver": {"iterations": int(solved.nit), "reason": str(solved.message), "optimalitycertified": False}}
    if model.penalty is None:
        raise ValueError("Feasible proposals require observed anchor priors.")
    from engine.quadratic import solve
    zero, empty = np.zeros((count, count)), np.zeros(count)
    base = [(model.metric, empty, -radius ** 2), (model.gram, 2 * model.cross, model.damage - damage)]
    base.extend((zero, -row, requirement - observed) for row, requirement, observed in zip(model.semantic, lower, model.scores))
    options, pieces = [], []
    boxlower = model.penalty + sum(change * (a if change >= 0 else b) for change, (a, b) in zip(model.slope, bounds))
    boxupper = model.penalty + sum(change * (b if change >= 0 else a) for change, (a, b) in zip(model.slope, bounds))
    span = radius * np.linalg.norm(model.slope / model.root)
    priorlower, priorupper = max(boxlower, model.penalty - span), min(boxupper, model.penalty + span)
    for piece in (("lower", "linear") if cap is None else ("lower", "linear", "upper")):
        if piece == "lower" and priorlower > 0 or piece == "linear" and (priorupper < 0 or cap is not None and priorlower > cap) or piece == "upper" and priorupper < cap:
            pieces.append({"piece": piece, "status": "empty prior interval", "iterations": 0, "dualbound": None, "optimalitycertified": True})
            continue
        intercept, slope = (0., np.zeros(count)) if piece == "lower" else (cap, np.zeros(count)) if piece == "upper" else (model.penalty, model.slope)
        constraints = list(base)
        if piece == "lower" or piece == "linear" and cap is not None:
            boundary = 0. if piece == "lower" else cap
            constraints.append((zero, model.slope, model.penalty - boundary))
        if piece in {"upper", "linear"}:
            boundary = cap if piece == "upper" else 0.
            constraints.append((zero, -model.slope, boundary - model.penalty))
        try:
            solved = solve(model.gram + ridge * model.metric, 2 * model.cross + slope, model.damage + intercept, constraints, bounds, iterations, checkpoint)
        except Exhausted:
            raise
        except (np.linalg.LinAlgError, RuntimeError, FloatingPointError, OverflowError) as error:
            pieces.append({"piece": piece, "status": "unresolved numerical solve", "iterations": 0, "dualbound": None, "optimalitycertified": False, "reason": str(error)})
            continue
        details = {key: value for key, value in solved.items() if key != "point"}
        details.update({"piece": piece, "status": "verified candidate" if solved["feasible"] else "unresolved numerical solve"})
        pieces.append(details)
        if solved["point"] is not None:
            move = solved["point"]
            point = model.anchor.copy()
            point[model.active] += move
            estimated, _, penalty = model.prediction(move, cap)
            options.append({"point": tuple(point), "move": move, "predicted": estimated + penalty, "start": model.damage + model.penalty, "objective": solved["objective"], "converged": solved["converged"], "piece": piece})
    if not options:
        return None
    qualified = [row for row in options if row["converged"]] or options
    tolerance = max([row.get("tolerance", 0.) for row in pieces] + [1e-12])
    smallest = min(row["objective"] for row in qualified)
    tied = [row for row in qualified if row["objective"] - smallest <= min(tolerance, 1e-10 * max(1., abs(smallest)))]
    selected = min(tied, key=lambda row: (row["point"], row["piece"]))
    certified = all(row["status"] == "empty prior interval" or row["dualbound"] is not None and selected["objective"] - row["dualbound"] <= tolerance for row in pieces)
    selected["solver"] = {"pieces": pieces, "iterations": sum(row["iterations"] for row in pieces), "optimalitycertified": bool(certified), "tolerance": tolerance, "tiedpieces": len(tied), "ties": "Deterministic parameter and piece ordering.", "scope": "Enumerated local proxy; numerical bounds do not certify nonlinear image optimality."}
    return selected

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
