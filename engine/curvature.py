# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import math
import numpy as np

class Guard:
    def __init__(self, semantic=.05, response=.25, condition=1000000., growth=4., critical=1e-8, enabled=True):
        values = (semantic, response, condition, growth, critical)
        if any(isinstance(value, bool) or not math.isfinite(value) or value <= 0 for value in values) or condition <= 1 or growth <= 1 or type(enabled) is not bool:
            raise ValueError("Curvature controls must be finite and positive.")
        self.semantic, self.response, self.condition = semantic, response, condition
        self.growth, self.critical, self.enabled = growth, critical, enabled
        self.multiplier, self.alerts, self.pending = 1., 0, False
        self.last = None

    def amplify(self):
        self.multiplier = min(1000000., self.multiplier * self.growth)
        self.alerts += 1
        self.pending = True

    def inspect(self, model, scales, damage, base):
        # Diagnose finite response errors before proposing a query.
        scales = np.asarray(scales, dtype=float)
        if scales.shape != model.scores.shape or not np.isfinite(scales).all() or np.any(scales <= 0) or not math.isfinite(base) or base <= 0 or not math.isfinite(damage) or damage <= 0:
            raise ValueError("Curvature normalization requires positive scales.")
        size = len(model.residual)
        semantic = model.fiterror[size:] / scales[:, None]
        semanticerror = float(np.max(np.abs(semantic))) if semantic.size else 0.
        protection = model.fiterror[:size]
        responseerror = float(np.max(np.linalg.norm(protection, axis=0))) / max(math.sqrt(damage), 1e-12) if protection.size else 0.
        actual = model.observed[size:] / scales[:, None]
        predicted = model.oldprediction[size:] / scales[:, None]
        floor = self.semantic * .1
        flips = int(np.sum((actual * predicted < 0) & (np.abs(actual) > floor) & (np.abs(predicted) > floor)))
        slope = float(np.linalg.norm((model.semantic / scales[:, None]) / model.root[None]))
        flat = slope <= self.critical and actual.size > 0 and float(np.max(np.abs(actual))) > floor
        reasons = []
        if semanticerror > self.semantic:
            reasons.append("semantic residual")
        if responseerror > self.response:
            reasons.append("protection residual")
        if flips:
            reasons.append("observed sign reversal")
        if flat:
            reasons.append("flat model with finite variation")
        if self.enabled and reasons:
            self.amplify()
        normalized = model.gram / np.outer(model.root, model.root)
        eigenvalues = np.linalg.eigvalsh((normalized + normalized.T) / 2)
        smallest, largest = max(0., float(eigenvalues[0])), max(0., float(eigenvalues[-1]))
        required = max(0., (largest - self.condition * smallest) / (self.condition - 1))
        damping = max(base * self.multiplier, required * (1 + 1e-12)) if self.enabled else base
        self.last = {"enabled": self.enabled, "semanticerror": semanticerror, "responseerror": responseerror, "signreversals": flips, "semanticslope": slope, "flat": bool(flat), "reasons": reasons, "multiplier": self.multiplier, "damping": damping, "hessianmin": smallest, "hessianmax": largest, "condition": (largest + damping) / (smallest + damping), "fallback": bool(self.enabled and self.pending), "scope": "Empirical model validity, not a curvature or convergence certificate."}
        return self.last

    def observe(self, model, move, residual, scores, scales, damage):
        predicted, semantic, _ = model.prediction(move, 0)
        response = float(np.linalg.norm(np.asarray(residual) - model.residual - model.response @ move)) / max(math.sqrt(damage), 1e-12)
        error = float(np.max(np.abs((np.asarray(scores) - semantic) / np.asarray(scales))))
        alert = self.enabled and (error > self.semantic or response > self.response)
        if alert:
            self.amplify()
        return {"semanticerror": error, "responseerror": response, "alert": bool(alert), "multiplier": self.multiplier, "predicteddamage": predicted}

    def accepted(self):
        self.multiplier = max(1., self.multiplier / 2)
        self.pending = False

    def metadata(self):
        return {"enabled": self.enabled, "alerts": self.alerts, "multiplier": self.multiplier, "pending": self.pending, "semanticthreshold": self.semantic, "responsethreshold": self.response, "hessianlimit": self.condition, "critical": self.critical}

def fallback(model, points, radius, preferred=None):
    # Query signed directions without claiming a true gradient.
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Fallback radii must be positive.")
    existing = [np.asarray(point, dtype=float) for point in points]
    limits = np.array([2., 2., 1.])
    directions = []
    if preferred is not None:
        value = np.asarray(preferred, dtype=float) * model.root
        if value.shape != model.root.shape or not np.isfinite(value).all():
            raise ValueError("Fallback directions must match active coordinates.")
        if np.linalg.norm(value) > 1e-12:
            directions.extend((-value / np.linalg.norm(value), value / np.linalg.norm(value)))
    gradient = (2 * model.cross + model.slope) / model.root
    if np.linalg.norm(gradient) > 1e-12:
        directions.extend((-gradient / np.linalg.norm(gradient), gradient / np.linalg.norm(gradient)))
    directions.extend((model.weak, -model.weak))
    for axis in range(len(model.active)):
        direction = np.eye(len(model.active))[axis]
        directions.extend((direction, -direction))
    for factor in (1., .5):
        for direction in directions:
            value = direction / model.root
            length = radius * factor
            for axis, change in zip(model.active, value):
                if abs(change) > 1e-14:
                    boundary = (limits[axis] - model.anchor[axis]) / change if change > 0 else -model.anchor[axis] / change
                    length = min(length, max(0., boundary))
            if length <= 1e-12:
                continue
            point = model.anchor.copy()
            point[model.active] += length * value
            if all(np.linalg.norm((point[model.active] - other[model.active]) * model.root) > 1e-10 for other in existing):
                return tuple(point)
    return None

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
