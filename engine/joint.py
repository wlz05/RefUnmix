# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import torch
from engine.response import Protection, Selection
from engine.wavelet import check, split

@dataclass(frozen=True)
class Config:
    strength: float = 1.0
    levels: int = 9
    checks: int = 3
    tolerance: float = .02
    scoretol: float = .02
    gain: float = 0.0
    slack: float = 0.0
    budget: float = .0005
    interaction: str = "full"
    decision: str = "joint"

    def validate(self):
        for name in ("strength", "tolerance", "scoretol", "gain", "slack", "budget"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not 0 < self.strength <= 2:
            raise ValueError("strength must be in (0, 2].")
        if type(self.levels) is not int or not 3 <= self.levels <= 33 or self.levels % 2 != 1:
            raise ValueError("levels must be an odd integer from three to thirty-three.")
        if type(self.checks) is not int or not 0 <= self.checks <= 32:
            raise ValueError("checks must be an integer from zero to thirty-two.")
        if self.interaction not in {"full", "additive"} or self.decision not in {"joint", "scalar", "corners", "dense"}:
            raise ValueError("Invalid interaction or decision mode.")

def scores(value):
    if not isinstance(value, dict) or "score" not in value:
        raise ValueError("The scorer must return a score dictionary.")
    result = {key: float(value[key]) for key in ("score", "global", "local", "target") if key in value}
    if any(not math.isfinite(number) for number in result.values()):
        raise ValueError("Selection scores must be finite.")
    return result

def coefficients(values):
    if len(values) != 4:
        raise ValueError("Four ordered factorial observations are required.")
    zero, low, high, both = values
    return zero, low - zero, high - zero, both - low - high + zero

def basis(low, high, interaction="full"):
    if interaction not in {"full", "additive"} or any(not math.isfinite(value) or not 0 <= value <= 1 for value in (low, high)):
        raise ValueError("Invalid response coordinates.")
    return torch.tensor([1., low, high, low * high if interaction == "full" else 0.], dtype=torch.float64)

class Surface:
    def __init__(self, images, values, protection):
        if len(images) != 4 or len(values) != 4:
            raise ValueError("Four images and four score dictionaries are required.")
        images = [image.detach().float().cpu() for image in images]
        for image in images:
            protection.feature(image)
        self.pixels = coefficients(images)
        self.values = [scores(value) for value in values]
        keys = set(self.values[0])
        if any(set(value) != keys for value in self.values):
            raise ValueError("All score dictionaries must have matching keys.")
        self.metrics = {key: torch.tensor(coefficients([value[key] for value in self.values]), dtype=torch.float64) for key in sorted(keys)}
        self.protection = protection
        if protection.valid:
            features = coefficients([protection.feature(image) for image in images])
            matrix = torch.stack(features).double()
            self.gram = matrix @ matrix.T
        else:
            self.gram = None

    def predict(self, low, high, interaction="full"):
        vector = basis(low, high, interaction)
        metrics = {key: float(value @ vector) for key, value in self.metrics.items()}
        damage = max(0., float(vector @ self.gram @ vector)) if self.gram is not None else None
        return {"scores": metrics, "damage": damage}

    def image(self, low, high, interaction="full"):
        vector = basis(low, high, interaction)
        return sum(float(weight) * pixels for weight, pixels in zip(vector, self.pixels))

    def calibration(self, image, values, interaction="full"):
        predicted = self.predict(.5, .5, interaction)
        actual = scores(values)
        if set(actual) != set(predicted["scores"]):
            raise ValueError("Calibration scores must match corner scores.")
        return {"rgb": float((image.detach().float().cpu() - self.image(.5, .5, interaction)).square().mean().sqrt()), "score": max(abs(actual[key] - predicted["scores"][key]) for key in actual), "predicted": predicted, "actual": {"scores": actual, "damage": self.protection.energy(image)}}

    def report(self):
        zero, low, high, cross = self.pixels
        energy = lambda value: float(value.square().mean())
        result = {"lowenergy": energy(low), "highenergy": energy(high), "interactionenergy": energy(cross), "interactionratio": energy(cross) / max(energy(low) + energy(high), 1e-12), "scoreinteraction": {key: float(value[3]) for key, value in self.metrics.items()}, "gram": self.gram.tolist() if self.gram is not None else None}
        if self.gram is not None:
            matrix = self.gram
            overlap = float(2 * matrix[1, 2])
            nonlinear = float(2 * matrix[0, 3] + 2 * matrix[1, 3] + 2 * matrix[2, 3] + matrix[3, 3])
            result.update({"damageoverlap": overlap, "damagenonlinear": nonlinear, "damageinteraction": overlap + nonlinear})
        else:
            result.update({"damageoverlap": None, "damagenonlinear": None, "damageinteraction": None})
        return result

def eligible(values, damage, reference, ceiling, config):
    if values["score"] < reference["score"] + config.gain - 1e-10:
        return False
    if any(values[key] < reference[key] - config.slack - 1e-10 for key in ("global", "local", "target") if key in reference):
        return False
    return damage is not None and ceiling is not None and damage <= ceiling + 1e-10

@torch.no_grad()
def select(base, low, high, render, score, origin, mask, config=None, transport=None):
    # Measure interactions before joint proposals.
    config = config or Config()
    config.validate()
    check(base)
    for field in (low, high):
        check(field)
        if field.shape != base.shape or field.device != base.device:
            raise ValueError("Band fields must share the baseline shape and device.")
    if min(base.shape[-2:]) < 2 or any(size % 2 for size in base.shape[-2:]):
        raise ValueError("Joint editing requires an even Haar lattice.")
    if split(low)[1].norm() > 1e-5 * low.norm().clamp_min(1) or split(high)[0].norm() > 1e-5 * high.norm().clamp_min(1):
        raise ValueError("Band fields must lie in their declared Haar subspaces.")
    protection = Protection(origin, mask)
    cache, rows = [], []

    def observe(field, label, weights=None):
        for entry in cache:
            if torch.equal(field, entry["field"]):
                rows.append({"label": label, "weights": weights, "cached": True, "observation": entry["id"], "scores": entry["scores"], "damage": entry["damage"]})
                return entry
        pixels = render(field).detach().float().cpu()
        protection.feature(pixels)
        entry = {"id": len(cache), "field": field.detach().clone(), "image": pixels, "scores": scores(score(pixels)), "damage": protection.energy(pixels), "label": label, "weights": weights}
        cache.append(entry)
        rows.append({"label": label, "weights": weights, "cached": False, "observation": entry["id"], "scores": entry["scores"], "damage": entry["damage"]})
        return entry

    baseline = observe(base, "base")
    reference = baseline["scores"]
    ceiling = baseline["damage"] + config.budget if protection.valid else None
    report = {"schema": 4, "method": "joint", "config": asdict(config), "baseline": {"scores": reference, "damage": baseline["damage"]}, "ceiling": ceiling, "protectedvalid": protection.valid, "coverage": protection.coverage, "notheorem": "Only measured outputs are accepted; no global optimality or semantic guarantee is claimed."}
    if not protection.valid or float(mask.mean()) <= 1e-8:
        report.update({"selected": "base", "fallback": True, "reason": "No valid edit and protection support.", "weights": None, "gain": 0., "audit": protection.audit(baseline["image"]), "evaluations": len(cache), "observations": rows, "verification": [], "constraintmet": False})
        return Selection(base, baseline["image"].to(origin.device), report)
    coordinates = [(0., 0.), (1., 0.), (0., 1.), (1., 1.)]
    fields = [config.strength * (left * low + right * high) for left, right in coordinates]
    corners = [observe(field, name, [config.strength * left, config.strength * right]) for field, name, (left, right) in zip(fields, ("zero", "low", "high", "both"), coordinates)]
    if any(set(entry["scores"]) != set(reference) for entry in corners):
        raise ValueError("The scorer must preserve its metric keys.")
    surface = Surface([entry["image"] for entry in corners], [entry["scores"] for entry in corners], protection)
    midpoint = observe(config.strength * .5 * (low + high), "center", [config.strength * .5] * 2)
    calibration = surface.calibration(midpoint["image"], midpoint["scores"], config.interaction)
    trusted = calibration["rgb"] <= config.tolerance and calibration["score"] <= config.scoretol
    report.update({"interaction": surface.report(), "calibration": calibration, "trusted": trusted, "original": {"scores": scores(score(origin)), "damage": protection.energy(origin), "selectable": False}, "zeronote": "Zero transport still includes reconstruction and optional target refinement."})
    if transport is not None:
        before = [transport(field).detach().float().cpu() for field in fields]
        presurface = Surface(before, [score(image) for image in before], protection)
        report["stages"] = {"before": presurface.report(), "after": surface.report(), "zeroreconstruction": protection.energy(before[0]), "zerorefined": corners[0]["damage"], "renders": 4}
    else:
        report["stages"] = None
    chosen = baseline
    verified = []

    def consider(entry):
        nonlocal chosen
        if not eligible(entry["scores"], entry["damage"], reference, ceiling, config):
            return
        currentvalid = eligible(chosen["scores"], chosen["damage"], reference, ceiling, config)
        if not currentvalid or entry["damage"] < chosen["damage"] - 1e-10 or abs(entry["damage"] - chosen["damage"]) <= 1e-10 and entry["scores"]["score"] > chosen["scores"]["score"] + 1e-10:
            chosen = entry

    allowed = [baseline, corners[0], corners[3], midpoint] if config.decision == "scalar" else [baseline, *corners, midpoint]
    for entry in allowed:
        consider(entry)
    candidates = []
    if config.decision != "corners":
        for left in range(config.levels):
            for right in range(config.levels):
                if config.decision == "scalar" and left != right:
                    continue
                a, b = left / (config.levels - 1), right / (config.levels - 1)
                field = config.strength * (a * low + b * high)
                if any(torch.equal(field, entry["field"]) for entry in cache):
                    continue
                predicted = surface.predict(a, b, config.interaction)
                if config.decision == "dense" or trusted and eligible(predicted["scores"], predicted["damage"], reference, ceiling, config):
                    candidates.append((predicted, a, b))
        candidates.sort(key=lambda item: (item[0]["damage"], -item[0]["scores"]["score"], item[1], item[2]))
    maximum = len(candidates) if config.decision == "dense" else config.checks
    for predicted, a, b in candidates[:maximum]:
        entry = observe(config.strength * (a * low + b * high), "verify", [config.strength * a, config.strength * b])
        accepted = eligible(entry["scores"], entry["damage"], reference, ceiling, config)
        verified.append({"weights": [config.strength * a, config.strength * b], "predicted": predicted, "actual": {"scores": entry["scores"], "damage": entry["damage"]}, "feasible": accepted})
        consider(entry)
    report.update({"selected": chosen["label"], "fallback": chosen is baseline, "weights": chosen["weights"], "gain": chosen["scores"]["score"] - reference["score"], "damagechange": chosen["damage"] - baseline["damage"], "scores": chosen["scores"], "audit": protection.audit(chosen["image"]), "evaluations": len(cache), "observations": rows, "verification": verified, "constraintmet": eligible(chosen["scores"], chosen["damage"], reference, ceiling, config), "proposed": len(candidates)})
    return Selection(chosen["field"], chosen["image"].to(origin.device), report)

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
