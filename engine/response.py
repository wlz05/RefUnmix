# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import math
import time
import numpy as np
import torch
import torch.nn.functional as functional
from engine.wavelet import check, dictionary, sources, split

@dataclass(frozen=True)
class Config:
    grid: int = 2
    step: float = 0.25
    radius: float = 0.75
    sparse: float = 0.001
    ridge: float = 0.001
    budget: float = 0.0005
    active: int = 4
    backtrack: int = 3
    coupling: str = "full"
    local: bool = True
    globalstage: bool = True
    bands: str = "both"
    probes: int = 0
    trust: str = "fixed"
    maps: int = 0

    def validate(self) -> None:
        for name in ("step", "radius", "sparse", "ridge", "budget"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not 0 < self.step <= min(1, self.radius) or self.radius > 8:
            raise ValueError("step must be in (0, min(1, radius)] and radius cannot exceed eight.")
        for name in ("grid", "active", "backtrack", "probes", "maps"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name in {"backtrack", "probes", "maps"} else 1):
                raise ValueError(f"Invalid {name}.")
        if self.coupling not in {"full", "diagonal"} or self.bands not in {"both", "low", "high"}:
            raise ValueError("Invalid coupling or band selection.")
        if type(self.local) is not bool or type(self.globalstage) is not bool:
            raise ValueError("Stage flags must be boolean.")
        if self.trust not in {"fixed", "curvature"} or self.maps > 64:
            raise ValueError("Invalid trust policy or influence map resolution.")

@dataclass
class Selection:
    field: torch.Tensor
    image: torch.Tensor
    report: dict

def region(base: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    # Freeze one edit support before evaluating any candidate.
    check(base)
    if mask is not None:
        check(mask)
        if mask.shape[1] != 1 or torch.any((mask < 0) | (mask > 1)).item():
            raise ValueError("Edit masks must have one channel and values in [0, 1].")
        return functional.interpolate(mask.to(device=base.device, dtype=torch.float32), base.shape[-2:], mode="area")
    activity = base.float().square().mean(dim=1, keepdim=True).sqrt()
    active = activity > 1e-8
    if not active.any().item():
        return torch.zeros_like(activity)
    scale = torch.quantile(activity[active], 0.90)
    return functional.avg_pool2d(functional.pad((activity / scale).clamp(0, 1), (1, 1, 1, 1), mode="replicate"), 3, 1)

class Protection:
    def __init__(self, origin: torch.Tensor, mask: torch.Tensor) -> None:
        check(origin)
        check(mask)
        if mask.shape[1] != 1 or torch.any((mask < 0) | (mask > 1)).item():
            raise ValueError("Protection masks must have one channel in [0, 1].")
        if origin.shape[1] != 3 or torch.any((origin < 0) | (origin > 1)).item():
            raise ValueError("Origin must contain RGB pixels in [0, 1].")
        self.origin = origin.detach().float().cpu()
        self.weight = 1 - functional.interpolate(mask.detach().float().cpu(), origin.shape[-2:], mode="bilinear", align_corners=False)
        mass = float(self.weight.sum())
        self.coverage = mass / self.weight.numel()
        self.valid = self.coverage >= 0.01 and mass >= 1.0
        self.factor = (self.weight / (mass * 3)).sqrt() if self.valid else torch.zeros_like(self.weight)

    def feature(self, image: torch.Tensor) -> torch.Tensor:
        check(image)
        if image.shape != self.origin.shape or torch.any((image < 0) | (image > 1)).item():
            raise ValueError("Rendered images must match origin RGB geometry and range.")
        return ((image.detach().float().cpu() - self.origin) * self.factor).flatten() if self.valid else torch.empty(0)

    def energy(self, image: torch.Tensor) -> float | None:
        return float(self.feature(image).square().sum()) if self.valid else None

    def audit(self, image: torch.Tensor) -> dict:
        if not self.valid:
            return {"detail": None, "cross": None, "coarse": None, "mse": None, "coverage": self.coverage}
        low, high = split(image.detach().float().cpu() - self.origin)
        low, high = low * self.factor, high * self.factor
        return {"detail": float(high.square().sum()), "cross": float(2 * (high * low).sum()), "coarse": float(low.square().sum()), "mse": self.energy(image), "coverage": self.coverage}

def solve(gradient: np.ndarray, matrix: np.ndarray, linear: np.ndarray, initial: float, ceiling: float, groups: list[str], config: Config) -> np.ndarray:
    # Search feasible rays under a joint budget.
    config.validate()
    gradient = np.asarray(gradient, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    linear = np.asarray(linear, dtype=np.float64)
    count = gradient.size
    if gradient.shape != (count,) or matrix.shape != (count, count) or linear.shape != (count,) or len(groups) != count:
        raise ValueError("Response system dimensions disagree.")
    if any(not np.isfinite(value).all() for value in (gradient, matrix, linear)) or not math.isfinite(initial) or not math.isfinite(ceiling):
        raise ValueError("Response systems must be finite.")
    if not np.allclose(matrix, matrix.T, atol=1e-9) or (count and np.linalg.eigvalsh(matrix).min() < -1e-7):
        raise ValueError("Response matrices must be symmetric positive semidefinite.")
    if ceiling < initial - 1e-9:
        raise ValueError("The response reference must be feasible.")
    weights = np.zeros(count)
    if not count:
        return weights
    membership = np.array([[group == key for group in groups] for key in sorted(set(groups))], dtype=np.float64)
    rays = [value for value in np.eye(count)]
    for left in range(count):
        for right in range(left + 1, count):
            for ratio in (0.25, 0.5, 0.75):
                value = np.zeros(count)
                value[left], value[right] = ratio, 1 - ratio
                rays.append(value)
    rays = np.stack(rays)
    groupsteps = rays @ membership.T
    curvature = np.maximum(np.einsum("ij,ij->i", rays @ matrix, rays), 0)
    lengths = (rays * rays).sum(axis=1)
    for _ in range(32):
        used = weights > 1e-10
        supports = (rays > 0) | used[None]
        upper = np.full(len(rays), max(0.0, config.radius - weights.sum()))
        remaining = 1 - membership @ weights
        for group in range(len(remaining)):
            room = np.divide(max(0.0, remaining[group]), groupsteps[:, group], out=np.full(len(rays), np.inf), where=groupsteps[:, group] > 0)
            upper = np.minimum(upper, room)
        upper[supports.sum(axis=1) > config.active] = 0
        intercept = min(0.0, initial + 2 * linear @ weights + weights @ matrix @ weights - ceiling)
        slope = 2 * rays @ (linear + matrix @ weights)
        discriminant = np.maximum(slope * slope - 4 * curvature * intercept, 0)
        safe = np.full(len(rays), np.inf)
        curved = curvature > 1e-14
        safe[curved] = (-slope[curved] + np.sqrt(discriminant[curved])) / (2 * curvature[curved])
        positive = curved & (slope >= 0) & (discriminant > 0)
        safe[positive] = -2 * intercept / (slope[positive] + np.sqrt(discriminant[positive]))
        straight = (~curved) & (slope > 1e-14)
        safe[straight] = -intercept / slope[straight]
        upper = np.maximum(0, np.minimum(upper, safe))
        ascent = rays @ (gradient - config.sparse - 2 * config.ridge * weights)
        optimum = np.maximum(ascent, 0) / (2 * config.ridge * lengths) if config.ridge > 0 else np.where(ascent > 0, upper, 0)
        steps = np.minimum(upper, optimum)
        gains = steps * ascent - config.ridge * steps * steps * lengths
        index = int(np.argmax(gains))
        if gains[index] <= 1e-12:
            break
        weights += steps[index] * rays[index]
    return weights

@torch.no_grad()
def select(anchor: torch.Tensor, base: torch.Tensor, pool: dict[float, torch.Tensor], render, score, origin: torch.Tensor, mask: torch.Tensor | None = None, config: Config | None = None, windows: list[torch.Tensor] | None = None) -> Selection:
    config = config or Config()
    config.validate()
    check(anchor)
    entries = sources(base, pool)
    if anchor.shape != base.shape or anchor.device != base.device:
        raise ValueError("Anchor and baseline must share shape and device.")
    support = region(base, mask)
    protect = Protection(origin, support)
    started = time.perf_counter()
    history = []
    evaluations = 0

    def evaluate(field, label):
        nonlocal evaluations
        image = render(field)
        feature = protect.feature(image)
        values = score(image)
        values = dict(values) if isinstance(values, dict) else {"score": float(values)}
        if "score" not in values or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values.values()):
            raise ValueError("Semantic scores must be finite numeric values including score.")
        values = {key: float(value) for key, value in values.items()}
        damage = float(feature.square().sum()) if protect.valid else None
        evaluations += 1
        record = {"candidate": label, "metrics": values, "damage": damage}
        history.append(record)
        return image.detach(), feature, values, record

    baseline, basefeature, basescore, baserecord = evaluate(base, "base")
    ceiling = (float(basefeature.square().sum()) if protect.valid else 0) + config.budget

    def feasible(record):
        return record["damage"] is None or record["damage"] <= ceiling + 1e-10

    baserecord["feasible"] = True if protect.valid else None
    bestfield, bestimage, bestscore = base.float(), baseline, basescore["score"]
    bestname, bestpair = "base", ["base", "base"]
    seen = [base.float()]
    if config.globalstage:
        parts = {key: split(value) for key, value in entries.items()}
        lows = list(parts) if config.bands in {"both", "low"} else ["base"]
        highs = list(parts) if config.bands in {"both", "high"} else ["base"]
        for low in lows:
            for high in highs:
                field = parts[low][0] + parts[high][1]
                if any(torch.allclose(field, other, rtol=0, atol=1e-8) for other in seen):
                    continue
                seen.append(field)
                image, _, values, record = evaluate(field, f"global:{low}:{high}")
                record["feasible"] = feasible(record) if protect.valid else None
                if feasible(record) and values["score"] > bestscore + 1e-10:
                    bestfield, bestimage, bestscore = field, image, values["score"]
                    bestname, bestpair = record["candidate"], [low, high]
    globalfield, globalimage, globalscore = bestfield, bestimage, bestscore
    globalname = bestname
    globalfeature = protect.feature(globalimage)
    globaldamage = float(globalfeature.square().sum())
    report = {"schema": 3, "method": "response", "auditkind": "protected RGB Haar residual energy", "config": asdict(config), "times": sorted(float(value) for value in pool), "mask": "provided" if mask is not None else "inferred", "protectionvalid": protect.valid, "ceiling": ceiling if protect.valid else None, "baseline": dict(baserecord), "global": {"candidate": globalname, "pair": bestpair, "score": globalscore}, "history": history, "atoms": [], "weights": [], "proposed": [], "responses": {}, "backtracking": []}
    bestweights = np.zeros(0)
    atoms = dictionary(base, globalfield, pool, config.grid, config.bands, windows) if config.local else []
    from engine.locality import shortlist
    atoms, report["shortlist"] = shortlist(atoms, support, config.probes)
    if len(atoms) > 128:
        raise ValueError("The pair solver supports at most 128 probes; reduce regions or set --probes.")
    if atoms:
        gradients, responses, probes = [], [], []
        for index, atom in enumerate(atoms):
            plusimage, plusfeature, plusvalues, plusrecord = evaluate(globalfield + config.step * atom.value, f"probe:{index}:plus")
            minusimage, minusfeature, minusvalues, minusrecord = evaluate(globalfield - config.step * atom.value, f"probe:{index}:minus")
            gradients.append((plusvalues["score"] - minusvalues["score"]) / (2 * config.step))
            responses.append((plusfeature - minusfeature) / (2 * config.step))
            plusrecord["feasible"] = feasible(plusrecord) if protect.valid else None
            minusrecord["feasible"] = feasible(minusrecord) if protect.valid else None
            probes.append({"scorecurve": (plusvalues["score"] + minusvalues["score"] - 2 * globalscore) / config.step ** 2, "responsecurve": float(((plusfeature + minusfeature - 2 * globalfeature) / config.step ** 2).norm())})
            if config.maps:
                influence = (plusimage.detach().float().cpu() - minusimage.detach().float().cpu()).norm(dim=1, keepdim=True) / (2 * config.step)
                energy = influence.square()
                probes[-1]["influence"] = functional.adaptive_avg_pool2d(influence, (min(config.maps, influence.shape[-2]), min(config.maps, influence.shape[-1])))[0, 0].tolist()
                probes[-1]["leakage"] = float((energy * protect.weight).sum() / energy.sum().clamp_min(1e-12))
            if feasible(plusrecord) and plusvalues["score"] > bestscore + 1e-10:
                bestfield, bestimage, bestscore = globalfield + config.step * atom.value, plusimage, plusvalues["score"]
                bestname = plusrecord["candidate"]
                bestweights = np.zeros(len(atoms))
                bestweights[index] = config.step
        response = torch.stack(responses).double()
        full = (response @ response.T).numpy()
        matrix = full if config.coupling == "full" else np.diag(np.diag(full))
        linear = (response @ globalfeature.double()).numpy()
        gradient = np.array(gradients)
        effective = config
        if config.trust == "curvature":
            ratios = [config.step * probe["responsecurve"] / max(float(value.norm()), 1e-8) for probe, value in zip(probes, responses)]
            radius = max(config.step, config.radius / math.sqrt(1 + max(ratios, default=0.)))
            effective = replace(config, radius=radius)
        report["trust"] = {"mode": config.trust, "radius": effective.radius, "note": "Probe curvature is a heuristic, not a certified remainder bound."}
        proposed = solve(gradient, matrix, linear, globaldamage, ceiling, [atom.group for atom in atoms], effective)
        report["atoms"] = [atom.metadata() for atom in atoms]
        report["proposed"] = proposed.tolist()
        report["responses"] = {"gradient": gradient.tolist(), "matrix": matrix.tolist(), "linear": linear.tolist(), "curvature": probes, "reference": globaldamage, "fullmatrix": full.tolist() if config.coupling != "full" else None}
        if bestweights.size == 0:
            bestweights = np.zeros(len(atoms))
        for iteration in range(config.backtrack + 1):
            weights = proposed * 0.5 ** iteration
            if not np.any(weights > 1e-10):
                break
            field = globalfield + sum((float(weight) * atom.value for weight, atom in zip(weights, atoms)), torch.zeros_like(globalfield))
            image, _, values, record = evaluate(field, f"joint:{iteration}")
            record["feasible"] = feasible(record) if protect.valid else None
            predicted = globaldamage + 2 * linear @ weights + weights @ matrix @ weights
            trace = {"factor": 0.5 ** iteration, "predicted": float(predicted) if protect.valid else None, "actual": record["damage"], "score": values["score"], "feasible": record["feasible"]}
            report["backtracking"].append(trace)
            if feasible(record) and values["score"] > bestscore + 1e-10:
                bestfield, bestimage, bestscore, bestname, bestweights = field, image, values["score"], record["candidate"], weights.copy()
    report.update({"selected": bestname, "score": bestscore, "gain": bestscore - basescore["score"], "localgain": bestscore - globalscore, "weights": bestweights.tolist(), "active": int((bestweights > 1e-10).sum()), "evaluations": evaluations, "seconds": time.perf_counter() - started, "audit": protect.audit(bestimage), "fallback": bestname == globalname, "identity": False})
    return Selection(bestfield.detach(), bestimage.detach(), report)

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
