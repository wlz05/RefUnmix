# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from engine.frequency import FrequencyRouter

@dataclass(frozen=True)
class FeedbackConfig:
    rounds: int = 1
    step: float = 0.25
    budget: float = 0.01
    penalty: float = 0.05
    local: float = 0.5
    tolerance: float = 1e-4
    slack: float = 0.01

    def validate(self) -> None:
        if isinstance(self.rounds, bool) or not isinstance(self.rounds, Integral) or not 0 <= self.rounds <= 3:
            raise ValueError("rounds must be an integer from zero to three.")
        values = (self.step, self.budget, self.penalty, self.local, self.tolerance, self.slack)
        if any(isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or value < 0 for value in values):
            raise ValueError("Feedback values must be finite and nonnegative.")
        if self.step == 0 or self.local > 1 or self.budget > 1:
            raise ValueError("step must be positive and local and budget must not exceed one.")

def options(parser) -> None:
    parser.add_argument("--clip", default=None)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--budget", type=float, default=0.01)
    parser.add_argument("--penalty", type=float, default=0.05)
    parser.add_argument("--local", type=float, default=0.5)
    parser.add_argument("--localtext", default=None)

def settings(args) -> FeedbackConfig:
    config = FeedbackConfig(rounds=args.rounds, step=args.step, budget=args.budget, penalty=args.penalty, local=args.local)
    config.validate()
    return config

def pixels(image: Image.Image) -> torch.Tensor:
    values = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0)

def average(value: torch.Tensor, weight: torch.Tensor) -> float:
    return float((value * weight).sum() / (weight.sum() * value.shape[1]).clamp_min(1e-8))

def correlate(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    def pool(value):
        return functional.avg_pool2d(functional.pad(value, (3, 3, 3, 3), mode="replicate"), 7, 1)

    first, second = pool(left), pool(right)
    covariance = pool(left * right) - first * second
    variance = (pool(left.square()) - first.square()).clamp_min(0) * (pool(right.square()) - second.square()).clamp_min(0)
    return (covariance / variance.sqrt().clamp_min(1e-6)).clamp(-1, 1), (variance > 1e-12).float()

def inspect(source: Image.Image, result: Image.Image, mask: torch.Tensor) -> dict[str, float]:
    # Compare frequencies within protected regions.
    if source.size != result.size:
        raise ValueError("Audit images must share one size.")
    original, edited = pixels(source), pixels(result)
    edit = functional.interpolate(mask.detach().float().cpu(), original.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)
    protect = 1.0 - edit
    interior = 1.0 - functional.max_pool2d(edit, 9, 1, 4)
    router = FrequencyRouter()
    low, nextlow = router.low(original), router.low(edited)
    high, nexthigh = original - low, edited - nextlow

    def residual(left, right):
        norm = lambda value: value.square().mean(dim=1, keepdim=True).sqrt()
        return norm(left - right) / (norm(left) + norm(right)).clamp_min(1e-4)

    first, second = router.activity(original), router.edge(original)
    nextfirst, nextsecond = router.activity(edited), router.edge(edited)
    reference, valid = correlate(first, second)
    forward, validforward = correlate(nextfirst, second)
    backward, validbackward = correlate(first, nextsecond)
    crossmask = interior * valid * validforward * validbackward
    return {"hh": average(residual(nexthigh, high), interior), "ll": average(residual(nextlow, low), interior), "hl": average(0.5 * (forward - reference).abs(), crossmask), "lh": average(0.5 * (backward - reference).abs(), crossmask), "edithh": average(residual(nexthigh, high), edit), "editll": average(residual(nextlow, low), edit), "damage": average((edited - original).square(), protect), "protectarea": float(protect.mean()), "auditarea": float(interior.mean()), "crossarea": float(crossmask.mean())}

class FeedbackRouter:
    def __init__(self, config: FeedbackConfig, ceiling: float = 1.5) -> None:
        config.validate()
        if not isfinite(ceiling) or ceiling <= 0:
            raise ValueError("ceiling must be positive and finite.")
        self.config = config
        self.ceiling = ceiling

    def run(self, render, anchor: torch.Tensor, source: Image.Image, maps: dict[str, torch.Tensor], judge):
        mask = maps["mask"]
        initial = (float(maps["lowweight"].mean()), float(maps["highweight"].mean()))
        selected = (float(maps["lowtime"].mean()), float(maps["hightime"].mean()))
        early, late = float(maps["earlytime"].mean()), float(maps["latetime"].mean())
        pairs = {"chosen": (maps["lowfield"], maps["highfield"], selected)}
        for first in ("early", "late"):
            for second in ("early", "late"):
                pairs[first + second] = (maps[first + "low"], maps[second + "high"], (early if first == "early" else late, early if second == "early" else late))
        history = []
        cache = {}
        interactions = []
        best = {"image": source, "latent": anchor, "field": torch.zeros_like(anchor), "score": 0.0, "semantic": 0.0, "pair": "chosen", "weights": initial, "times": selected, "name": "source"}

        def evaluate(label, field, pair, weights):
            nonlocal best
            identity = (pairs[pair][2], tuple(round(value, 7) for value in weights)) if label != "base" else ("base",)
            if identity in cache:
                return cache[identity]
            latent, image = render(field)
            semantic = judge(image)
            audit = inspect(source, image, mask)
            gain = (1.0 - self.config.local) * semantic["global"] + self.config.local * semantic["local"] + 0.05 * semantic["direction"]
            cost = sum(audit[key] for key in ("hh", "hl", "lh", "ll")) / 4.0
            score = gain - self.config.penalty * cost
            valid = all(isfinite(float(value)) for value in (*semantic.values(), *audit.values(), gain, score))
            feasible = valid and audit["damage"] <= self.config.budget and semantic["global"] >= -self.config.slack and semantic["local"] >= -self.config.slack
            accepted = feasible and gain > self.config.tolerance and score > best["score"] + self.config.tolerance and gain >= best["semantic"] - self.config.slack
            row = {"name": label, "pair": pair if label != "base" else None, "weights": list(weights) if label != "base" else None, "times": list(pairs[pair][2]) if label != "base" else None, **semantic, **audit, "semantic": gain, "score": score, "feasible": feasible, "accepted": accepted}
            row = {key: (None if isinstance(value, float) and not isfinite(value) else value) for key, value in row.items()}
            history.append(row)
            cache[identity] = row
            if accepted:
                best = {"image": image, "latent": latent, "field": field, "score": score, "semantic": gain, "pair": pair, "weights": weights, "times": pairs[pair][2], "name": label}
            return row

        evaluate("base", maps["base"], "chosen", initial)
        for pair, (low, high, _) in pairs.items():
            evaluate(pair, initial[0] * low + initial[1] * high, pair, initial)
        for iteration in range(self.config.rounds):
            pair, center = best["pair"], best["weights"]
            low, high, _ = pairs[pair]
            step = self.config.step / (2 ** iteration)
            centerrow = evaluate("center", center[0] * low + center[1] * high, pair, center)
            responses = {}
            for first, second in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                weights = (min(self.ceiling, max(0.0, center[0] + first * step)), min(self.ceiling, max(0.0, center[1] + second * step)))
                responses[first, second] = evaluate("correction", weights[0] * low + weights[1] * high, pair, weights)
            if center[0] + step <= self.ceiling and center[1] + step <= self.ceiling:
                points = (responses[1, 1], responses[1, 0], responses[0, 1], centerrow)
                if all(point["semantic"] is not None for point in points):
                    interactions.append({"round": iteration, "pair": pair, "semantic": (points[0]["semantic"] - points[1]["semantic"] - points[2]["semantic"] + points[3]["semantic"]) / step ** 2, "damage": (points[0]["damage"] - points[1]["damage"] - points[2]["damage"] + points[3]["damage"]) / step ** 2})
        report = {"selected": best["name"], "accepted": best["name"] != "source", "evaluations": len(history), "weights": list(best["weights"]) if best["name"] not in {"source", "base"} else None, "times": list(best["times"]) if best["name"] not in {"source", "base"} else None, "budget": self.config.budget, "mask": "provided" if bool(maps["explicit"].item()) else "estimated", "history": history, "interaction": interactions, "audit": inspect(source, best["image"], mask)}
        report["baseline"] = {key: history[0][key] for key in ("semantic", "score", "damage", "feasible")}
        report["improvement"] = best["score"] - history[0]["score"] if history[0]["score"] is not None else None
        return best["latent"], best["image"], best["field"], report

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
