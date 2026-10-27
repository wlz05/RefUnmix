# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from dataclasses import asdict, dataclass, field
import math
import re
import numpy as np
import torch
import torch.nn.functional as functional
from engine.limit import Exhausted
from engine.session import tensor

@dataclass(frozen=True)
class Config:
    candidates: int = 12
    step: float = .5
    minimum: float = 1 / 16
    slack: float = .02
    priorslack: float = 0.
    semanticscale: float = 1.
    damagescale: float = 1.
    priorscale: float = 1.
    axes: str = "all"
    eta: float = 0.
    slacks: dict = field(default_factory=dict)
    scales: dict = field(default_factory=dict)
    algorithm: str = "trust"
    priorcontrol: str = "bounded"
    priorweight: float = .1
    secant: float = .0001
    ridge: float = .001
    condition: float = 1000000.
    curvature: float = .05
    responseerror: float = .25
    hessian: float = 1000000.
    damping: float = 4.
    critical: float = 1e-8
    monitor: bool = True
    solversteps: int = 250
    certificatestep: int = 150
    certificate: bool = True
    radiusprofile: dict = field(default_factory=dict)
    compensation: bool = False
    release: bool = True
    geometry: str = "output"
    ranktol: float = 1e-6
    frontier: bool = False
    frontiersteps: int = 150
    selection: str = "attain"
    gainloss: float = 0.
    goals: tuple = ()
    archive: bool = False

    def validate(self):
        if type(self.candidates) is not int or not 1 <= self.candidates <= 10000:
            raise ValueError("Candidate counts must be positive and bounded.")
        for key in ("step", "minimum", "semanticscale", "damagescale", "priorscale", "secant", "ridge", "condition", "curvature", "responseerror", "hessian", "damping", "critical"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Search steps and scales must be positive.")
        if self.hessian <= 1 or self.damping <= 1 or type(self.monitor) is not bool:
            raise ValueError("Damping and Hessian controls are invalid.")
        if isinstance(self.ranktol, bool) or not isinstance(self.ranktol, (int, float)) or not math.isfinite(self.ranktol) or not 0 < self.ranktol < 1:
            raise ValueError("Response rank tolerances must be in (0, 1).")
        if any(type(value) is not int or not 1 <= value <= 10000 for value in (self.solversteps, self.certificatestep, self.frontiersteps)) or type(self.certificate) is not bool or type(self.compensation) is not bool or type(self.frontier) is not bool:
            raise ValueError("Local solver and certificate controls are invalid.")
        if self.radiusprofile:
            from engine.radius import profile
            profile(self.radiusprofile)
        elif not isinstance(self.radiusprofile, dict):
            raise ValueError("Radius profiles must be mappings.")
        if isinstance(self.eta, bool) or not isinstance(self.eta, (int, float)) or not math.isfinite(self.eta) or self.minimum > self.step or self.step > 2 or not 0 <= self.eta <= 1:
            raise ValueError("Search steps or reference coefficients are invalid.")
        if self.axes not in {"all", "field", "path", "none"}:
            raise ValueError("Unknown active control coordinates.")
        if self.algorithm not in {"trust", "poll", "random"} or self.priorcontrol not in {"bounded", "hard", "soft"}:
            raise ValueError("Unknown search or prior ablation.")
        if self.selection not in {"attain", "protect"} or isinstance(self.gainloss, bool) or not isinstance(self.gainloss, (int, float)) or not math.isfinite(self.gainloss) or self.gainloss < 0:
            raise ValueError("Attainment selection controls are invalid.")
        if self.geometry not in {"output", "latent", "attenuate"}:
            raise ValueError("Unknown compensation geometry.")
        if self.geometry != "output" and (not self.release or self.algorithm != "trust" or self.selection != "attain"):
            raise ValueError("Geometry ablations require attainment release queries.")
        if type(self.archive) is not bool or type(self.release) is not bool:
            raise ValueError("Candidate archival must be an explicit Boolean.")
        if not isinstance(self.goals, (tuple, list)) or any(not isinstance(key, str) or not key or not key.isascii() or key.startswith("retain") for key in self.goals) or len(set(self.goals)) != len(self.goals):
            raise ValueError("Editing goals must be unique English non-retention names.")
        if self.compensation and self.algorithm != "trust":
            raise ValueError("Compensation probes require trust response models.")
        if self.frontier and self.algorithm != "trust":
            raise ValueError("Frontier diagnostics require trust response models.")
        if isinstance(self.priorweight, bool) or not isinstance(self.priorweight, (int, float)) or not math.isfinite(self.priorweight) or not 0 < self.priorweight < 1:
            raise ValueError("Prior weights must be strictly between zero and one.")
        for value in (self.slack, self.priorslack):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Proxy slacks must be finite and nonnegative.")
        for mapping, positive in ((self.slacks, False), (self.scales, True)):
            if not isinstance(mapping, dict):
                raise ValueError("Component controls must be named mappings.")
            for key, value in mapping.items():
                if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or positive and value == 0:
                    raise ValueError("Component slacks and scales are invalid.")

def feedback(value):
    if not isinstance(value, dict):
        raise ValueError("Control proxies must return named components.")
    keys = [key for key in value if key in {"global", "target"} or re.fullmatch(r"(?:local|attribute|retain)\d+", key)]
    if not keys:
        keys = ["score"] if "score" in value else []
    if not keys:
        raise ValueError("At least one control component is required.")
    result = {}
    for key in keys:
        if isinstance(value[key], bool) or not isinstance(value[key], (float, int)) or not math.isfinite(value[key]):
            raise ValueError("Control components must be finite numbers.")
        result[key] = float(value[key])
    return result

def protection(origin, weight):
    tensor(origin, 3)
    if weight is None:
        raise ValueError("Declare fixed protection weights before searching.")
    tensor(weight, 1)
    if torch.any((weight < 0) | (weight > 1)).item():
        raise ValueError("Protection weights must be in [0, 1].")
    fixed = functional.interpolate(weight.detach().to(origin.device).double(), origin.shape[-2:], mode="bilinear", align_corners=False).clone()
    mass = float(fixed.sum())

    def damage(image):
        tensor(image, 3)
        if image.shape != origin.shape:
            raise ValueError("Candidate pixels must match source geometry.")
        return float(((image.double() - origin.double()).square() * fixed).sum()) / mass if mass > 0 else None

    return damage, mass

@dataclass
class Candidate:
    key: str
    point: tuple | None
    latent: torch.Tensor
    image: torch.Tensor
    scores: dict
    damage: float | None
    prior: float | None = None
    feasible: bool = False
    complete: bool = False
    violation: float = 0.
    reasons: tuple = ()
    penalty: float | None = None
    objective: float | None = None
    taskgain: float = 0.

    def row(self):
        return {"key": self.key, "point": list(self.point) if self.point is not None else None, "scores": self.scores, "damage": self.damage, "prior": self.prior, "penalty": self.penalty, "objective": self.objective, "taskgain": self.taskgain, "feasible": self.feasible, "complete": self.complete, "violation": self.violation, "reasons": list(self.reasons)}

@dataclass
class Result:
    image: torch.Tensor
    latent: torch.Tensor
    report: dict

def select(path, score, weight, prior, config=None):
    # Propose from paid responses and verify real outputs.
    from engine.trust import Model, geometry, propose
    from engine.curvature import Guard, fallback
    from engine.certificate import identify
    from engine.radius import initialize
    from engine.leverage import analyze, probe
    from engine.attainment import gain, choose, propose as attained
    from engine.release import propose as released
    config = config or Config()
    config.validate()
    session = path.session
    damage, mass = protection(session.origin, weight)
    if mass <= 0:
        damage = lambda image: 0.
    rawlatent, rawimage = path.raw()
    raw = Candidate("raw", None, rawlatent, rawimage, feedback(score(rawimage)), damage(rawimage), feasible=True, complete=True)
    if config.archive:
        session.artifacts = getattr(session, "artifacts", {})
        session.artifacts["raw"] = rawimage.detach().cpu().clone()
    if (config.slacks.keys() | config.scales.keys() | set(config.goals)) - raw.scores.keys():
        raise ValueError("Component overrides must match frozen score names.")
    names = list(raw.scores)
    goals = list(config.goals) if config.goals else [key for key in names if not key.startswith("retain")]
    if not goals:
        raise ValueError("Declare at least one editing goal distinct from retention.")
    indices = [names.index(key) for key in goals]
    reference = np.array([raw.scores[key] for key in names])
    lower = np.array([raw.scores[key] - config.slacks.get(key, config.slack) for key in names])
    scales = np.array([config.scales.get(key, config.semanticscale) for key in names])
    candidates, history, visited = [], [], set()
    stopped = "candidate limit"
    if raw.damage == 0 and config.selection == "protect":
        return Result(rawimage, rawlatent, {"selected": "raw", "fallback": True, "protectedvalid": mass > 0, "reason": "No reducible protected damage.", "scores": raw.scores, "damage": raw.damage, "rawdamage": raw.damage, "evaluations": 1, "candidates": [], "constraintmet": True, "prior": {"queried": False}})
    requested = (0, 1, 2) if config.axes == "all" else (0, 1) if config.axes == "field" else (2,) if config.axes == "path" else ()
    try:
        raw.prior = prior(rawlatent)
        raw.penalty, raw.objective = 0., raw.damage
        references = (0., 1.) if 2 in path.axes(requested) else (config.eta,)
        path.initialize(references) if hasattr(path, "cost") else path.initialize()
    except Exhausted as error:
        return Result(rawimage, rawlatent, {"selected": "raw", "fallback": True, "protectedvalid": mass > 0, "reason": str(error), "scores": raw.scores, "damage": raw.damage, "rawdamage": raw.damage, "evaluations": 1, "candidates": [], "constraintmet": True, "prior": prior.metadata(), "initialization": {"evaluated": 0, "complete": False}, "budget": session.limit.report() if session.limit is not None else None})
    cap = config.priorweight * raw.damage

    def costs(point):
        samples, calls = path.cost(point) if hasattr(path, "cost") else (0, 0)
        extra = prior.cost() if hasattr(prior, "cost") else (0, 0)
        return samples + extra[0], calls + extra[1]

    def affordable(point, reserve=False):
        limit = session.limit
        if limit is None or not hasattr(path, "cost"):
            return True
        samples, calls = costs(point)
        if reserve and (samples > 0 or calls > 0):
            extra = prior.cost() if hasattr(prior, "cost") else (0, 0)
            samples += 2 * (path.native.cost(0.) + extra[0])
            calls += 2 * (path.native.queries() + extra[1])
        return limit.used + samples <= limit.maximum and (limit.calls is None or limit.forwards + calls <= limit.calls)

    def complete(candidate):
        candidate.prior = prior(candidate.latent)
        excess = 0. if raw.prior is None else max(0., candidate.prior - raw.prior)
        if config.priorcontrol == "hard":
            if excess > config.priorslack:
                candidate.reasons += ("prior",)
                candidate.violation += (excess - config.priorslack) / config.priorscale
            candidate.penalty = 0.
        elif config.priorcontrol == "soft":
            candidate.penalty = cap * excess / config.priorscale
        else:
            fraction = 1 / (1 + config.priorscale / excess) if excess > 0 else 0.
            candidate.penalty = min(np.nextafter(cap, 0.), cap * fraction) if cap > 0 else 0.
        candidate.objective = candidate.damage + candidate.penalty
        candidate.feasible = not candidate.reasons
        candidate.complete = True

    def visit(point):
        if len(point) != 3 or any(not math.isfinite(float(value)) for value in point):
            raise ValueError("Proposed controls must contain three finite coordinates.")
        point = path.canonical(tuple(min(limit, max(0., float(value))) for value, limit in zip(point, (2., 2., 1.))))
        if point in visited or len(candidates) >= config.candidates:
            return None
        if session.limit is not None:
            session.limit.require(*costs(point))
        visited.add(point)
        latent, pixels = path.candidate(point)
        scores = feedback(score(pixels))
        if scores.keys() != raw.scores.keys():
            raise ValueError("Control components must remain fixed across candidates.")
        current = Candidate(str(len(candidates)), point, latent, pixels, scores, damage(pixels))
        current.taskgain = gain([scores[key] for key in names], reference, scales, indices)
        reasons = []
        for index, key in enumerate(names):
            excess = lower[index] - scores[key]
            current.violation += max(0., excess) / scales[index]
            if excess > 0:
                reasons.append("semantic:" + key)
        excess = current.damage - raw.damage
        current.violation += max(0., excess) / config.damagescale
        if excess > 0:
            reasons.append("protection")
        current.reasons = tuple(reasons)
        candidates.append(current)
        if config.archive:
            session.artifacts["candidate" + current.key] = pixels.detach().cpu().clone()
        if not reasons:
            complete(current)
        return current

    def center():
        feasible = [candidate for candidate in candidates if candidate.feasible]
        if feasible:
            if config.selection == "attain":
                best = max(row.taskgain for row in feasible)
                band = [row for row in feasible if row.taskgain >= best - config.gainloss]
                return min(band, key=lambda row: (row.objective, int(row.key)))
            return min(feasible, key=lambda row: (row.objective, int(row.key)))
        return min(candidates, key=lambda row: (row.violation, int(row.key))) if candidates else None

    def better(candidate, anchor):
        if not anchor.feasible:
            return bool(candidate.feasible or candidate.violation < anchor.violation)
        if not candidate.feasible:
            return False
        if config.selection == "attain":
            if candidate.taskgain > anchor.taskgain + config.gainloss:
                return True
            if candidate.taskgain < max(0., anchor.taskgain - config.gainloss):
                return False
        return bool(candidate.objective < anchor.objective)

    active = path.axes(requested)
    continuous = tuple(axis for axis in active if axis != 2 or path.native.continuous)
    initial = (0., .5, 1.) if 2 in active and path.native.continuous else (0., 1.) if 2 in active else (config.eta,)
    radius, radiusinfo = initialize(config.radiusprofile, config.step, path.native, path.mode if hasattr(path, "mode") else "latent", continuous)
    previous, latentprevious = None, None
    guard = Guard(config.curvature, config.responseerror, config.hessian, config.damping, config.critical, config.monitor)
    lastmove = None
    initialleverage = None
    rescuetrace = {"considered": 0, "eligible": 0, "proposed": 0, "budgetblocked": 0, "skipped": {}}
    generator = np.random.default_rng(path.seed ^ 0xC91)
    fixed = functional.interpolate(weight.detach().to(session.origin.device).double(), session.origin.shape[-2:], mode="bilinear", align_corners=False)
    root = (fixed / mass).sqrt() if mass > 0 else torch.zeros_like(fixed)
    residuals = {}

    def residual(candidate):
        if candidate.key not in residuals:
            residuals[candidate.key] = ((candidate.image.double() - session.origin.double()) * root).detach().cpu().numpy().reshape(-1)
        return residuals[candidate.key]

    middle = .5 if 2 in continuous else initial[0]
    plan = [(1., 1., middle)]
    if 2 in continuous:
        plan.append((1., 1., 0.))
    for axis in continuous:
        if axis != 2:
            point = [1., 1., middle]
            point[axis] += radius
            plan.append(tuple(point))
    required = len(plan)
    plan.extend((1., 1., eta) for eta in initial if eta != middle)
    for axis in continuous:
        if axis != 2:
            point = [1., 1., middle]
            point[axis] -= radius
            plan.append(tuple(point))
    initialization = {"planned": len(set(path.canonical(point) for point in plan)), "required": required, "evaluated": 0, "skipped": 0, "complete": False, "scope": "Independent directions precede optional symmetric probes; reserve two endpoint queries when affordable."}
    initializing = True
    try:
        for index, point in enumerate(plan):
            if session.limit is not None:
                session.limit.require()
            canonical = path.canonical(point)
            if canonical in visited:
                continue
            optional = index >= required
            reserved = config.candidates >= required + 2 and len(candidates) >= config.candidates - 2
            if not affordable(canonical, optional) or optional and reserved:
                initialization["skipped"] += 1
                continue
            visit(point)
        initialization["evaluated"] = len(candidates)
        initialization["complete"] = all(path.canonical(point) in visited for point in plan[:required])
        initializing = False
        anchor = center()
        while len(candidates) < config.candidates and radius >= config.minimum and continuous:
            if anchor is None:
                stopped = "no visited exploration anchor"
                break
            if config.algorithm == "random":
                point = list(anchor.point)
                for axis in active:
                    point[axis] = float(generator.uniform(0, 1 if axis == 2 else 2)) if axis != 2 or path.native.continuous else float(generator.integers(0, 2))
                current = visit(point)
                history.append({"kind": "random", "point": point, "visited": current is not None})
                anchor = center()
                continue
            if config.algorithm == "poll":
                initialanchor = anchor
                for axis in continuous:
                    for sign in (-1, 1):
                        point = list(initialanchor.point)
                        point[axis] += sign * radius
                        visit(point)
                anchor = center()
                improves = better(anchor, initialanchor)
                history.append({"kind": "poll", "radius": radius, "center": list(initialanchor.point), "next": list(anchor.point), "improves": improves})
                if not improves:
                    radius *= .5
                continue
            observed = [row for row in candidates if all(abs(row.point[axis] - anchor.point[axis]) < 1e-10 for axis in range(3) if axis not in continuous)]
            model = Model(anchor.point, [row.point for row in observed], [residual(row) for row in observed], [[row.scores[key] for key in names] for row in observed], [row.penalty for row in observed], continuous, config.secant, previous)
            previous = model.fitted
            if config.geometry == "latent":
                from engine.geometry import response as latentfit
                model.latentresponse, model.latentmetadata = latentfit(model, [row.point for row in observed], [row.latent.detach().double().cpu().numpy().reshape(-1) for row in observed], config.secant, latentprevious)
                latentprevious = model.latentresponse
            checkpoint = session.limit.require if session.limit is not None else None
            if checkpoint is not None:
                checkpoint()
            certificate = identify(model.response, model.residual, model.semantic, model.metric, lower - model.scores, radius, raw.damage, config.certificatestep, checkpoint) if config.certificate else {"status": "disabled", "infeasible": False}
            monitoring = guard.inspect(model, scales, max(raw.damage, 1e-8), config.ridge * max(raw.damage, 1e-8))
            repair = model.rank < len(continuous) or model.condition is None or model.condition > config.condition
            try:
                leverage = analyze(model.response, model.semantic, model.residual, model.metric, model.active, names, scales, config.damagescale, config.ranktol)
                leverage.update({"geometryresolved": not repair, "fittrusted": not repair and not monitoring["reasons"], "enabled": config.compensation})
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                leverage = {"active": False, "geometryresolved": False, "fittrusted": False, "enabled": config.compensation, "reason": str(error)}
            if initialleverage is None:
                initialleverage = {**leverage, "center": list(anchor.point), "stage": "first fitted model after frozen initial probes"}
                if config.frontier:
                    initialleverage["frontier"] = {"active": leverage.get("active", False), "fittrusted": leverage.get("fittrusted", False), "status": "Inactive or untrusted initial response model.", "components": {}}
                    if leverage.get("active") and leverage.get("fittrusted"):
                        from engine.frontier import analyze as capacity
                        try:
                            initialleverage["frontier"] = capacity(model, lower, scales, raw.damage, radius, names, iterations=config.frontiersteps, checkpoint=checkpoint)
                            from engine.attainment import capacity as balanced
                            initialleverage["frontier"]["balanced"] = balanced(model, radius, lower, raw.damage, reference, scales, indices, iterations=config.frontiersteps, checkpoint=checkpoint)
                            initialleverage["frontier"]["iterations"] += initialleverage["frontier"]["balanced"]["iterations"]
                        except Exhausted as error:
                            initialleverage["frontier"]["status"] = "Diagnostic interrupted by the shared time budget: " + str(error)
                            raise
                        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                            initialleverage["frontier"]["status"] = "Unverified diagnostic: " + str(error)
            rescue = not repair and monitoring["fallback"]
            if repair or rescue:
                proposed = None
            elif config.selection == "attain" and anchor.feasible:
                proposed = released(model, radius, lower, raw.damage, reference, scales, indices, monitoring["damping"], config.gainloss, config.ranktol, checkpoint, config.geometry) if config.release and leverage.get("fittrusted") else None
                if proposed is None or path.canonical(proposed["point"]) in visited:
                    proposed = attained(model, radius, lower, scales, raw.damage, config.damagescale, None if config.priorcontrol == "soft" else cap, monitoring["damping"], reference, indices, config.gainloss, config.solversteps, checkpoint)
            else:
                proposed = propose(model, radius, lower, scales, raw.damage, config.damagescale, None if config.priorcontrol == "soft" else cap, monitoring["damping"], not anchor.feasible, config.solversteps, checkpoint)
            if config.monitor and not repair and not rescue and (proposed is None or not proposed["converged"] or float(np.linalg.norm(proposed["move"] * model.root)) <= config.critical or path.canonical(proposed["point"]) in visited):
                guard.amplify()
                rescue = True
            compensated = None
            if config.compensation:
                rescuetrace["considered"] += 1
                reason = "unresolved geometry" if repair else "regular proposal available" if not rescue else "untrusted fit" if not leverage.get("fittrusted") else None
                if reason is None:
                    rescuetrace["eligible"] += 1
                    compensated = probe(model, leverage, [row.point for row in observed], radius, lower, scales, raw.damage, config.damagescale)
                    if compensated is None:
                        reason = "no improving unvisited direction"
                    else:
                        rescuetrace["proposed"] += 1
                if reason is not None:
                    rescuetrace["skipped"][reason] = rescuetrace["skipped"].get(reason, 0) + 1
            point = geometry(model, [row.point for row in observed], radius) if repair else compensated if compensated is not None else fallback(model, [row.point for row in observed], radius, lastmove) if rescue else proposed["point"] if proposed is not None else None
            if point is None:
                history.append({"kind": "model", "radius": radius, "model": model.metadata(), "certificate": certificate, "curvature": monitoring, "leverage": leverage, "reason": "No unvisited safeguarded proposal." if rescue else "No valid local proposal."})
                radius *= .5
                continue
            budgetfallback = False
            if not affordable(point):
                if compensated is not None:
                    rescuetrace["budgetblocked"] += 1
                # Reuse one cached endpoint when a joint query cannot fit.
                fieldpoint, referencepoint = list(point), list(anchor.point)
                fieldpoint[2], referencepoint[2] = anchor.point[2], point[2]
                alternatives = [fieldpoint, referencepoint]
                for axis in continuous:
                    for sign in (-1, 1):
                        trial = list(anchor.point)
                        trial[axis] = min(1. if axis == 2 else 2., max(0., trial[axis] + sign * radius / model.root[list(model.active).index(axis)]))
                        alternatives.append(trial)
                point = next((path.canonical(value) for value in alternatives if path.canonical(value) not in visited and affordable(path.canonical(value))), None)
                if point is None:
                    stopped = "No unvisited candidate fits the remaining model budget."
                    history.append({"kind": "budget", "radius": radius, "reason": stopped, "queried": False})
                    break
                budgetfallback, rescue, compensated, proposed = True, not repair, None, None
            current = visit(point)
            record = {"kind": "geometry" if repair else "compensation" if compensated is not None else "fallback" if rescue else "release" if proposed is not None and proposed["solver"].get("operator") == "release" else "trust", "radius": radius, "center": list(anchor.point), "point": list(point), "model": model.metadata(), "certificate": certificate, "curvature": monitoring, "leverage": leverage}
            record["candidate"] = current.key if current is not None else None
            record["budgetfallback"] = budgetfallback
            if proposed is not None:
                record["solver"] = proposed["solver"]
            if config.geometry == "latent":
                record["latentgeometry"] = model.latentmetadata
            if current is None:
                record["reason"] = "Duplicate effective candidate."
                radius *= .5
            elif repair:
                accepted = better(current, anchor)
                if accepted:
                    anchor = current
                record["accepted"] = accepted
                record["reason"] = "Geometry query retained as a real response."
            else:
                move = np.asarray(current.point)[model.active] - np.asarray(anchor.point)[model.active]
                lastmove = move
                predicted, semantic, penalty = model.prediction(move, None if config.priorcontrol == "soft" else cap)
                observederror = guard.observe(model, move, residual(current), [current.scores[key] for key in names], scales, max(raw.damage, 1e-8))
                # Compare predicted and observed changes before acceptance.
                record["verification"] = {"predictedgain": gain(semantic, reference, scales, indices) - anchor.taskgain, "actualgain": current.taskgain - anchor.taskgain, "predicteddamagechange": predicted - anchor.damage, "actualdamagechange": current.damage - anchor.damage, "feasible": current.feasible}
                record.update({"predicted": predicted, "actual": current.damage, "responseerror": float(np.linalg.norm(residual(current) - model.residual - model.response @ move)), "semanticerror": float(np.linalg.norm(np.array([current.scores[key] for key in names]) - semantic)), "observation": observederror, "piece": "cached endpoint poll" if budgetfallback else "compensated reference poll" if compensated is not None else "signed response poll" if rescue else proposed["piece"], "converged": None if rescue else proposed["converged"]})
                if rescue:
                    accepted = better(current, anchor)
                    record.update({"accepted": accepted, "ratio": None, "reason": "Real response acceptance without a reliable model ratio."})
                    if accepted:
                        anchor = current
                        lastmove = None
                        guard.accepted()
                    else:
                        radius *= .5
                elif anchor.feasible:
                    decrease = proposed["start"] - proposed["predicted"]
                    gainphase = proposed.get("phase") == "attainment"
                    actualdecrease = current.taskgain - anchor.taskgain if gainphase else anchor.objective - current.objective
                    ratio = actualdecrease / decrease if decrease > 1e-14 and current.feasible else None
                    accepted = bool(ratio is not None and ratio >= .1 and better(current, anchor))
                    record["phase"] = proposed.get("phase", "protection")
                    record.update({"ratio": ratio, "predicteddecrease": decrease, "accepted": accepted})
                    if accepted:
                        anchor = current
                        lastmove = None
                        guard.accepted()
                        if ratio >= .75 and float(np.linalg.norm(move * model.root)) >= .8 * radius:
                            radius = min(1., 2 * radius)
                    else:
                        if config.monitor:
                            if not observederror["alert"]:
                                guard.amplify()
                        else:
                            radius *= .5
                else:
                    accepted = bool(current.violation < anchor.violation)
                    record.update({"accepted": accepted, "actualviolation": current.violation, "predictedviolation": proposed["predicted"]})
                    if accepted:
                        anchor = current
                        lastmove = None
                        guard.accepted()
                    else:
                        if config.monitor:
                            if not observederror["alert"]:
                                guard.amplify()
                        else:
                            radius *= .5
            history.append(record)
        if not continuous:
            stopped = "fixed or ineffective control coordinates"
        elif radius < config.minimum:
            stopped = "minimum trust radius; empirical stagnation, not certified convergence"
    except Exhausted as error:
        stopped = str(error)
        if initializing:
            initialization["evaluated"] = len(candidates)
    feasible = [raw] + [candidate for candidate in candidates if candidate.feasible]
    selected, attainment = choose(feasible, config.gainloss, config.selection)
    report = {"selected": selected.key, "point": list(selected.point) if selected.point else None, "fallback": selected is raw, "protectedvalid": mass > 0, "constraintmet": True, "reason": stopped, "scores": selected.scores, "rawscores": raw.scores, "damage": selected.damage, "rawdamage": raw.damage, "damagechange": selected.damage - raw.damage, "penalty": selected.penalty, "objective": selected.objective, "maximumpriorbias": cap if config.priorcontrol == "bounded" else None, "gain": sum(selected.scores.values()) - sum(raw.scores.values()), "evaluations": len(candidates) + 1, "candidates": [candidate.row() for candidate in candidates], "raw": raw.row(), "history": history, "active": list(continuous), "config": asdict(config), "prior": prior.metadata(), "path": path.metadata(), "backend": path.native.metadata(), "budget": session.limit.report() if session.limit is not None else None, "scope": "Minimum actual bounded-prior objective among evaluated feasible candidates."}
    report["initialization"] = initialization
    report["protection"] = {"active": mass > 0, "mass": mass, "scope": "Empty protection imposes zero damage and preserves semantic search."}
    report["curvature"] = guard.metadata()
    report["release"] = {"enabled": config.release and config.algorithm == "trust" and config.selection == "attain", "geometry": config.geometry, "queries": sum(row.get("kind") == "release" for row in history), "accepted": sum(row.get("kind") == "release" and row.get("accepted") is True for row in history), "scope": "Compensation proposals share protected-output verification and attainment selection."}
    report["rescue"] = {**rescuetrace, "enabled": config.compensation, "queries": sum(row.get("kind") == "compensation" for row in history), "accepted": sum(row.get("kind") == "compensation" and row.get("accepted") is True for row in history), "selected": any(row.get("kind") == "compensation" and row.get("candidate") == selected.key for row in history), "scope": "Optional rescue probes are separate from the main release operator."}
    from engine.activity import prior as prioractivity
    report["prioractivity"] = prioractivity(raw, candidates, selected, config, prior.metadata(), getattr(prior, "mode", "off" if raw.prior is None else "unknown"))
    report["position"] = {"task": "Editing attainability under a fixed output protection budget", "method": "Frozen-generator inference-time reference-field control", "reference": "RefUnmix calibration is separate from the native raw editor", "coordinates": "Frequency amplitudes parameterize controls, not semantic categories", "cost": "Single-step backbones do not imply a single-query complete editor"}
    report["initialleverage"] = initialleverage
    report["radiusprofile"] = radiusinfo
    report["attainment"] = {**attainment, "goals": goals, "scales": {key: float(scales[names.index(key)]) for key in goals}, "retention": [key for key in names if key.startswith("retain")], "zero": "No weighted averaging of editing and retention requirements."}
    changes = {key: (selected.scores[key] - raw.scores[key]) / scales[names.index(key)] for key in goals}
    report["attainment"].update({"changes": changes, "bottlenecks": [key for key, value in changes.items() if value <= selected.taskgain + 1e-10]})
    report["commonpool"] = {mode: {"selected": chosen.key, **detail} for mode in ("attain", "protect") for chosen, detail in [choose(feasible, config.gainloss, mode)]}
    report["candidatearchive"] = {"enabled": config.archive, "images": len(candidates) + 1 if config.archive else 0, "scope": "Same-pool selector comparison; artifact labels never feed back into search."}
    report["scope"] = "Minimum actual bounded cost within the measured attainment band." if config.selection == "attain" else "Minimum actual bounded cost among evaluated feasible candidates."
    report["termination"] = {"reason": stopped, "radius": radius, "stationaritycertified": False, "globaloptimalitycertified": False}
    return Result(selected.image, selected.latent, report)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
