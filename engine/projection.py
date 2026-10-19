# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import math
import numpy as np
import torch
from engine.response import Protection, Selection
from engine.secant import complete, defect
from engine.wavelet import check, gates, split

@dataclass(frozen=True)
class Config:
    grid: int = 2
    step: float = .25
    radius: float = .75
    ridge: float = .001
    backtrack: int = 3
    slack: float = 0.
    minreduce: float = 0.
    ranktol: float = 1e-6
    lock: str = "cone"
    coupling: str = "full"
    basis: str = "frequency"
    calibration: str = "secant"
    updates: int = 2
    defect: float = .1
    factorial: bool = False

    def validate(self):
        for key in ("step", "radius", "ridge", "slack", "minreduce", "ranktol", "defect"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative.")
        if not 0 < self.step <= 1 or self.radius > 4 or self.ridge <= 0 or not 0 < self.ranktol < 1:
            raise ValueError("Invalid finite difference, radius, ridge, or rank tolerance.")
        if type(self.grid) is not int or not 1 <= self.grid <= 8 or type(self.backtrack) is not int or not 0 <= self.backtrack <= 16:
            raise ValueError("Invalid spatial grid or backtracking count.")
        if self.lock not in {"cone", "semantic", "none"} or self.coupling not in {"full", "diagonal"} or self.basis not in {"frequency", "channel"}:
            raise ValueError("Invalid semantic lock or response coupling.")
        if self.calibration not in {"fixed", "shrink", "secant"} or type(self.updates) is not int or not 0 <= self.updates <= 8 or type(self.factorial) is not bool:
            raise ValueError("Invalid response calibration or diagnostic setting.")

def build(base, pool=None, grid=2, basis="frequency"):
    # Confine directions to aligned spatial blocks.
    check(base)
    if any(size % 2 for size in base.shape[-2:]):
        raise ValueError("Projection requires an even Haar lattice.")
    if basis not in {"frequency", "channel"} or basis == "channel" and base.shape[1] < 2:
        raise ValueError("Channel control requires at least two latent channels.")
    windows = gates(base, grid)
    if pool is not None and not isinstance(pool, dict):
        raise ValueError("Donor fields must be supplied as a dictionary.")
    donors = [("base", base)]
    for key, field in (pool or {}).items():
        check(field)
        if field.shape != base.shape or field.device != base.device:
            raise ValueError("Donor fields must match the baseline shape and device.")
        donors.append((str(key), field - base))
    scale = float(base.double().square().mean().sqrt())
    report = {"scale": scale, "grid": grid, "kind": basis, "atoms": [], "discarded": [], "metric": "Latent RMS relative to the baseline field."}
    if scale <= 1e-10:
        return [], report
    if len(donors) * len(windows) * 2 > 64:
        raise ValueError("The requested dictionary exceeds sixty-four raw directions.")
    if basis == "frequency":
        divided = [(key, split(field)) for key, field in donors]
        names = ("low", "high")
    else:
        channel = (torch.arange(base.shape[1], device=base.device) < base.shape[1] // 2).float()[None, :, None, None]
        divided = [(key, (field.float() * channel, field.float() * (1 - channel))) for key, field in donors]
        names = ("first", "second")
    output = []
    for region, window in enumerate(windows):
        for band, name in enumerate(names):
            local = []
            for key, fields in divided:
                value = (window * fields[band]).double()
                amplitude = float(value.square().mean().sqrt())
                if amplitude <= max(1e-10, scale * 1e-7):
                    report["discarded"].append({"region": region, "band": name, "donor": key, "reason": "empty"})
                    continue
                for _ in range(2):
                    for previous in local:
                        value -= (value * previous).mean() * previous
                norm = float(value.square().mean().sqrt())
                if norm <= amplitude * 1e-6:
                    report["discarded"].append({"region": region, "band": name, "donor": key, "reason": "dependent"})
                    continue
                unit = value / norm
                local.append(unit)
                output.append((unit * scale).float())
                report["atoms"].append({"region": region, "band": name, "donor": key, "inputrms": amplitude, "remainingrms": norm, "normalization": scale / norm})
    return output, report

def solve(response, residual, semantics, config=None, metric=None):
    # Solve semantic constraints within the physical radius.
    config = config or Config()
    config.validate()
    response, residual, semantics = (np.asarray(value, dtype=np.float64) for value in (response, residual, semantics))
    if response.ndim != 2 or residual.shape != (response.shape[0],) or semantics.ndim != 2 or semantics.shape[1] != response.shape[1]:
        raise ValueError("Response, residual, and semantic dimensions disagree.")
    if any(not np.isfinite(value).all() for value in (response, residual, semantics)):
        raise ValueError("Projection systems must contain finite values.")
    count = response.shape[1]
    metric = np.eye(count) if metric is None else np.asarray(metric, dtype=np.float64)
    if metric.shape != (count, count) or not np.isfinite(metric).all() or not np.allclose(metric, metric.T, rtol=1e-8, atol=1e-10):
        raise ValueError("The field metric must be finite and symmetric.")
    initial = float(residual @ residual)
    if not math.isfinite(initial):
        raise ValueError("Residual energy exceeds numerical range.")
    if count:
        try:
            transform = np.linalg.solve(np.linalg.cholesky(metric).T, np.eye(count))
        except np.linalg.LinAlgError as error:
            raise ValueError("The field metric must be positive definite.") from error
    else:
        transform = np.empty((0, 0))
    tangent = semantics @ transform
    rownorms = np.linalg.norm(tangent, axis=1)
    active = rownorms > 1e-10
    tangent = tangent[active] / rownorms[active, None]
    if config.lock == "cone" and tangent.shape[0] > 8:
        raise ValueError("Cone solving supports at most eight active semantic rows.")

    def nullspace(rows):
        if not rows.size:
            return np.eye(count), np.empty(0), 0
        _, singular, right = np.linalg.svd(rows, full_matrices=True)
        rank = int(np.count_nonzero(singular > config.ranktol * singular[0]))
        return right[rank:].T, singular, rank

    fullnull, singular, semanticrank = nullspace(tangent)
    gram = response.T @ response
    fitted = gram if config.coupling == "full" else np.diag(np.diag(gram))
    whitened = transform.T @ fitted @ transform
    gradient = transform.T @ (response.T @ residual)
    weights, bestvalue = np.zeros(count), initial
    best = {"rank": semanticrank if config.lock != "none" else 0, "nullity": fullnull.shape[1] if config.lock != "none" else count, "dual": 0., "condition": None, "active": np.flatnonzero(active).tolist() if config.lock != "none" else [], "descent": 0.}
    tested, feasible = 0, 0
    if config.lock == "cone":
        faces = range(1 << tangent.shape[0])
    elif config.lock == "semantic":
        faces = [(1 << tangent.shape[0]) - 1]
    else:
        faces = [0]
    if config.radius > 0 and count:
        for face in faces:
            selected = [index for index in range(tangent.shape[0]) if face & (1 << index)]
            null, _, rank = nullspace(tangent[selected])
            tested += 1
            if not null.shape[1]:
                continue
            curvature = null.T @ whitened @ null
            linear = null.T @ gradient
            eigenvalues, eigenvectors = np.linalg.eigh((curvature + curvature.T) * .5)
            eigenvalues = np.maximum(eigenvalues, 0.)
            projected = eigenvectors.T @ linear

            def coordinates(value):
                return -eigenvectors @ (projected / (eigenvalues + config.ridge + value))

            coordinate, dual = coordinates(0.), 0.
            if np.linalg.norm(coordinate) > config.radius:
                lower, upper = 0., max(config.ridge, float(np.linalg.norm(linear)) / config.radius)
                for _ in range(80):
                    middle = (lower + upper) * .5
                    if np.linalg.norm(coordinates(middle)) > config.radius:
                        lower = middle
                    else:
                        upper = middle
                dual = upper
                coordinate = coordinates(dual)
            whitenedweight = null @ coordinate
            if config.lock == "cone" and np.any(tangent @ whitenedweight < -1e-9 * max(1., config.radius)):
                continue
            feasible += 1
            value = initial + float(2 * gradient @ whitenedweight + whitenedweight @ whitened @ whitenedweight + config.ridge * whitenedweight @ whitenedweight)
            if value < bestvalue:
                weights, bestvalue = transform @ whitenedweight, value
                best = {"rank": rank, "nullity": null.shape[1], "dual": dual, "condition": float((eigenvalues[-1] + config.ridge + dual) / (eigenvalues[0] + config.ridge + dual)), "active": np.flatnonzero(active)[selected].tolist(), "descent": float(np.linalg.norm(linear))}
    predicted = float(np.square(residual + response @ weights).sum())
    norm = math.sqrt(max(0., float(weights @ metric @ weights)))
    semanticchange = semantics @ weights
    lockerror = float(np.max(np.abs(semanticchange))) if semantics.shape[0] else 0.
    violation = max(0., -float(semanticchange.min())) if semanticchange.size else 0.
    if config.lock == "cone":
        lockerror = violation
    modelvalue = initial + float(2 * weights @ (response.T @ residual) + weights @ fitted @ weights + config.ridge * norm ** 2)
    report = {**best, "semanticrank": semanticrank, "initial": initial, "predicted": predicted, "objective": predicted + config.ridge * norm ** 2, "surrogate": modelvalue, "modeldecrease": initial - modelvalue, "norm": norm, "lock": config.lock, "lockerror": lockerror, "violation": violation, "semanticchange": semanticchange.tolist(), "singular": singular.tolist(), "inactive": np.flatnonzero(~active).tolist(), "coupling": config.coupling, "faces": tested, "feasiblefaces": feasible}
    if not np.isfinite(weights).all() or any(not math.isfinite(value) for value in (predicted, norm, lockerror, modelvalue)):
        raise RuntimeError("Projection produced nonfinite coefficients.")
    return weights, report

def measured(value):
    if not isinstance(value, dict) or "score" not in value:
        raise ValueError("The semantic scorer must return a score dictionary.")
    result = {}
    for key, number in value.items():
        if key in {"score", "global", "local", "target"} or key.startswith("local") and key[5:].isdigit():
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
                raise ValueError("Semantic responses must be finite numbers.")
            result[key] = float(number)
    return result

def factorial(base, vectors, atoms, weights, render, protection, image, output, transport=None, score=None, baseline=None, current=None, slack=0., minreduce=0.):
    # Measure mixed effects with matched interventions.
    names = sorted(set(atom["band"] for atom in atoms))
    if len(names) != 2:
        return {"measured": False, "reason": "Both basis families are required.", "evaluations": 0}
    changes = [sum((float(weight) * vector for weight, vector, atom in zip(weights, vectors, atoms) if atom["band"] == name), torch.zeros_like(base)) for name in names]
    if any(float(change.double().square().mean()) <= 1e-24 for change in changes):
        return {"measured": False, "reason": "Both interventions must be nonzero.", "evaluations": 0}
    first, second = (render(base + change).detach().float().cpu() for change in changes)
    reference = protection.feature(image).double().numpy()
    response = [protection.feature(value).double().numpy() for value in (first, second, output)]
    interaction = response[2] - response[0] - response[1] + reference
    pixels = output - first - second + image
    report = {"measured": True, "groups": names, "rgb": float(pixels.double().square().mean().sqrt()), "protected": float(np.linalg.norm(interaction)), "relative": float(np.linalg.norm(interaction)) / max(float(np.linalg.norm(response[2] - reference)), 1e-8), "evaluations": 2, "decodes": 2}
    left, right = response[0] - reference, response[1] - reference
    overlap = float(2 * left @ right)
    nonlinear = float(2 * (reference + left + right) @ interaction + interaction @ interaction)
    difference = float(response[2] @ response[2] - response[0] @ response[0] - response[1] @ response[1] + reference @ reference)
    report.update({"overlap": overlap, "nonlinear": nonlinear, "damageinteraction": difference, "identityerror": abs(difference - overlap - nonlinear)})
    if score is not None:
        scorecalls = 2 + int(baseline is None) + int(current is None)
        baseline = measured(score(image)) if baseline is None else measured(baseline)
        current = measured(score(output)) if current is None else measured(current)
        singles = [measured(score(value)) for value in (first, second)]
        if any(set(value) != set(baseline) for value in [current, *singles]):
            raise ValueError("Factorial semantic keys must remain fixed.")
        semantic = {key: current[key] - singles[0][key] - singles[1][key] + baseline[key] for key in baseline}
        damage = float(reference @ reference)
        interventions = []
        for label, feature, scores in zip([*names, "joint"], response, [*singles, current]):
            energy = float(feature @ feature)
            safe = all(scores[key] >= baseline[key] - slack - 1e-10 for key in baseline)
            improves = energy < damage - max(minreduce, 1e-10)
            interventions.append({"label": label, "damage": energy, "damagechange": energy - damage, "semanticvalid": safe, "improves": improves, "accepted": safe and improves, "scores": scores})
        report.update({"semanticinteraction": semantic, "interventions": interventions, "singlevalidjointinvalid": all(row["accepted"] for row in interventions[:2]) and not interventions[2]["accepted"], "semanticscores": scorecalls})
    if transport is not None:
        plain = [transport(field).detach().float().cpu() for field in (base, base + changes[0], base + changes[1], base + changes[0] + changes[1])]
        features = [protection.feature(value).double().numpy() for value in plain]
        before = features[3] - features[1] - features[2] + features[0]
        report.update({"transport": float(np.linalg.norm(before)), "refinementchange": float(np.linalg.norm(interaction - before)), "transportdecodes": 4, "decodes": 6})
    return report

@torch.no_grad()
def select(base, render, score, origin, mask, config=None, pool=None, transport=None, admit=None):
    # Verify joint repairs within a fixed query budget.
    config = config or Config()
    config.validate()
    if admit is not None and not callable(admit):
        raise ValueError("The external output guard must be callable.")
    check(base)
    protection = Protection(origin, mask)
    image = render(base).detach().float().cpu()
    residual = protection.feature(image).double().numpy()
    baseline = measured(score(image))
    damage = protection.energy(image)
    referencevalid = True if admit is None else bool(admit(image, baseline))
    report = {"schema": 6, "method": "projection", "config": asdict(config), "baseline": {"scores": baseline, "damage": damage}, "protectedvalid": protection.valid, "coverage": protection.coverage, "evaluations": 1, "verification": [], "responses": [], "feedback": {"mode": config.calibration, "updates": 0, "models": []}, "fallback": True, "selected": "base", "weights": [], "gain": 0., "damagechange": 0. if protection.valid else None, "constraintmet": False, "audit": protection.audit(image), "scores": baseline, "guarantee": "Local predictions are not global guarantees; every retained output passes measured checks."}

    def finish(field, pixels, reason):
        report["reason"] = reason
        return Selection(field, pixels.to(origin.device), report)

    if not protection.valid or float(mask.mean()) <= 1e-8:
        return finish(base, image, "No valid edit and protection support.")
    vectors, construction = build(base, pool, config.grid, config.basis)
    report["basis"] = construction
    if not vectors or config.radius == 0:
        return finish(base, image, "No usable field directions or zero trust radius.")
    keys = [key for key in baseline if key != "score"] or ["score"]
    if any(key.startswith("local") and key[5:].isdigit() for key in keys):
        keys = [key for key in keys if key != "local"]
    columns, semantic = [], []
    for index, vector in enumerate(vectors):
        plus = render(base + config.step * vector).detach().float().cpu()
        minus = render(base - config.step * vector).detach().float().cpu()
        report["evaluations"] += 2
        positive, negative = measured(score(plus)), measured(score(minus))
        if set(positive) != set(baseline) or set(negative) != set(baseline):
            raise ValueError("Semantic metric keys must remain fixed across probes.")
        first, second = protection.feature(plus).double().numpy(), protection.feature(minus).double().numpy()
        response = (first - second) / (2 * config.step)
        columns.append(response)
        semantic.append([(positive[key] - negative[key]) / (2 * config.step) for key in keys])
        report["responses"].append({"atom": index, "norm": float(np.linalg.norm(response)), "curvature": float(np.linalg.norm(first - 2 * residual + second)) / config.step ** 2})
    matrix = np.stack(columns, axis=1)
    constraints = np.asarray(semantic, dtype=np.float64).T
    flattened = torch.stack([vector.detach().double().cpu().flatten() for vector in vectors])
    metric = (flattened @ flattened.T / (base.numel() * construction["scale"] ** 2)).numpy()
    weights, solution = solve(matrix, residual, constraints, config, metric)
    report.update({"solver": solution, "semanticrows": keys, "semanticmatrix": constraints.tolist(), "responsegram": (matrix.T @ matrix).tolist(), "fieldmetric": metric.tolist(), "proposal": weights.tolist()})
    if solution["norm"] <= 1e-12:
        return finish(base, image, "The measured semantic constraints contain no useful repair direction.")
    selectedfield, selectedimage = base, image
    selecteddamage, selectedscores = damage, baseline
    originalmatrix, originalconstraints = matrix.copy(), constraints.copy()
    observedweights, observedpixels, observedscores = [], [], []
    model, scale = 0, 1.
    for attempt in range(config.backtrack + 1):
        correction = sum(float(weight) * vector for weight, vector in zip(weights, vectors))
        field = base + correction
        output = render(field).detach().float().cpu()
        report["evaluations"] += 1
        current = measured(score(output))
        if set(current) != set(baseline):
            raise ValueError("Verification metrics must match baseline metrics.")
        actual = protection.energy(output)
        predictedchange = matrix @ weights
        actualchange = protection.feature(output).double().numpy() - residual
        predicted = float(np.square(residual + predictedchange).sum())
        semanticvalid = all(current[key] >= baseline[key] - config.slack - 1e-10 for key in baseline)
        externalvalid = True if admit is None else bool(admit(output, current))
        admissible = semanticvalid and externalvalid and actual < damage - max(config.minreduce, 1e-10)
        predictedsemantic = constraints @ weights
        actualsemantic = np.array([current[key] - baseline[key] for key in keys])
        pixeldefect, semanticdefect = defect(actualchange, predictedchange), defect(actualsemantic, predictedsemantic)
        report["verification"].append({"attempt": attempt, "model": model, "scale": scale, "weights": weights.tolist(), "predicted": predicted, "actual": actual, "pixelerror": abs(predicted - actual), "responseerror": pixeldefect["absolute"], "responsedefect": pixeldefect["relative"], "semanticprediction": predictedsemantic.tolist(), "semanticchange": actualsemantic.tolist(), "semanticerror": float(np.max(np.abs(predictedsemantic - actualsemantic))), "semanticdefect": semanticdefect["relative"], "semanticvalid": semanticvalid, "scores": current, "accepted": admissible, "agreement": (damage - actual) / max(damage - predicted, 1e-12)})
        initialpixel, initialsemantic = defect(actualchange, originalmatrix @ weights), defect(actualsemantic, originalconstraints @ weights)
        report["verification"][-1].update({"semanticresponseerror": semanticdefect["absolute"], "initialresponseerror": initialpixel["absolute"], "initialsemanticerror": initialsemantic["absolute"], "responseimprovement": initialpixel["absolute"] - pixeldefect["absolute"], "semanticimprovement": initialsemantic["absolute"] - semanticdefect["absolute"]})
        report["verification"][-1]["referencevalid"] = externalvalid
        if config.factorial and attempt == 0:
            report["factorial"] = factorial(base, vectors, construction["atoms"], weights, render, protection, image, output, transport, score, baseline, current, config.slack, config.minreduce)
            report["evaluations"] += report["factorial"]["evaluations"]
        if admissible and (not referencevalid or actual < selecteddamage):
            selectedfield, selectedimage, selecteddamage, selectedscores = field, output, actual, current
            referencevalid = True
            report.update({"weights": weights.tolist(), "selected": "projected", "fallback": False, "constraintmet": True, "scale": scale, "selectedmodel": model})
        observedweights.append(weights.copy())
        observedpixels.append(actualchange)
        observedscores.append(actualsemantic)
        if attempt == config.backtrack:
            break
        if not admissible and config.calibration != "fixed" and model < config.updates and max(pixeldefect["relative"], semanticdefect["relative"]) > config.defect:
            fit = None
            if config.calibration == "secant":
                memory = config.updates + 1
                chords = np.stack(observedweights[-memory:], axis=1)
                targets = np.concatenate((np.stack(observedpixels[-memory:], axis=1), np.stack(observedscores[-memory:], axis=1)), axis=0)
                fitted, fit = complete(np.concatenate((originalmatrix, originalconstraints), axis=0), chords, targets, metric, config.ranktol)
                matrix, constraints = fitted[:residual.size], fitted[residual.size:]
            model += 1
            localconfig = replace(config, radius=config.radius * .5 ** model)
            weights, solution = solve(matrix, residual, constraints, localconfig, metric)
            report["feedback"]["updates"] = model
            report["feedback"]["models"].append({"model": model, "after": attempt, "radius": localconfig.radius, "fit": fit, "solver": solution, "semanticmatrix": constraints.tolist(), "responsegram": (matrix.T @ matrix).tolist()})
            scale = 1.
            if solution["norm"] <= 1e-12:
                break
        else:
            weights, scale = .5 * weights, .5 * scale
    report.update({"gain": selectedscores["score"] - baseline["score"], "damagechange": selecteddamage - damage, "scores": selectedscores, "audit": protection.audit(selectedimage)})
    return finish(selectedfield, selectedimage, "Verified signed field repair." if not report["fallback"] else "All projected repairs failed actual output checks.")

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
