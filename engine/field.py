# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from dataclasses import dataclass
import math
import numpy as np
import torch
from engine.limit import Exhausted
from engine.session import number, tensor

KINDS = ("pullback", "chord", "instant", "average", "blend", "ect", "zero")

def norm(value):
    scale = float(value.abs().max())
    return scale * float((value / scale).square().sum().sqrt()) if scale > 0 else 0.

def measure(displacement, request, prediction=None):
    # Compare responses in the same clean latent units.
    tensor(displacement)
    tensor(request)
    if displacement.shape != request.shape or displacement.device != request.device:
        raise ValueError("Response diagnostics require matching latent geometry.")
    actual, target = displacement.double(), request.double()
    error = actual - target
    targetnorm, actualnorm = norm(target), norm(actual)
    errornorm = norm(error)
    ratio = errornorm / targetnorm if targetnorm > 0 else None
    cosine = float(((actual / actualnorm) * (target / targetnorm)).sum().clamp(-1, 1)) if min(targetnorm, actualnorm) > 0 else None
    result = {"actualfit": float(error.square().sum()), "actualfitmean": float(error.square().mean()), "targetnorm": targetnorm, "responsenorm": actualnorm, "relativefit": ratio if ratio is not None and math.isfinite(ratio) else None, "responsecosine": cosine, "responsestatus": "zero request" if targetnorm == 0 else "zero response" if actualnorm == 0 else "measured"}
    result["relativefitstatus"] = "zero request" if targetnorm == 0 else "finite" if math.isfinite(ratio) else "overflow"
    if errornorm > 0 and (result["actualfit"] == 0 or result["actualfitmean"] == 0):
        result.update({"actualfit": result["actualfit"] or None, "actualfitmean": None, "fitstatus": "underflow"})
    else:
        result["fitstatus"] = "finite"
    if prediction is not None:
        tensor(prediction)
        if prediction.shape != actual.shape or prediction.device != actual.device:
            raise ValueError("Predicted responses must match measured endpoints.")
        gap = actual - prediction.double()
        gapnorm = norm(gap)
        ratio = gapnorm / targetnorm if targetnorm > 0 else None
        result.update({"modelgap": float(gap.square().sum()), "modelgapmean": float(gap.square().mean()), "relativegap": ratio if ratio is not None and math.isfinite(ratio) else None})
        result["relativegapstatus"] = "zero request" if targetnorm == 0 else "finite" if math.isfinite(ratio) else "overflow"
        if gapnorm > 0 and (result["modelgap"] == 0 or result["modelgapmean"] == 0):
            result.update({"modelgap": result["modelgap"] or None, "modelgapmean": None, "gapstatus": "underflow"})
        else:
            result["gapstatus"] = "finite"
    if any(isinstance(value, float) and not math.isfinite(value) for value in result.values()):
        raise ValueError("Response diagnostics exceed numerical range.")
    return result

@dataclass
class Field:
    value: torch.Tensor
    report: dict
    neutral: torch.Tensor | None = None
    request: torch.Tensor | None = None
    prediction: torch.Tensor | None = None
    endpoint: torch.Tensor | None = None
    baseline: torch.Tensor | None = None
    pixels: torch.Tensor | None = None
    scores: dict | None = None

    def observe(self, endpoint, reference=None, request=None):
        # Diagnose a paid endpoint without another generation query.
        neutral = self.neutral if reference is None else reference
        target = self.request if request is None else request
        if endpoint is None or neutral is None:
            return
        tensor(endpoint)
        tensor(neutral)
        if endpoint.shape != neutral.shape or endpoint.device != neutral.device:
            raise ValueError("Observed field endpoints must match the cached reference.")
        if target is None:
            length = norm(endpoint.double() - neutral.double())
            self.report.update(responsenorm=length, responsestatus="zero response" if length == 0 else "measured without request")
            return
        self.report.update(measure(endpoint.double() - neutral.double(), target, self.prediction))
        self.report["observed"] = "Native field endpoint before reference control."

def options(kind, ridge, probe, radius, tolerance, geometry="coupled", weight=.5, shrink=.7):
    if kind not in KINDS:
        raise ValueError("Unknown field construction.")
    for value, label in ((ridge, "Field ridge"), (probe, "Field probe"), (tolerance, "Field tolerance")):
        if number(value, label, 0) <= 0:
            raise ValueError(label + " must be positive.")
    number(radius, "Field radius", 0)
    number(weight, "Temporal mixture weight", 0, 1)
    number(shrink, "ECT shrinkage", 0, 1)
    if tolerance >= 1:
        raise ValueError("Field tolerance must be smaller than one.")
    if geometry not in {"coupled", "scalar"} or geometry == "scalar" and kind != "pullback":
        raise ValueError("Norm-matched geometry requires a pullback field.")

def compatible(editor, kind):
    # Keep the ECT transport on its declared VP parameterization.
    if kind == "ect" and (getattr(editor, "metadata", {}).get("family") != "vp" or getattr(editor, "kind", None) != "epsilon"):
        raise ValueError("ECT transport requires an epsilon-prediction VP backbone.")

def temporal(matrix, value, tolerance=1e-6):
    # Test whether a field is another temporal weighting.
    matrix, value = (np.asarray(item, dtype=float) for item in (matrix, value))
    if matrix.ndim != 2 or matrix.shape[1] not in (1, 2) or not matrix.shape[0] or value.shape != (matrix.shape[0],) or any(not np.isfinite(item).all() for item in (matrix, value)):
        raise ValueError("Temporal decomposition requires one or two finite contrast columns.")
    if not 0 < number(tolerance, "Temporal rank tolerance") < 1:
        raise ValueError("Temporal rank tolerance must be between zero and one.")
    scales = np.max(np.abs(matrix), axis=0)
    targetscale = float(np.max(np.abs(value)))
    matrix = matrix / np.where(scales > 0, scales, 1.)
    value = value / targetscale if targetscale else value
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, value, rcond=None)
    candidates = [np.zeros(matrix.shape[1])]
    if np.all(coefficients >= 0):
        candidates.append(coefficients)
    for index in range(matrix.shape[1]):
        column = matrix[:, index]
        square = float(column @ column)
        candidate = np.zeros(matrix.shape[1])
        candidate[index] = max(0., float(column @ value) / square) if square else 0.
        candidates.append(candidate)
    nonnegative = min(candidates, key=lambda item: math.hypot(*(matrix @ item - value)))
    length = math.hypot(*value)
    spanratio = math.hypot(*(matrix @ coefficients - value)) / length if length else None
    coneratio = math.hypot(*(matrix @ nonnegative - value)) / length if length else None
    span = spanratio ** 2 if spanratio is not None else None
    cone = coneratio ** 2 if coneratio is not None else None
    spanstatus = "zero field" if span is None else "underflow" if span == 0 and spanratio > 0 else "finite"
    conestatus = "zero field" if cone is None else "underflow" if cone == 0 and coneratio > 0 else "finite"

    def restore(weights):
        # Restore coefficient units without intermediate overflow.
        restored = []
        for weight, scale in zip(weights, scales):
            if weight == 0 or targetscale == 0:
                restored.append(0.)
                continue
            if scale == 0:
                raise ValueError("A zero temporal column has a nonzero coefficient.")
            first, exponent = math.frexp(float(weight))
            second, targetexponent = math.frexp(targetscale)
            divisor, columnexponent = math.frexp(float(scale))
            try:
                coefficient = math.ldexp(first * second / divisor, exponent + targetexponent - columnexponent)
            except OverflowError as error:
                raise ValueError("Temporal coefficients exceed numerical range.") from error
            if coefficient == 0 or not math.isfinite(coefficient):
                raise ValueError("Temporal coefficients exceed numerical range.")
            restored.append(coefficient)
        return np.asarray(restored)

    coefficients, nonnegative = restore(coefficients), restore(nonnegative)
    try:
        strength = math.fsum(nonnegative)
    except OverflowError as error:
        raise ValueError("Temporal mixture strength exceeds numerical range.") from error
    if not math.isfinite(strength) or any(item is not None and not math.isfinite(item) for item in (span, cone)):
        raise ValueError("Temporal decomposition exceeds numerical range.")
    return {"rank": int(rank), "coefficients": coefficients.tolist(), "nonnegativeweights": nonnegative.tolist(), "spanerror": None if spanstatus == "underflow" else span, "coneerror": None if conestatus == "underflow" else cone, "spanstatus": spanstatus, "conestatus": conestatus, "nonnegative": coneratio <= tolerance if coneratio is not None else None, "strength": strength, "weight": float(nonnegative[-1] / strength) if strength > 0 and len(nonnegative) == 2 else None, "status": "measured" if length else "zero field", "scope": "Column-scaled input-family membership, separate from output response and task completion."}

def orthogonal(values, tolerance=1e-6):
    # Keep only resolved directions in clean latent units.
    if not values or not 0 < number(tolerance, "Rank tolerance") < 1:
        raise ValueError("Field bases need values and a relative tolerance.")
    reference = tensor(values[0])
    for value in values:
        tensor(value)
        if value.shape != reference.shape or value.device != reference.device:
            raise ValueError("Field basis geometry must match.")
    matrix = torch.stack([value.detach().reshape(-1).to(device="cpu", dtype=torch.float64) for value in values], dim=1)
    left, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
    keep = singular > max(float(singular[0]) * tolerance, np.finfo(float).tiny)
    basis = left[:, keep].numpy()
    for index in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, index])))
        if basis[pivot, index] < 0:
            basis[:, index] *= -1
    return basis, singular[keep].tolist()

def capacity(response, target, anchor, tolerance=1e-6):
    # Separate scalar mismatch from resolved directional capacity.
    response, target, anchor = (np.asarray(value, dtype=float) for value in (response, target, anchor))
    if response.ndim != 2 or not all(response.shape) or target.shape != (response.shape[0],) or anchor.shape != (response.shape[1],) or any(not np.isfinite(value).all() for value in (response, target, anchor)):
        raise ValueError("Response capacity requires finite aligned arrays.")
    if isinstance(tolerance, bool) or not 0 < number(tolerance, "Response rank tolerance") < 1:
        raise ValueError("Response rank tolerance must be between zero and one.")
    magnitude = float(np.max(np.abs(response)))
    left, singular, right = np.linalg.svd(response / magnitude if magnitude else response, full_matrices=False)
    kept = singular > tolerance * singular[0]
    left, singular, right = left[:, kept], singular[kept], right[kept]
    report = {"rank": int(kept.sum()), "tolerance": tolerance, "lineerror": None, "spanerror": None, "directionroom": None, "linecosine": None, "status": "zero request", "scope": "Normalized squared errors in the resolved linear response; unconstrained capacity before damping and the correction ball."}
    targetscale = float(np.max(np.abs(target)))
    if targetscale == 0:
        return report
    normalized = target / targetscale
    normalized /= math.hypot(*normalized)
    projected = left @ (left.T @ normalized)
    anchorscale = float(np.max(np.abs(anchor)))
    baseline = left @ (singular * (right @ (anchor / anchorscale))) if anchorscale else np.zeros_like(target)
    length = math.hypot(*baseline)
    cosine = float(np.clip(normalized @ (baseline / length), -1., 1.)) if length else None
    line = cosine * (baseline / length) if length else np.zeros_like(target)
    lineerror = float(np.clip((normalized - line) @ (normalized - line), 0., 1.))
    spanerror = float(np.clip((normalized - projected) @ (normalized - projected), 0., 1.))
    report.update({"lineerror": lineerror, "spanerror": spanerror, "directionroom": max(0., lineerror - spanerror), "linecosine": cosine, "status": "zero response" if not len(singular) else "zero baseline" if not length else "measured"})
    return report

def solve(response, target, anchor, ridge=.05, radius=1., bound=None):
    # Solve one coupled inverse response problem inside a ball.
    response, target, anchor = (np.asarray(value, dtype=float) for value in (response, target, anchor))
    if response.ndim != 2 or target.shape != (response.shape[0],) or anchor.shape != (response.shape[1],) or not len(anchor) or not len(target) or any(not np.isfinite(value).all() for value in (response, target, anchor)):
        raise ValueError("Pullback response geometry must be finite and aligned.")
    if number(ridge, "Field ridge", 0) <= 0:
        raise ValueError("Field ridge must be positive.")
    radius = number(radius, "Field radius", 0)
    gram = response.T @ response
    damping = ridge * max(float(np.trace(gram)) / len(anchor), 1e-12)
    hessian = gram + damping * np.eye(len(anchor))
    residual = target - response @ anchor
    right = response.T @ residual
    if any(not np.isfinite(value).all() for value in (hessian, right, residual)):
        raise ValueError("Pullback normal equations exceed numerical range.")
    eigenvalues, vectors = np.linalg.eigh(hessian)
    if eigenvalues[0] <= 0:
        raise ValueError("Pullback normal equations must be positive definite.")
    rotated = vectors.T @ right
    bound = radius * math.hypot(*anchor) if bound is None else number(bound, "Correction bound", 0)
    if not math.isfinite(bound):
        raise ValueError("Pullback radius exceeds numerical range.")
    multiplier = 0.
    if bound == 0:
        move = np.zeros_like(anchor)
    else:
        move = vectors @ (rotated / eigenvalues)
        if math.hypot(*move) > bound:
            lower, upper = 0., max(math.hypot(*right) / bound, float(eigenvalues[-1]))
            if not math.isfinite(upper):
                raise ValueError("Pullback boundary multipliers exceed numerical range.")
            for _ in range(100):
                middle = lower + (upper - lower) / 2
                length = math.hypot(*(rotated / (eigenvalues + middle)))
                if length > bound:
                    lower = middle
                else:
                    upper = middle
            multiplier = upper
            move = vectors @ (rotated / (eigenvalues + multiplier))
    coefficients = anchor + move
    before = float(residual @ residual)
    error = response @ coefficients - target
    after = float(error @ error)
    objective = after + damping * float(move @ move)
    report = {"before": before, "after": after, "objective": objective, "damping": damping, "multiplier": multiplier, "condition": float(eigenvalues[-1] / eigenvalues[0]), "bound": bound, "movement": math.hypot(*move), "coefficients": coefficients.tolist(), "anchor": anchor.tolist(), "gram": gram.tolist(), "scope": "Optimality concerns this measured linear response model, not the nonlinear generator or image quality."}
    report["correction"] = move.tolist()
    if not np.isfinite(coefficients).all() or any(not math.isfinite(value) for value in (before, after, objective, multiplier, bound, damping, report["movement"], report["condition"])):
        raise ValueError("Pullback solution exceeds numerical range.")
    if objective > before + 1e-9 * max(1., before):
        raise RuntimeError("Pullback solve increased its convex objective.")
    return coefficients, report

@torch.no_grad()
def build(session, native, kind="pullback", start=.9, delta=.15, ridge=.05, probe=.25, radius=1., tolerance=1e-6, geometry="coupled", weight=.5, shrink=.7, repair=0):
    options(kind, ridge, probe, radius, tolerance, geometry, weight, shrink)
    if type(repair) is not int or not 0 <= repair <= 2 or repair and (kind != "pullback" or geometry != "coupled"):
        raise ValueError("Field repair requires zero to two rounds of coupled pullback construction.")
    compatible(session.editor, kind)
    start, delta = number(start, "Field time", 0, 1), number(delta, "Field interval", 0, 1)
    if start <= 0 or delta >= start:
        raise ValueError("Field times require zero <= delta < start <= one.")
    if kind == "ect" and delta != 0:
        raise ValueError("ECT transport uses one temporal support; set delta to zero.")
    if kind == "zero":
        return Field(torch.zeros_like(session.anchor, dtype=torch.float32), {"requested": kind, "actual": kind, "times": [], "status": "constructed", "probes": 0, "space": "clean latent displacement", "scope": "Explicit zero field; native target and reference conditions remain active."})
    if kind == "chord":
        return Field(tensor(session.field(start, delta)).detach().clone(), {"requested": kind, "actual": kind, "times": [start] if delta == 0 else [start, start - delta], "status": "constructed", "probes": 0})
    first = tensor(session.difference(session.editor.index(start))).detach().clone()
    if kind == "ect":
        variance = float(session.editor.alphas[session.editor.index(start)])
        if not 0 < variance <= 1:
            raise ValueError("ECT requires VP coefficients in (0, 1].")
        alpha, sigma = math.sqrt(variance), math.sqrt(1 - variance)
        gain = (math.atan2(sigma, alpha) / sigma if sigma else 1.) - shrink
        value = tensor((gain * first.double()).float())
        return Field(value, {"requested": kind, "actual": kind, "times": [start], "status": "constructed", "probes": 0, "gain": gain, "shrink": shrink, "scope": "ECT Equation 22 transport; direct editing requires raw mode and no cleanup."})
    if kind == "instant":
        return Field(first, {"requested": kind, "actual": kind, "times": [start], "status": "constructed", "probes": 0})
    second = first if delta == 0 else tensor(session.difference(session.editor.index(start - delta))).detach().clone()
    if first.shape != second.shape or first.device != second.device:
        raise ValueError("Conditional contrasts must share latent geometry.")
    request = tensor((first.double() + second.double()).mul(.5).float())
    report = {"requested": kind, "actual": kind, "times": [start] if delta == 0 else [start, start - delta], "status": "constructed", "probes": 0, "space": "clean latent displacement", "request": "paired conditional contrast consensus", "usesclip": False}
    report["geometry"] = geometry
    if repair:
        report["repair"] = {"requested": repair, "accepted": 0, "endpoints": 0, "status": "skipped", "reason": "Initial construction is unavailable or degenerate.", "calls": 0, "samples": 0, "endpointcached": False}
    if kind == "blend":
        value = tensor(((1 - weight) * first.double() + weight * second.double()).float())
        report.update({"weight": weight, "request": "fixed temporal mixture"})
        return Field(value, report)
    if kind == "average":
        return Field(request, report)
    if native is None or native.session is not session:
        raise ValueError("Pullback fields require the matched native continuation.")
    if session.scale == 0 or not torch.any(request != 0).item() or radius == 0 or not session.cleanup:
        report.update({"status": "degenerate", "reason": "Zero request, disabled field, zero radius, or identity continuation."})
        return Field(request, report)
    # Retain independent contrast directions for the response fit.
    basis, singular = orthogonal([first] if delta == 0 else [first, second], tolerance)
    rank = basis.shape[1]
    original = request.detach().reshape(-1).to(device="cpu", dtype=torch.float64).numpy()
    anchor = basis.T @ original
    requestnorm, projectionnorm = float(np.linalg.norm(original)), float(np.linalg.norm(anchor))
    report.update({"rank": rank, "singular": singular, "ridge": ridge, "radius": radius, "requestnorm": requestnorm, "projectionnorm": projectionnorm})
    if projectionnorm <= tolerance * requestnorm:
        report.update({"status": "degenerate", "reason": "Request projection is below the relative rank tolerance."})
        return Field(torch.zeros_like(request), report)
    step = probe * requestnorm / math.sqrt(rank)
    if not math.isfinite(step) or step <= 0:
        raise ValueError("Pullback probe size is invalid.")
    report["probe"] = step
    limit = session.limit
    if limit is not None:
        try:
            limit.require((rank + 2) * native.cost(0.), (rank + 2) * native.queries())
        except Exhausted as error:
            report.update({"actual": "average", "status": "budget", "reason": str(error)})
            return Field(request, report)
    previous = limit.phase if limit is not None else None
    before = (session.calls, session.samples)
    try:
        if limit is not None:
            limit.phase = "field"
        neutral = tensor(native.endpoint(torch.zeros_like(request)))
        if neutral.shape != request.shape or neutral.device != request.device:
            raise ValueError("Pullback references must match the clean request geometry.")
        # Measure endpoint responses with the same cached noise.
        columns = []
        for index in range(rank):
            direction = torch.from_numpy((step * basis[:, index]).reshape(request.shape)).to(device=request.device, dtype=torch.float32)
            value = tensor(native.endpoint(direction))
            if value.shape != neutral.shape or value.device != neutral.device:
                raise ValueError("Pullback endpoints must preserve native geometry.")
            columns.append(((value.double() - neutral.double()) / step).reshape(-1).cpu().numpy())
        response = np.stack(columns, axis=1)
        report["capacity"] = capacity(response, session.scale * original, anchor, tolerance)
        coefficients, diagnostics = solve(response, session.scale * original, anchor, ridge, radius)
        if geometry == "scalar":
            length = float(np.linalg.norm(anchor))
            coefficients = anchor * (float(np.linalg.norm(coefficients)) / length) if length > 0 else np.zeros_like(anchor)
        applied = response @ coefficients - session.scale * original
        diagnostics["appliedcoefficients"] = coefficients.tolist()
        diagnostics["appliedafter"] = float(applied @ applied)
        diagnostics["appliedcorrection"] = (coefficients - anchor).tolist()
        length = float(np.linalg.norm(coefficients))
        cosine = float(np.clip(coefficients @ anchor / (length * projectionnorm), -1, 1)) if length > 0 else None
        correction = coefficients - anchor
        perpendicular = correction - anchor * float(correction @ anchor) / projectionnorm ** 2
        report["alignment"] = {"fieldnorm": length, "normratio": length / projectionnorm, "cosine": cosine, "angle": math.degrees(math.acos(cosine)) if cosine is not None else None, "orthogonalratio": float(np.linalg.norm(perpendicular)) / projectionnorm, "singularratio": singular[-1] / singular[0] if rank > 1 else None, "scope": "Input geometry relative to the projected mean request; not semantic correctness."}
        value = torch.from_numpy((basis @ coefficients).reshape(request.shape)).to(device=request.device, dtype=torch.float32)
        tensor(value)
        contrasts = np.stack([item.detach().double().reshape(-1).cpu().numpy() for item in ([first] if delta == 0 else [first, second])], axis=1)
        report["temporal"] = temporal(contrasts, value.detach().double().reshape(-1).cpu().numpy(), tolerance)
        report.update({"probes": rank + 1, "calls": session.calls - before[0], "samples": session.samples - before[1], "model": diagnostics, "neutralcached": True})
        target = torch.from_numpy((session.scale * original).reshape(request.shape)).to(device=request.device, dtype=torch.float64)
        predicted = torch.from_numpy((response @ coefficients).reshape(request.shape)).to(device=request.device, dtype=torch.float64)
        result = Field(value, report, neutral.detach().clone(), target, predicted)
        if repair:
            refine(result, native, [first, second], repair, ridge, probe, radius, tolerance)
            report["initialtemporal"] = report["temporal"]
            report["temporal"] = temporal(contrasts, result.value.detach().double().reshape(-1).cpu().numpy(), tolerance)
            report.update(calls=session.calls - before[0], samples=session.samples - before[1], probes=rank + 1 + report["repair"]["endpoints"])
            report["initialdiagnostics"] = "Model, capacity, and alignment describe the initial linear construction."
        return result
    finally:
        if limit is not None:
            limit.phase = previous

@torch.no_grad()
def refine(field, native, contrasts, rounds, ridge, probe, radius, tolerance):
    # Refit the endpoint response at the current finite field.
    session, limit = native.session, native.session.limit
    history = []
    report = {"requested": rounds, "accepted": 0, "endpoints": 0, "history": history, "status": "unchanged", "usesclip": False, "scope": "Measured latent request fit; decoded protection and semantic control follow separately."}
    field.report["repair"] = report
    before = session.calls, session.samples
    initial = field.value.detach().clone()
    scale = norm(initial.double())
    report["initialnorm"] = scale
    field.report["baselinefieldnorm"] = scale
    bound = radius * scale
    if scale == 0 or not math.isfinite(bound) or bound <= 0:
        report.update(status="zero radius", calls=0, samples=0, endpointcached=False)
        return
    try:
        if limit is not None:
            limit.require(native.cost(0.), native.queries())
        field.endpoint = tensor(native.endpoint(field.value)).detach().clone()
        report["endpoints"] += 1
        if field.endpoint.shape != field.value.shape or field.endpoint.device != field.value.device:
            raise ValueError("Field repair endpoints must preserve latent geometry.")
        field.baseline = field.endpoint.detach().clone()
        report["initialerror"] = norm(field.request - (field.endpoint.double() - field.neutral.double()))
        for iteration in range(rounds):
            residual = field.request - (field.endpoint.double() - field.neutral.double())
            error = norm(residual)
            if error <= tolerance * norm(field.request):
                report["status"] = "resolved"
                break
            # Add an output residual direction only after measuring its input response.
            directions = [*contrasts, field.value, (residual * (scale / error)).float()]
            basis, _ = orthogonal(directions, tolerance)
            rank = basis.shape[1]
            step = probe * scale / math.sqrt(rank)
            if not math.isfinite(step) or step <= 0:
                raise ValueError("Field repair probes exceed numerical range.")
            if limit is not None:
                limit.require((rank + 2) * native.cost(0.), (rank + 2) * native.queries())
            columns = []
            for index in range(rank):
                direction = torch.from_numpy((step * basis[:, index]).reshape(initial.shape)).to(initial.device, torch.float32)
                endpoint = tensor(native.endpoint(tensor(field.value + direction)))
                report["endpoints"] += 1
                if endpoint.shape != initial.shape or endpoint.device != initial.device:
                    raise ValueError("Field repair probes must preserve latent geometry.")
                columns.append(((endpoint.double() - field.endpoint.double()) / step).reshape(-1).cpu().numpy())
            response = np.stack(columns, axis=1)
            target = residual.reshape(-1).cpu().numpy()
            move, fitted = solve(response, target, np.zeros(rank), ridge, bound=bound)
            correction = torch.from_numpy((basis @ move).reshape(initial.shape)).to(initial.device, torch.float32)
            candidate = tensor(field.value + correction)
            predicted = norm(torch.from_numpy(response @ move - target))
            row = {"round": iteration + 1, "rank": rank, "before": error, "predicted": predicted, "movement": norm(correction.double()), "accepted": False, "model": fitted}
            history.append(row)
            if predicted >= error or torch.equal(candidate, field.value):
                row["reason"], report["status"] = "No predicted improvement.", "stationary"
                break
            endpoint = tensor(native.endpoint(candidate)).detach().clone()
            report["endpoints"] += 1
            if endpoint.shape != initial.shape or endpoint.device != initial.device:
                raise ValueError("Field repair candidates must preserve latent geometry.")
            actual = norm(field.request - (endpoint.double() - field.neutral.double()))
            row["after"] = actual
            row["accepted"] = actual < error * (1 - tolerance)
            if not row["accepted"]:
                row["reason"], report["status"] = "Measured endpoint fit did not improve.", "rejected"
                break
            field.prediction = field.endpoint.double() - field.neutral.double() + torch.from_numpy((response @ move).reshape(initial.shape)).to(initial.device)
            field.value, field.endpoint = candidate.detach().clone(), endpoint
            report["accepted"] += 1
            report["status"] = "accepted"
        report["movement"] = norm(field.value.double() - initial.double())
        report["finalerror"] = norm(field.request - (field.endpoint.double() - field.neutral.double()))
    except Exhausted as error:
        report.update({"status": "budget", "reason": str(error)})
    finally:
        if field.endpoint is not None:
            report["finalerror"] = norm(field.request - (field.endpoint.double() - field.neutral.double()))
        report.update({"calls": session.calls - before[0], "samples": session.samples - before[1], "endpointcached": field.endpoint is not None})

@torch.no_grad()
def rescale(field, factor):
    # Scale the constructed field without rebuilding its inverse response.
    factor = number(factor, "Field amplitude factor", 0)
    baseline = tensor(field.value).detach().clone()
    field.report["radial"] = {"factor": factor, "anchorfieldnorm": norm(baseline.double()), "scope": "Fixed coupled field multiplied after construction; native scale is unchanged."}
    if factor == 1:
        return
    field.value = tensor((baseline.double() * factor).float()).detach().clone()
    field.baseline = None
    field.endpoint = field.neutral.detach().clone() if factor == 0 and field.neutral is not None else None
    field.pixels, field.scores, field.prediction = None, None, None
    if "temporal" in field.report:
        field.report["initialtemporal"] = field.report.pop("temporal")

@torch.no_grad()
def redirect(field, native, score, mask, gain=.02, penalty=1., checks=3, match=False, family="residual", probe=.25, radius=1., ridge=.05, tolerance=1e-6, slack=.02, slacks=None, feedback=False):
    # Redistribute residuals using measured semantic and protected responses.
    from engine.geometry import tangent, adoption, matching
    from engine.poll import feedback as feedbackscores
    import torch.nn.functional as functional
    if not callable(score) or type(checks) is not int or not 1 <= checks <= 8 or type(match) is not bool or type(feedback) is not bool or family not in {"residual", "temporal"}:
        raise ValueError("Tangent refinement requires a scorer and valid query controls.")
    for value, label in ((gain, "Tangent gain"), (probe, "Tangent probe"), (ridge, "Tangent ridge")):
        if number(value, label, 0) <= 0:
            raise ValueError(label + " must be positive.")
    number(penalty, "Tangent protection weight", 0)
    number(radius, "Tangent radius", 0)
    number(slack, "Tangent semantic slack", 0)
    slacks = {} if slacks is None else slacks
    if not isinstance(slacks, dict) or any(not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for key, value in slacks.items()):
        raise ValueError("Tangent semantic slacks must be named finite nonnegative values.")
    if not 0 < number(tolerance, "Tangent tolerance") < 1:
        raise ValueError("Tangent tolerance must be between zero and one.")
    session, limit = native.session, native.session.limit
    baseline = tensor(field.value).detach().clone()
    length = norm(baseline.double())
    report = {"status": "baseline", "accepted": False, "alpha": 0., "gain": 0., "calls": 0, "samples": 0, "decodes": 0, "checks": 0, "probes": 0, "endpoints": 0, "evaluations": 0, "rounding": 0, "budgetstopped": False, "family": family, "matched": match, "normtolerance": .01, "history": [], "usesclip": True, "fieldmasking": False, "scope": "Frozen control-score gains and actual native-output protection; independent official scoring follows selection."}
    field.report["tangent"] = report
    report.update(feedback=feedback, updates=[])
    if length == 0 or radius == 0 or session.scale == 0 or field.request is None or field.neutral is None:
        report["reason"] = "A resolved nonzero coupled baseline is required."
        return
    for value in (field.request, field.neutral):
        if tensor(value).shape != baseline.shape or value.device != baseline.device:
            raise ValueError("Tangent requests and references must match the field geometry.")
    tensor(mask, 1)
    if torch.any((mask < 0) | (mask > 1)).item():
        raise ValueError("Tangent measurement masks must be in [0, 1].")
    weights = functional.interpolate((1 - mask.detach()).to(session.origin.device, torch.float64), session.origin.shape[-2:], mode="bilinear", align_corners=False)
    mass = float(weights.sum())
    root = (weights / mass).sqrt() if mass else torch.zeros_like(weights)
    def residual(pixels):
        if tensor(pixels, 3).shape != session.origin.shape or pixels.device != session.origin.device:
            raise ValueError("Protected images must match the source geometry.")
        return ((pixels.double() - session.origin.double()) * root).reshape(-1).cpu().numpy()
    def selected(values, names):
        if not isinstance(values, dict) or any(key not in values or isinstance(values[key], bool) or not isinstance(values[key], (int, float)) or not math.isfinite(values[key]) for key in names):
            raise ValueError("Tangent control scores must preserve finite named heads.")
        return np.array([values[key] for key in names], dtype=float)
    before = session.calls, session.samples, session.decodes
    previous = limit.phase if limit is not None else None
    try:
        if limit is not None:
            limit.phase = "tangent"
        endpoint = field.endpoint
        if endpoint is None:
            endpoint = tensor(native.endpoint(baseline)).detach().clone()
            report["endpoints"] += 1
        if tensor(endpoint).shape != baseline.shape or endpoint.device != baseline.device:
            raise ValueError("Tangent endpoints must match the baseline geometry.")
        field.endpoint = field.baseline = endpoint.detach().clone()
        field.report["baselinefieldnorm"] = length
        image = session.decode(endpoint).clamp(0, 1)
        r = residual(image)
        scores = score(image)
        report["evaluations"] += 1
        field.pixels, field.scores = image.detach().clone(), dict(scores)
        floors = feedbackscores(scores)
        if slacks.keys() - floors.keys():
            raise ValueError("Tangent semantic slacks must match the frozen score heads.")
        names = ["target"] if "target" in floors else ["score"]
        if scores.get("localviews", 0) > 0 and "local" in scores:
            names.append("local")
        current = selected(scores, names)
        damage = float(r @ r)
        displacement = (endpoint.double() - field.neutral.double()).reshape(-1).cpu().numpy()
        responselevel = float(np.linalg.norm(displacement))
        request = field.request.double() - (endpoint.double() - field.neutral.double())
        pool = [baseline]
        for time in field.report.get("times", []):
            contrast = session.difference(session.editor.index(time))
            if norm(contrast.double()) > 0:
                pool.append((contrast.double() * (length / norm(contrast.double()))).float())
        if family == "residual" and norm(request) > 0:
            pool.append((request * (length / norm(request))).float())
        basis, _ = orthogonal(pool, tolerance)
        count = basis.shape[1]
        anchor = basis.T @ baseline.double().reshape(-1).cpu().numpy() / length
        if np.linalg.norm(basis @ anchor - baseline.double().reshape(-1).cpu().numpy() / length) > tolerance:
            report["reason"] = "The resolved dictionary does not preserve the protection anchor."
            return
        report.update(rank=count, baselinefieldnorm=length, baselineendpointnorm=responselevel, baselinedamage=damage, damage=damage, gain=0., inputnormratio=1., endpointnormratio=1. if responselevel > 0 else None, protectedvalid=mass > 0, heads=names, desired=[gain] * len(names))
        if count < 2:
            report["reason"] = "The measured dictionary has no lateral direction."
            return
        _, _, vectors = np.linalg.svd((anchor / np.linalg.norm(anchor))[None], full_matrices=True)
        lateral = vectors[1:].T
        width = lateral.shape[1]
        step = probe / math.sqrt(width)
        if limit is not None:
            limit.require((width + 1) * native.cost(0.), (width + 1) * native.queries())
        edit, protect, latent = [], [], []
        for index in range(width):
            direction = torch.from_numpy((length * step * basis @ lateral[:, index]).reshape(baseline.shape)).to(baseline.device, torch.float32)
            measured = tensor(native.endpoint(tensor(baseline + direction)))
            report["endpoints"] += 1
            report["probes"] += 1
            if measured.shape != baseline.shape or measured.device != baseline.device:
                raise ValueError("Tangent probes must preserve baseline geometry.")
            pixels = session.decode(measured).clamp(0, 1)
            observedresidual = residual(pixels)
            observed = score(pixels)
            report["evaluations"] += 1
            edit.append((selected(observed, names) - current) / step)
            protect.append((observedresidual - r) / step)
            latent.append(((measured.double() - endpoint.double()) / step).reshape(-1).cpu().numpy())
        edit, protect, latent = (np.stack(values, axis=1) @ lateral.T for values in (edit, protect, latent))
        protectscale = math.sqrt(damage) if damage > 0 else 1.
        report.update(feedback=feedback, updates=[])
        def plan():
            value, allocation = tangent(edit / gain, protect / protectscale, np.ones(len(names)), anchor, np.eye(count), penalty, ridge, radius, tolerance, r / protectscale)
            report["allocation"] = allocation
            direction = torch.from_numpy((length * basis @ value).reshape(baseline.shape)).to(baseline.device, torch.float32)
            relative = norm(direction.double()) / length
            defect = abs(float((baseline.double() * direction.double()).sum())) / (length * length)
            if defect > tolerance * max(relative, np.finfo(float).tiny):
                raise ValueError("The applied residual direction lost its tangent constraint.")
            effect = protect @ value
            maximum, coefficients = adoption(r, effect)
            report.update(tangenterror=defect, directionnorm=relative, protection=coefficients, maximum=maximum)
            if maximum <= 0 or not np.any(value != 0):
                report["reason"] = "No positive predicted adoption satisfies baseline protection."
                return []
            predicted = (edit @ value) / gain
            divisor = float(predicted @ predicted + penalty * (effect @ effect) / (protectscale * protectscale) + allocation["model"]["damping"] * (value @ value))
            numerator = float(predicted.sum() - penalty * (r @ effect) / (protectscale * protectscale))
            alpha = min(maximum, max(0., numerator / divisor)) if divisor > 0 else 0.
            report["amplitudemodel"] = {"linear": numerator, "quadratic": divisor, "unmatched": alpha}
            intervals = [(0., maximum)]
            if match:
                intervals = matching(displacement, latent @ value, maximum, report["normtolerance"])
                report["normintervals"] = [list(row) for row in intervals]
                if not intervals:
                    report["reason"] = "No positive norm-matched adoption fits the protection interval."
                    return []
            if alpha <= 0 or np.min(predicted) <= 0:
                report["reason"] = "No common positive predicted target gain."
                return []
            amplitudes = [min(high, max(low, alpha)) for low, high in intervals]
            amplitudes += [value * .5 ** index for value in amplitudes[:] for index in range(1, checks + 1)]
            amplitudes += [value for row in intervals for value in (row[1], (row[0] + row[1]) / 2, row[0])]
            amplitudes = sorted(set(value for value in amplitudes if value > 0 and any(low <= value <= high for low, high in intervals)), key=lambda value: (divisor * value * value - 2 * numerator * value, -value))
            return [(alpha, value, direction, predicted) for alpha in amplitudes]

        best, seen, proposals = 0., [], plan()
        while proposals and len(report["history"]) < checks:
            actualalpha, value, direction, predicted = proposals.pop(0)
            candidate = tensor(baseline + actualalpha * direction)
            applied = candidate.double() - baseline.double()
            appliednorm = norm(applied) / length
            applieddefect = abs(float((baseline.double() * applied).sum())) / (length * length)
            if appliednorm == 0 or applieddefect > tolerance * appliednorm or any(torch.equal(candidate, previous) for previous in seen):
                report["rounding"] += 1
                continue
            seen.append(candidate.detach().clone())
            measured = tensor(native.endpoint(candidate))
            report["endpoints"] += 1
            if measured.shape != baseline.shape or measured.device != baseline.device:
                raise ValueError("Tangent candidates must preserve baseline geometry.")
            pixels = session.decode(measured).clamp(0, 1)
            observedresidual = residual(pixels)
            observed = score(pixels)
            report["evaluations"] += 1
            changes = (selected(observed, names) - current) / gain
            controlled = feedbackscores(observed)
            if controlled.keys() != floors.keys():
                raise ValueError("Tangent score heads changed during verification.")
            observedcost = float(observedresidual @ observedresidual)
            level = norm(measured.double() - field.neutral.double())
            ratio = level / responselevel if responselevel > 0 else None
            feasible = observedcost <= damage + 1e-10 * max(damage, np.finfo(float).tiny) and all(controlled[key] >= floors[key] - slacks.get(key, slack) for key in floors)
            matched = not match or ratio is not None and abs(ratio - 1) <= .01
            quality = float(changes.min())
            accepted = bool(feasible and matched and quality > max(best, tolerance))
            violations = [key for key in floors if controlled[key] < floors[key] - slacks.get(key, slack)]
            if observedcost > damage + 1e-10 * max(damage, np.finfo(float).tiny):
                violations.append("protection")
            if not matched:
                violations.append("norm")
            if quality <= tolerance:
                violations.append("target gain")
            report["history"].append({"alpha": actualalpha, "gain": quality, "changes": changes.tolist(), "predicted": (actualalpha * predicted).tolist(), "model": len(report["updates"]), "damage": observedcost, "endpointnorm": level, "endpointnormratio": ratio, "tangenterror": applieddefect, "violations": violations, "feasible": bool(feasible), "matched": bool(matched), "accepted": accepted})
            if accepted:
                field.value, field.endpoint, field.prediction = candidate.detach().clone(), measured.detach().clone(), None
                field.pixels, field.scores = pixels.detach().clone(), dict(observed)
                best = quality
                report.update(status="accepted", accepted=True, alpha=actualalpha, gain=quality, damage=observedcost, endpointnormratio=ratio)
                from copy import deepcopy
                report["selectedallocation"] = deepcopy(report["allocation"])
            if feedback and len(report["history"]) < checks:
                from engine.theory import fit
                chord = lateral.T @ (basis.T @ applied.reshape(-1).cpu().numpy() / length)
                square = float(chord @ chord)
                if square <= np.finfo(float).tiny:
                    break
                observations = (changes * gain, observedresidual - r, (measured.double() - endpoint.double()).reshape(-1).cpu().numpy())
                updated, errors = [], {}
                previousmodel = edit, protect, latent
                try:
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        lengthchord = math.sqrt(square)
                        for name, matrix, observedchange in zip(("semantic", "protect", "latent"), previousmodel, observations):
                            response = matrix @ lateral
                            error = observedchange - response @ chord
                            errors[name] = float(np.linalg.norm(error))
                            fitted = fit(response, (chord / lengthchord)[:, None], (observedchange / lengthchord)[:, None], ridge) @ lateral.T
                            if not np.isfinite(fitted).all() or not math.isfinite(errors[name]):
                                raise ValueError("The corrected response exceeds numerical range.")
                            updated.append(fitted)
                        edit, protect, latent = updated
                        proposals = plan()
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                    edit, protect, latent = previousmodel
                    report["feedbackfailure"] = str(error)
                    break
                report["updates"].append({"check": len(report["history"]), "error": errors, "queries": 0, "scope": "Regularized secant correction from an already paid candidate."})
    except Exhausted as error:
        report["reason"] = str(error)
        report["budgetstopped"] = True
    finally:
        report.update(calls=session.calls - before[0], samples=session.samples - before[1], decodes=session.decodes - before[2], checks=len(report["history"]), inputnormratio=norm(field.value.double()) / length)
        report["tangenterror"] = abs(float((baseline.double() * (field.value.double() - baseline.double())).sum())) / (length * length)
        if limit is not None:
            limit.phase = previous

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
