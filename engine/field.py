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

    def observe(self, endpoint, reference=None, request=None):
        # Diagnose a paid endpoint without another generation query.
        neutral = self.neutral if reference is None else reference
        target = self.request if request is None else request
        if endpoint is None or neutral is None or target is None:
            return
        tensor(endpoint)
        tensor(neutral)
        if endpoint.shape != neutral.shape or endpoint.device != neutral.device:
            raise ValueError("Observed field endpoints must match the cached reference.")
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

def solve(response, target, anchor, ridge=.05, radius=1.):
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
    bound = radius * math.hypot(*anchor)
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
def build(session, native, kind="pullback", start=.9, delta=.15, ridge=.05, probe=.25, radius=1., tolerance=1e-6, geometry="coupled", weight=.5, shrink=.7):
    options(kind, ridge, probe, radius, tolerance, geometry, weight, shrink)
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
        return Field(value, report, neutral.detach().clone(), target, predicted)
    finally:
        if limit is not None:
            limit.phase = previous

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
