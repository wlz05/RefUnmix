# -*- coding: utf-8 -*-
# RefUnmix: Separating Prompt Response from Shared Transport in One-Step Image Editing

import math
import torch
import torch.nn.functional as functional
from engine.session import tensor

def average(value, window):
    return functional.avg_pool2d(functional.pad(value, (window // 2,) * 4, mode='replicate'), window, 1)

def endpoint(session, args, anchors, noises, tick, embeddings):
    # Align native target responses with adjacent source reference projections.
    keys = args.keystrength if args.referencephase in {"endpoint", "both"} else 0.
    values = args.valuestrength if args.referencephase in {"endpoint", "both"} else 0.
    if not keys and not values:
        return session.predict(anchors, noises, tick, embeddings)
    if anchors.shape[0] % 2:
        raise ValueError("Reference endpoints require target-source pairs.")
    order = torch.arange(anchors.shape[0], device=anchors.device).reshape(-1, 2).flip(1).flatten()
    result = session.predict(anchors.index_select(0, order), noises.index_select(0, order), tick, embeddings.index_select(0, order), keys, values)
    return result.index_select(0, order)

@torch.no_grad()
def calibrate(anchor, source, contrast, ridge=.5, window=5, prior=.8):
    # Transfer source-calibrated channel responses to the conditional innovation.
    inputs = torch.cat(source, 1).double()
    changes = torch.cat(contrast, 1).double()
    channels = anchor.shape[1]
    count = inputs.shape[1]
    identity = torch.eye(channels, device=anchor.device, dtype=inputs.dtype)
    initial = torch.cat((prior * identity, (1 - prior) * identity), dim=1)
    residual = anchor - torch.einsum('oi,bihw->bohw', initial, inputs)
    mean = average(inputs, window)
    centered = average((inputs[:, :, None] * inputs[:, None, :]).flatten(1, 2), window).reshape(1, count, count, *anchor.shape[-2:]) - mean[:, :, None] * mean[:, None, :]
    residualmean = average(residual, window)
    cross = average((residual[:, :, None] * inputs[:, None, :]).flatten(1, 2), window).reshape(1, channels, count, *anchor.shape[-2:]) - residualmean[:, :, None] * mean[:, None, :]
    matrix = centered.permute(0, 3, 4, 1, 2)
    matrix = (matrix + matrix.transpose(-1, -2)) * .5
    scale = matrix.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-6) / count
    matrix = matrix + (ridge * scale + 1e-5)[..., None, None] * torch.eye(count, device=anchor.device, dtype=inputs.dtype)
    coefficients = torch.linalg.solve(matrix, cross.permute(0, 3, 4, 2, 1)).transpose(-1, -2)
    correction = torch.einsum('bhwoi,bihw->bohw', coefficients, changes)
    result = torch.einsum('oi,bihw->bohw', initial, changes) + correction
    tensor(result)
    return result.float(), {'ridge': ridge, 'window': window, 'prior': prior, 'correctionnorm': float(correction.double().norm()), 'fieldnorm': float(result.double().norm()), 'coefficientnorm': float(coefficients.double().square().mean().sqrt())}

@torch.no_grad()
def innovation(anchor, source, contrast, ridge=.5, window=5, prior=.8):
    # Separate target innovation from measured source reconstruction uncertainty.
    request = prior * contrast[0] + (1 - prior) * contrast[1]
    errors = [value - anchor for value in source]
    channels = anchor.shape[1]
    signal = average((request[:, :, None] * request[:, None, :]).flatten(1, 2), window).reshape(1, channels, channels, *anchor.shape[-2:]).double()
    nuisance = sum(average((value[:, :, None] * value[:, None, :]).flatten(1, 2), window).reshape_as(signal).double() for value in errors) / len(errors)
    matrix = (signal + ridge * nuisance).permute(0, 3, 4, 1, 2)
    matrix = (matrix + matrix.transpose(-1, -2)) * .5
    epsilon = matrix.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-8) * .01 + 1e-6
    identity = torch.eye(channels, device=anchor.device, dtype=matrix.dtype)
    solved = torch.linalg.solve(matrix + epsilon[..., None, None] * identity, request.permute(0, 2, 3, 1).double()[..., None])
    result = (signal.permute(0, 3, 4, 1, 2) @ solved)[..., 0].permute(0, 3, 1, 2).float()
    tensor(result)
    return result, {'ridge': ridge, 'window': window, 'prior': prior, 'fieldnorm': float(result.double().norm()), 'requestnorm': float(request.double().norm())}

@torch.no_grad()
def controls(origin, conditional, shared, weight, pixels, penalty):
    # Compare allocations using the same measured endpoint tensors.
    from engine.responsibility import resolve
    source = (origin.double() * 255).round()
    exported = lambda value: (value.double().clamp(0, 1) * 255).round()
    joint, _ = resolve(origin, conditional, shared, weight, penalty)
    images = {"gate": origin + weight * (conditional + shared), "prior": origin + conditional + weight * shared, "joint": joint, "strong": origin + conditional + shared}
    images = {name: exported(value).float() / 255 for name, value in images.items()}
    target = float((exported(pixels) - source).square().sum())
    ray = 255 * (conditional.double() + shared.double())
    value = lambda scale: (source + scale * ray).clamp(0, 255).round()
    cost = lambda scale: float((value(scale) - source).square().sum())
    lower, upper = 0., 1.
    for _ in range(20):
        if cost(upper) >= target:
            break
        upper *= 2
    if cost(upper) >= target and target > 0:
        for _ in range(36):
            middle = (lower + upper) * .5
            if cost(middle) <= target:
                lower = middle
            else:
                upper = middle
    elif target > 0:
        lower = upper
    images["radial"] = value(lower).float() / 255
    actual = cost(lower)
    report = {"radialscale": lower, "normspace": "Decoded endpoint RGB response, measured after PNG quantization", "requestedenergy": target, "radialenergy": actual, "relativegap": abs(actual - target) / max(target, 1.), "matched": abs(actual - target) <= 1e-3 * max(target, 1.), "extraqueries": 0, "scope": "Diagnostic allocations from identical endpoints; no mask or score selects outputs."}
    return images, report

@torch.no_grad()
def edit(session, args, metadata):
    # Construct one calibrated conditional field without candidate selection.
    conditioning = {"mode": "raw"}
    if args.singlecondition not in {"raw", "extrapolated"}:
        from engine.condition import refine
        session.target, conditioning = refine(session, metadata["source"], metadata["target"], args.singlecondition, args.conditiongain)
    fieldtarget = session.target
    if args.singlecondition == "extrapolated":
        fieldtarget = session.source + args.conditiongain * (session.target - session.source)
        conditioning = {"mode": "extrapolated", "gain": args.conditiongain, "scope": "Field queries only; native continuation retains the original target condition."}
    metadata["conditioning"] = conditioning
    ticks = [session.editor.index(args.start)] if args.delta == 0 else [session.editor.index(args.start), session.editor.index(args.start - args.delta)]
    times = [args.start] if args.delta == 0 else [args.start, args.start - args.delta]
    source, target, calibrations = [], [], []
    for tick in ticks:
        count = session.noises.shape[0]
        anchors = session.anchor.expand(2 * count, -1, -1, -1)
        noises = torch.cat((session.noises, session.noises))
        embeddings = torch.cat((session.source.expand(count, -1, -1), fieldtarget.expand(count, -1, -1)))
        if args.singlemode == 'recenter':
            initial = session.predict(session.anchor.expand(count, -1, -1, -1), session.noises, tick, session.source.expand(count, -1, -1))
            variance = session.editor.alphas[tick]
            shift = args.recenter * variance.sqrt() / (1 - variance).sqrt().clamp_min(1e-6) * (session.anchor - initial)
            corrected = session.noises + shift
            noises = torch.cat((corrected, corrected))
        prediction = session.predict(anchors, noises, tick, embeddings, args.keystrength if args.referencephase in {"field", "both"} else 0., args.valuestrength if args.referencephase in {"field", "both"} else 0.)
        if args.singlemode == 'recenter':
            calibrations.append({'tick': tick, 'before': float((initial - session.anchor).double().norm()), 'after': float((prediction[:count] - session.anchor).double().norm()), 'noiseshift': float(shift.double().norm())})
        source.append(prediction[:count].mean(0, keepdim=True))
        target.append(prediction[count:].mean(0, keepdim=True))
    if args.carry:
        session.noises = corrected.detach().clone()
    metadata["noisereference"] = "calibrated" if args.carry else "sampled"
    metadata["sourcecalibration"] = calibrations
    if args.singleend == 'decoded':
        images = [session.decode(right) - session.decode(left) for left, right in zip(source, target)]
        field = images[0] if len(images) == 1 else args.singleprior * images[0] + (1 - args.singleprior) * images[1]
        pixels = (session.origin + args.singlegain * field).clamp(0, 1)
        report = {**metadata, 'method': 'single', 'renderer': 'decoded', 'field': {'requested': 'decoded', 'actual': 'decoded', 'times': [args.start] if args.delta == 0 else [args.start, args.start - args.delta], 'gain': args.singlegain, 'fieldnorm': float(field.double().norm())}, 'stats': session.stats(), 'selected': 'single', 'controller': {'enabled': False}, 'candidates': [], 'explicitregion': False, 'scope': 'One native conditional RGB field and one image update; no candidate search.'}
        return pixels, report, session
    if len(source) == 1:
        source.append(source[0]);target.append(target[0])
    contrast = [right - left for left, right in zip(source, target)]
    for tick, value in zip(ticks, contrast):
        session.cache[tick] = value
    if args.singlemode == 'calibrated':
        field, details = calibrate(session.anchor, source, contrast, args.singleridge, args.singlewindow, args.singleprior)
    else:
        field = args.singleprior * contrast[0] + (1 - args.singleprior) * contrast[1]
        details = {'prior': args.singleprior}
    if args.singlemode == 'coherent':
        mean = (contrast[0] + contrast[1]) * .5
        disagreement = (contrast[0] - contrast[1]) * .5
        signal = average(mean.square().mean(1, keepdim=True), 3)
        uncertainty = average(disagreement.square().mean(1, keepdim=True), 3)
        reliability = (1 - uncertainty / signal.clamp_min(1e-8)).clamp(0, 1)
        field = reliability * contrast[0]
        details = {'agreement': float(reliability.mean()), 'uncertainty': float(uncertainty.mean()), 'signal': float(signal.mean())}
    if args.singlemode == 'innovation':
        field, details = innovation(session.anchor, source, contrast, args.singleridge, args.singlewindow, args.singleprior)
    if args.singlecone:
        request = args.singleprior * contrast[0] + (1 - args.singleprior) * contrast[1]
        square = request.double().square().sum(1, keepdim=True)
        deficit = (square - (field.double() * request.double()).sum(1, keepdim=True)).clamp_min(0)
        field = field + (deficit / square.clamp_min(1e-12)).float() * request
    if args.singlemode == 'pullback':
        from engine.field import build
        from engine.native import create
        native = create(session, 1, 1., False, 'auto')
        constructed = build(session, native, 'pullback', args.start, args.delta, args.fieldridge, args.fieldprobe, args.fieldradius, args.fieldtolerance, 'coupled')
        field, details = constructed.value, constructed.report
    if args.singlebias:
        common = args.singleprior * source[0].double() + (1 - args.singleprior) * source[1].double() - session.anchor.double()
        original = field.double()
        square = float(common.square().sum())
        cross = float((common * original).sum())
        coefficient = cross / ((1 + args.biasridge) * square) if square > 1e-20 else 0.
        localreport = None
        if args.biasgeometry == "local":
            from engine.unmix import project
            field, localreport = project(original, common, args.biaswindow, args.biasridge)
        else:
            field = original - coefficient * common
        before, after = float(original.norm()), float(field.norm())
        if args.biasnorm and after > 1e-12:
            field = field * (before / after)
        details['sourcequotient'] = {'coefficient': coefficient, 'biasnorm': math.sqrt(square), 'correlation': cross / max(math.sqrt(square) * before, 1e-20), 'before': before, 'after': float(field.norm()), 'matched': args.biasnorm, 'normspace': 'input field', 'geometry': args.biasgeometry, 'local': localreport}
        field = field.float()
    field = field * args.singlegain
    transported = session.anchor + field
    if args.singleend == 'responsibility':
        from engine.response import region
        from engine.responsibility import resolve
        anchors = torch.cat((session.anchor, transported, transported))
        predictions = session.predict(anchors, session.noises[:1].expand(3, -1, -1, -1), session.editor.index(session.end), torch.cat((session.source, session.source, session.target)), args.keystrength if args.referencephase in {"endpoint", "both"} else 0., args.valuestrength if args.referencephase in {"endpoint", "both"} else 0., 3)
        weight = region(predictions[2:3] - predictions[1:2]).pow(args.responsepower)
        positive = weight.pow(args.responsesharp)
        negative = (1 - weight).pow(args.responsesharp)
        weight = positive / (positive + negative).clamp_min(1e-12)
        weight = functional.interpolate(weight, session.origin.shape[-2:], mode="bilinear", align_corners=False)
        reference, shared, targetimage = [session.decode(predictions[index:index + 1]) for index in range(3)]
        conditional = targetimage - shared
        advection = shared - reference
        pixels, response = resolve(session.origin, conditional, advection, weight, args.responsepenalty, args.innovationgain)
        if args.responsecontrols:
            session.artifacts, response["controls"] = controls(session.origin, conditional, advection, weight, pixels, args.responsepenalty)
        report = {**metadata, 'method': 'single', 'renderer': 'responsibility', 'field': {'requested': args.singlemode, 'actual': args.singlemode, 'times': times, 'gain': args.singlegain, 'calibration': details, 'response': response}, 'stats': session.stats(), 'selected': 'single', 'controller': {'enabled': False}, 'candidates': [], 'explicitregion': False, 'scope': 'One field update from matched source-zero, source-field and target-field endpoints.'}
        return pixels, report, session
    if args.singleend == 'anchored':
        anchors = torch.cat((transported, session.anchor))
        predictions = endpoint(session, args, anchors, session.noises[:1].expand(2, -1, -1, -1), session.editor.index(session.end), torch.cat((session.target, session.source)))
        field = session.decode(predictions[:1]) - session.decode(predictions[1:])
        pixels = (session.origin + field).clamp(0, 1)
        report = {**metadata, 'method': 'single', 'renderer': 'anchored', 'field': {'requested': args.singlemode, 'actual': args.singlemode, 'times': times, 'gain': args.singlegain, 'calibration': details, 'fieldnorm': float(field.double().norm())}, 'stats': session.stats(), 'selected': 'single', 'controller': {'enabled': False}, 'candidates': [], 'explicitregion': False, 'scope': 'A fixed native conditional response field with one image update.'}
        return pixels, report, session
    if args.singleend == 'response':
        anchors = torch.cat((transported, transported, session.anchor, session.anchor))
        embeddings = torch.cat((session.target, session.source, session.target, session.source))
        predictions = endpoint(session, args, anchors, session.noises[:1].expand(4, -1, -1, -1), session.editor.index(session.end), embeddings)
        first = (predictions[2:3] - predictions[3:4]).double()
        second = (predictions[:1] - predictions[1:2]).double()
        advection = (predictions[1:2] - predictions[3:4]).double()
        energy = advection.square().sum(1, keepdim=True) if args.singlemetric == 'advection' else (first.square() + second.square()).sum(1, keepdim=True) * .5
        damping = args.singleprice * energy + 1e-8
        a = first.square().sum(1, keepdim=True) + damping
        b = (first * second).sum(1, keepdim=True)
        c = second.square().sum(1, keepdim=True) + damping
        x = (first * advection).sum(1, keepdim=True)
        y = (second * advection).sum(1, keepdim=True)
        determinant = (a * c - b.square()).clamp_min(1e-16)
        projected = first * ((c * x - b * y) / determinant) + second * ((a * y - b * x) / determinant)
        response = (second + projected).float()
        if args.singleroute == 'cone':
            from engine.cone import project
            basis = field if args.singlebasis == 'input' else first
            response, conereport = project(basis, second, advection, args.singlenorm, args.singlegeometry)
            details['cone'] = conereport
        latent = session.anchor + response
        details['response'] = {'conditionalnorm': float(second.norm()), 'advectionnorm': float(advection.norm()), 'projectednorm': float(projected.norm()), 'price': args.singleprice}
    elif args.singleend == 'quotient':
        from engine.coupling import bounded
        from engine.response import region
        anchors = torch.cat((transported, session.anchor))
        predictions = endpoint(session, args, anchors, session.noises[:1].expand(2, -1, -1, -1), session.editor.index(session.end), torch.cat((session.target, session.source)))
        latent = predictions[:1]
        pixels = session.decode(latent)
        reconstruction = session.decode(session.anchor)
        reference = session.decode(predictions[1:])
        columns = torch.stack((reconstruction - session.origin, reference - reconstruction)).double()
        mask = region(contrast[0])
        weight = 1 - functional.interpolate(mask, pixels.shape[-2:], mode='bilinear', align_corners=False)
        root = weight.double().sqrt()
        vectors = (columns * root).flatten(1)
        gram = (vectors @ vectors.T).cpu().numpy()
        linear = (columns * weight.double() * (pixels.double() - session.origin.double())).sum((1, 2, 3, 4)).cpu().numpy()
        beta = bounded(gram, linear)
        if args.singlequotient == 'global':
            pixels = (pixels.double() - (columns * torch.as_tensor(beta, device=pixels.device)[:, None, None, None, None]).sum(0)).float().clamp(0, 1)
        else:
            from engine.quotient import field as quotientfield
            neutral = constructed.neutral if args.singlemode == 'pullback' else session.predict(session.anchor, session.noises[:1], session.editor.index(session.end), session.target)
            innovation = session.decode(neutral) - reference
            pixels, quotientreport = quotientfield(session.origin, pixels, columns, weight, args.singlequotient, innovation, args.singlepatch)
            details['quotient'] = quotientreport
        report = {**metadata, 'method': 'single', 'renderer': 'quotient', 'field': {'requested': args.singlemode, 'actual': args.singlemode, 'times': times, 'gain': args.singlegain, 'calibration': details, 'referencecoefficients': beta.tolist()}, 'stats': session.stats(), 'selected': 'single', 'controller': {'enabled': False}, 'candidates': [], 'explicitregion': False, 'scope': 'Fixed-budget source-quotiented output field with one image update and no selection.'}
        return pixels, report, session
    elif args.singleend == 'paired':
        anchors = torch.cat((transported, session.anchor))
        noises = session.noises[:1].expand(2, -1, -1, -1)
        predictions = endpoint(session, args, anchors, noises, session.editor.index(session.end), torch.cat((session.target, session.source)))
        latent = session.anchor + predictions[:1] - predictions[1:]
    elif args.singleend == 'target':
        latent = session.predict(transported, session.noises[:1], session.editor.index(session.end), session.target)
    else:
        latent = transported
    pixels = session.decode(latent)
    if args.singlelift:
        pixels = session.origin + pixels - session.decode(session.anchor)
    pixels = pixels.clamp(0, 1)
    report = {**metadata, 'method': 'single', 'renderer': 'anchored' if args.singlelift else 'native', 'field': {'requested': args.singlemode, 'actual': args.singlemode, 'times': times, 'gain': args.singlegain, 'calibration': details}, 'stats': session.stats(), 'selected': 'single', 'controller': {'enabled': False}, 'candidates': [], 'explicitregion': False, 'scope': 'One field update, fixed paired prediction branches, and no semantic search.'}
    return pixels, report, session

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
