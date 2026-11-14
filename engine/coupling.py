# -*- coding: utf-8 -*-
# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

import itertools
import math
import numpy as np
import torch
import torch.nn.functional as functional
from engine.limit import Exhausted
from engine.session import tensor

def bounded(matrix, linear):
    # Solve the small box-constrained reference projection exactly.
    count = len(linear)
    best, loss = np.zeros(count), 0.
    for states in itertools.product((-1, 0, 1), repeat=count):
        value = np.array([max(0, state) for state in states], dtype=float)
        free = np.array([index for index, state in enumerate(states) if state == -1])
        fixed = np.array([index for index, state in enumerate(states) if state != -1])
        if len(free):
            rhs = linear[free] - (matrix[np.ix_(free, fixed)] @ value[fixed] if len(fixed) else 0.)
            value[free] = np.linalg.lstsq(matrix[np.ix_(free, free)], rhs, rcond=1e-10)[0]
        if np.any(value < -1e-8) or np.any(value > 1 + 1e-8):
            continue
        value = np.clip(value, 0, 1)
        current = float(value @ matrix @ value - 2 * linear @ value)
        if current < loss:
            best, loss = value, current
    return best

@torch.no_grad()
def transport(field, native, score, mask, gains=(1., 1.5, 2.), mode='joint', bound='unit', selection='alignment', axis='radial'):
    # Remove measured shared drift before pricing stronger semantic transport.
    session = native.session
    origin = tensor(session.origin, 3)
    baseline = tensor(field.pixels, 3)
    if mode not in {'joint', 'vae', 'source'} or not gains or any(not math.isfinite(x) or x < 1 or x > 4 for x in gains):
        raise ValueError('Coupling requires bounded positive transport gains.')
    before = session.stats()
    previous = session.limit.phase if session.limit is not None else None
    if session.limit is not None:
        session.limit.phase = 'coupling'
    report = {'bound': bound, 'selection': selection, 'axis': axis, 'accepted': False, 'gain': 1., 'coefficients': [], 'history': [], 'mode': mode, 'usesgroundtruth': False}
    area = functional.interpolate(mask.float(), origin.shape[-2:], mode='bilinear', align_corners=False)
    weight = (1 - area).clamp(0, 1).double()
    mass = float(weight.sum()) * 3
    if mass <= 1e-8:
        weight = torch.ones_like(weight)
        mass = float(weight.sum()) * 3
    def damage(image):
        return float(((image.double() - origin.double()).square() * weight).sum()) / mass
    def merit(values):
        if selection == 'direction':
            return .5 * values.get('local', 0.) + .5 * values.get('localdirection', 0.) + .1 * values['target']
        return .75 * values.get('local', values['target']) + .25 * values['target']
    reference = dict(field.scores)
    floor = damage(baseline)
    best, bestlatent, bestfield, bestscore = baseline, field.endpoint, field.value, reference
    bestmerit, bestdamage = merit(reference), floor
    try:
        reconstruction = session.decode(session.anchor).clamp(0, 1)
        source = field.source
        if source is None:
            source = native.endpoint(torch.zeros_like(field.value), 1.)
        decoded = session.decode(source).clamp(0, 1)
        columns = [reconstruction - origin, decoded - reconstruction]
        columns = columns[:1] if mode == 'vae' else columns[1:] if mode == 'source' else columns
        vectors = torch.stack(columns).double()
        weighted = (vectors * weight.sqrt()).reshape(len(columns), -1)
        gram = (weighted @ weighted.T).cpu().numpy() / mass
        report['gram'] = gram.tolist()
        direction = field.value
        if axis == 'residual' and field.request is not None and field.neutral is not None:
            direction = field.request - (field.endpoint - field.neutral)
        elif axis == 'conditional':
            direction = session.difference(session.editor.index(field.report['times'][0]))
        length = float(direction.double().norm())
        direction = (direction * (float(field.value.double().norm()) / length)).float() if length > 1e-10 else field.value
        for gain in sorted(set(gains)):
            actualfield = field.value + (gain - 1) * direction
            try:
                latent = field.endpoint if gain == 1 else native.endpoint(actualfield)
                pixels = baseline if gain == 1 else session.decode(latent).clamp(0, 1)
            except Exhausted:
                report['budgetstopped'] = True
                break
            raw = (pixels.double() - origin.double())
            linear = (vectors * raw * weight).sum(dim=(1, 2, 3, 4)).cpu().numpy() / mass
            limits = np.ones(len(columns))
            if bound == 'gain' and mode != 'vae':
                limits[-1] = gain
            coefficients = limits * bounded(gram * np.outer(limits, limits), linear * limits)
            for fraction in (0., .5, 1.):
                actual = coefficients * fraction
                correction = (vectors * torch.as_tensor(actual, device=origin.device)[:, None, None, None, None]).sum(0)
                candidate = (pixels.double() - correction).float().clamp(0, 1)
                cost = damage(candidate)
                if cost > floor + 1e-10:
                    report['history'].append({'gain': gain, 'coefficients': actual.tolist(), 'damage': cost, 'feasible': False})
                    continue
                scores = score(candidate)
                feasible = scores['target'] >= reference['target'] - .001 and scores.get('local', 0.) >= reference.get('local', 0.) - .001
                value = merit(scores)
                accepted = feasible and (value > bestmerit + 1e-6 or value >= bestmerit - 1e-6 and cost < bestdamage)
                report['history'].append({'gain': gain, 'coefficients': actual.tolist(), 'damage': cost, 'feasible': feasible, 'target': scores['target'], 'local': scores.get('local'), 'accepted': accepted})
                if accepted:
                    best, bestlatent, bestfield, bestscore = candidate, latent, actualfield, scores
                    bestmerit, bestdamage = value, cost
                    report.update(accepted=True, gain=gain, coefficients=actual.tolist())
        report.update(baselinecost=floor, cost=bestdamage, semanticgain=bestmerit - merit(reference))
    except Exhausted:
        report['budgetstopped'] = True
    finally:
        report['costs'] = {key: session.stats()[key] - before[key] for key in ('calls', 'samples', 'decodes')}
        if session.limit is not None:
            session.limit.phase = previous
    field.value, field.endpoint, field.pixels, field.scores = bestfield, bestlatent, best, bestscore
    field.report['coupling'] = report
    return best

# Author: Weiliang Zhu; Other main author: Yeming Dai, Yaming Lv
# Email: wlzchina05@gmail.com
