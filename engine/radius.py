# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import math

def bound(gradient, curvature, third, ratio, span=1.):
    # Bound a declared smooth directional Taylor remainder.
    values = (gradient, curvature, third, ratio, span)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values) or min(gradient, curvature, third) < 0 or min(ratio, span) <= 0:
        raise ValueError("Directional radius constants must be finite and nonnegative.")
    if third == 0:
        return {"radius": span, "uncapped": None, "threshold": None, "branch": "zero remainder", "ratio": ratio, "scope": "A directional pre-clipping bound, not a joint image certificate."}
    if gradient == 0 and curvature == 0:
        return {"radius": 0., "uncapped": 0., "threshold": None, "branch": "no quadratic signal", "ratio": ratio, "scope": "A directional pre-clipping bound, not a joint image certificate."}
    first = .5 * (math.log(6.) + math.log(ratio) + math.log(gradient) - math.log(third)) if gradient > 0 else -math.inf
    second = math.log(3.) + math.log(ratio) + math.log(curvature) - math.log(third) if curvature > 0 else -math.inf
    maximum = max(first, second)
    radius = span if maximum >= math.log(span) else math.exp(maximum)
    threshold = math.log(2. / 3.) + math.log(third) + math.log(gradient) - 2 * math.log(curvature) if gradient > 0 and curvature > 0 else -math.inf if curvature > 0 else math.inf
    return {"radius": radius, "uncapped": math.exp(maximum) if maximum < 709 else None, "threshold": math.exp(threshold) if threshold < 709 else None, "branch": "gradient" if first >= second else "curvature", "ratio": ratio, "scope": "A directional pre-clipping bound, not a joint image certificate."}

def profile(value):
    if not isinstance(value, dict) or value.keys() - {"gradient", "curvature", "third", "ratio", "span", "verified"} or {"gradient", "curvature", "third", "ratio"} - value.keys() or type(value.get("verified", False)) is not bool:
        raise ValueError("Radius profiles require declared directional constants.")
    return bound(value["gradient"], value["curvature"], value["third"], value["ratio"], value.get("span", 1.))

def initialize(value, default, native, mode, active):
    if not value:
        return default, {"mode": "fixed", "radius": default, "certified": False}
    result = profile(value)
    if 2 not in active:
        return default, {**result, "mode": "inactive reference coordinate", "initial": default, "certified": False}
    if not native.affine or mode != "latent":
        raise ValueError("Directional profiles require an affine latent reference path.")
    radius = min(default, result["radius"])
    return radius, {**result, "mode": "declared directional profile", "default": default, "initial": radius, "boundsdeclared": value.get("verified", False), "certified": False, "scope": "Initial radius hint only; joint directions, reanchoring, clipping, and endpoint probes remain actually verified."}

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
