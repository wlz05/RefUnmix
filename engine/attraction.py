# -*- coding: utf-8 -*-
# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

import math
import torch
from engine.session import tensor

def attract(field, anchor, target, weight=.5):
    # Add a bounded pull toward the target-conditioned field prediction.
    for value in (field, anchor, target):
        tensor(value, 4)
    if any(value.shape != field.shape or value.device != field.device for value in (anchor, target)):
        raise ValueError("Attraction tensors must share latent geometry and device.")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Target attraction weight must be finite and in [0, 1].")
    residual = target - anchor
    result = field if weight == 0 else field + weight * residual
    tensor(result, 4)
    return result, {"weight": weight, "initialnorm": float(field.double().norm()), "residualnorm": float(residual.double().norm()), "finalnorm": float(result.double().norm()), "source": "Existing target-conditioned field-query prediction", "extraqueries": 0}

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
