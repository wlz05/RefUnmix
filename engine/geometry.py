# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import math
import numpy as np
from engine.theory import fit

def response(model, points, values, regularization=.0001, initial=None):
    # Reuse the same measured chords in clean latent space.
    points, values = np.asarray(points, dtype=float), np.asarray(values, dtype=float)
    active = np.asarray(model.active)
    anchor = np.asarray(model.anchor, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or values.ndim != 2 or values.shape[0] != len(points) or values.shape[1] < 1 or active.ndim != 1 or active.dtype.kind not in "iu" or len(set(active.tolist())) != len(active) or any(axis not in (0, 1, 2) for axis in active) or not len(active):
        raise ValueError("Latent responses require matching measured control rows.")
    if anchor.shape != (3,) or any(not np.isfinite(value).all() for value in (points, values, anchor)) or isinstance(regularization, bool) or not isinstance(regularization, (int, float)) or not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("Latent response inputs must be finite.")
    offsets = points[:, active] - anchor[active]
    centers = np.flatnonzero(np.linalg.norm(offsets, axis=1) < 1e-12)
    indices = np.asarray(model.indices)
    if len(centers) != 1 or indices.ndim != 1 or indices.dtype.kind not in "iu" or len(set(indices.tolist())) != len(indices) or np.any(indices < 0) or np.any(indices >= len(points)):
        raise ValueError("Latent fits must reuse one visited anchor and its neighbors.")
    center = int(centers[0])
    chosen = [int(index) for index in indices if index != center]
    directions = offsets[chosen].T
    if directions.shape != model.directions.shape or not np.allclose(directions, model.directions, atol=1e-12, rtol=1e-12):
        raise ValueError("Latent and output fits must share identical chords.")
    observed = (values[chosen] - values[center]).T
    previous = np.zeros((values.shape[1], len(active))) if initial is None else np.asarray(initial, dtype=float)
    fitted = fit(previous, directions, observed, regularization)
    error = float(np.linalg.norm(observed - fitted @ directions))
    if not np.isfinite(fitted).all() or not math.isfinite(error):
        raise ValueError("Latent response fits exceed numerical range.")
    return fitted, {"neighbors": indices.tolist(), "fiterror": error, "dimension": values.shape[1], "queries": 0, "scope": "Unweighted clean-latent endpoint geometry; not an external editor reproduction."}

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
