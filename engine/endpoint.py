# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import numpy as np

def design(observed=("sourcezero", "targetzero", "targetfield")):
    # Identify effect contrasts from the observed endpoint rows.
    names = ("sourcezero", "targetzero", "sourcefield", "targetfield")
    if not isinstance(observed, (list, tuple)) or any(name not in names for name in observed) or len(set(observed)) != len(observed):
        raise ValueError("Endpoint designs require unique matched endpoint names.")
    matrix = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]], dtype=float)
    rows = matrix[[names.index(name) for name in observed]]
    _, singular, vectors = np.linalg.svd(rows, full_matrices=True)
    rank = int(np.sum(singular > 1e-12))
    inverse = np.linalg.pinv(rows)
    projector = inverse @ rows
    contrasts = {"bias": [1, 0, 0, 0], "condition": [0, 1, 0, 0], "field": [0, 0, 1, 0], "interaction": [0, 0, 0, 1], "biascondition": [1, 1, 0, 0], "fieldinteraction": [0, 0, 1, 1]}
    identifiable, coefficients = {}, {}
    for name, contrast in contrasts.items():
        value = np.asarray(contrast, dtype=float)
        identifiable[name] = bool(np.linalg.norm(value - value @ projector) <= 1e-10)
        if identifiable[name]:
            coefficients[name] = (value @ inverse).tolist()
    return {"observed": list(observed), "matrix": rows.tolist(), "rank": rank, "nullity": 4 - rank, "kernel": vectors[rank:].tolist(), "identifiable": identifiable, "coefficients": coefficients, "fullinverse": np.linalg.inv(matrix).tolist(), "scope": "Exact finite endpoint contrasts, including nonlinear interactions; rank does not imply pure semantic effects."}

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
