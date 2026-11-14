# -*- coding: utf-8 -*-
# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

from difflib import SequenceMatcher
import numpy as np
import torch

@torch.no_grad()
def refine(session, source, target, mode="projected", gain=1.5):
    # Keep shared context fixed outside the changed-token semantic span.
    editor = session.editor
    encoded = editor.tokenizer([source, target], padding="max_length", truncation=True, max_length=editor.tokens, return_tensors="pt").input_ids
    special = set(editor.tokenizer.all_special_ids)
    positions = [[i for i, value in enumerate(row) if int(value) not in special] for row in encoded]
    values = [[int(encoded[side, i]) for i in positions[side]] for side in range(2)]
    left, right = session.source[0], session.target[0]
    eos = int(editor.tokenizer.eos_token_id)
    ends = [next((i for i, value in enumerate(row) if int(value) == eos), len(row) - 1) for row in encoded]
    aligned = left[min(ends[0], left.shape[0] - 1)].expand_as(right).clone()
    aligned[0] = left[0]
    shared = torch.ones(right.shape[0], dtype=torch.bool, device=right.device)
    columns = []
    common = []
    operations = SequenceMatcher(a=values[0], b=values[1], autojunk=False).get_opcodes()
    for tag, a, b, c, d in operations:
        if tag == "equal":
            for i, j in zip(positions[0][a:b], positions[1][c:d]):
                aligned[j] = left[i]
                common.append(left[i])
    background = torch.stack(common).mean(0) if common else (left[0] + right[0]) * .5
    for tag, a, b, c, d in operations:
        if tag == "equal":
            continue
        old, new = positions[0][a:b], positions[1][c:d]
        oldmean = left[old].mean(0) if old else background
        newmean = right[new].mean(0) if new else background
        columns.append(newmean - oldmean)
        for k, j in enumerate(new):
            aligned[j] = left[old[min(k, len(old) - 1)]] if old else background
            shared[j] = False
    for j in range(ends[1], right.shape[0]):
        aligned[j] = left[min(ends[0] + j - ends[1], left.shape[0] - 1)]
    if not columns:
        return session.source.clone(), {"mode": mode, "rank": 0, "gain": gain, "changed": 0}
    matrix = torch.stack(columns, 1).double().cpu().numpy()
    basis, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    keep = singular > max(float(singular[0]) * 1e-5, 1e-10)
    basis = torch.as_tensor(basis[:, keep], device=right.device, dtype=right.dtype)
    residual = right - aligned
    restored = residual.clone()
    restored[shared] = (residual[shared] @ basis) @ basis.T if mode == "projected" else 0.
    result = aligned + gain * restored
    if not torch.isfinite(result).all():
        raise RuntimeError("Refined text conditions must remain finite.")
    report = {"mode": mode, "rank": int(keep.sum()), "gain": gain, "changed": int((~shared).sum()), "originalnorm": float(residual.norm()), "retainednorm": float(restored.norm())}
    return result[None], report

# Author: Weiliang Zhu; Other main author: Yeming Dai, Yaming Lv
# Email: wlzchina05@gmail.com
