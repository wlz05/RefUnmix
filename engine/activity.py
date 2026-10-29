# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from dataclasses import replace
from engine.attainment import choose

def prior(raw, candidates, selected, config, metadata, mode):
    # Compare prior decisions on the same observed candidate pool.
    observed = [raw] + [row for row in candidates if row.complete]
    counterfactual = [replace(row, feasible=not any(reason != "prior" for reason in row.reasons), penalty=0., objective=row.damage) for row in observed]
    without, detail = choose(counterfactual, config.gainloss, config.selection)
    changed = without.key != selected.key
    positive = sum(bool(row.prior is not None and raw.prior is not None and row.prior > raw.prior) for row in candidates if row.complete)
    penalties = sum(bool(row.penalty is not None and row.penalty > 0) for row in candidates if row.complete)
    vetoes = sum("prior" in row.reasons for row in candidates)
    reason = "disabled" if mode == "off" else "same-pool selection changed" if changed else "no candidate excess" if positive == 0 else "zero bounded penalty cap" if config.priorcontrol == "bounded" and config.priorweight * raw.damage == 0 else "attainment band has one candidate" if config.selection == "attain" and detail["bandcount"] == 1 and vetoes == 0 else "same-pool selection unchanged"
    return {"enabled": mode != "off", "mode": mode, "control": config.priorcontrol, "evaluations": metadata.get("evaluations", 0), "positiveexcess": positive, "penalized": penalties, "vetoed": vetoes, "selectedwithprior": selected.key, "selectedwithoutprior": without.key, "selectionchanged": changed, "bandwithoutprior": detail["bandcount"], "reason": reason, "scope": "Same observed pool only; probe cost and search trajectories require separate paired runs."}

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
