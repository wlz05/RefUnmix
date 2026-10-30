# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import math
from pathlib import Path
from data import clean, locate
from engine.reproduce import digest
from engine.serial import read
from evaluate import aggregate, declared, state

RADIAL = {"zero": 0., "half": .5, "weak": .75, "anchor": 1., "strong": 1.25, "large": 1.5, "double": 2.}
CONDITIONAL = {"full", "anchor", "expanded", "plain", "residual", "feedback", "matched"}

def index(rows):
    declared({"rows": rows})
    return {(row["key"], row["seed"]): row for row in rows}

def blend(arms, weights):
    # Apply one cohort-level interpolation to every request and seed.
    if not weights or any(name not in arms or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for name, value in weights.items()) or not math.isclose(sum(weights.values()), 1., abs_tol=1e-12):
        raise ValueError("Curve weights must be finite convex weights on known arms.")
    weights = {name: value for name, value in weights.items() if value > 0}
    mapped = {name: index(arms[name]) for name in weights}
    identities = next(iter(mapped.values())).keys()
    if any(rows.keys() != identities for rows in mapped.values()):
        raise ValueError("Curve interpolation requires identical request and seed sets.")
    output = []
    for identity in sorted(identities):
        originals = [rows[identity] for rows in mapped.values()]
        row = {key: originals[0][key] for key in ("key", "seed", "group", "report") if key in originals[0]}
        row.update(metrics={}, status={})
        for key in sorted(set().union(*(value["metrics"] for value in originals))):
            states = [state(mapped[name][identity], key) for name in weights]
            signs = set(states) & {-1, 1}
            if 0 in states or len(signs) > 1:
                row["metrics"][key] = None
                row["status"][key] = "undefined interpolation"
            elif signs:
                row["metrics"][key] = None
                row["status"][key] = "positive infinity" if 1 in signs else "negative infinity"
            else:
                value = math.fsum(weight * mapped[name][identity]["metrics"][key] for name, weight in weights.items())
                if not math.isfinite(value):
                    raise ValueError("Curve interpolation exceeds numerical range.")
                row["metrics"][key] = value
        output.append(row)
    return output

def crossings(points, value, axis):
    # Retain every adjacent crossing without assuming monotone editing.
    if value is None or not math.isfinite(value):
        return []
    result = []
    for point in points:
        coordinate = point["metrics"].get(axis, {}).get("mean")
        if coordinate is not None and math.isclose(coordinate, value, rel_tol=1e-12, abs_tol=1e-12):
            result.append({point["name"]: 1.})
    for left, right in zip(points, points[1:]):
        first, second = (point["metrics"].get(axis, {}).get("mean") for point in (left, right))
        if first is None or second is None or first == second or not min(first, second) < value < max(first, second):
            continue
        weight = (value - first) / (second - first)
        result.append({left["name"]: 1 - weight, right["name"]: weight})
    return result

def paired(reference, comparison, repeats, seed):
    from audit import paired as compare
    return compare(reference, comparison, repeats, seed)

def compare(arms, radial=RADIAL, reference="full", baseline="anchor", matched="matched", inference=None, repeats=2000, seed=42, tolerance=.01):
    if type(repeats) is not int or not 1 <= repeats <= 100000 or type(seed) is not int or isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("Curve statistics require valid bootstrap and norm settings.")
    if not isinstance(radial, dict) or not radial or any(name not in arms or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for name, value in radial.items()) or len(set(radial.values())) != len(radial):
        raise ValueError("The radial family requires unique finite amplitude factors.")
    if any(name not in arms for name in (reference, baseline, matched)):
        raise ValueError("Curve comparisons require reference, anchor, and matched arms.")
    mapped = {name: index(rows) for name, rows in arms.items()}
    identities = mapped[baseline].keys()
    if any(rows.keys() != identities for rows in mapped.values()):
        raise ValueError("Every curve arm must contain the same requests and seeds.")
    for identity in identities:
        expected = mapped[baseline][identity]
        for rows in mapped.values():
            for key in expected["metrics"].keys() | rows[identity]["metrics"].keys():
                if bool(state(expected, key)) != bool(state(rows[identity], key)):
                    raise ValueError("Metric coverage must remain fixed across curve arms.")
    points = [{"name": name, "factor": radial[name], **aggregate(arms[name])} for name in sorted(radial, key=radial.get)]
    target = aggregate(arms[reference])
    output = {"schema": 1, "reference": reference, "baseline": baseline, "curve": points, "target": target, "scope": "Postselection diagnostic curves of fixed configurations; no per-request selection or deployment recipe is derived."}
    for label, axis, minimize in (("equaledit", "edited", True), ("equalbudget", "mse", False)):
        goal = target["metrics"].get(axis, {}).get("mean")
        candidates = []
        for weights in crossings(points, goal, axis):
            rows = blend(arms, weights)
            point = aggregate(rows)
            quality = point["metrics"].get("mse" if minimize else "edited", {}).get("mean")
            if quality is not None:
                candidates.append({"weights": weights, "point": point, "quality": quality, "rows": rows})
        chosen = (min if minimize else max)(candidates, key=lambda value: value["quality"]) if candidates else None
        discrete = []
        for point in points:
            coordinate = point["metrics"].get(axis, {}).get("mean")
            if goal is not None and coordinate is not None and (coordinate >= goal if minimize else coordinate <= goal):
                discrete.append(point["name"])
        output[label] = {"axis": axis, "target": goal, "status": "matched" if chosen else "outside measured range or undefined", "crossings": [{"weights": value["weights"], "point": value["point"]} for value in candidates], "discretefeasible": discrete, "interpolation": "Fixed weights describe a cohort curve, not a newly generated image; paired intervals condition on these weights."}
        if chosen:
            output[label].update(weights=chosen["weights"], point=chosen["point"], paired=paired(arms[reference], chosen["rows"], repeats, seed))
        eligible = [point for point in points if point["name"] in discrete and point["metrics"].get("mse" if minimize else "edited", {}).get("mean") is not None]
        if eligible:
            selected = (min if minimize else max)(eligible, key=lambda point: point["metrics"]["mse" if minimize else "edited"]["mean"])
            output[label]["discrete"] = {"name": selected["name"], "point": selected, "constraint": "At least the target CLIP-Edited." if minimize else "At most the target protected MSE.", "paired": paired(arms[reference], arms[selected["name"]], repeats, seed)}
    output["normmatched"] = {"arm": matched, "point": aggregate(arms[matched]), "paired": paired(arms[baseline], arms[matched], repeats, seed), "tolerance": tolerance}
    if inference is not None:
        raw = {name: index(rows) for name, rows in inference.items()}
        if raw.keys() != arms.keys() or any(rows.keys() != identities for rows in raw.values()):
            raise ValueError("Endpoint diagnostics require all scored inference requests.")
        ratios, coverage = [], {"matched": 0, "zeroanchor": 0}
        for identity in identities:
            anchor = raw[baseline][identity]["report"]["field"].get("responsenorm")
            value = raw[matched][identity]["report"]["field"].get("responsenorm")
            if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item < 0 for item in (anchor, value)):
                raise ValueError("Every norm comparison needs measured finite endpoint norms.")
            if anchor == 0:
                if value != 0:
                    raise ValueError("A zero response anchor cannot match a nonzero endpoint.")
                ratio = None
                coverage["zeroanchor"] += 1
            else:
                ratio = value / anchor
                if abs(ratio - 1) > tolerance + 1e-10:
                    raise ValueError("An exported matched endpoint violates its norm tolerance.")
                coverage["matched"] += 1
            ratios.append({"key": identity[0], "seed": identity[1], "metrics": {"ratio": ratio}})
        output["normmatched"].update(coverage=coverage, ratios=aggregate(ratios), scope="All requests remain in quality results, including baseline fallbacks and zero anchors.")
    if "feedback" in arms:
        output["feedback"] = {"point": aggregate(arms["feedback"]), "paired": paired(arms[reference], arms["feedback"], repeats, seed)}
        if inference is not None:
            traces = [row.get("report", {}).get("field", {}).get("tangent", {}) for row in inference["feedback"]]
            output["feedback"]["activity"] = {"requests": len(traces), "updated": sum(bool(trace.get("updates")) for trace in traces), "accepted": sum(trace.get("accepted") is True for trace in traces), "updates": sum(len(trace.get("updates", [])) for trace in traces), "checks": sum(trace.get("checks", 0) for trace in traces), "updatequeries": sum(row.get("queries", 0) for trace in traces for row in trace.get("updates", []))}
    return output

def conditional(arms, inference=None, repeats=2000, seed=42):
    # Compare protection rules within the same measured field dictionary.
    if set(arms) != CONDITIONAL:
        raise ValueError("Conditional comparisons require the complete seven-arm matrix.")
    verified = compare(arms, radial={"anchor": 1.}, inference=inference, repeats=repeats, seed=seed)
    pairs = {"dictionary": ("plain", "expanded"), "rule": ("expanded", "full"), "capacity": ("residual", "full"), "feedback": ("full", "feedback"), "anchor": ("anchor", "full")}
    result = {"schema": 1, "arms": {name: aggregate(rows) for name, rows in arms.items()}, "comparisons": {name: {"reference": first, "comparison": second, **paired(arms[first], arms[second], repeats, seed)} for name, (first, second) in pairs.items()}, "normmatched": verified["normmatched"], "scope": "Fixed configurations with model-space projection and realized control-head retention; final official image metrics are independently scored."}
    if inference is not None:
        traces = [row["report"]["field"].get("tangent", {}) for row in inference["full"]]
        locks = [trace.get("conditionalcapacity", {}) for trace in traces]
        history = [row for trace in traces for row in trace.get("history", [])]
        result["activity"] = {"requests": len(traces), "queried": sum(trace.get("checks", 0) > 0 for trace in traces), "accepted": sum(trace.get("accepted") is True for trace in traces), "repairable": sum(lock.get("nullity", 0) > 0 for lock in locks), "repaired": sum(lock.get("status") == "repaired" and lock.get("after", 0) < lock.get("before", 0) for lock in locks), "repairchecks": sum(row.get("kind") == "conditional" for row in history), "seedchecks": sum(row.get("kind") == "seed" for row in history), "targetrejections": sum("realized target" in row.get("violations", []) for row in history), "noqueryrepairs": sum(lock.get("queries") == 0 for lock in locks), "sourceendpoints": sum(trace.get("sources", 0) for trace in traces), "sourcecalls": sum(trace.get("shared", {}).get("queries", 0) for trace in traces)}
        result["activity"].update(selectedrepair=sum(trace.get("selectedkind") == "conditional" for trace in traces), retainedseed=sum(trace.get("selectedkind") == "seed" for trace in traces))
    return result

def load(summary, metrics, repeats=2000, seed=42):
    # Bind official scores to the exact inference files and anchor fields.
    suitepath = locate(Path("."), clean(summary, "summary"))
    suitehash = digest(suitepath)
    suite = read(suitepath)
    kind = suite.get("settings", {}).get("suite")
    if suite.get("schema") != 1 or kind not in {"amplitude", "conditional"} or not isinstance(suite.get("variants"), dict):
        raise ValueError("Response analysis requires an amplitude or conditional suite summary.")
    expected = CONDITIONAL if kind == "conditional" else set(RADIAL) | {"full", "matched", "feedback"}
    if set(suite["variants"]) != expected:
        raise ValueError("The complete response matrix is required for analysis.")
    directory = locate(Path("."), clean(metrics, "metrics"))
    arms, inference, protocol, mapping, hashes = {}, {}, None, None, {suitepath: suitehash}
    for name, variant in suite["variants"].items():
        source = locate(Path("."), clean(variant["path"], "variant"))
        scoredpath = directory / (name + ".json")
        hashes.update({source: digest(source), scoredpath: digest(scoredpath)})
        payload, scored = read(source), read(scoredpath)
        seeds = declared(payload)
        if seeds != sorted(suite.get("seeds", [])) or declared(scored) != seeds or scored.get("inputsha256") != hashes[source]:
            raise ValueError("Curve scores must match the declared seeds and exact inference summary.")
        signature = payload.get("controlsignature", {}).get("sha256")
        if not isinstance(signature, str) or len(signature) != 64 or any(letter not in "0123456789abcdef" for letter in signature) or signature != suite.get("design", {}).get("variants", {}).get(name, {}).get("sha256"):
            raise ValueError("A curve arm differs from its declared configuration signature.")
        backend = scored.get("backend", {})
        if scored.get("protocol") != "PIE external metric adapter" or not backend.get("complete") or any(not isinstance(value, str) or len(value) != 64 or any(letter not in "0123456789abcdef" for letter in value) for value in (backend.get("sha256"), scored.get("mappingsha256"))):
            raise ValueError("Curve comparisons require complete official PIE metrics.")
        identity = backend["sha256"]
        if protocol is not None and (identity != protocol or scored.get("mappingsha256") != mapping):
            raise ValueError("All curve arms must use the same official calculator and mapping.")
        protocol, mapping = identity, scored.get("mappingsha256")
        inference[name] = payload["rows"]
        observed = index(payload["rows"])
        arms[name] = []
        for row in scored["rows"]:
            key = row["key"], row["seed"]
            if key not in observed:
                raise ValueError("Official curve rows differ from the inference population.")
            original = observed[key]
            arms[name].append({"key": key[0], "seed": key[1], "group": original.get("group", key[0]), "report": original["report"], "metrics": row["table"], "status": row.get("status", {})})
        if index(arms[name]).keys() != observed.keys():
            raise ValueError("Official curve scores are missing inference requests.")
    if set(arms) != expected:
        raise ValueError("The complete response matrix is required for analysis.")
    anchor = index(inference["anchor"])
    for name, rows in inference.items():
        settings = suite["design"]["variants"][name]["settings"]
        factor = RADIAL.get(name, 1.) if kind == "amplitude" else 1.
        if settings.get("fieldfactor") != factor:
            raise ValueError("The amplitude matrix changed a declared radial factor.")
        for row in rows:
            report = row["report"]
            reference = anchor[(row["key"], row["seed"])]["report"]
            for key in ("anchorfieldsha256", "protectionsha256", "source", "target", "imagesha256", "masksha256"):
                if key not in report or key not in reference or report[key] != reference[key]:
                    raise ValueError("Amplitude comparisons changed the field anchor, request, or protection support.")
            if name in RADIAL and (kind == "amplitude" or name == "anchor"):
                first, second = reference.get("fieldnorm"), report.get("fieldnorm")
                if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in (first, second)) or not math.isclose(second, RADIAL[name] * first, rel_tol=1e-5, abs_tol=1e-12):
                    raise ValueError("A radial output changed the constructed field direction or amplitude.")
    result = (conditional if kind == "conditional" else compare)(arms, inference=inference, repeats=repeats, seed=seed)
    if any(digest(path) != value for path, value in hashes.items()):
        raise ValueError("A curve input changed during analysis.")
    result.update(protocolsha256=protocol, mappingsha256=mapping, suitehash=suitehash)
    result["cost"] = {name: value.get("cost") for name, value in suite["variants"].items()}
    return result

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
