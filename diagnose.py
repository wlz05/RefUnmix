# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from data import clean, locate, records

def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number

def fraction(value: str) -> float:
    number = float(value)
    if not 0.0 < number < 1.0:
        raise argparse.ArgumentTypeError("value must be between zero and one")
    return number

def positivefloat(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return number

def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RefUnmix field diagnostics.")
    parser.add_argument("--model", default="./models")
    parser.add_argument("--dataset", default="./pie")
    parser.add_argument("--mapping", default="./mapping_file.json")
    parser.add_argument("--images", default="./annotation_images")
    parser.add_argument("--masks", default=None)
    parser.add_argument("--output", default="./diagnostics")
    parser.add_argument("--precision", type=str.casefold, choices=("fp32",), default="fp32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=positive, default=100)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--start", type=float, default=0.90)
    parser.add_argument("--delta", type=float, default=0.15)
    parser.add_argument("--epsilon", type=positivefloat, nargs="+", default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--eta", type=positivefloat, nargs="+", default=[0.005, 0.01, 0.02])
    parser.add_argument("--tail", type=fraction, default=0.80)
    parser.add_argument("--bins", type=positive, default=10)
    parser.add_argument("--bootstrap", type=positive, default=1000)
    parser.add_argument("--mode", type=str.casefold, choices=("surprisal", "relative", "invariant", "magnitude"), default="surprisal")
    parser.add_argument("--support", type=float, default=0.50)
    parser.add_argument("--floor", type=float, default=0.10)
    parser.add_argument("--gain", type=float, default=1.00)
    parser.add_argument("--cap", type=float, default=1.50)
    parser.add_argument("--prox", type=float, default=0.50)
    parser.add_argument("--pivot", type=float, default=None)
    parser.add_argument("--signal", type=float, default=None)
    parser.add_argument("--rough", type=float, default=None)
    parser.add_argument("--path", type=float, default=1.0)
    parser.add_argument("--grid", type=float, default=0.0)
    parser.add_argument("--relay", type=float, default=1.0)
    parser.add_argument("--reach", type=int, default=8)
    parser.add_argument("--band", type=int, default=3)
    parser.add_argument("--feather", type=int, default=2)
    parser.add_argument("--route", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--edge", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scalar", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()

def ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    sortedvalues = values[order]
    result = torch.empty_like(values)
    index = 0
    total = values.numel()
    while index < total:
        stop = index
        while stop + 1 < total and sortedvalues[stop + 1] == sortedvalues[index]:
            stop += 1
        result[order[index:stop + 1]] = float(index + stop) * 0.5
        index = stop + 1
    return result

def spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() < 2 or left.numel() != right.numel():
        return 0.0
    first = ranks(left.flatten().double())
    second = ranks(right.flatten().double())
    first = first - first.mean()
    second = second - second.mean()
    denominator = first.norm() * second.norm()
    if denominator <= 0.0:
        return 0.0
    return float((first * second).sum() / denominator)

def averageprecision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    target = labels.flatten().double()
    positives = target.sum()
    if positives <= 0 or positives >= target.numel():
        return float("nan")
    order = torch.argsort(scores.flatten(), descending=True)
    ordered = target[order]
    found = ordered.cumsum(dim=0)
    precision = found / torch.arange(1, target.numel() + 1, dtype=torch.float64, device=target.device)
    return float((precision * ordered).sum() / positives)

def bucket(values: torch.Tensor, edges: list[float], index: int) -> torch.Tensor:
    low = edges[index]
    high = edges[index + 1]
    if index + 1 == len(edges) - 1:
        return (values >= low) & (values <= high)
    return (values >= low) & (values < high)

def partition(total: int) -> set[int]:
    if total < 2:
        return set()
    generator = torch.Generator().manual_seed(17)
    count = max(1, round(0.20 * total))
    return set(torch.randperm(total, generator=generator)[:count].tolist())

@torch.no_grad()
def calibrate(editor, items, config, seed: int):
    if config.signal is not None and config.rough is not None and config.pivot is not None:
        return config, {"signal": config.signal, "rough": config.rough, "pivot": config.pivot, "images": 0}
    from engine.router import ThreadRouter
    router = ThreadRouter(config.router())
    amplitudes: list[torch.Tensor] = []
    differences: list[torch.Tensor] = []
    roughness: list[torch.Tensor] = []
    for item in items:
        from engine.format import read
        image = read(item.image)
        anchor = editor.image(image)
        source = editor.prompt(item.source)
        target = editor.prompt(item.target)
        generator = torch.Generator(device=editor.device).manual_seed(int(seed))
        noise = torch.randn(anchor.shape, generator=generator, device=editor.device, dtype=editor.dtype)
        probe = editor.probe(anchor, source, target, [noise], config.start, config.delta)
        current = probe.current.mean(dim=0).float()
        previous = probe.previous.mean(dim=0).float()
        base = probe.fields.mean(dim=0).float()
        amplitudes.append((0.5 * (router.norm(current) + router.norm(previous))).flatten().cpu())
        differences.append(router.norm(current - previous).flatten().cpu())
        roughness.append(router.gradient(base).flatten().cpu())
    amplitude = torch.cat(amplitudes).double()
    difference = torch.cat(differences).double()
    gridraw = torch.cat(roughness).double()
    epsilon = config.router().epsilon
    positive = amplitude[amplitude > epsilon]
    signal = float(config.signal) if config.signal is not None else float(torch.quantile(positive, 0.10)) if positive.numel() else epsilon
    value = float(torch.quantile(gridraw, 0.95))
    rough = float(config.rough) if config.rough is not None else value if value > epsilon else float(gridraw.max().clamp_min(epsilon))
    pathchange = torch.where(amplitude + difference <= epsilon, torch.zeros_like(difference), difference / (amplitude + signal)).clamp(0.0, 2.0)
    gridchange = (gridraw / max(rough, epsilon)).clamp(0.0, 1.0)
    change = config.path * pathchange + config.grid * gridchange
    value = float(torch.quantile(change, 0.50))
    positive = change[change > epsilon]
    pivot = float(config.pivot) if config.pivot is not None else value if value > epsilon else float(torch.quantile(positive, 0.50)) if positive.numel() else epsilon
    locked = replace(config, signal=max(signal, epsilon), rough=max(rough, epsilon), pivot=max(pivot, epsilon))
    return locked, {"signal": locked.signal, "rough": locked.rough, "pivot": locked.pivot, "images": len(items)}

@torch.no_grad()
def measure(editor, image: Image.Image, source: str, target: str, config, seed: int, epsilons, etas=(0.01,)):
    from engine.router import ThreadRouter
    anchor = editor.image(image)
    sourceembed = editor.prompt(source)
    targetembed = editor.prompt(target)
    generator = torch.Generator(device=editor.device).manual_seed(int(seed))
    noise = torch.randn(anchor.shape, generator=generator, device=editor.device, dtype=editor.dtype)
    probe = editor.probe(anchor, sourceembed, targetembed, [noise], config.start, config.delta)
    base = probe.fields.mean(dim=0)
    routed = ThreadRouter(config.router())(base, probe.current.mean(dim=0), probe.previous.mean(dim=0), probe.target.mean(dim=0), probe.reference.mean(dim=0), anchor)
    reference = editor.reference(anchor, sourceembed, targetembed, noise, config)
    routedreference = editor.reference(anchor, sourceembed, targetembed, noise, config, routed=True)
    maps = dict(routed.maps)
    surprisal = (probe.target.mean(dim=0).float() - probe.reference.mean(dim=0).float()).abs().mean(dim=1, keepdim=True)
    for mode in ("surprisal", "relative", "invariant", "magnitude"):
        worker = ThreadRouter(replace(config.router(), mode=mode))
        maps["edit" + mode] = worker.evidencemap(base.float(), probe.current.mean(dim=0).float(), probe.previous.mean(dim=0).float(), probe.target.mean(dim=0).float(), probe.reference.mean(dim=0).float(), surprisal)
    epsilonvalues = [float(epsilons)] if isinstance(epsilons, (int, float)) else [float(value) for value in epsilons]
    etavalues = [float(etas)] if isinstance(etas, (int, float)) else [float(value) for value in etas]
    directions = {format(value, ".8g"): 0.5 * config.scale * config.scale * editor.direction(anchor, sourceembed, targetembed, noise, config, value) for value in epsilonvalues}
    sensitivities = {format(value, ".8g"): 0.5 * config.scale * config.scale * editor.timesensitivity(anchor, sourceembed, targetembed, noise, config, value) for value in etavalues}
    maps["direction"] = directions[format(epsilonvalues[len(epsilonvalues) // 2], ".8g")]
    maps["timesensitivity"] = sensitivities[format(etavalues[len(etavalues) // 2], ".8g")]
    maps["theorybound"] = maps["direction"] + maps["timesensitivity"]
    return maps, reference, routedreference, {"epsilon": directions, "eta": sensitivities}

def summary(values: list[float]) -> dict[str, float]:
    usable = [value for value in values if value == value]
    if not usable:
        return {"mean": float("nan"), "std": float("nan"), "count": 0}
    tensor = torch.tensor(usable, dtype=torch.float64)
    return {"mean": float(tensor.mean()), "std": float(tensor.std(unbiased=False)), "count": len(usable)}

def maskfile(root: Path, relative: Path) -> Path | None:
    for candidate in (relative, relative.with_suffix(".png"), relative.with_suffix(".jpg"), relative.with_suffix(".jpeg")):
        path = locate(root, candidate)
        if path.is_file():
            return path
    return None

def masktensor(path: Path, height: int, width: int) -> torch.Tensor:
    method = getattr(Image, "Resampling", Image).NEAREST
    with Image.open(path) as handle:
        image = ImageOps.fit(handle.convert("L"), (width, height), method=method)
    return torch.from_numpy(np.asarray(image, dtype="uint8").copy()).reshape(1, 1, height, width) > 127

def supportscore(scores: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    labels = target.to(device=scores.device)
    count = int(labels.sum())
    if count <= 0 or count >= labels.numel():
        return float("nan"), float("nan")
    order = torch.argsort(scores.flatten(), descending=True)
    prediction = torch.zeros(scores.numel(), dtype=torch.bool, device=scores.device)
    prediction[order[:count]] = True
    prediction = prediction.reshape_as(labels)
    union = (prediction | labels).sum()
    iou = float((prediction & labels).sum() / union) if union > 0 else float("nan")
    return averageprecision(scores, labels), iou

def regression(trainx: torch.Tensor, trainy: torch.Tensor, testx: torch.Tensor) -> torch.Tensor:
    mean = trainx.mean(dim=0, keepdim=True)
    spread = trainx.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    train = (trainx - mean) / spread
    test = (testx - mean) / spread
    train = torch.cat((train, torch.ones(train.shape[0], 1, dtype=train.dtype)), dim=1)
    test = torch.cat((test, torch.ones(test.shape[0], 1, dtype=test.dtype)), dim=1)
    ridge = torch.eye(train.shape[1], dtype=train.dtype) * 1e-4
    ridge[-1, -1] = 0.0
    weight = torch.linalg.solve(train.T @ train + ridge, train.T @ trainy)
    return test @ weight

def calibration(prediction: torch.Tensor, target: torch.Tensor, scale: float, bins: int = 10) -> float:
    prediction = prediction.clamp_min(0.0)
    edges = torch.quantile(prediction, torch.linspace(0.0, 1.0, bins + 1, dtype=prediction.dtype))
    error = 0.0
    for index in range(bins):
        mask = bucket(prediction, edges.tolist(), index)
        if bool(mask.any()):
            error += float(mask.float().mean()) * abs(float(prediction[mask].mean() - target[mask].mean()))
    return error / max(scale, 1e-8)

def metrics(prediction: torch.Tensor, target: torch.Tensor, labels: torch.Tensor, scale: float) -> dict[str, float]:
    variance = ((target - target.mean()) ** 2).sum()
    r2 = float(1.0 - ((target - prediction) ** 2).sum() / variance) if variance > 0.0 else float("nan")
    return {"r2": r2, "ap": averageprecision(prediction, labels), "spearman": spearman(prediction, target), "ece": calibration(prediction, target, scale)}

def paired(base: list[dict[str, float]], added: list[dict[str, float]], count: int) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    generator = torch.Generator().manual_seed(17)
    for key in ("r2", "ap", "spearman", "ece"):
        values = []
        for first, second in zip(base, added):
            value = first[key] - second[key] if key == "ece" else second[key] - first[key]
            if math.isfinite(value):
                values.append(value)
        if not values:
            result[key] = {"mean": float("nan"), "low": float("nan"), "high": float("nan")}
            continue
        tensor = torch.tensor(values, dtype=torch.float64)
        draws = torch.randint(0, len(values), (count, len(values)), generator=generator)
        means = tensor[draws].mean(dim=1)
        result[key] = {"mean": float(tensor.mean()), "low": float(torch.quantile(means, 0.025)), "high": float(torch.quantile(means, 0.975))}
    return result

def nested(samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], bootstrap: int = 1000) -> dict[str, object]:
    if len(samples) < 2:
        return {"status": "at least two images are required"}
    testids = partition(len(samples))
    trainx = torch.cat([sample[0] for index, sample in enumerate(samples) if index not in testids]).double()
    trainy = torch.cat([sample[1] for index, sample in enumerate(samples) if index not in testids]).double()
    tests = [sample for index, sample in enumerate(samples) if index in testids]
    testx = torch.cat([sample[0] for sample in tests]).double()
    testy = torch.cat([sample[1] for sample in tests]).double()
    labels = torch.cat([sample[2] for sample in tests]).double()
    models = {"edit": (0, 1), "grid": (0, 1, 2), "lite": (0, 1, 2, 3), "oracle": (0, 1, 2, 3, 4)}
    result: dict[str, object] = {"trainimages": len(samples) - len(testids), "testimages": len(testids)}
    scores: dict[str, dict[str, float]] = {}
    imagescores: dict[str, list[dict[str, float]]] = {}
    scale = float(torch.quantile(trainy.abs(), 0.95).clamp_min(1e-8))
    for name, columns in models.items():
        prediction = regression(trainx[:, columns], trainy, testx[:, columns])
        scores[name] = metrics(prediction, testy, labels, scale)
        imagescores[name] = []
        start = 0
        for features, target, label in tests:
            stop = start + target.numel()
            imagescores[name].append(metrics(prediction[start:stop], target.double(), label.double(), scale))
            start = stop
    result["models"] = scores
    result["pathgain"] = paired(imagescores["grid"], imagescores["lite"], bootstrap)
    result["oraclegain"] = paired(imagescores["lite"], imagescores["oracle"], bootstrap)
    return result

def main() -> None:
    args = arguments()
    from engine.model import EditConfig, Editor
    items = records(args.dataset, args.mapping, args.images)[:args.limit]
    config = EditConfig(noise=1, start=args.start, delta=args.delta, scale=args.scale, cleanup=False, mode=args.mode, support=args.support, floor=args.floor, gain=args.gain, cap=args.cap, prox=args.prox, pivot=args.pivot, signal=args.signal, rough=args.rough, path=args.path, grid=args.grid, relay=args.relay, reach=args.reach, band=args.band, feather=args.feather, route=args.route, edge=args.edge, scalar=args.scalar)
    config.validate()
    editor = Editor.load(args.model, args.device, args.precision)
    testids = partition(len(items))
    calibrationitems = [item for index, item in enumerate(items) if index not in testids] or items
    config, locked = calibrate(editor, calibrationitems, config, args.seed)
    root = clean(args.output, "output")
    root.mkdir(parents=True, exist_ok=True)
    maskroot = locate(Path("."), clean(args.masks, "masks")) if args.masks is not None else None
    if maskroot is not None and not maskroot.is_dir():
        raise FileNotFoundError(maskroot)
    keys = ("evidence", "surprisal", "magnitude", "agreement", "probechange", "pathchange", "gridroughness", "direction", "timesensitivity", "theorybound")
    correlations: dict[str, list[float]] = {key: [] for key in keys}
    precisions: dict[str, list[float]] = {key: [] for key in ("evidence", "surprisal", "magnitude", "probechange", "pathchange", "direction", "theorybound")}
    errors: dict[str, list[float]] = {key: [] for key in ("error", "twoerror", "halferror", "heunerror", "heunstep", "ratio")}
    stability: dict[str, list[float]] = {"twofour": [], "heunfour": [], "converged": [], "routedratio": []}
    relayvalues: dict[str, list[float]] = {"delta": [], "cosine": [], "active": []}
    scans: dict[str, dict[str, list[float]]] = {"epsilon": {format(value, ".8g"): [] for value in args.epsilon}, "eta": {format(value, ".8g"): [] for value in args.eta}}
    gatea: dict[str, dict[str, list[float]]] = {mode: {"ap": [], "iou": []} for mode in ("surprisal", "relative", "invariant", "magnitude")}
    rows: list[list[float]] = []
    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for index, item in enumerate(items, start=1):
        from engine.format import read
        image = read(item.image)
        maps, reference, routedreference, scan = measure(editor, image, item.source, item.target, config, args.seed, args.epsilon, args.eta)
        error = reference["error"]
        cutoff = torch.quantile(error.flatten().float(), args.tail)
        label = error >= cutoff
        features = torch.stack((maps["surprisal"].flatten(), maps["magnitude"].flatten(), maps["gridroughness"].flatten(), maps["pathchange"].flatten(), maps["theorybound"].flatten()), dim=1)
        samples.append((features.cpu(), error.flatten().cpu(), label.flatten().cpu()))
        maskpath = maskfile(maskroot, item.relative) if maskroot is not None else None
        if maskpath is not None:
            targetmask = masktensor(maskpath, error.shape[-2], error.shape[-1])
            for mode in gatea:
                ap, iou = supportscore(maps["edit" + mode], targetmask)
                gatea[mode]["ap"].append(ap)
                gatea[mode]["iou"].append(iou)
        for key in keys:
            correlations[key].append(spearman(maps[key], error))
        for key in precisions:
            precisions[key].append(averageprecision(maps[key], label))
        for key in errors:
            errors[key].append(float(reference[key].mean()))
        stability["twofour"].append(spearman(reference["twoerror"], reference["error"]))
        stability["heunfour"].append(spearman(reference["heunstep"], reference["error"]))
        stability["converged"].append(float(reference["halferror"].mean() < reference["twoerror"].mean()))
        stability["routedratio"].append(float(routedreference["error"].mean() / (reference["error"].mean() + config.router().epsilon)))
        for family in scans:
            for value in scans[family]:
                scans[family][value].append(spearman(scan[family][value], error))
        band = maps["band"] > 0.0
        relayvalues["delta"].append(float(maps["relaydelta"]))
        relayvalues["active"].append(float((maps["relay"] > 0.0).float().mean()))
        relayvalues["cosine"].append(float(maps["relaycos"][band].mean()) if bool(band.any()) else float("nan"))
        change = maps["probechange"]
        edges = torch.quantile(change.flatten().double(), torch.linspace(0.0, 1.0, args.bins + 1, dtype=torch.float64, device=change.device)).tolist()
        for position in range(args.bins):
            mask = bucket(change, edges, position)
            if bool(mask.any()):
                rows.append([index, position, edges[position], edges[position + 1], int(mask.sum()), float(error[mask].mean()), float(maps["direction"][mask].mean())])
        print(f"{index}/{len(items)} {item.relative}")
    report = {"images": len(items), "target": "same-field full step versus four quarter steps", "calibration": locked, "gatea": {mode: {metric: summary(values) for metric, values in metrics.items()} for mode, metrics in gatea.items()}, "spearman": {key: summary(value) for key, value in correlations.items()}, "averageprecision": {key: summary(value) for key, value in precisions.items()}, "reference": {"errors": {key: summary(value) for key, value in errors.items()}, "checks": {key: summary(value) for key, value in stability.items()}}, "scan": {family: {value: summary(scores) for value, scores in values.items()} for family, values in scans.items()}, "increment": nested(samples, args.bootstrap), "relay": {key: summary(value) for key, value in relayvalues.items()}}
    (root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    header = "image,bin,low,high,count,error,direction\n"
    (root / "bins.csv").write_text(header + "\n".join(",".join(str(value) for value in row) for row in rows), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
