# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import torch
import torch.nn.functional as functional
from engine.wavelet import check, gates, split

@dataclass(frozen=True)
class Config:
    mode: str = "baseline"
    blend: float = 0.5
    threshold: float = 0.4
    floor: float = 1e-8
    components: int = 8
    minimum: int = 2
    small: int = 2
    dilation: int = 1
    partition: str = "grid"
    regions: int = 4
    leaf: int = 1
    penalty: float = 0.001

    def validate(self):
        if self.mode not in {"baseline", "activity", "attention", "hybrid"} or self.partition not in {"grid", "adaptive"}:
            raise ValueError("Unknown localization or partition mode.")
        for name in ("blend", "threshold", "floor", "penalty"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Localization parameters must be finite and nonnegative.")
        if self.blend > 1 or not 0 < self.threshold <= 1 or self.floor == 0:
            raise ValueError("Localization thresholds are outside their valid ranges.")
        for name in ("components", "minimum", "small", "dilation", "regions", "leaf"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name in {"small", "dilation"} else 1):
                raise ValueError("Localization counts must be nonnegative integers.")

def normalize(value, floor=1e-8):
    # Normalize active values without erasing tiny edit regions.
    check(value)
    if not math.isfinite(floor) or floor <= 0 or torch.any(value < 0).item():
        raise ValueError("Normalization requires nonnegative values and a positive floor.")
    value = value.detach().float()
    active = value > floor
    if not active.any().item():
        return torch.zeros_like(value)
    scale = torch.quantile(value[active], .95).clamp_min(floor)
    return (value / scale).clamp(0, 1) * active

def components(value, threshold, limit, minimum, small):
    # Retain components and reserve small edit regions.
    scores = value[0, 0].detach().cpu().numpy()
    bitmap = scores >= threshold
    height, width = bitmap.shape
    seen, found = set(), []
    for row in range(height):
        for column in range(width):
            if not bitmap[row, column] or (row, column) in seen:
                continue
            stack, points = [(row, column)], []
            seen.add((row, column))
            while stack:
                y, x = stack.pop()
                points.append((y, x))
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yy < height and 0 <= xx < width and bitmap[yy, xx] and (yy, xx) not in seen:
                        seen.add((yy, xx))
                        stack.append((yy, xx))
            strength = sum(float(scores[y, x]) for y, x in points)
            found.append((strength, points))
    found.sort(key=lambda item: (-item[0], item[1][0]))
    large = [item for item in found if len(item[1]) >= minimum]
    tiny = [item for item in found if len(item[1]) < minimum][:min(small, limit)]
    kept = large[:max(0, limit - len(tiny))] + tiny
    support = torch.zeros_like(value, device="cpu")
    for _, points in kept:
        for row, column in points:
            support[0, 0, row, column] = float(scores[row, column])
    return support.to(value.device), {"detected": len(found), "retained": len(kept), "small": len(tiny)}

def locate(base, pool, explicit=None, attention=None, config=None):
    from engine.response import region
    config = config or Config()
    config.validate()
    check(base)
    report = {"config": asdict(config), "attention": attention is not None, "explicit": explicit is not None}
    if explicit is not None or config.mode == "baseline":
        mask = region(base, explicit)
        report.update({"source": "explicit" if explicit is not None else "baseline", "uncertain": not bool(mask.any()), "coverage": float(mask.mean())})
        return mask, report
    if not pool:
        raise ValueError("Multi-time localization requires candidate fields.")
    values = []
    for field in pool.values():
        check(field)
        if field.shape != base.shape or field.device != base.device:
            raise ValueError("Localization fields must share shape and device.")
        values.append(field.float().square().mean(1, keepdim=True))
    activity = normalize(torch.quantile(torch.stack(values), .5, dim=0), config.floor)
    textual = None
    if attention is not None:
        check(attention)
        if attention.shape[1] != 1 or torch.any(attention < 0).item():
            raise ValueError("Attention maps must be nonnegative single-channel tensors.")
        textual = functional.interpolate(attention.to(base.device), base.shape[-2:], mode="bilinear", align_corners=False)
        textual = normalize((textual - torch.quantile(textual.flatten(), .1)).clamp_min(0), config.floor)
        if not textual.any().item():
            textual = None
    if config.mode == "attention":
        signal = textual if textual is not None else torch.zeros_like(activity)
        source = "attention" if textual is not None else "unavailable"
    elif config.mode == "hybrid" and textual is not None:
        signal, source = config.blend * textual + (1 - config.blend) * activity, "hybrid"
    else:
        signal, source = activity, "activity"
    mask, details = components(signal, config.threshold, config.components, config.minimum, config.small)
    if config.dilation:
        radius = config.dilation
        mask = functional.max_pool2d(mask, 2 * radius + 1, 1, radius)
    mask = functional.avg_pool2d(functional.pad(mask, (1, 1, 1, 1), mode="replicate"), 3, 1)
    report.update({**details, "source": source, "fallback": config.mode == "hybrid" and textual is None, "uncertain": not bool(mask.any()), "coverage": float(mask.mean()), "signalpeak": float(signal.max())})
    return mask, report

def partition(base, pool, mask, grid=2, config=None):
    config = config or Config()
    config.validate()
    check(base)
    check(mask)
    if mask.shape != (1, 1, *base.shape[-2:]) or torch.any((mask < 0) | (mask > 1)).item():
        raise ValueError("Partition support must match the latent grid.")
    if config.partition == "grid":
        windows = gates(base, grid)
        return windows, {"mode": "grid", "count": len(windows)}
    height, width = base.shape[-2:]

    def poolblocks(value):
        return functional.avg_pool2d(functional.pad(value.float(), (0, width % 2, 0, height % 2), mode="replicate"), 2)

    features = [poolblocks(mask)]
    for field in pool.values():
        check(field)
        if field.shape != base.shape or field.device != base.device:
            raise ValueError("Partition fields must share shape and device.")
        for band in split(field):
            features.append(normalize(poolblocks(band.square().mean(1, keepdim=True)), config.floor) / math.sqrt(max(1, 2 * len(pool))))
    features = torch.cat(features, dim=1).detach().double().cpu()
    rows, columns = features.shape[-2:]
    leaves = [(0, 0, rows, columns)]

    def cost(box):
        y, x, bottom, right = box
        part = features[..., y:bottom, x:right]
        return float((part - part.mean(dim=(-2, -1), keepdim=True)).square().sum())

    while len(leaves) < config.regions:
        options = []
        for index, (y, x, bottom, right) in enumerate(leaves):
            if bottom - y < 2 * config.leaf or right - x < 2 * config.leaf or len(leaves) + 3 > config.regions:
                continue
            midy, midx = (y + bottom) // 2, (x + right) // 2
            children = [(a, b, c, d) for a, c in ((y, midy), (midy, bottom)) for b, d in ((x, midx), (midx, right))]
            gain = cost(leaves[index]) - sum(cost(child) for child in children) - 3 * config.penalty
            options.append((gain, index, children))
        if not options:
            break
        gain, index, children = max(options, key=lambda item: (item[0], -item[1]))
        if gain <= 0:
            break
        leaves[index:index + 1] = children
    windows = []
    for y, x, bottom, right in leaves:
        coarse = torch.zeros((1, 1, rows, columns), device=base.device)
        coarse[..., y:bottom, x:right] = 1
        windows.append(coarse.repeat_interleave(2, -2).repeat_interleave(2, -1)[..., :height, :width])
    return windows, {"mode": "adaptive", "count": len(windows), "boxes": [list(box) for box in leaves], "objective": sum(cost(box) for box in leaves) + config.penalty * len(leaves)}

def shortlist(atoms, support, limit=0):
    # Retain diverse candidates and opposite responses.
    if type(limit) is not int or limit < 0:
        raise ValueError("Probe limit must be a nonnegative integer.")
    check(support)
    if support.shape[1] != 1 or torch.any((support < 0) | (support > 1)).item():
        raise ValueError("Shortlist support must be a single-channel probability-range map.")
    if not limit or len(atoms) <= limit:
        return atoms, {"available": len(atoms), "retained": len(atoms), "complete": True, "dropped": []}
    vectors, relevance = [], []
    for atom in atoms:
        value = atom.value.detach().float().cpu()
        vectors.append(value.flatten())
        relevance.append(float((value.square() * support.detach().cpu()).sum()) / max(float(value.square().sum()), 1e-12))
    values = torch.stack(vectors)
    norms = values.norm(dim=1).clamp_min(1e-12)
    directions = values / norms[:, None]
    kept, groups, times, regions, bands = [], set(), set(), set(), set()
    while len(kept) < limit:
        choices = []
        for index, atom in enumerate(atoms):
            if index in kept:
                continue
            distance = 1. if not kept else float((1 - directions[index] @ directions[kept].T).clamp(0, 2).min())
            amplitude = 1. if not kept else min(1., float(torch.abs(torch.log(norms[index] / norms[kept])).min()))
            coverage = int(atom.region not in regions) + int(atom.band not in bands) + int(bool(set((atom.time, *atom.aliases)) - times)) + int(atom.group not in groups)
            choices.append((coverage, distance + .25 * amplitude + .25 * relevance[index], -index))
        _, _, negative = max(choices)
        index = -negative
        atom = atoms[index]
        kept.append(index)
        groups.add(atom.group)
        times.update((atom.time, *atom.aliases))
        regions.add(atom.region)
        bands.add(atom.band)
    dropped = [atom.metadata() for index, atom in enumerate(atoms) if index not in kept]
    return [atoms[index] for index in kept], {"available": len(atoms), "retained": len(kept), "complete": False, "dropped": dropped, "sources": sorted(times), "note": "Budgeted coverage and diversity cannot preserve every useful joint combination."}

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
