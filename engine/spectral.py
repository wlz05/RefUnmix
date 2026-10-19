# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
import math
import statistics
import time
from dataclasses import asdict
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageOps
from dataset import annotation
from engine.frequency import FrequencyRouter
from engine.model import EditConfig, Result

def validate(value):
    if value.get("operator") == "bands":
        if set(value) != {"operator", "edit", "lowtime", "hightime"}:
            raise ValueError("Band candidates have unsupported fields.")
        for name in ("lowtime", "hightime"):
            number = value[name]
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not 0 < number <= 1:
                raise ValueError("Band times must be finite and in (0, 1].")
        config = EditConfig(**value["edit"])
        if config.method != "base":
            raise ValueError("Band candidates require an unmasked base editor.")
    else:
        config = EditConfig(**value)
    config.validate()
    return config

def describe(value):
    config = validate(value)
    common = {"method": config.method, "end": config.end, "cleanup": config.cleanup}
    if value.get("operator") == "bands":
        return {**common, "operator": "bands", "lowtime": value["lowtime"], "hightime": value["hightime"]}
    return {**common, "operator": "time", "start": config.start, "delta": config.delta, "second": round(config.start - config.delta, 8)}

def grid(times, pivot=0.75, end=0.30, cleanup=True, noise=1):
    # Keep both conditional axes and the same-time diagonal.
    if not times or any(isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1 for value in [*times, pivot]):
        raise ValueError("Band times and the pivot must be in (0, 1].")
    config = EditConfig(method="base", end=end, cleanup=cleanup, noise=noise)
    config.validate()
    native = asdict(config)
    configs = {"native": native}
    for low in sorted(set([*times, pivot])):
        for high in sorted(set([*times, pivot])):
            configs[f"pair{len(configs):03d}"] = {"operator": "bands", "edit": dict(native), "lowtime": low, "hightime": high}
    return configs

def masks(region, size, device):
    # Exclude Haar blocks that cross an annotation boundary.
    value = torch.as_tensor(np.asarray(region.convert("L"), dtype=np.float32).copy() / 255.0, device=device)[None, None]
    value = functional.interpolate(value, size=size, mode="area")
    block = FrequencyRouter().low(value)
    return (block >= 1.0 - 1e-6).float(), (block <= 1e-6).float()

def energy(value, mask=None):
    squared = value.float().square()
    if mask is None:
        return float(squared.mean())
    mask = mask.expand_as(squared)
    count = float(mask.sum())
    return float((squared * mask).sum()) / count if count > 0 else None

def spectrum(original, target, region):
    if original.size != target.size or original.size != region.size:
        raise ValueError("RGB frequency comparisons require aligned image sizes.")

    def tensor(image):
        return torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0).permute(2, 0, 1)[None]

    source, edited = tensor(original), tensor(target)
    inside, outside = masks(region, source.shape[-2:], source.device)
    router = FrequencyRouter()
    result = {"space": "rgb", "editarea": float(inside.mean()), "protectarea": float(outside.mean())}
    for name, project in (("low", router.low), ("high", router.high)):
        first, second = project(source), project(edited)
        difference = second - first
        result[name] = {"original": energy(first), "result": energy(second), "change": energy(difference), "edit": energy(difference, inside), "protected": energy(difference, outside)}
    return result

class Sampler:
    @torch.inference_mode()
    def __init__(self, editor, image, source, target, seed, noise):
        EditConfig(method="base", noise=noise).validate()
        if not isinstance(image, Image.Image) or not source.strip() or not target.strip():
            raise ValueError("Sampling requires an image and nonempty prompts.")
        if type(seed) is not int or not -(2 ** 63) <= seed < 2 ** 64:
            raise ValueError("The seed is outside the supported range.")
        self.editor = editor
        self.anchor = editor.image(image)
        self.source = editor.prompt(source.strip())
        self.target = editor.prompt(target.strip())
        generator = torch.Generator(device=editor.device).manual_seed(seed)
        self.noises = [torch.randn(self.anchor.shape, generator=generator, device=editor.device, dtype=editor.dtype) for _ in range(noise)]
        self.cache = {}
        self.router = FrequencyRouter()
        editor.calls, editor.samples = 0, 0

    def tick(self, value):
        return int(self.editor.tick(value, 1).item())

    def remember(self, time, target, source):
        self.cache[self.tick(time)] = {"target": target, "source": source, "change": target - source}

    def prediction(self, time):
        tick = self.tick(time)
        if tick not in self.cache:
            probe = self.editor.probe(self.anchor, self.source, self.target, self.noises, time, 0.0)
            self.remember(time, probe.target, probe.reference)
        return self.cache[tick]

    def base(self, config):
        first, second = config.start, config.start - config.delta
        if self.tick(first) not in self.cache and self.tick(second) not in self.cache:
            probe = self.editor.probe(self.anchor, self.source, self.target, self.noises, first, config.delta)
            self.remember(first, probe.target, probe.reference)
            self.remember(second, probe.targetlow, probe.referencelow)
            return probe.fields.mean(dim=0)
        current = self.prediction(first)["change"]
        previous = self.prediction(second)["change"]
        return ((config.delta * current + first * previous) / (first + config.delta)).mean(dim=0)

    @torch.inference_mode()
    def render(self, value):
        config = validate(value)
        if config.method != "base" or config.noise != len(self.noises):
            raise ValueError("Cached candidates must retain base mode and noise count.")
        calls, samples = self.editor.calls, self.editor.samples
        if value.get("operator") == "bands":
            times = [value["lowtime"], value["hightime"]]
            low = self.prediction(times[0])["change"].mean(dim=0).float()
            high = self.prediction(times[1])["change"].mean(dim=0).float()
            field = (self.router.low(low) + self.router.high(high)).to(self.anchor.dtype)
        else:
            times = [config.start, config.start - config.delta]
            field = self.base(config)
        if not torch.isfinite(field).all():
            raise RuntimeError("The candidate field contains nonfinite values.")
        latent = self.anchor + config.scale * field
        if config.cleanup:
            latent = self.editor.predict(latent, self.noises[0], config.end, self.target)
        images = self.editor.decode(latent)
        ticks = sorted({self.tick(value) for value in times})
        report = {"calls": self.editor.calls - calls, "samples": self.editor.samples - samples, "ticks": ticks, "requiredsamples": 2 * len(ticks) * len(self.noises) + int(config.cleanup), "cached": True}
        return Result(images, latent.detach().cpu(), {"field": field.detach().cpu()}, report)

    @torch.inference_mode()
    def diagnose(self, time, region):
        # Separate target edits from reconstruction errors.
        if self.tick(time) not in self.cache:
            raise ValueError("Diagnostics must not introduce new model queries.")
        prediction = self.cache[self.tick(time)]
        source = prediction["source"].float().mean(dim=0)
        target = prediction["target"].float().mean(dim=0)
        anchor = self.anchor.float()
        inside, outside = masks(region, anchor.shape[-2:], anchor.device)
        result = {"space": "latent", "time": time, "tick": self.tick(time), "editarea": float(inside.mean()), "protectarea": float(outside.mean()), "draws": len(self.noises)}
        for name, project in (("low", self.router.low), ("high", self.router.high)):
            raw = project(target - anchor)
            error = project(source - anchor)
            change = project(target - source)
            draws = prediction["change"].float().flatten(0, 1)
            variance = energy(project(draws) - project(draws).mean(dim=0, keepdim=True)) if len(self.noises) > 1 else None
            result[name] = {"original": energy(project(anchor)), "prediction": energy(project(target)), "predchange": energy(raw), "reconstruction": energy(error), "correction": energy(change), "edit": energy(change, inside), "protected": energy(change, outside), "variance": variance}
        return result

def profiles(report, pivot, selecting=True):
    # Report optima without assuming independent bands.
    pairs = {(row["config"]["lowtime"], row["config"]["hightime"]): row for row in report["table"] if row["config"].get("operator") == "bands"}
    origin = pairs.get((pivot, pivot))

    def brief(row):
        return {key: row[key] for key in ("id", "gain", "feasible")} | {"lowtime": row["config"]["lowtime"], "hightime": row["config"]["hightime"]}

    low = [row for (first, second), row in pairs.items() if second == pivot]
    high = [row for (first, second), row in pairs.items() if first == pivot]
    same = [row for (first, second), row in pairs.items() if first == second]

    def best(values):
        values = [row for row in values if row["feasible"]]
        if not selecting or not values:
            return None
        return min(values, key=lambda row: (-row["gain"], row["config"]["lowtime"] != pivot, row["config"]["hightime"] != pivot, row["id"]))

    lowbest, highbest, samebest = best(low), best(high), best(same)
    independent = pairs.get((lowbest["config"]["lowtime"], highbest["config"]["hightime"])) if lowbest and highbest else None
    matrix = []
    for (first, second), row in pairs.items():
        left, right = pairs.get((first, pivot)), pairs.get((pivot, second))
        semantic, damage = None, None
        if origin and left and right:
            semantic = row["semantic"] - left["semantic"] - right["semantic"] + origin["semantic"]
            if all(value["mse"] is not None for value in (row, left, right, origin)):
                damage = row["mse"] - left["mse"] - right["mse"] + origin["mse"]
        matrix.append({**brief(row), "semanticinteraction": semantic, "damageinteraction": damage})
    return {"pivot": pivot, "lowcurve": [brief(row) for row in low], "highcurve": [brief(row) for row in high], "samecurve": [brief(row) for row in same], "lowchoice": brief(lowbest) if lowbest else None, "highchoice": brief(highbest) if highbest else None, "samechoice": brief(samebest) if samebest else None, "independent": brief(independent) if independent else None, "matrix": matrix, "warning": "Frequency change is not semantic correctness; conditional choices depend on the fixed opposite band."}

def controls(report, pivot):
    profile = profiles(report, pivot)
    pairs = {(row["config"]["lowtime"], row["config"]["hightime"]): row["id"] for row in report["table"] if row["config"].get("operator") == "bands"}
    ids = [report["selected"], pairs.get((pivot, pivot))]
    ids.extend(profile[key]["id"] for key in ("lowchoice", "highchoice", "samechoice", "independent") if profile[key] is not None)
    for (low, high), name in pairs.items():
        if name == report["selected"]:
            ids.extend((pairs.get((low, pivot)), pairs.get((pivot, high))))
    return list(dict.fromkeys(name for name in ids if name is not None and name != "native"))

def predictions(rows):
    # Count each image-seed-time diagnostic only once.
    def compatible(first, second):
        if isinstance(first, dict) and isinstance(second, dict):
            return set(first) == set(second) and all(compatible(first[key], second[key]) for key in first)
        if isinstance(first, float) and isinstance(second, float):
            return math.isclose(first, second, rel_tol=1e-4, abs_tol=1e-8)
        return first == second

    unique = {}
    for row in rows:
        for probe in row.get("probes", []):
            key = (row["key"], row["seed"], probe["time"])
            if key in unique and not compatible(unique[key], probe):
                raise ValueError("Repeated prediction diagnostics disagree.")
            unique.setdefault(key, probe)
    times = sorted({key[2] for key in unique})
    result = []
    for time in times:
        probes = [value for key, value in unique.items() if key[2] == time]
        entry = {"time": time, "pairs": len(probes), "space": "latent"}
        for band in ("low", "high"):
            values = {}
            for name in probes[0][band]:
                numbers = [probe[band][name] for probe in probes if probe[band][name] is not None]
                values[name] = statistics.fmean(numbers) if numbers else None
            entry[band] = values
        result.append(entry)
    return result

@torch.inference_mode()
def collect(editor, judge, items, configs, seeds, groups, local=0.5, completed=None):
    from engine.timing import Metric, synchronize
    completed = completed or set()
    for item in items:
        from engine.format import orient, read
        source = read(item.image)
        region = annotation(item)
        if region is None:
            raise ValueError("Band evaluation requires an edit mask.")
        region = region.resize(source.size, Image.Resampling.NEAREST)
        source, region = orient(source, region)
        original = ImageOps.fit(source, (editor.size, editor.size), method=Image.Resampling.LANCZOS)
        region = ImageOps.fit(region, original.size, method=Image.Resampling.NEAREST)
        metric = None
        for seed in seeds:
            pending = [name for name in configs if (item.key, seed, name) not in completed]
            if not pending:
                continue
            if metric is None:
                metric = Metric(judge, original, item.target, region, local)
                originalscore = metric(original)["semantic"]
            synchronize(editor.device)
            started = time.perf_counter()
            sampler = Sampler(editor, source, item.source, item.target, seed, validate(configs[pending[0]]).noise)
            sampler.base(validate(configs["native"]))
            warmcalls, warmsamples = editor.calls, editor.samples
            synchronize(editor.device)
            preparation = time.perf_counter() - started
            diagnostics = {}
            for index, name in enumerate(pending):
                synchronize(editor.device)
                started = time.perf_counter()
                result = sampler.render(configs[name])
                if index == 0:
                    result.report["calls"] += warmcalls
                    result.report["samples"] += warmsamples
                synchronize(editor.device)
                elapsed = time.perf_counter() - started + (preparation if index == 0 else 0.0)
                synchronize(judge.device)
                started = time.perf_counter()
                scores = metric(result.images[0])
                final = spectrum(original, result.images[0], region)
                view = describe(configs[name])
                times = [view["lowtime"], view["hightime"]] if view["operator"] == "bands" else [view["start"], view["second"]]
                probes = []
                for point in dict.fromkeys(times):
                    if point not in diagnostics:
                        diagnostics[point] = sampler.diagnose(point, region)
                    probes.append(diagnostics[point])
                synchronize(judge.device)
                synchronize(editor.device)
                scoring = time.perf_counter() - started
                yield {"key": item.key, "group": groups[item.key], "category": item.category, "seed": seed, "config": name, **scores, "original": originalscore, "editgain": scores["semantic"] - originalscore, "seconds": elapsed, "scoreseconds": scoring, **result.report, "frequency": final, "probes": probes}

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
