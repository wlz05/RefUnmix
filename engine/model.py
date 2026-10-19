# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from PIL import Image, ImageOps
from data import clean, locate
from engine.frequency import FrequencyRouter
from engine.feedback import FeedbackConfig, FeedbackRouter
from engine.router import RouterConfig, ThreadRouter

@dataclass(frozen=True)
class EditConfig:
    method: str = "thread"
    noise: int = 1
    start: float = 0.90
    end: float = 0.30
    delta: float = 0.15
    scale: float = 1.0
    cleanup: bool = True
    mode: str = "surprisal"
    support: float = 0.50
    floor: float = 0.10
    gain: float = 1.0
    cap: float = 1.50
    prox: float = 0.50
    pivot: Optional[float] = None
    signal: Optional[float] = None
    rough: Optional[float] = None
    reach: int = 8
    band: int = 3
    feather: int = 2
    path: float = 1.0
    grid: float = 0.0
    relay: float = 1.0
    route: bool = True
    edge: bool = False
    scalar: bool = False
    preserve: float = 0.50
    couple: float = 0.25

    def __post_init__(self) -> None:
        if isinstance(self.method, str):
            object.__setattr__(self, "method", self.method.strip().casefold())

    def router(self) -> RouterConfig:
        return RouterConfig(
            mode=self.mode,
            support=self.support,
            floor=self.floor,
            gain=self.gain,
            cap=self.cap,
            prox=self.prox,
            pivot=self.pivot,
            signal=self.signal,
            rough=self.rough,
            reach=self.reach,
            band=self.band,
            feather=self.feather,
            path=self.path,
            grid=self.grid,
            relay=self.relay,
            route=self.route,
            edge=self.edge,
            scalar=self.scalar,
        )

    def validate(self) -> None:
        if self.method not in {"thread", "base", "frequency", "feedback"}:
            raise ValueError("method must be thread, base, frequency, or feedback.")
        if not isinstance(self.noise, Integral) or isinstance(self.noise, bool) or self.noise < 1:
            raise ValueError("noise must be a positive integer.")
        values = (self.start, self.end, self.delta, self.scale)
        try:
            valid = all(isinstance(value, Real) and not isinstance(value, bool) and isfinite(float(value)) for value in values)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("Edit values must be finite numbers.")
        if not 0.0 < self.start <= 1.0 or not 0.0 <= self.end <= 1.0:
            raise ValueError("start must be in (0, 1] and refinement time in [0, 1].")
        if not 0.0 <= self.delta < self.start:
            raise ValueError("delta must be smaller than start.")
        if self.scale < 0.0:
            raise ValueError("scale must be nonnegative.")
        if any(not isinstance(value, Real) or isinstance(value, bool) or not isfinite(float(value)) or value < 0.0 for value in (self.preserve, self.couple)):
            raise ValueError("Frequency penalties must be nonnegative finite numbers.")
        if not isinstance(self.cleanup, bool):
            raise TypeError("cleanup must be boolean.")
        self.router().validate()

@dataclass
class Probe:
    fields: torch.Tensor
    current: torch.Tensor
    previous: torch.Tensor
    target: torch.Tensor
    reference: torch.Tensor
    targetlow: torch.Tensor
    referencelow: torch.Tensor

@dataclass
class Result:
    images: list[Image.Image]
    latent: torch.Tensor
    maps: dict[str, torch.Tensor]
    report: Optional[dict] = None

class Editor:
    def __init__(self, unet, scheduler, vae, tokenizer, text, device: str, dtype: torch.dtype, size: int = 512) -> None:
        # Validate model metadata before inference.
        if not isinstance(size, Integral) or isinstance(size, bool) or size < 8 or size % 8:
            raise ValueError("size must be a positive multiple of eight.")
        self.unet = unet.to(device=device, dtype=dtype).eval()
        self.scheduler = scheduler
        self.vae = vae.to(device=device, dtype=dtype).eval()
        self.tokenizer = tokenizer
        self.text = text.to(device=device, dtype=dtype).eval()
        self.device = torch.device(device)
        self.dtype = dtype
        self.size = int(size)
        steps = getattr(getattr(scheduler, "config", None), "num_train_timesteps", 0)
        if not isinstance(steps, Integral) or isinstance(steps, bool):
            raise ValueError("Scheduler timesteps are invalid.")
        self.maxtime = int(steps) - 1
        if self.maxtime < 0 or not hasattr(scheduler, "alphas_cumprod") or len(scheduler.alphas_cumprod) <= self.maxtime:
            raise ValueError("Scheduler timesteps are invalid.")
        self.alphas = scheduler.alphas_cumprod.to(self.device, torch.float32)
        if not torch.isfinite(self.alphas).all().item() or torch.any((self.alphas <= 0.0) | (self.alphas > 1.0)).item():
            raise ValueError("Scheduler coefficients must be finite and in (0, 1].")
        self.factor = float(getattr(self.vae.config, "scaling_factor", 1.0))
        if not isfinite(self.factor) or self.factor <= 0.0:
            raise ValueError("VAE scaling factor must be finite and positive.")
        self.kind = str(getattr(scheduler.config, "prediction_type", "epsilon")).strip().casefold().replace("_", "")
        if self.kind not in {"epsilon", "sample", "vprediction"}:
            raise ValueError("Scheduler prediction type is unsupported.")
        limit = getattr(tokenizer, "model_max_length", 77)
        textlimit = getattr(getattr(text, "config", None), "max_position_embeddings", limit)
        if not isinstance(limit, Integral) or not isinstance(textlimit, Integral) or limit < 1 or textlimit < 1:
            raise ValueError("Tokenizer length is invalid.")
        self.tokens = min(int(limit), int(textlimit), 512)

    @classmethod
    def load(cls, root: str = "./models", device: Optional[str] = None, precision: str = "fp32", size: int = 512) -> "Editor":
        # Load local model components with validated settings.
        precision = precision.strip().casefold() if isinstance(precision, str) else ""
        types = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        if precision not in types:
            raise ValueError("precision must be fp32, fp16, or bf16.")
        path = locate(Path("."), clean(root, "model"))
        dtype = types[precision]
        target = torch.device(str(device).strip().casefold() if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        if target.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available.")
        if target.type == "cuda" and target.index is not None and target.index >= torch.cuda.device_count():
            raise ValueError("CUDA device index is unavailable.")
        if target.type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise ValueError("The selected CUDA device does not support bf16.")
        if target.type == "cpu" and dtype != torch.float32:
            raise ValueError("CPU inference requires fp32 precision.")
        parts = {name: locate(path, Path(name)) for name in ("unet", "scheduler", "vae", "tokenizer", "text")}
        if not parts["text"].is_dir():
            parts["text"] = locate(path, Path("text_encoder"))
        missing = [str(value) for value in parts.values() if not value.is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing model folders: {missing}")
        try:
            from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
            from transformers import AutoTokenizer, CLIPTextModel
        except ImportError as error:
            raise RuntimeError("Model loading requires diffusers and transformers.") from error
        return cls(UNet2DConditionModel.from_pretrained(parts["unet"], torch_dtype=dtype, local_files_only=True), DDPMScheduler.from_pretrained(parts["scheduler"], local_files_only=True), AutoencoderKL.from_pretrained(parts["vae"], torch_dtype=dtype, local_files_only=True), AutoTokenizer.from_pretrained(parts["tokenizer"], local_files_only=True), CLIPTextModel.from_pretrained(parts["text"], torch_dtype=dtype, local_files_only=True), target, dtype, size)

    def image(self, source: Image.Image) -> torch.Tensor:
        from engine.format import fit
        source = fit(source, (self.size, self.size))
        array = np.asarray(source, dtype=np.float32).copy() / 127.5 - 1.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device, self.dtype)
        return self.vae.encode(tensor).latent_dist.mode() * self.factor

    def prompt(self, value: str) -> torch.Tensor:
        tokens = self.tokenizer([value], padding="max_length", truncation=True, max_length=self.tokens, return_tensors="pt")
        hidden = self.text(input_ids=tokens.input_ids.to(self.device))[0]
        return hidden.to(self.device, self.dtype)

    def tick(self, value: float, batch: int) -> torch.Tensor:
        if not isinstance(value, Real) or isinstance(value, bool) or not isfinite(float(value)) or not isinstance(batch, Integral) or isinstance(batch, bool) or batch < 1:
            raise ValueError("time must be finite and batch must be positive.")
        index = max(0, min(self.maxtime, round(self.maxtime * value)))
        return torch.full((batch,), index, device=self.device, dtype=torch.long)

    def schedule(self, tensor: torch.Tensor, tick: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.alphas[tick]
        alpha = values.sqrt().view(-1, 1, 1, 1).to(tensor.dtype).clamp_min(torch.finfo(tensor.dtype).tiny)
        sigma = (1.0 - values).sqrt().view(-1, 1, 1, 1).to(tensor.dtype)
        return alpha, sigma

    def recover(self, sample: torch.Tensor, estimate: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # Recover clean latents for each prediction type.
        if estimate.shape != sample.shape:
            raise RuntimeError("Model output shape does not match its input.")
        sample32 = sample.float()
        estimate32 = estimate.float()
        alpha32 = alpha.float()
        sigma32 = sigma.float()
        if self.kind == "epsilon":
            clean = (sample32 - sigma32 * estimate32) / alpha32
        elif self.kind == "sample":
            clean = estimate32
        else:
            clean = alpha32 * sample32 - sigma32 * estimate32
        return clean.to(sample.dtype)

    def predict(self, anchor: torch.Tensor, noise: torch.Tensor, time: float, embed: torch.Tensor) -> torch.Tensor:
        tick = self.tick(time, anchor.shape[0])
        alpha, sigma = self.schedule(anchor, tick)
        sample = alpha * anchor + sigma * noise
        self.calls = getattr(self, "calls", 0) + 1
        self.samples = getattr(self, "samples", 0) + sample.shape[0]
        estimate = self.unet(sample=sample, timestep=tick, encoder_hidden_states=embed, return_dict=False)[0]
        return self.recover(sample, estimate, alpha, sigma)

    def probe(self, anchor: torch.Tensor, source: torch.Tensor, target: torch.Tensor, noises: list[torch.Tensor], time: float, delta: float) -> Probe:
        # Build fields and predictions with shared noise.
        batch = anchor.shape[0]
        high = self.tick(time, batch)
        low = self.tick(time - delta, batch)
        ahigh, shigh = self.schedule(anchor, high)
        alow, slow = self.schedule(anchor, low)
        samples: list[torch.Tensor] = []
        embeds: list[torch.Tensor] = []
        ticks: list[torch.Tensor] = []
        alphas: list[torch.Tensor] = []
        sigmas: list[torch.Tensor] = []
        shared = torch.equal(high, low)
        for noise in noises:
            first = ahigh * anchor + shigh * noise
            second = alow * anchor + slow * noise
            samples.extend((first, first) if shared else (first, first, second, second))
            embeds.extend((source, target) if shared else (source, target, source, target))
            ticks.extend((high, high) if shared else (high, high, low, low))
            alphas.extend((ahigh, ahigh) if shared else (ahigh, ahigh, alow, alow))
            sigmas.extend((shigh, shigh) if shared else (shigh, shigh, slow, slow))
        joined = torch.cat(samples)
        self.calls = getattr(self, "calls", 0) + 1
        self.samples = getattr(self, "samples", 0) + joined.shape[0]
        estimate = self.unet(sample=joined, timestep=torch.cat(ticks), encoder_hidden_states=torch.cat(embeds), return_dict=False)[0]
        clean = self.recover(joined, estimate, torch.cat(alphas), torch.cat(sigmas)).reshape(len(noises), 2 if shared else 4, *anchor.shape)
        if shared:
            sourcehigh, targethigh = clean.unbind(dim=1)
            sourcelow, targetlow = sourcehigh, targethigh
        else:
            sourcehigh, targethigh, sourcelow, targetlow = clean.unbind(dim=1)
        current = targethigh - sourcehigh
        previous = targetlow - sourcelow
        fields = (delta * current + time * previous) / (time + delta)
        return Probe(fields=fields, current=current, previous=previous, target=targethigh, reference=sourcehigh, targetlow=targetlow, referencelow=sourcelow)

    def field(self, anchor: torch.Tensor, source: torch.Tensor, target: torch.Tensor, noises: list[torch.Tensor], config: EditConfig, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        probe = self.probe(anchor, source, target, noises, config.start, config.delta)
        base = probe.fields.mean(dim=0)
        if config.method == "base":
            return base, {"base": base, "field": base}
        if config.method in {"frequency", "feedback"}:
            router = FrequencyRouter(config.preserve, config.couple, config.cap)
            routed = router(anchor, probe.current.mean(dim=0), probe.previous.mean(dim=0), probe.reference.mean(dim=0), probe.referencelow.mean(dim=0), config.start, config.start - config.delta, mask)
            maps = dict(routed.maps)
            maps["base"] = base
            maps["field"] = routed.field
            maps["target"] = routed.target
            return routed.field, maps
        router = ThreadRouter(config.router())
        routed = router(
            base,
            probe.current.mean(dim=0),
            probe.previous.mean(dim=0),
            probe.target.mean(dim=0),
            probe.reference.mean(dim=0),
            anchor,
        )
        maps = dict(routed.maps)
        maps["base"] = base
        maps["field"] = routed.field
        maps["mask"] = routed.mask
        return routed.field, maps

    def flow(self, anchor: torch.Tensor, source: torch.Tensor, target: torch.Tensor, noise: torch.Tensor, config: EditConfig) -> torch.Tensor:
        return self.probe(anchor, source, target, [noise], config.start, config.delta).fields.mean(dim=0)

    @torch.no_grad()
    def reference(self, anchor: torch.Tensor, source: torch.Tensor, target: torch.Tensor, noise: torch.Tensor, config: EditConfig, routed: bool = False) -> dict[str, torch.Tensor]:
        if not isinstance(routed, bool):
            raise TypeError("routed must be boolean.")

        def sample(point: torch.Tensor) -> torch.Tensor:
            return self.field(point, source, target, [noise], config)[0] if routed else self.flow(point, source, target, noise, config)

        field = sample(anchor)
        full = anchor + config.scale * field
        half = anchor + 0.5 * config.scale * field
        half = half + 0.5 * config.scale * sample(half)
        quarter = anchor + 0.25 * config.scale * field
        for _ in range(3):
            quarter = quarter + 0.25 * config.scale * sample(quarter)
        trial = anchor + config.scale * field
        heun = anchor + 0.5 * config.scale * (field + sample(trial))
        twoerror = self.mapnorm(full - half)
        fullerror = self.mapnorm(full - quarter)
        halferror = self.mapnorm(half - quarter)
        heunerror = self.mapnorm(heun - quarter)
        heunstep = self.mapnorm(full - heun)
        ratio = halferror / (twoerror + config.router().epsilon)
        return {"full": full, "half": half, "quarter": quarter, "heun": heun, "error": fullerror, "twoerror": twoerror, "halferror": halferror, "heunerror": heunerror, "heunstep": heunstep, "ratio": ratio}

    @torch.no_grad()
    def direction(self, anchor: torch.Tensor, source: torch.Tensor, target: torch.Tensor, noise: torch.Tensor, config: EditConfig, epsilon: float = 1e-3) -> torch.Tensor:
        if not isinstance(epsilon, Real) or isinstance(epsilon, bool) or not isfinite(float(epsilon)) or epsilon <= 0.0:
            raise ValueError("epsilon must be positive and finite.")
        field = self.flow(anchor, source, target, noise, config)
        baselevel = field.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        anchorlevel = anchor.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1.0)
        active = baselevel > config.router().epsilon
        step = torch.where(active, float(epsilon) * anchorlevel / baselevel.clamp_min(config.router().epsilon), torch.ones_like(baselevel))
        moved = torch.where(active.expand_as(anchor), anchor + step * field, anchor)
        derivative = (self.flow(moved, source, target, noise, config) - field) / step
        derivative = torch.where(active.expand_as(derivative), derivative, torch.zeros_like(derivative))
        return self.mapnorm(derivative)

    @torch.no_grad()
    def timesensitivity(self, anchor: torch.Tensor, source: torch.Tensor, target: torch.Tensor, noise: torch.Tensor, config: EditConfig, eta: float = 1e-2) -> torch.Tensor:
        if not isinstance(eta, Real) or isinstance(eta, bool) or not isfinite(float(eta)) or eta <= 0.0:
            raise ValueError("eta must be positive and finite.")
        if config.start - eta <= config.delta or config.start + eta > 1.0:
            raise ValueError("eta exceeds the central probe range.")
        plus = self.probe(anchor, source, target, [noise], config.start + eta, config.delta).fields.mean(dim=0)
        minus = self.probe(anchor, source, target, [noise], config.start - eta, config.delta).fields.mean(dim=0)
        return self.mapnorm((plus - minus) / (2.0 * eta))

    def mapnorm(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(tensor.float().square().sum(dim=1, keepdim=True).clamp_min(0.0))

    def decode(self, latent: torch.Tensor) -> list[Image.Image]:
        if not torch.isfinite(latent).all().item():
            raise ValueError("latent must contain finite values.")
        pixels = self.vae.decode(latent / self.factor).sample.float().clamp(-1.0, 1.0).add(1.0).div(2.0)
        arrays = pixels.mul(255.0).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(array) for array in arrays]

    @torch.no_grad()
    def edit(self, image: Image.Image, source: str, target: str, config: Optional[EditConfig] = None, seed: int = 42, region: Optional[Image.Image] = None, polarity: str = "edit", judge=None, feedback: Optional[FeedbackConfig] = None, localtext: Optional[str] = None) -> Result:
        # Apply the routed edit and optional refinement.
        config = config or EditConfig()
        config.validate()
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL image.")
        if not isinstance(source, str) or not isinstance(target, str) or not source.strip() or not target.strip():
            raise ValueError("source and target prompts must be nonempty strings.")
        if not isinstance(seed, Integral) or isinstance(seed, bool) or not -(2 ** 63) <= seed < 2 ** 64:
            raise ValueError("seed is outside the supported integer range.")
        source = source.strip()
        target = target.strip()
        polarity = polarity.strip().casefold()
        if polarity not in {"edit", "protect"}:
            raise ValueError("polarity must be edit or protect.")
        if polarity == "protect" and region is None:
            raise ValueError("Protect polarity requires an explicit region mask.")
        if region is not None and (not isinstance(region, Image.Image) or region.size != image.size):
            raise ValueError("region must be a mask with the source image size.")
        if region is not None and config.method not in {"frequency", "feedback"}:
            raise ValueError("Explicit regions require frequency or feedback mode.")
        if config.method == "feedback" and judge is None:
            raise ValueError("Feedback mode requires a semantic judge.")
        self.calls, self.samples = 0, 0
        from engine.format import orient
        image, region = orient(image, region)
        anchor = self.image(image)
        mask = None
        if region is not None:
            resized = ImageOps.fit(region.convert("L"), (self.size, self.size), method=Image.Resampling.BILINEAR)
            mask = torch.from_numpy(np.asarray(resized, dtype=np.float32).copy() / 255.0).reshape(1, 1, self.size, self.size).to(self.device)
            mask = 1.0 - mask if polarity == "protect" else mask
            mask = torch.nn.functional.interpolate(mask, anchor.shape[-2:], mode="area")
        if config.method == "feedback" and source.casefold() == target.casefold():
            reference = ImageOps.fit(image.convert("RGB"), (self.size, self.size), method=Image.Resampling.LANCZOS)
            zero = torch.zeros_like(anchor).cpu()
            return Result([reference], anchor.detach().cpu(), {"base": zero, "field": zero}, {"selected": "source", "accepted": False, "reason": "identical prompts", "calls": 0, "samples": 0, "evaluations": 0})
        sourceembed = self.prompt(source)
        targetembed = self.prompt(target)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        noises = [torch.randn(anchor.shape, generator=generator, device=self.device, dtype=self.dtype) for _ in range(config.noise)]
        field, maps = self.field(anchor, sourceembed, targetembed, noises, config, mask)
        if not torch.isfinite(field).all().item():
            raise RuntimeError("The edit field contains nonfinite values.")
        if config.method == "feedback":
            reference = ImageOps.fit(image.convert("RGB"), (self.size, self.size), method=Image.Resampling.LANCZOS)
            objective = judge.prepare(reference, source, target, maps["mask"], localtext)

            def render(candidate):
                value = anchor + config.scale * candidate.to(anchor.dtype)
                if config.cleanup:
                    value = self.predict(value, noises[0], config.end, targetembed)
                return value, self.decode(value)[0]

            latent, edited, field, report = FeedbackRouter(feedback or FeedbackConfig(), config.cap).run(render, anchor, reference, maps, objective)
            maps["field"] = field
            for key in ("lowweight", "highweight", "lowtime", "hightime"):
                maps["initial" + key] = maps.pop(key)
            if report["weights"] is not None:
                for key, value in zip(("lowweight", "highweight", "lowtime", "hightime"), report["weights"] + report["times"]):
                    maps[key] = torch.tensor(value)
            maps.update({key: torch.tensor(value) for key, value in report["audit"].items()})
            report.update({"calls": self.calls, "samples": self.samples, "polarity": polarity})
            return Result([edited], latent.detach().cpu(), {key: value.detach().cpu() for key, value in maps.items()}, report)
        latent = anchor + config.scale * field
        if config.method == "frequency":
            router = FrequencyRouter(config.preserve, config.couple, config.cap)
            maps.update({key + "pre": value for key, value in router.audit(anchor, latent, maps["target"], maps["mask"]).items()})
        if config.cleanup:
            refined = self.predict(latent, noises[0], config.end, targetembed)
            latent = maps["mask"] * refined + (1.0 - maps["mask"]) * anchor if config.method == "thread" else refined
        if config.method == "frequency":
            maps.update(router.audit(anchor, latent, maps["target"], maps["mask"]))
        return Result(self.decode(latent), latent.detach().cpu(), {key: value.detach().cpu() for key, value in maps.items()}, {"calls": self.calls, "samples": self.samples})

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
