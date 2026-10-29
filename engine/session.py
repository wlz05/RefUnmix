# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
from math import isfinite
from numbers import Integral, Real
from pathlib import Path, PureWindowsPath
from time import perf_counter
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageOps
from engine.reference import center, decomposition
from engine.format import orient as alignment

def number(value, label, minimum=None, maximum=None):
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{label} must be finite.")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} exceeds numerical range.") from error
    if not isfinite(result):
        raise ValueError(f"{label} must be finite.")
    if minimum is not None and result < minimum or maximum is not None and result > maximum:
        raise ValueError(f"{label} is outside its supported range.")
    return result

def relative(value):
    text = str(value).strip().replace("\\", "/")
    path = Path(text)
    if not text or path.is_absolute() or PureWindowsPath(text).drive or ".." in path.parts:
        raise ValueError("Model paths must be local and relative.")
    try:
        path.resolve().relative_to(Path.cwd().resolve())
    except ValueError as error:
        raise ValueError("Model paths must stay inside the working directory.") from error
    return path

def tensor(value, channels=None):
    if not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[0] != 1 or min(value.shape) < 1:
        raise ValueError("Expected a nonempty tensor with shape 1CHW.")
    if channels is not None and value.shape[1] != channels:
        raise ValueError("Tensor channel count is invalid.")
    if not value.is_floating_point() or not torch.isfinite(value).all().item():
        raise ValueError("Tensor values must be finite floating point numbers.")
    return value

def rms(value):
    # Keep diagnostics stable across latent scales.
    value = tensor(value).double()
    scale = float(value.abs().max())
    return scale * float((value / scale).square().mean().sqrt()) if scale > 0 else 0.

def toimage(pixels):
    values = tensor(pixels, 3).detach().float().clamp(0, 1)
    array = values[0].mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)

def orient(image, mask=None):
    # Apply source orientation to images and masks together.
    return alignment(image, mask)

class Editor:
    def __init__(self, unet, scheduler, vae, tokenizer, text, device="cpu", dtype=torch.float32, size=512):
        if not isinstance(size, Integral) or isinstance(size, bool) or size < 8 or size % 8:
            raise ValueError("size must be a positive multiple of eight.")
        self.device, self.dtype, self.size = torch.device(device), dtype, int(size)
        self.metadata = {"family": "vp", "weights": None}
        self.textcalls = 0
        if dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            raise ValueError("Model dtype is unsupported.")
        self.unet = unet.to(device=self.device, dtype=dtype).eval()
        self.vae = vae.to(device=self.device, dtype=dtype).eval()
        self.text = text.to(device=self.device, dtype=dtype).eval()
        for model in (self.unet, self.vae, self.text):
            model.requires_grad_(False)
        self.scheduler, self.tokenizer = scheduler, tokenizer
        steps = getattr(scheduler.config, "num_train_timesteps", 0)
        if not isinstance(steps, Integral) or isinstance(steps, bool) or steps < 2:
            raise ValueError("Scheduler timesteps are invalid.")
        self.maxtime = int(steps) - 1
        self.alphas = torch.as_tensor(scheduler.alphas_cumprod, device=self.device, dtype=torch.float32)
        if self.alphas.ndim != 1 or len(self.alphas) != steps or not torch.isfinite(self.alphas).all().item() or torch.any((self.alphas <= 0) | (self.alphas > 1)).item():
            raise ValueError("Scheduler coefficients must be finite and in (0, 1].")
        self.factor = number(getattr(vae.config, "scaling_factor", 1.0), "VAE scale", 1e-12)
        self.kind = str(getattr(scheduler.config, "prediction_type", "epsilon")).strip().casefold().replace("_", "")
        if self.kind not in {"epsilon", "sample", "vprediction"}:
            raise ValueError("Scheduler prediction type is unsupported.")
        limit = getattr(tokenizer, "model_max_length", 77)
        textlimit = getattr(getattr(text, "config", None), "max_position_embeddings", limit)
        if any(not isinstance(item, Integral) or isinstance(item, bool) or item < 1 for item in (limit, textlimit)):
            raise ValueError("Tokenizer length is invalid.")
        self.tokens = min(int(limit), int(textlimit), 512)

    @classmethod
    def load(cls, root="./models", device=None, precision="fp32", size=512):
        # Load only local pretrained components.
        types = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        precision = str(precision).strip().casefold()
        if precision not in types:
            raise ValueError("precision must be fp32, fp16, or bf16.")
        selected = torch.device(str(device).strip().casefold() if device is not None else "cuda" if torch.cuda.is_available() else "cpu")
        if selected.type not in {"cpu", "cuda"}:
            raise ValueError("Only CPU and CUDA inference are supported.")
        if selected.type == "cuda" and (not torch.cuda.is_available() or selected.index is not None and selected.index >= torch.cuda.device_count()):
            raise ValueError("The selected CUDA device is unavailable.")
        if selected.type == "cpu" and precision != "fp32":
            raise ValueError("CPU inference requires fp32 precision.")
        if selected.type == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("The selected CUDA device does not support bf16.")
        root = relative(root)
        from data import locate
        parts = {name: locate(root, Path(name)) for name in ("unet", "scheduler", "vae", "tokenizer", "text")}
        if not parts["text"].is_dir():
            parts["text"] = locate(root, Path("text_encoder"))
        missing = [str(path) for path in parts.values() if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing model folders: {missing}")
        try:
            from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
            from transformers import AutoTokenizer, CLIPTextModel
        except ImportError as error:
            raise RuntimeError("Model loading requires diffusers and transformers.") from error
        options = {"torch_dtype": types[precision], "local_files_only": True}
        from engine.checkpoint import configuration, fingerprint
        result = cls(UNet2DConditionModel.from_pretrained(parts["unet"], **options), DDPMScheduler.from_pretrained(parts["scheduler"], local_files_only=True), AutoencoderKL.from_pretrained(parts["vae"], **options), AutoTokenizer.from_pretrained(parts["tokenizer"], local_files_only=True), CLIPTextModel.from_pretrained(parts["text"], **options), selected, types[precision], size)
        result.metadata.update({"weights": fingerprint(root), "configuration": configuration(root)})
        return result

    def index(self, time):
        return round(self.maxtime * number(time, "time", 0, 1))

    def prompt(self, value):
        self.textcalls += 1
        tokens = self.tokenizer([value], padding="max_length", truncation=True, max_length=self.tokens, return_tensors="pt")
        return self.text(input_ids=tokens.input_ids.to(self.device))[0].to(self.dtype)

    def encode(self, pixels):
        return self.vae.encode((pixels * 2 - 1).to(self.dtype)).latent_dist.mode() * self.factor

    def decode(self, latent):
        return self.vae.decode((latent / self.factor).to(self.dtype)).sample.float()

    def recover(self, sample, estimate, alpha, sigma):
        if estimate.shape != sample.shape:
            raise RuntimeError("Model output shape does not match its input.")
        sample, estimate, alpha, sigma = (value.float() for value in (sample, estimate, alpha, sigma))
        if self.kind == "epsilon":
            clean = (sample - sigma * estimate) / alpha
        elif self.kind == "sample":
            clean = estimate
        else:
            clean = alpha * sample - sigma * estimate
        if not torch.isfinite(clean).all().item():
            raise RuntimeError("Model prediction contains nonfinite values.")
        return clean

    def predict(self, anchor, noise, tick, embed):
        alpha = self.alphas[tick].sqrt()
        sigma = (1 - self.alphas[tick]).sqrt()
        sample = alpha * anchor.float() + sigma * noise.float()
        return self.state(sample, tick, embed)

    def state(self, sample, tick, embed):
        alpha = self.alphas[tick].sqrt()
        sigma = (1 - self.alphas[tick]).sqrt()
        original = sample.float()
        sample = sample.to(self.dtype)
        ticks = torch.full((sample.shape[0],), tick, device=self.device, dtype=torch.long)
        estimate = self.unet(sample=sample, timestep=ticks, encoder_hidden_states=embed, return_dict=False)[0]
        return self.recover(original, estimate, alpha, sigma)

    @torch.no_grad()
    def prepare(self, image, source, target, seed=42, noise=1, end=0.30, cleanup=True, scale=1.0, region=None, polarity="edit"):
        started = perf_counter()
        textbase = self.textcalls
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL image.")
        if any(not isinstance(value, str) or not value.strip() for value in (source, target)):
            raise ValueError("Prompts must be nonempty strings.")
        if not isinstance(seed, Integral) or isinstance(seed, bool) or not -(2 ** 63) <= seed < 2 ** 64:
            raise ValueError("seed is outside the supported integer range.")
        if not isinstance(noise, Integral) or isinstance(noise, bool) or noise < 1:
            raise ValueError("noise must be a positive integer.")
        end, scale = number(end, "end", 0, 1), number(scale, "scale", 0)
        if not isinstance(cleanup, bool):
            raise ValueError("cleanup must be boolean.")
        polarity = str(polarity).strip().casefold()
        if polarity not in {"edit", "protect"} or polarity == "protect" and region is None:
            raise ValueError("Mask polarity must be edit or protect with an explicit region.")
        if region is not None and (not isinstance(region, Image.Image) or region.size != image.size):
            raise ValueError("region must be a PIL mask with the source image size.")
        image, region = orient(image, region)
        resized = ImageOps.fit(image.convert("RGB"), (self.size, self.size), method=Image.Resampling.LANCZOS)
        origin = torch.from_numpy(np.asarray(resized, dtype=np.float32).copy()).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255
        anchor = self.encode(origin)
        tensor(anchor)
        mask = None
        area = None
        if region is not None:
            resized = ImageOps.fit(region.convert("L"), (self.size, self.size), method=Image.Resampling.BILINEAR)
            mask = torch.from_numpy(np.asarray(resized, dtype=np.float32).copy()).reshape(1, 1, self.size, self.size).to(self.device) / 255
            area = 1 - mask if polarity == "protect" else mask.clone()
            mask = functional.interpolate(mask, anchor.shape[-2:], mode="area")
            mask = 1 - mask if polarity == "protect" else mask
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        noises = torch.randn((int(noise), *anchor.shape[1:]), generator=generator, device=self.device, dtype=self.dtype)
        result = Session(self, anchor, origin, mask, self.prompt(source.strip()), self.prompt(target.strip()), noises, end, cleanup, scale)
        result.area = area
        result.started = started
        result.textbase = textbase
        return result

class Session:
    def __init__(self, editor, anchor, origin, mask, source, target, noises, end, cleanup, scale):
        self.editor, self.anchor, self.origin, self.mask = editor, anchor, origin, mask
        self.source, self.target, self.noises = source, target, noises
        self.end, self.cleanup, self.scale = end, cleanup, scale
        self.cache, self.calls, self.samples, self.decodes = {}, 0, 0, 0
        self.references, self.renders = {}, {"raw": 0, "latent": 0, "pixel": 0}
        self.collector, self.attentions, self.attentionlog = None, {}, {}
        self.started = perf_counter()
        self.limit = None
        self.area = None
        self.encodes = 1
        self.textbase = max(0, editor.textcalls - 2)
        self.artifacts = {}

    def localization(self, source, target, words=None, maximum=1024):
        from engine.attention import Collector, tokens
        if self.cache:
            raise ValueError("Enable attention capture before requesting any fields.")
        indices = tokens(self.editor.tokenizer, source, target, self.editor.tokens, words)
        self.collector = Collector(self.editor.unet, indices, self.anchor.shape[-2:], self.noises.shape[0], maximum)

    def attention(self):
        values = [value for value in self.attentions.values() if value is not None]
        return torch.stack(values).mean(0).to(self.anchor.device) if values else None

    def predict(self, anchor, noises, tick, embeddings):
        if self.limit is not None:
            self.limit.consume(anchor.shape[0])
        self.calls += 1
        self.samples += anchor.shape[0]
        return self.editor.predict(anchor, noises, tick, embeddings)

    def state(self, sample, tick, embeddings):
        if self.limit is not None:
            self.limit.consume(sample.shape[0])
        self.calls += 1
        self.samples += sample.shape[0]
        return self.editor.state(sample, tick, embeddings)

    def velocity(self, sample, tick, embeddings):
        if not callable(getattr(self.editor, "velocity", None)):
            raise ValueError("This editor has no native velocity interface.")
        if self.limit is not None:
            self.limit.consume(sample.shape[0])
        self.calls += 1
        self.samples += sample.shape[0]
        return self.editor.velocity(sample, tick, embeddings)

    @torch.no_grad()
    def difference(self, tick):
        # Share source anchors and noise across times.
        if tick not in self.cache:
            count = self.noises.shape[0]
            anchor = self.anchor.expand(2 * count, -1, -1, -1)
            noise = torch.cat((self.noises, self.noises))
            embeddings = torch.cat((self.source.expand(count, -1, -1), self.target.expand(count, -1, -1)))
            if self.collector is None:
                clean = self.predict(anchor, noise, tick, embeddings)
            else:
                with self.collector:
                    clean = self.predict(anchor, noise, tick, embeddings)
                self.attentions[tick] = self.collector.result()
                self.attentionlog[tick] = self.collector.report()
            self.cache[tick] = (clean[count:] - clean[:count]).mean(dim=0, keepdim=True)
        return self.cache[tick]

    @torch.no_grad()
    def field(self, time, delta=0.0):
        time, delta = number(time, "time", 0, 1), number(delta, "delta", 0, 1)
        if delta > 0 and delta >= time:
            raise ValueError("Positive delta must be smaller than time.")
        current = self.difference(self.editor.index(time))
        if delta == 0:
            return current.clone()
        previous = self.difference(self.editor.index(time - delta))
        return (delta * current + time * previous) / (time + delta)

    @torch.no_grad()
    def decode(self, latent):
        # Decode unclipped RGB for reference subtraction.
        tensor(latent)
        if self.limit is not None:
            self.limit.require()
        self.decodes += 1
        pixels = self.editor.decode(latent)
        tensor(pixels, 3)
        if pixels.shape != self.origin.shape:
            raise RuntimeError("Decoded image size differs from the source image.")
        return pixels.add(1).div(2)

    def enabled(self, cleanup):
        if cleanup is not None and not isinstance(cleanup, bool):
            raise ValueError("The cleanup override must be boolean.")
        return self.cleanup if cleanup is None else cleanup

    def neutral(self, enabled, pixels=False):
        # Cache the matched zero-field branch.
        key = self.editor.index(self.end) if enabled else None
        if key not in self.references:
            latent = self.predict(self.anchor.float(), self.noises[:1], key, self.target) if enabled else self.anchor.float()
            self.references[key] = {"latent": latent.detach().clone(), "pixels": None}
        entry = self.references[key]
        if pixels and entry["pixels"] is None:
            entry["pixels"] = self.decode(entry["latent"]).detach().clone()
        return entry

    @torch.no_grad()
    def render(self, field, cleanup=None, reference="raw"):
        # Render raw, latent-centered, or pixel-centered outputs.
        tensor(field)
        if field.shape != self.anchor.shape:
            raise ValueError("Edit fields must match the original latent shape.")
        reference = str(reference).strip().casefold()
        if reference not in self.renders:
            raise ValueError("Reference mode must be raw, latent, or pixel.")
        enabled = self.enabled(cleanup)
        latent = self.anchor.float() + self.scale * field.to(device=self.anchor.device, dtype=torch.float32)
        tensor(latent)
        if enabled:
            latent = self.predict(latent, self.noises[:1], self.editor.index(self.end), self.target)
        tensor(latent)
        if reference == "latent":
            latent = self.anchor.float() + (latent - self.neutral(enabled)["latent"])
        pixels = self.decode(latent)
        self.renders[reference] += 1
        if reference == "pixel":
            return center(self.origin, pixels, self.neutral(enabled, pixels=True)["pixels"])
        return pixels.clamp(0, 1)

    @torch.no_grad()
    def diagnostic(self, field, mask=None):
        # Measure paired drift without assuming editing errors.
        tensor(field)
        if field.shape != self.anchor.shape:
            raise ValueError("Diagnostic fields must match the source latent shape.")
        before = {key: getattr(self, key) for key in ("calls", "samples", "decodes")}
        latent = self.anchor.float() + self.scale * field.to(device=self.anchor.device, dtype=torch.float32)
        tensor(latent)
        if self.cleanup:
            latent = self.predict(latent, self.noises[:1], self.editor.index(self.end), self.target)
        edited = self.decode(latent)
        neutral = self.neutral(self.cleanup, pixels=True)["pixels"]
        report = decomposition(self.origin, edited, neutral, mask)
        reconstruction = self.neutral(False, pixels=True)["pixels"]
        report.update({"reconstruction": float((reconstruction.double() - self.origin.double()).square().mean()), "refinementdrift": float((neutral.double() - reconstruction.double()).square().mean()), "zeropixelerror": float((center(self.origin, neutral, neutral) - self.origin).abs().max()), "seedmatched": True, "fieldscale": self.scale, "tick": self.editor.index(self.end) if self.cleanup else None, "cost": {key: getattr(self, key) - before[key] for key in before}})
        return report

    def stats(self):
        return {"calls": self.calls, "samples": self.samples, "decodes": self.decodes, "encodes": self.encodes, "textcalls": self.editor.textcalls - self.textbase, "ticks": sorted(self.cache), "elapsed": perf_counter() - self.started, "attention": self.attentionlog, "renders": dict(self.renders), "references": [{"cleanup": key is not None, "tick": key, "decoded": entry["pixels"] is not None} for key, entry in self.references.items()], "budget": self.limit.report() if self.limit is not None else None}

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
