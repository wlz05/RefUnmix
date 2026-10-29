# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import math
import torch
from engine.session import tensor

class Prior:
    def __init__(self, session, probes=2, seed=42, mode="ddpm", probabilities=None, variance=None):
        if type(probes) is not int or probes < 1 or mode not in {"ddpm", "rec", "flow", "off"}:
            raise ValueError("Invalid consistency mode or probe count.")
        self.session, self.probes, self.mode = session, probes, mode
        self.evaluations = 0
        self.seed = int(seed)
        self.ticks, self.noises, self.weights = [], [], []
        self.probabilities = []
        if mode == "flow":
            if getattr(session.editor, "metadata", {}).get("family") != "flow" or variance is not None:
                raise ValueError("Flow consistency requires a native flow editor without DDPM variance.")
            from engine.flow import Consistency
            self.flow = Consistency(session, probes, seed, probabilities)
            self.ticks, self.noises, self.weights, self.probabilities = self.flow.ticks, self.flow.noises, self.flow.weights, self.flow.probabilities
            return
        self.variance = variance or getattr(session.editor.scheduler.config, "variance_type", "fixed_small")
        if mode == "off":
            return
        if getattr(session.editor, "metadata", {}).get("family") == "flow":
            raise ValueError("VP consistency cannot query a native flow editor.")
        alphas = session.editor.alphas.detach().double().cpu()
        if alphas.ndim != 1 or len(alphas) < 2 or not torch.isfinite(alphas).all().item() or torch.any(alphas <= 0).item() or torch.any(alphas > 1).item() or torch.any(alphas[1:] >= alphas[:-1]).item() or torch.any(alphas[1:] >= 1).item():
            raise ValueError("Consistency requires decreasing positive VP coefficients.")
        probability = torch.ones(len(alphas) - 1, dtype=torch.float64) if probabilities is None else torch.as_tensor(probabilities, dtype=torch.float64, device="cpu").detach().clone()
        if probability.shape != (len(alphas) - 1,) or not torch.isfinite(probability).all().item() or torch.any(probability <= 0).item():
            raise ValueError("Prior probabilities must cover every interior tick.")
        probability /= probability.max()
        probability /= probability.sum()
        if torch.any(probability <= 0).item():
            raise ValueError("Prior probabilities underflow after normalization.")
        self.probabilities = probability.tolist()
        chooser = torch.Generator(device="cpu").manual_seed(self.seed)
        self.ticks = (torch.multinomial(probability, probes, replacement=True, generator=chooser) + 1).tolist()
        generator = torch.Generator(device=session.anchor.device).manual_seed(self.seed ^ 0x531)
        self.noises = [torch.randn(session.anchor.shape, generator=generator, device=session.anchor.device, dtype=torch.float32) for _ in self.ticks]
        if mode == "ddpm" and self.variance not in {"fixed_small", "fixed_large"}:
            raise ValueError("DDPM consistency requires a fixed reverse variance.")
        if mode == "ddpm" and self.variance == "fixed_small" and float(alphas[0]) >= 1:
            raise ValueError("Zero reverse variance excludes the DDPM prior.")
        for tick in self.ticks:
            cumulative, previous = float(alphas[tick]), float(alphas[tick - 1])
            step = cumulative / previous
            beta = 1 - step
            variancevalue = beta * (1 - previous) / (1 - cumulative) if self.variance == "fixed_small" else beta
            if mode == "ddpm" and variancevalue <= 0:
                raise ValueError("Reverse variances must be strictly positive.")
            weight = beta ** 2 / (2 * variancevalue * step * (1 - cumulative)) / float(probability[tick - 1]) if mode == "ddpm" else 1.
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("Consistency weights must be finite and positive.")
            self.weights.append(weight)

    def cost(self):
        count = self.probes if self.mode != "off" else 0
        return count, count

    @torch.no_grad()
    def __call__(self, latent, phase="prior"):
        # Reuse fixed probes across candidate comparisons.
        tensor(latent)
        session = self.session
        if latent.shape != session.anchor.shape or latent.device != session.anchor.device:
            raise ValueError("Consistency latents must match the source latent.")
        if self.mode == "off":
            return None
        if self.mode == "flow":
            result = self.flow(latent, phase)
            self.evaluations = self.flow.evaluations
            return result
        previousphase = session.limit.phase if session.limit is not None else None
        if session.limit is not None:
            session.limit.require(self.probes, self.probes)
            session.limit.phase = phase
        values = []
        try:
            for tick, noise, weight in zip(self.ticks, self.noises, self.weights):
                prediction = session.predict(latent, noise, tick, session.target)
                cumulative = session.editor.alphas[tick].double()
                error = (prediction.double() - latent.double()).square().mean() * cumulative / (1 - cumulative)
                values.append(weight * error)
            result = float(torch.stack(values).mean())
            if not math.isfinite(result):
                raise RuntimeError("Consistency produced a nonfinite value.")
            self.evaluations += 1
            return result
        finally:
            if session.limit is not None:
                session.limit.phase = previousphase

    def metadata(self):
        if self.mode == "flow":
            return self.flow.metadata()
        return {"kind": self.mode, "probes": self.probes if self.mode != "off" else 0, "ticks": self.ticks, "seed": self.seed, "weights": self.weights, "probabilities": self.probabilities, "variance": self.variance, "evaluations": self.evaluations, "note": "Finite denoising consistency is not likelihood or quality."}

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
