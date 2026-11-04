# -*- coding: utf-8 -*-
# RefUnmix: Separating Prompt Response from Shared Transport in One-Step Image Editing

from numbers import Integral
import torch
from engine.session import Editor as Base, number

class Editor(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxtime = 1000
        self.alphas = None
        self.kind = "velocity"
        self.metadata.update({"family": "linear", "architecture": "instaflow", "native": {"prediction": "noise-to-data velocity", "bridge": "(1 - sigma) * anchor + sigma * noise", "clock": "1000 * sigma", "recovery": "state + sigma * velocity", "steps": 1, "cfg": False}})

    def sigma(self, tick):
        if not isinstance(tick, Integral) or isinstance(tick, bool) or not 0 <= tick <= self.maxtime:
            raise ValueError("InstaFlow ticks must be integers from 0 to 1000; use index(sigma).")
        return int(tick) / self.maxtime

    def inputs(self, state, embedding):
        if not isinstance(state, torch.Tensor) or state.ndim != 4 or min(state.shape) < 1 or not state.is_floating_point() or not torch.isfinite(state).all():
            raise ValueError("InstaFlow states require finite nonempty NCHW tensors.")
        if not isinstance(embedding, torch.Tensor) or embedding.ndim != 3 or min(embedding.shape) < 1 or not embedding.is_floating_point() or not torch.isfinite(embedding).all():
            raise ValueError("InstaFlow conditions require finite nonempty token batches.")
        if state.shape[0] != embedding.shape[0] or state.device != self.device or embedding.device != self.device:
            raise ValueError("InstaFlow state and condition batches must share the model device.")

    def recover(self, sample, estimate, alpha, sigma):
        raise ValueError("InstaFlow uses state() for velocity recovery, not VP noise recovery.")

    def velocity(self, state, tick, embedding):
        # The official model clock decreases from 1000 to zero.
        self.inputs(state, embedding)
        self.sigma(tick)
        ticks = torch.full((state.shape[0],), float(tick), device=self.device, dtype=torch.float32)
        result = self.unet(sample=state.to(self.dtype), timestep=ticks, encoder_hidden_states=embedding.to(self.dtype), return_dict=False)[0]
        if not isinstance(result, torch.Tensor) or result.shape != state.shape or not result.is_floating_point() or not torch.isfinite(result).all():
            raise RuntimeError("InstaFlow returned an invalid velocity batch.")
        return result.float()

    def state(self, sample, tick, embedding):
        # Positive velocity advances the noise-to-data ODE.
        self.inputs(sample, embedding)
        sigma = self.sigma(tick)
        result = sample.float() + sigma * self.velocity(sample, tick, embedding)
        if not torch.isfinite(result).all():
            raise RuntimeError("InstaFlow clean predictions must remain finite.")
        return result

    def predict(self, anchor, noise, tick, embedding, keys=0., values=0., layout=2):
        if number(keys, "key strength", 0) or number(values, "value strength", 0):
            raise ValueError("InstaFlow does not implement VP reference-attention hooks.")
        self.inputs(anchor, embedding)
        if not isinstance(noise, torch.Tensor) or noise.shape != anchor.shape or noise.device != anchor.device or not noise.is_floating_point() or not torch.isfinite(noise).all():
            raise ValueError("InstaFlow noise must match the finite anchor batch.")
        sigma = self.sigma(tick)
        sample = (1 - sigma) * anchor.float() + sigma * noise.float()
        return self.state(sample, tick, embedding)

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
