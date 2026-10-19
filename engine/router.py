# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Optional
import torch
import torch.nn.functional as functional

@dataclass(frozen=True)
class RouterConfig:
    mode: str = "surprisal"
    support: float = 0.50
    floor: float = 0.10
    gain: float = 1.00
    cap: float = 1.50
    prox: float = 0.50
    pivot: Optional[float] = None
    signal: Optional[float] = None
    rough: Optional[float] = None
    reach: int = 8
    band: int = 3
    feather: int = 2
    path: float = 1.00
    grid: float = 0.00
    relay: float = 1.00
    route: bool = True
    edge: bool = False
    scalar: bool = False
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            object.__setattr__(self, "mode", self.mode.strip().casefold())

    def validate(self) -> None:
        if self.mode not in {"surprisal", "relative", "invariant", "magnitude"}:
            raise ValueError("mode must be surprisal, relative, invariant, or magnitude.")
        values = (self.support, self.floor, self.gain, self.cap, self.prox, self.path, self.grid, self.relay, self.epsilon)
        try:
            valid = all(isinstance(value, Real) and not isinstance(value, bool) and isfinite(float(value)) for value in values)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("Router values must be finite numbers.")
        if not 0.0 < self.support < 1.0:
            raise ValueError("support must be between zero and one.")
        if not 0.0 <= self.floor <= 1.0 <= self.gain:
            raise ValueError("gain values must satisfy 0 <= floor <= 1 <= gain.")
        if self.cap <= 0.0 or self.relay <= 0.0 or self.epsilon <= 0.0:
            raise ValueError("cap, relay, and epsilon must be positive.")
        if not 0.0 <= self.prox <= 1.0:
            raise ValueError("prox must be in [0, 1].")
        for name, value in (("pivot", self.pivot), ("signal", self.signal), ("rough", self.rough)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or value <= 0.0):
                raise ValueError(f"{name} must be a positive finite number.")
        radius = (self.reach, self.band, self.feather)
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in radius):
            raise ValueError("reach, band, and feather must be nonnegative integers.")
        if self.reach < 1 or self.reach < self.band:
            raise ValueError("reach must be positive and at least the band radius.")
        if self.path < 0.0 or self.grid < 0.0 or self.path + self.grid <= 0.0:
            raise ValueError("probe weights must be nonnegative with a positive sum.")
        if not isinstance(self.route, bool) or not isinstance(self.edge, bool) or not isinstance(self.scalar, bool):
            raise TypeError("route switches must be boolean.")
        if self.scalar and not self.route:
            raise ValueError("scalar routing requires route to be enabled.")

@dataclass
class RouterOutput:
    field: torch.Tensor
    mask: torch.Tensor
    maps: dict[str, torch.Tensor]

class ThreadRouter:
    # Route fields using local probe stability.
    def __init__(self, config: RouterConfig) -> None:
        config.validate()
        self.config = config

    def __call__(self, base: torch.Tensor, current: torch.Tensor, previous: torch.Tensor, target: torch.Tensor, reference: torch.Tensor, latent: torch.Tensor) -> RouterOutput:
        self.validateinputs(base, current, previous, target, reference, latent)
        dtype = base.dtype
        base = base.float()
        current = current.float()
        previous = previous.float()
        target = target.float()
        reference = reference.float()
        latent = latent.float()
        magnitude = self.relative(self.norm(base))
        surprisal = (target - reference).abs().mean(dim=1, keepdim=True)
        evidence = self.evidencemap(base, current, previous, target, reference, surprisal)
        probechange, pathchange, gridroughness = self.probemap(base, current, previous)
        trust = 1.0 / (1.0 + probechange / self.pivotscale(probechange))
        if not self.config.route:
            trust = torch.ones_like(trust)
        if self.config.scalar:
            numerator = (evidence * trust).mean(dim=(2, 3), keepdim=True)
            denominator = evidence.mean(dim=(2, 3), keepdim=True) + self.config.epsilon
            trust = (numerator / denominator).clamp(0.0, 1.0).expand_as(trust)
        gate = evidence * (self.config.floor + (self.config.gain - self.config.floor) * trust)
        support = (gate / self.config.gain >= self.config.support).float()
        boundary = self.boundarymap(evidence)
        band, reached = self.bandmap(support)
        cap = self.config.cap * self.robustscale(self.norm(base), 0.95).clamp_min(self.config.epsilon)
        direct = self.clip(gate * base, cap)
        if self.config.edge:
            relay = self.relayfield(base, direct, support, latent, cap)
            relayweight = (band * evidence * trust).clamp(0.0, 1.0)
            field = (1.0 - relayweight) * direct + relayweight * relay
        else:
            relay = direct
            relayweight = torch.zeros_like(evidence)
            field = direct
        anchor = (1.0 - gate / self.config.gain).clamp(0.0, 1.0)
        mask = self.smooth((gate / self.config.gain + self.config.prox * relayweight).clamp(0.0, 1.0), self.config.feather)
        relaycos = (relay * base).sum(dim=1, keepdim=True) / (self.norm(relay) * self.norm(base) + self.config.epsilon)
        relaydelta = (band * self.norm(relay - direct)).sum() / ((band * self.norm(direct)).sum() + self.config.epsilon)
        maps = {"evidence": evidence, "surprisal": surprisal, "magnitude": magnitude, "agreement": self.agreementmap(current, previous), "probechange": probechange, "pathchange": pathchange, "gridroughness": gridroughness, "trust": trust, "boundary": boundary, "support": support, "band": band, "reached": reached, "gate": gate, "relay": relayweight, "anchor": anchor, "cap": cap, "relaycos": relaycos, "relaydelta": relaydelta.detach().reshape(1)}
        output = field.to(dtype=dtype)
        resultmask = mask.to(dtype=dtype)
        if not torch.isfinite(output).all().item() or not torch.isfinite(resultmask).all().item():
            raise FloatingPointError("Router computation produced nonfinite values.")
        if any(not torch.isfinite(value).all().item() for value in maps.values()):
            raise FloatingPointError("Router maps contain nonfinite values.")
        return RouterOutput(field=output, mask=resultmask, maps=maps)

    def validateinputs(self, base: torch.Tensor, current: torch.Tensor, previous: torch.Tensor, target: torch.Tensor, reference: torch.Tensor, latent: torch.Tensor) -> None:
        tensors = [base, current, previous, target, reference, latent]
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError("Router inputs must be tensors.")
        shape = base.shape
        if base.ndim != 4 or any(size < 1 for size in shape):
            raise ValueError("Fields must have a positive four dimensional shape.")
        if any(tensor.shape != shape for tensor in tensors):
            raise ValueError("All router tensors must share one shape.")
        if any(not tensor.is_floating_point() for tensor in tensors):
            raise TypeError("Router tensors must use floating point values.")
        if any(tensor.device != base.device for tensor in tensors):
            raise ValueError("Router tensors must share one device.")
        if any(not torch.isfinite(tensor).all().item() for tensor in tensors):
            raise ValueError("Router tensors must contain finite values.")

    def evidencemap(self, base: torch.Tensor, current: torch.Tensor, previous: torch.Tensor, target: torch.Tensor, reference: torch.Tensor, surprisal: torch.Tensor) -> torch.Tensor:
        if self.config.mode == "surprisal":
            return self.relative(surprisal)
        if self.config.mode == "relative":
            value = self.norm(target - reference) / (self.norm(target) + self.norm(reference) + self.config.epsilon)
            return value.clamp(0.0, 1.0)
        if self.config.mode == "invariant":
            ratio = self.norm(current + previous) / (self.norm(current - previous) + self.config.epsilon)
            return (ratio / (ratio + 1.0)).clamp(0.0, 1.0)
        return self.relative(self.norm(base))

    def probemap(self, base: torch.Tensor, current: torch.Tensor, previous: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        amplitude = 0.5 * (self.norm(current) + self.norm(previous))
        difference = self.norm(current - previous)
        if self.config.signal is None:
            floor = 0.05 * self.quantile(amplitude, 0.50)
        else:
            floor = torch.full_like(amplitude[:, :, :1, :1], float(self.config.signal))
        denominator = amplitude + floor.clamp_min(self.config.epsilon)
        pathchange = difference / denominator
        quiet = (amplitude <= self.config.epsilon) & (difference <= self.config.epsilon)
        pathchange = torch.where(quiet, torch.zeros_like(pathchange), pathchange).clamp(0.0, 2.0)
        raw = self.gradient(base)
        if self.config.rough is None:
            scale = self.robustscale(raw, 0.95)
        else:
            scale = torch.full_like(raw[:, :, :1, :1], float(self.config.rough))
        gridroughness = torch.where(scale > self.config.epsilon, raw / scale.clamp_min(self.config.epsilon), torch.zeros_like(raw)).clamp(0.0, 1.0)
        return self.config.path * pathchange + self.config.grid * gridroughness, pathchange, gridroughness

    def agreementmap(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        value = (current * previous).sum(dim=1, keepdim=True) / (self.norm(current) * self.norm(previous) + self.config.epsilon)
        return value.clamp(-1.0, 1.0)

    def boundarymap(self, evidence: torch.Tensor) -> torch.Tensor:
        dx = functional.pad(evidence[:, :, :, 1:] - evidence[:, :, :, :-1], (0, 1, 0, 0))
        dy = functional.pad(evidence[:, :, 1:, :] - evidence[:, :, :-1, :], (0, 0, 0, 1))
        magnitude = torch.sqrt(dx.square() + dy.square())
        scale = self.robustscale(magnitude, 0.95)
        return torch.where(scale > self.config.epsilon, magnitude / scale.clamp_min(self.config.epsilon), torch.zeros_like(magnitude)).clamp(0.0, 1.0)

    def bandmap(self, support: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reached = support.clone()
        soft = support.clone()
        for step in range(1, self.config.reach + 1):
            grow = self.neighbors(reached).amax(dim=2)
            newly = (grow - reached).clamp(min=0.0)
            if step <= self.config.band:
                soft = soft + newly * (1.0 - float(step) / float(self.config.band + 1))
            reached = (reached + newly).clamp(max=1.0)
        return (soft - support).clamp(0.0, 1.0), reached

    def relayfield(self, base: torch.Tensor, direct: torch.Tensor, support: torch.Tensor, latent: torch.Tensor, cap: torch.Tensor) -> torch.Tensor:
        if not self.config.edge:
            return direct
        affinity = self.affinity(latent)
        value = direct
        weight = support
        for _ in range(self.config.reach):
            shifted = self.neighbors(torch.cat((value, weight), dim=1))
            weights = shifted[:, -1:]
            incoming = (affinity * weights).sum(dim=2)
            aggregate = (affinity * weights * shifted[:, :-1]).sum(dim=2)
            fresh = (incoming > 0.0) & (weight <= 0.0)
            value = torch.where(fresh, aggregate / (incoming + self.config.epsilon), value)
            weight = torch.where(fresh, torch.ones_like(weight), weight)
        relay = (self.config.relay * base + weight * value) / (self.config.relay + weight + self.config.epsilon)
        return self.clip(relay, cap)

    def affinity(self, latent: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = latent.shape
        neighbors = self.neighbors(latent)
        distance = (neighbors - latent.unsqueeze(2)).square().sum(dim=1, keepdim=True)
        valid = self.neighbors(torch.ones(batch, 1, height, width, device=latent.device, dtype=latent.dtype))
        flat = ((distance * valid).sum(dim=2) / (valid.sum(dim=2) + self.config.epsilon)).flatten(start_dim=1)
        bandwidth = torch.quantile(flat, 0.50, dim=1, keepdim=True).clamp_min(self.config.epsilon).view(batch, 1, 1, 1, 1)
        return torch.exp(-distance / bandwidth) * valid

    def neighbors(self, tensor: torch.Tensor) -> torch.Tensor:
        padded = functional.pad(tensor, (1, 1, 1, 1))
        height, width = tensor.shape[-2:]
        parts: list[torch.Tensor] = []
        for row in (-1, 0, 1):
            for column in (-1, 0, 1):
                if row != 0 or column != 0:
                    parts.append(padded[:, :, 1 + row:1 + row + height, 1 + column:1 + column + width])
        return torch.stack(parts, dim=2)

    def gradient(self, tensor: torch.Tensor) -> torch.Tensor:
        padded = functional.pad(tensor, (1, 1, 1, 1), mode="replicate")
        dx = 0.5 * (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2])
        dy = 0.5 * (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1])
        return torch.sqrt(dx.square().sum(dim=1, keepdim=True) + dy.square().sum(dim=1, keepdim=True))

    def smooth(self, tensor: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return tensor
        size = 2 * radius + 1
        channels = tensor.shape[1]
        points = torch.arange(-radius, radius + 1, device=tensor.device, dtype=tensor.dtype)
        kernel = torch.exp(-0.5 * (points / max(float(radius) * 0.5, 0.5)).square())
        kernel = kernel / kernel.sum()
        horizontal = kernel.view(1, 1, 1, size).expand(channels, 1, 1, size)
        vertical = kernel.view(1, 1, size, 1).expand(channels, 1, size, 1)
        result = functional.conv2d(functional.pad(tensor, (radius, radius, 0, 0), mode="replicate"), horizontal, groups=channels)
        return functional.conv2d(functional.pad(result, (0, 0, radius, radius), mode="replicate"), vertical, groups=channels)

    def relative(self, magnitude: torch.Tensor) -> torch.Tensor:
        scale = self.robustscale(magnitude, 0.95)
        return torch.where(scale > self.config.epsilon, magnitude / scale.clamp_min(self.config.epsilon), torch.zeros_like(magnitude)).clamp(0.0, 1.0)

    def pivotscale(self, change: torch.Tensor) -> torch.Tensor:
        if self.config.pivot is not None:
            return torch.full_like(change[:, :, :1, :1], float(self.config.pivot))
        return self.robustscale(change, 0.50).clamp_min(self.config.epsilon)

    def robustscale(self, tensor: torch.Tensor, value: float) -> torch.Tensor:
        scale = self.quantile(tensor, value)
        maximum = tensor.flatten(start_dim=2).amax(dim=2, keepdim=True).unsqueeze(-1)
        return torch.where(scale > self.config.epsilon, scale, maximum)

    def quantile(self, tensor: torch.Tensor, value: float) -> torch.Tensor:
        result = torch.quantile(tensor.flatten(start_dim=2).float(), value, dim=2, keepdim=True)
        return result.unsqueeze(-1).to(dtype=tensor.dtype)

    def norm(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(tensor.square().sum(dim=1, keepdim=True).clamp_min(0.0))

    def clip(self, tensor: torch.Tensor, cap: torch.Tensor) -> torch.Tensor:
        return tensor * torch.minimum(torch.ones_like(cap), cap / (self.norm(tensor) + self.config.epsilon))

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
