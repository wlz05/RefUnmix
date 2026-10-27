# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import torch
from engine.session import number, rms, tensor
from engine.wavelet import split
from engine.endpoint import design

def coordinates(point, bounded=True):
    try:
        low, high, eta = point
    except (TypeError, ValueError) as error:
        raise ValueError("Control points need three finite numeric coordinates.") from error
    return tuple(number(value, label, 0 if bounded else None, limit if bounded else None) for value, label, limit in zip((low, high, eta), ("low", "high", "eta"), (2., 2., 1.)))

class Path:
    def __init__(self, native, base, mode="latent", seed=42, neutral=None):
        tensor(base)
        if base.shape != native.session.anchor.shape or any(size % 2 for size in base.shape[-2:]):
            raise ValueError("Path control requires matching even latent geometry.")
        if mode not in {"latent", "rgb", "mix", "random", "cfg"}:
            raise ValueError("Unknown reference path representation.")
        self.native, self.session, self.base, self.mode = native, native.session, base.detach().clone(), mode
        self.low, self.high = split(base)
        self.empty = self.session.scale == 0 or not torch.any(self.base != 0).item()
        self.fields, self.references, self.pixels = {}, {}, {}
        if neutral is not None:
            tensor(neutral)
            if neutral.shape != base.shape or neutral.device != base.device:
                raise ValueError("Cached references must match the native field geometry.")
            self.references[0.] = neutral.detach().clone()
            self.fields[(0., 0.)] = self.references[0.]
        self.hits, self.seed, self.direction = 0, int(seed), None
        self.images = {}

    def phase(self):
        if self.session.limit is not None:
            self.session.limit.phase = "backend"

    def target(self, low=1., high=1.):
        low, high = number(low, "low amplitude", 0, 2), number(high, "high amplitude", 0, 2)
        key = (low, high)
        if key not in self.fields:
            self.phase()
            zero = self.empty or low == high == 0
            self.fields[key] = self.references[0.] if zero and 0. in self.references else self.native.endpoint(low * self.low + high * self.high)
            if zero:
                self.references[0.] = self.fields[(0., 0.)] = self.fields[key]
        else:
            self.hits += 1
        return self.fields[key]

    def zero(self, eta):
        eta = number(eta, "eta", 0, 1)
        if eta not in self.references:
            self.phase()
            if eta == 0 and (0., 0.) in self.fields:
                self.references[eta] = self.fields[(0., 0.)]
            elif eta in {0., 1.} or not self.native.affine:
                self.references[eta] = self.native.endpoint(torch.zeros_like(self.base), eta)
            else:
                self.references[eta] = (1 - eta) * self.zero(0.) + eta * self.zero(1.)
            if eta == 0:
                self.fields[(0., 0.)] = self.references[eta]
        else:
            self.hits += 1
        return self.references[eta]

    def initialize(self, references=(0., 1.)):
        self.target()
        if self.mode == "cfg":
            return
        for eta in references:
            self.zero(eta)
        if self.mode == "random":
            self.randomize()

    def cost(self, point):
        # Reserve all uncached endpoints before a candidate query.
        low, high, eta = coordinates(point, False)
        zero = self.empty or low == high == 0
        endpoints = [] if (low, high) in self.fields or zero and 0. in self.references else [0.]
        if self.mode == "cfg":
            if eta != 0:
                endpoints.append(eta)
        else:
            needed = (0., 1.) if self.mode == "random" or self.mode == "mix" and eta not in {0., 1.} or self.native.affine and eta not in {0., 1.} else (eta,)
            endpoints.extend(value for value in needed if value not in self.references and not (value == 0 and ((0., 0.) in self.fields or zero)))
        return sum(self.native.cost(value) for value in endpoints), len(endpoints) * self.native.queries()

    def randomize(self):
        if self.direction is not None:
            return
        delta = self.zero(0.) - self.zero(1.)
        generator = torch.Generator(device=delta.device).manual_seed(self.seed ^ 0xA13)
        random = torch.randn(delta.shape, generator=generator, device=delta.device, dtype=torch.float32)
        self.direction = random * delta.double().norm().to(random.dtype) / random.double().norm().clamp_min(1e-30).to(random.dtype)

    def canonical(self, point):
        low, high, eta = coordinates(point, False)
        low = low if self.session.scale != 0 and torch.any(self.low != 0).item() else 1.
        high = high if self.session.scale != 0 and torch.any(self.high != 0).item() else 1.
        if not self.session.cleanup or torch.equal(self.session.source, self.session.target) or self.native.cfg and self.native.guidance == 0 or self.mode != "cfg" and self.native.affine and 0. in self.references and 1. in self.references and torch.equal(self.references[0.], self.references[1.]):
            eta = 0.
        return tuple(round(float(value), 12) for value in (low, high, eta))

    def axes(self, requested):
        active = []
        for axis in requested:
            if axis in {0, 1} and self.session.scale == 0:
                continue
            if axis == 0 and not torch.any(self.low != 0).item() or axis == 1 and not torch.any(self.high != 0).item():
                continue
            if axis == 2 and self.canonical((1., 1., 1.))[2] == 0:
                continue
            active.append(axis)
        return tuple(active)

    def latent(self, point):
        low, high, eta = point
        target = self.target(low, high)
        if self.mode == "cfg":
            if eta == 0:
                return target
            self.phase()
            return self.native.endpoint(low * self.low + high * self.high, eta)
        if self.mode == "mix":
            reference = self.zero(eta) if eta in {0., 1.} else (1 - eta) * self.zero(0.) + eta * self.zero(1.)
        elif self.mode == "random":
            self.randomize()
            reference = self.zero(0.) - eta * self.direction
        else:
            reference = self.zero(eta)
        # Remove the matched reference before restoring the source anchor.
        return self.session.anchor.float() + (target - reference)

    def candidate(self, point):
        # Decode every visited finite amplitude candidate.
        point = coordinates(point)
        if self.session.limit is not None:
            self.session.limit.require()
        latent = self.latent(point)
        pixels = None if self.mode == "rgb" else self.session.decode(latent)
        if self.mode == "rgb":
            target = self.target(*point[:2])
            key = ("target", *point[:2])
            if key not in self.pixels:
                self.pixels[key] = self.session.decode(target)
            if point[2] not in self.pixels:
                self.pixels[point[2]] = self.session.decode(self.zero(point[2]))
            pixels = self.session.origin + self.pixels[key] - self.pixels[point[2]]
            self.session.encodes += 1
            latent = self.session.editor.encode(pixels.clamp(0, 1))
        return tensor(latent).detach().clone(), tensor(pixels, 3).clamp(0, 1).detach().clone()

    def raw(self):
        latent = self.target()
        return latent.detach().clone(), self.session.decode(latent).clamp(0, 1).detach().clone()

    def mechanism(self, decoded=False):
        # Separate reconstruction, condition, field, and interaction.
        previous = self.session.limit.phase if self.session.limit is not None else None
        try:
            self.phase()
            zero, target, field = self.zero(1.), self.zero(0.), self.target()
            if self.session.limit is not None:
                self.session.limit.phase = "diagnostic"
            sourcefield = self.native.endpoint(self.base, 1.)
            values = {"bias": zero.double() - self.session.anchor.double(), "condition": target.double() - zero.double(), "field": sourcefield.double() - zero.double(), "interaction": field.double() - sourcefield.double() - target.double() + zero.double()}
            residual = field.double() - self.session.anchor.double() - sum(values.values(), torch.zeros_like(field, dtype=torch.float64))
            report = {"latent": {key: rms(value) for key, value in values.items()}, "identityerror": float(residual.abs().max()), "pathzero": float(values["condition"].abs().max()) <= 1e-8, "reference": {"targetremoved": "bias and condition", "sourceremoved": "bias"}, "cost": "Fourth endpoint is diagnostic, not controller input."}
            report["design"] = design(("sourcezero", "targetzero", "sourcefield", "targetfield"))
            if decoded:
                images = {"sourcezero": self.session.decode(zero), "targetzero": self.session.decode(target), "sourcefield": self.session.decode(sourcefield), "targetfield": self.session.decode(field)}
                bias = images["sourcezero"].double() - self.session.origin.double()
                condition = images["targetzero"].double() - images["sourcezero"].double()
                effect = images["sourcefield"].double() - images["sourcezero"].double()
                interaction = images["targetfield"].double() - images["sourcefield"].double() - images["targetzero"].double() + images["sourcezero"].double()
                report["rgb"] = {key: rms(value) for key, value in {"bias": bias, "condition": condition, "field": effect, "interaction": interaction}.items()}
                report["rgbcondition"] = report["rgb"]["condition"]
                report["decoded"] = True
                self.images.update({key: image.clamp(0, 1) for key, image in images.items()})
                self.session.artifacts.update(self.images)
            return report
        finally:
            if self.session.limit is not None:
                self.session.limit.phase = previous

    def metadata(self):
        count = len(self.fields) - int((0., 0.) in self.fields)
        observed = [name for name, known in (("sourcezero", 1. in self.references), ("targetzero", 0. in self.references), ("targetfield", (1., 1.) in self.fields)) if known]
        return {"mode": self.mode, "fields": count, "references": len(self.references), "cachehits": self.hits, "conditionrms": rms(self.references[0.].double() - self.references[1.].double()) if 0. in self.references and 1. in self.references else None, "affineequivalence": self.mode == "mix" and self.native.affine, "design": design(observed), "scope": "Latent frequencies are controls, not semantic partitions."}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
