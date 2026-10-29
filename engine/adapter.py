# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import importlib.util
import hashlib
from pathlib import Path
import torch
from engine.session import number, tensor
from data import clean, locate

class Adapter:
    def __init__(self, session, generator, inputs, context=None, fields=False, continuous=True):
        if not isinstance(generator, torch.nn.Module) or not callable(inputs) or type(fields) is not bool or type(continuous) is not bool:
            raise ValueError("Adapters require frozen generators and legitimate input maps.")
        self.session = session
        self.generator = generator.eval().requires_grad_(False)
        self.inputs = inputs
        self.context = dict(context or {})
        self.fields = fields
        self.continuous = continuous
        self.affine, self.cfg, self.guidance = False, False, 1.
        self.steps, self.endpoints = 1, 0
        self.path = "embedding" if continuous else "discrete"
        self.origin = tensor(inputs(torch.zeros_like(session.anchor))).detach().clone()
        if self.origin.device != session.anchor.device:
            raise ValueError("Native inputs must use the metered session device.")
        self.cache = {self.key(torch.zeros_like(session.anchor)): self.origin}

    def key(self, field):
        return hashlib.sha256(field.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()

    def cost(self, eta):
        return 1

    def queries(self):
        return 1

    @torch.no_grad()
    def endpoint(self, field, eta=0.):
        # Map clean fields through the declared native input.
        tensor(field)
        eta = number(eta, "eta", 0, 1)
        if field.shape != self.session.anchor.shape or field.device != self.session.anchor.device:
            raise ValueError("Adapter fields or reference coefficients are invalid.")
        if not self.fields and torch.any(field != 0).item():
            raise ValueError("This generator exposes only zero-field diagnostics.")
        if not self.continuous and eta not in {0., 1.}:
            raise ValueError("The generator exposes only endpoint conditions.")
        key = self.key(field)
        if key not in self.cache:
            self.cache[key] = tensor(self.inputs(self.session.scale * field)).detach().clone()
        sample = self.cache[key]
        if sample.shape != self.origin.shape or sample.device != self.origin.device:
            raise ValueError("Native input maps must preserve matched geometry.")
        embedding = (1 - eta) * self.session.target + eta * self.session.source
        if self.session.limit is not None:
            self.session.limit.consume(sample.shape[0])
        self.session.calls += 1
        self.session.samples += sample.shape[0]
        output = tensor(self.generator(sample, embedding, **self.context))
        if output.shape != self.session.anchor.shape or output.device != self.session.anchor.device:
            raise ValueError("Generators must return matching clean latent units.")
        self.endpoints += 1
        return output.detach().clone()

    def metadata(self):
        return {"kind": "native", "steps": 1, "referencepath": self.path, "continuous": self.continuous, "affine": False, "conditional": True, "fields": self.fields, "endpoints": self.endpoints, "inputshape": list(self.origin.shape), "mappedinputs": len(self.cache), "scope": "Caller-supplied native inversion and clean-output generator; inversion queries must be metered, not fabricated."}

def load(value, session):
    path = locate(Path("."), clean(value, "adapter"))
    if path.suffix != ".py" or not path.stem.isascii() or not path.stem.isalpha() or not path.stem.islower():
        raise ValueError("Adapters must use lowercase single-word Python filenames.")
    spec = importlib.util.spec_from_file_location("refunmixadapter", path)
    if spec is None or spec.loader is None:
        raise ValueError("The native adapter cannot be loaded.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "create", None)):
        raise ValueError("Native adapter files must expose create(session).")
    native = module.create(session)
    if not isinstance(native, Adapter) or native.session is not session:
        raise ValueError("Native adapters must preserve the metered session.")
    return native

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
