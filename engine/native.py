# -*- coding: utf-8 -*-
# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

import torch
from engine.session import number, tensor

def load(root="./models", backbone="vp", device=None, precision="fp32", size=512):
    # Editing and native generation load the same local backbone components.
    if backbone == "instaflow":
        from engine.linear import Editor
        editor = Editor.load(root, device, precision, size)
        editor.metadata["root"] = str(root)
        return editor
    if backbone not in {"vp", "sdturbo", "swiftbrush"}:
        raise ValueError("Unknown one-step native backbone.")
    from engine.session import Editor
    editor = Editor.load(root, device, precision, size)
    if backbone == "vp":
        return editor
    if editor.kind != "epsilon":
        raise ValueError("SD-Turbo and SwiftBrush-v2 require epsilon-prediction components.")
    editor.metadata.update({"architecture": backbone, "root": str(root), "native": {"prediction": editor.kind, "bridge": "alpha * anchor + sigma * noise", "clock": "round((training steps - 1) * time)", "recovery": "(state - sigma * predicted noise) / alpha"}})
    return editor

@torch.no_grad()
def direct(session, args, metadata):
    # Native: Q(V(P_te(E(X), c_t; noise))), with no editing field.
    latent = session.predict(session.anchor, session.noises[:1], session.editor.index(session.end), session.target)
    pixels = session.decode(latent).clamp(0, 1)
    report = {**metadata, "method": "native", "renderer": "source-anchored native prediction", "selected": "native", "endpointtime": session.end, "editingfield": False, "stats": session.stats(), "controller": {"enabled": False}, "candidates": [], "explicitregion": False, "scope": "One target-conditioned native prediction from the encoded source, with no editing field or allocation."}
    return pixels, report, session

@torch.no_grad()
def naive(session, args, metadata):
    # Match the delta-zero ChordEdit control with one target refinement.
    field = session.field(args.start, 0.)
    native = create(session, 1, 1., False, "prediction")
    latent = native.endpoint(field)
    pixels = session.decode(latent).clamp(0, 1)
    report = {**metadata, "method": "naive", "renderer": "instantaneous field", "selected": "naive", "field": {"start": args.start, "delta": 0., "end": args.end, "scale": session.scale, "conditioning": "original source and target prompts"}, "backend": native.metadata(), "stats": session.stats(), "controller": {"enabled": False}, "candidates": [], "explicitregion": False, "scope": "Naive editing control: instantaneous prompt difference and one target refinement through the selected native backbone."}
    return pixels, report, session

def create(session, steps=1, guidance=1., cfg=False, path="auto"):
    if getattr(session.editor, "metadata", {}).get("family") == "flow":
        from engine.flow import Native as Flow
        return Flow(session, steps, guidance, cfg, path)
    return Native(session, steps, guidance, cfg, path)

class Native:
    def __init__(self, session, steps=1, guidance=1., cfg=False, path="prediction"):
        self.session = session
        self.guidance = number(guidance, "guidance", 0)
        if type(steps) is not int or steps < 1 or type(cfg) is not bool:
            raise ValueError("Native steps and CFG capability are invalid.")
        if not cfg and self.guidance != 1:
            raise ValueError("Nonunit guidance requires an explicit CFG capability.")
        if getattr(session.editor, "metadata", {}).get("family") == "linear" and (steps != 1 or cfg):
            raise ValueError("The InstaFlow one-step checkpoint requires one unguided prediction.")
        if path == "auto":
            path = "prediction" if cfg else "embedding"
        if path not in {"prediction", "embedding", "discrete"}:
            raise ValueError("Unknown native reference capability.")
        if path == "embedding" and session.source.shape != session.target.shape:
            raise ValueError("Embedding paths require matching native shapes.")
        self.path = path
        self.continuous = path != "discrete"
        self.steps, self.cfg = steps, cfg
        self.unconditional = None
        self.endpoints = 0
        tick = session.editor.index(session.end)
        if steps > tick + 1:
            raise ValueError("Continuation steps exceed distinct scheduler ticks.")
        self.ticks = [tick] if steps == 1 else torch.linspace(tick, 0, steps).round().long().tolist()
        self.affine = steps == 1 and path == "prediction" or not session.cleanup
        self.mode = "clean" if steps == 1 else "ddim"

    def cost(self, eta):
        conditions = 1 if eta in {0., 1.} or self.path == "embedding" else 2
        return (conditions + int(self.cfg and self.guidance != 1)) * self.steps if self.session.cleanup else 0

    def queries(self):
        return self.steps if self.session.cleanup else 0

    def conditions(self, eta):
        if self.path == "discrete" and eta not in {0., 1.}:
            raise ValueError("This backend exposes only discrete references.")
        values = [self.session.target] if eta == 0 else [self.session.source] if eta == 1 else [self.session.target, self.session.source]
        weights = [1.] if len(values) == 1 else [1 - eta, eta]
        if self.path == "embedding" and 0 < eta < 1:
            values, weights = [(1 - eta) * self.session.target + eta * self.session.source], [1.]
        if self.cfg and self.guidance != 1:
            if self.unconditional is None:
                self.unconditional = self.session.editor.prompt("")
            values.append(self.unconditional)
            weights = [weight * self.guidance for weight in weights] + [1 - self.guidance]
        return torch.cat(values), weights

    def predict(self, value, tick, eta, noisy=False):
        embeddings, weights = self.conditions(eta)
        count = len(weights)
        if noisy:
            outputs = self.session.state(value.expand(count, -1, -1, -1), tick, embeddings)
        else:
            outputs = self.session.predict(value.expand(count, -1, -1, -1), self.session.noises[:1].expand(count, -1, -1, -1), tick, embeddings)
        return sum((weight * output[None] for weight, output in zip(weights, outputs)), torch.zeros_like(value, dtype=torch.float32))

    @torch.no_grad()
    def endpoint(self, field, eta=0.):
        # Continue matched conditions on the native state.
        tensor(field)
        eta = number(eta, "eta", 0, 1)
        if not self.continuous and eta not in {0., 1.}:
            raise ValueError("Continuous references are unavailable.")
        session = self.session
        if field.shape != session.anchor.shape or field.device != session.anchor.device:
            raise ValueError("Native fields must match the source latent.")
        if session.limit is not None:
            session.limit.require(self.cost(eta), self.queries())
        latent = tensor(session.anchor.float() + session.scale * field.float())
        if not session.cleanup:
            self.endpoints += 1
            return latent
        if self.steps == 1:
            result = self.predict(latent, self.ticks[0], eta)
        else:
            alpha = session.editor.alphas[self.ticks[0]].sqrt()
            sigma = (1 - session.editor.alphas[self.ticks[0]]).sqrt()
            state = tensor(alpha * latent + sigma * session.noises[:1].float())
            for index, tick in enumerate(self.ticks):
                clean = self.predict(state, tick, eta, noisy=True)
                if index + 1 == len(self.ticks):
                    result = clean
                    break
                alpha = session.editor.alphas[tick].sqrt()
                sigma = (1 - session.editor.alphas[tick]).sqrt()
                if float(sigma) <= 0:
                    raise ValueError("DDIM continuation requires positive noise scales.")
                epsilon = (state - alpha * clean) / sigma
                following = session.editor.alphas[self.ticks[index + 1]]
                state = tensor(following.sqrt() * clean + (1 - following).sqrt() * epsilon)
        tensor(result)
        self.endpoints += 1
        return result.detach().clone()

    def metadata(self):
        return {"kind": self.mode, "steps": self.steps, "ticks": self.ticks, "cfg": self.cfg, "guidance": self.guidance, "referencepath": self.path, "continuous": self.continuous, "affine": self.affine, "conditional": self.session.cleanup, "endpoints": self.endpoints, "prediction": getattr(self.session.editor, "metadata", {}).get("native"), "sampler": "Deterministic DDIM without latent clipping." if self.steps > 1 else "One clean prediction without latent clipping."}

# Author: Weiliang Zhu; Other main author: Yeming Dai, Yaming Lv
# Email: wlzchina05@gmail.com
