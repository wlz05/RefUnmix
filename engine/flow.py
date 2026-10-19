# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

import hashlib
import math
from numbers import Integral
import torch
from engine.session import Editor as Base, number, relative, tensor
from engine.native import Native as Original

def pack(value):
    if value.ndim != 4 or any(size % 2 for size in value.shape[-2:]):
        raise ValueError("Flux packing requires even BCHW latents.")
    batch, channels, height, width = value.shape
    return value.reshape(batch, channels, height // 2, 2, width // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(batch, height * width // 4, channels * 4)

def unpack(value, shape):
    batch, channels, height, width = shape
    if value.shape != (batch, height * width // 4, channels * 4) or height % 2 or width % 2:
        raise ValueError("Packed velocities must match the source lattice.")
    return value.reshape(batch, height // 2, width // 2, channels, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(shape)

def identifiers(height, width, device, dtype):
    if height % 2 or width % 2:
        raise ValueError("Flux image positions require an even lattice.")
    rows, columns = torch.meshgrid(torch.arange(height // 2, device=device), torch.arange(width // 2, device=device), indexing="ij")
    return torch.stack((torch.zeros_like(rows), rows, columns), dim=-1).reshape(-1, 3).to(dtype)

def stability(integral, lipschitz, time):
    # Bound continuous endpoint error under declared assumptions.
    integral = number(integral, "residual integral", 0)
    lipschitz = number(lipschitz, "Lipschitz bound", 0)
    time = number(time, "flow interval", 0, 1)
    if integral == 0 or time == 0:
        return 0.
    try:
        coefficient = time if lipschitz == 0 else math.expm1(2 * lipschitz * time) / (2 * lipschitz)
    except OverflowError as error:
        raise ValueError("The declared endpoint bound overflows.") from error
    result = math.sqrt(coefficient * integral)
    if not math.isfinite(result):
        raise ValueError("The declared endpoint bound overflows.")
    return result

class Editor(Base):
    def __init__(self, pipeline, family="flux", device="cpu", dtype=torch.float32, size=512, guidance=3.5, consistency=False):
        if family not in {"flux", "sd3"} or not isinstance(size, Integral) or isinstance(size, bool) or size < 16 or size % 16 or type(consistency) is not bool:
            raise ValueError("Flow editors require a declared family and even VAE lattice.")
        if dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            raise ValueError("Flow precision is unsupported.")
        self.pipeline, self.family = pipeline, family
        self.device, self.dtype, self.size = torch.device(device), dtype, int(size)
        self.guidance = number(guidance, "flow guidance", 0)
        self.velocityprobes = consistency
        self.scheduler, self.vae, self.unet = pipeline.scheduler, pipeline.vae, pipeline.transformer
        config = self.scheduler.config
        if any(bool(getattr(config, name, False)) for name in ("stochastic_sampling", "invert_sigmas")):
            raise ValueError("Only data-to-noise deterministic Euler conventions are supported.")
        steps = getattr(config, "num_train_timesteps", 1000)
        if not isinstance(steps, Integral) or isinstance(steps, bool) or steps < 2:
            raise ValueError("Flow timestep scales are invalid.")
        self.training = int(steps)
        self.maxtime = self.training
        self.sigmas = torch.linspace(0, 1, self.training + 1, device=self.device, dtype=torch.float32)
        self.factor = number(getattr(self.vae.config, "scaling_factor", 1.), "VAE scale", 1e-12)
        self.shift = number(getattr(self.vae.config, "shift_factor", 0.) or 0., "VAE shift")
        self.textcalls, self.pools, self.textids = 0, {}, {}
        self.tokens = 512 if family == "flux" else 256
        self.tokenizer = pipeline.tokenizer
        self.kind = "velocity"
        self.metadata = {"family": "flow", "architecture": family, "weights": None, "velocityprobes": consistency, "times": "Actual sigma in [0, 1]; model clock mapping is explicit.", "guidanceembedding": self.guidance if family == "flux" and getattr(self.unet.config, "guidance_embeds", False) else None}
        modules = [self.unet, self.vae] + [getattr(pipeline, name, None) for name in ("text_encoder", "text_encoder_2", "text_encoder_3")]
        for module in modules:
            if module is not None:
                module.to(device=self.device, dtype=dtype).eval().requires_grad_(False)

    @classmethod
    def load(cls, root="./models", device=None, precision="fp32", size=512, family="flux", guidance=3.5, consistency=False):
        # Load a local native pipeline without network fallback.
        choices = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        precision = str(precision).strip().casefold()
        if precision not in choices or family not in {"flux", "sd3"}:
            raise ValueError("Unknown flow precision or architecture.")
        selected = torch.device(str(device).strip().casefold() if device is not None else "cuda" if torch.cuda.is_available() else "cpu")
        if selected.type not in {"cpu", "cuda"} or selected.type == "cpu" and precision != "fp32":
            raise ValueError("Flow devices or CPU precision are unsupported.")
        if selected.type == "cuda" and (not torch.cuda.is_available() or selected.index is not None and selected.index >= torch.cuda.device_count()):
            raise ValueError("The selected CUDA device is unavailable.")
        if selected.type == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("The selected CUDA device does not support bf16.")
        root = relative(root)
        if not (root / "model_index.json").is_file():
            raise FileNotFoundError("Flow loading requires a complete local Diffusers pipeline.")
        for name in ("transformer", "vae", "scheduler"):
            if not relative(root / name).is_dir():
                raise FileNotFoundError(root / name)
        try:
            from diffusers import FluxPipeline, StableDiffusion3Pipeline
        except ImportError as error:
            raise RuntimeError("Flow loading requires native Diffusers pipeline support.") from error
        pipeline = (FluxPipeline if family == "flux" else StableDiffusion3Pipeline).from_pretrained(root, torch_dtype=choices[precision], local_files_only=True)
        if type(pipeline.scheduler).__name__ != "FlowMatchEulerDiscreteScheduler":
            raise ValueError("Flow editing requires the declared native Euler velocity scheduler.")
        from engine.checkpoint import configuration, fingerprint
        result = cls(pipeline, family, selected, choices[precision], size, guidance, consistency)
        result.metadata.update({"weights": fingerprint(root), "configuration": configuration(root)})
        return result

    def key(self, embedding):
        return hashlib.sha256(embedding.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()

    def register(self, embedding, pool, positions=None):
        if embedding.ndim != 3 or embedding.shape[0] != 1 or pool.ndim != 2 or pool.shape[0] != 1 or not torch.isfinite(embedding).all() or not torch.isfinite(pool).all():
            raise ValueError("Flow conditions require finite token and pooled embeddings.")
        key = self.key(embedding)
        if key in self.pools and not torch.equal(self.pools[key], pool):
            raise ValueError("Identical token embeddings have ambiguous pooled conditions.")
        if self.family == "flux" and positions is None:
            raise ValueError("Flux conditions require matched text positions.")
        if positions is not None:
            if positions.ndim != 2 or positions.shape != (embedding.shape[1], 3) or not torch.isfinite(positions).all():
                raise ValueError("Flux text positions must match token geometry.")
            if key in self.textids and not torch.equal(self.textids[key], positions):
                raise ValueError("Matched token embeddings require identical text positions.")
        self.pools[key] = pool.detach().clone()
        if positions is not None:
            self.textids[key] = positions.detach().clone()
        return embedding

    @torch.no_grad()
    def prompt(self, value):
        count = sum(getattr(self.pipeline, name, None) is not None for name in ("text_encoder", "text_encoder_2", "text_encoder_3"))
        self.textcalls += max(1, count)
        options = {"prompt": value, "device": self.device, "num_images_per_prompt": 1, "max_sequence_length": self.tokens}
        if self.family == "flux":
            embedding, pool, positions = self.pipeline.encode_prompt(**options)
            if positions.ndim == 3 and positions.shape[0] == 1:
                positions = positions[0]
        else:
            embedding, _, pool, _ = self.pipeline.encode_prompt(**options, do_classifier_free_guidance=False)
            positions = None
        return self.register(embedding.to(self.device, self.dtype), pool.to(self.device, self.dtype), None if positions is None else positions.to(self.device, self.dtype))

    def blend(self, source, target, eta):
        first, second = self.key(source), self.key(target)
        positions = None
        if self.family == "flux":
            if not torch.equal(self.textids[first], self.textids[second]):
                raise ValueError("Embedding paths require matched Flux text positions.")
            positions = self.textids[first]
        embedding = (1 - eta) * target + eta * source
        pool = (1 - eta) * self.pools[second] + eta * self.pools[first]
        return self.register(embedding, pool, positions)

    def encode(self, pixels):
        return (self.vae.encode((pixels * 2 - 1).to(self.dtype)).latent_dist.mode() - self.shift) * self.factor

    def decode(self, latent):
        return self.vae.decode((latent / self.factor + self.shift).to(self.dtype)).sample.float()

    def velocity(self, state, tick, embedding):
        if state.ndim != 4 or embedding.ndim != 3 or state.shape[0] != embedding.shape[0] or state.device != self.sigmas.device or not torch.isfinite(state).all() or not torch.isfinite(embedding).all():
            raise ValueError("Flow states and conditions must have matched finite batches.")
        if isinstance(tick, bool) or isinstance(tick, Integral) and not 0 <= tick <= self.training:
            raise ValueError("Flow ticks must lie on the declared actual sigma grid.")
        sigma = float(self.sigmas[int(tick)]) if isinstance(tick, Integral) else number(tick, "sigma", 0, 1)
        keys = [self.key(row[None]) for row in embedding]
        if any(key not in self.pools for key in keys):
            raise ValueError("Pooled conditions must be registered before velocity queries.")
        pools = torch.cat([self.pools[key] for key in keys])
        times = torch.full((state.shape[0],), sigma * self.training, device=self.device, dtype=self.dtype)
        options = {"encoder_hidden_states": embedding.to(self.dtype), "pooled_projections": pools.to(self.dtype), "return_dict": False}
        if self.family == "flux":
            positions = [self.textids[key] for key in keys]
            if any(not torch.equal(positions[0], row) for row in positions[1:]):
                raise ValueError("Batched Flux conditions require matched text positions.")
            guidance = torch.full((state.shape[0],), self.guidance, device=self.device, dtype=torch.float32) if getattr(self.unet.config, "guidance_embeds", False) else None
            options.update({"hidden_states": pack(state.to(self.dtype)), "timestep": times / 1000, "guidance": guidance, "txt_ids": positions[0], "img_ids": identifiers(*state.shape[-2:], self.device, self.dtype)})
        else:
            options.update({"hidden_states": state.to(self.dtype), "timestep": times})
        output = self.unet(**options)[0]
        result = unpack(output, state.shape) if self.family == "flux" else output
        if result.shape != state.shape or not torch.isfinite(result).all():
            raise RuntimeError("Native flow velocities have invalid geometry or values.")
        return result.float()

    def state(self, state, tick, embedding):
        sigma = self.sigmas[tick]
        return state.float() - sigma * self.velocity(state, tick, embedding)

    def predict(self, anchor, noise, tick, embedding):
        sigma = self.sigmas[tick]
        state = (1 - sigma) * anchor.float() + sigma * noise.float()
        return self.state(state, tick, embedding)

class Native(Original):
    def __init__(self, session, steps=1, guidance=1., cfg=False, path="auto"):
        if type(steps) is not int or steps < 1 or steps > session.editor.training or type(cfg) is not bool:
            raise ValueError("Flow continuation steps or CFG capability are invalid.")
        self.session, self.steps, self.cfg = session, steps, cfg
        self.guidance = number(guidance, "guidance", 0)
        if not cfg and self.guidance != 1:
            raise ValueError("Nonunit guidance requires explicit true CFG.")
        if path == "auto":
            path = "prediction" if cfg else "embedding"
        if path not in {"embedding", "prediction", "discrete"}:
            raise ValueError("Unknown flow reference path.")
        self.path, self.continuous = path, path != "discrete"
        self.affine = steps == 1 and path == "prediction" or not session.cleanup
        self.unconditional, self.endpoints = None, 0
        start = float(session.editor.sigmas[session.editor.index(session.end)])
        if session.cleanup and not 0 < start < 1:
            raise ValueError("Editable flow continuation requires an interior starting sigma.")
        self.sigmas = torch.linspace(start, 0, steps + 1, device=session.anchor.device, dtype=torch.float32)

    def conditions(self, eta):
        if self.path == "embedding" and 0 < eta < 1:
            self.session.editor.blend(self.session.source, self.session.target, eta)
        return super().conditions(eta)

    @torch.no_grad()
    def endpoint(self, field, eta=0.):
        # Integrate native velocities using actual sigma increments.
        tensor(field)
        eta = number(eta, "eta", 0, 1)
        session = self.session
        if field.shape != session.anchor.shape or field.device != session.anchor.device or not self.continuous and eta not in {0., 1.}:
            raise ValueError("Flow fields or reference coordinates are invalid.")
        if session.limit is not None:
            session.limit.require(self.cost(eta), self.queries())
        clean = session.anchor.float() + session.scale * field.float()
        if not session.cleanup:
            self.endpoints += 1
            return clean
        embeddings, weights = self.conditions(eta)
        count = len(weights)
        start = self.sigmas[0]
        state = (1 - start) * clean + start * session.noises[:1].float()
        for index in range(self.steps):
            sigma = float(self.sigmas[index])
            output = session.velocity(state.expand(count, -1, -1, -1), sigma, embeddings)
            velocity = sum((weight * row[None] for weight, row in zip(weights, output)), torch.zeros_like(state))
            state = state + (self.sigmas[index + 1] - self.sigmas[index]) * velocity
        self.endpoints += 1
        return tensor(state).detach().clone()

    def metadata(self):
        return {"kind": "flow", "architecture": self.session.editor.family, "steps": self.steps, "sigmas": self.sigmas.cpu().tolist(), "cfg": self.cfg, "guidance": self.guidance, "referencepath": self.path, "continuous": self.continuous, "affine": self.affine, "conditional": self.session.cleanup, "endpoints": self.endpoints, "sampler": "Deterministic Euler on explicitly declared actual sigma levels.", "scope": "Few-step numerical integration does not certify a distilled checkpoint's quality."}

class Consistency:
    def __init__(self, session, probes=2, seed=42, probabilities=None):
        if type(probes) is not int or probes < 1 or not getattr(session.editor, "velocityprobes", False):
            raise ValueError("Flow consistency requires a declared valid velocity-probe checkpoint.")
        self.session, self.probes, self.seed = session, probes, int(seed)
        self.evaluations, self.mode = 0, "flow"
        grid = session.editor.sigmas.detach().double().cpu()
        if grid.ndim != 1 or len(grid) < 3 or not torch.isfinite(grid).all() or torch.any(grid[1:] <= grid[:-1]) or grid[0] != 0 or grid[-1] != 1:
            raise ValueError("Flow consistency requires a declared monotone actual sigma grid.")
        count = len(grid) - 2
        probabilities = torch.ones(count, dtype=torch.float64) if probabilities is None else torch.as_tensor(probabilities, dtype=torch.float64)
        if probabilities.shape != (count,) or not torch.isfinite(probabilities).all() or torch.any(probabilities <= 0):
            raise ValueError("Flow probes must cover all interior actual sigma levels.")
        probabilities = probabilities / probabilities.max()
        probabilities /= probabilities.sum()
        if torch.any(probabilities <= 0):
            raise ValueError("Flow probabilities underflow after normalization.")
        chooser = torch.Generator(device="cpu").manual_seed(self.seed)
        self.ticks = (torch.multinomial(probabilities, probes, replacement=True, generator=chooser) + 1).tolist()
        self.probabilities = probabilities.tolist()
        self.weights = [1 / (count * float(probabilities[tick - 1])) for tick in self.ticks]
        if any(not math.isfinite(value) or value <= 0 for value in self.weights):
            raise ValueError("Flow importance weights must be finite and positive.")
        generator = torch.Generator(device=session.anchor.device).manual_seed(self.seed ^ 0x531)
        self.noises = [torch.randn(session.anchor.shape, generator=generator, device=session.anchor.device, dtype=torch.float32) for _ in self.ticks]

    @torch.no_grad()
    def __call__(self, latent, phase="prior"):
        # Compare native velocity with the sampled pair displacement.
        tensor(latent)
        session = self.session
        if latent.shape != session.anchor.shape or latent.device != session.anchor.device:
            raise ValueError("Flow consistency latents must match source geometry.")
        previous = session.limit.phase if session.limit is not None else None
        if session.limit is not None:
            session.limit.require(self.probes, self.probes)
            session.limit.phase = phase
        values = []
        try:
            for tick, noise, weight in zip(self.ticks, self.noises, self.weights):
                sigma = session.editor.sigmas[tick]
                state = (1 - sigma) * latent.float() + sigma * noise
                velocity = session.velocity(state, tick, session.target)
                values.append(weight * (velocity.double() - (noise.double() - latent.double())).square().mean())
            result = float(torch.stack(values).mean())
            if not math.isfinite(result):
                raise RuntimeError("Flow consistency produced nonfinite values.")
            self.evaluations += 1
            return result
        finally:
            if session.limit is not None:
                session.limit.phase = previous

    def metadata(self):
        return {"kind": "flow", "probes": self.probes, "ticks": self.ticks, "sigmas": [float(self.session.editor.sigmas[tick]) for tick in self.ticks], "seed": self.seed, "weights": self.weights, "probabilities": self.probabilities, "evaluations": self.evaluations, "velocitydirection": "data-to-noise", "noise": "Shared independent Gaussian endpoint per probe.", "note": "Conditional pair velocity residual is not likelihood, manifold distance, or convergence proof."}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
