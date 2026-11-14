# -*- coding: utf-8 -*-
# RefUnmix: Reference-Grounded Response Unmixing for One-Step Image Editing

from types import MethodType
import torch

def fractions(sample, timestep):
    ticks = torch.as_tensor(timestep, device=sample.device).reshape(-1)
    if ticks.numel() not in (1, sample.shape[0]) or not torch.isfinite(ticks).all() or torch.any((ticks < 0) | (ticks > 1000)):
        raise ValueError("The InstaFlow clock must match the finite sample batch.")
    return (ticks.to(sample.dtype) / 1000).reshape(-1, 1, 1, 1)

def coefficients(self, sample, timestep):
    sigma = fractions(sample, timestep)
    if torch.any(sigma >= 1):
        raise ValueError("The official editing recovery interface requires sigma below one.")
    return 1 - sigma, sigma

def recovery(state, velocity, sigma):
    return state + sigma * velocity

def activate(pipeline):
    # The adapter is algebraic; official transport, masks and selection remain intact.
    original = pipeline.unet.forward
    def forward(module, sample, timestep, encoder_hidden_states, return_dict=True, **kwargs):
        result = original(sample=sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states, return_dict=return_dict, **kwargs)
        velocity = result.sample if return_dict else result[0]
        sigma = fractions(sample, timestep)
        equivalent = sample - (1 - sigma) * velocity
        if not torch.isfinite(equivalent).all():
            raise FloatingPointError("The native flow head returned nonfinite values.")
        return type(result)(sample=equivalent) if return_dict else (equivalent, *result[1:])
    pipeline._cross_native_forward = original
    pipeline.unet.forward = MethodType(forward, pipeline.unet)
    pipeline._max_unet_timestep = 1000
    pipeline._get_alpha_sigma = MethodType(coefficients, pipeline)
    return {"corruption": "(1-sigma)*anchor+sigma*noise", "clock": "round(1000*sigma)", "nativeoutput": "noise-to-data velocity", "clean": "state+sigma*velocity", "equivalenthead": "state-(1-sigma)*velocity", "identity": "(state-sigma*equivalenthead)/(1-sigma)=state+sigma*velocity", "scope": "Only native corruption, model clock and prediction parameterization change."}

def check():
    generator = torch.Generator().manual_seed(42)
    for batch in (1, 2, 4):
        anchor = torch.randn((batch, 4, 8, 8), generator=generator, dtype=torch.float64)
        noise = torch.randn(anchor.shape, generator=generator, dtype=anchor.dtype)
        velocity = torch.randn(anchor.shape, generator=generator, dtype=anchor.dtype)
        for tick in (0, 1, 300, 750, 900, 999):
            sigma = fractions(anchor, tick)
            state = (1 - sigma) * anchor + sigma * noise
            head = state - (1 - sigma) * velocity
            torch.testing.assert_close((state - sigma * head) / (1 - sigma), recovery(state, velocity, sigma), rtol=1e-10, atol=1e-10)
        state = noise
        torch.testing.assert_close(recovery(state, velocity, fractions(state, 1000)), noise + velocity)
    for invalid in (-1, 1001, float("nan")):
        try:
            fractions(anchor, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("An invalid native clock was accepted.")
    print("CROSS_FLOW_CPU_ORACLE_PASSED", flush=True)

# Author: Weiliang Zhu; Other main author: Yeming Dai, Yaming Lv
# Email: wlzchina05@gmail.com
