# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as functional

def check(value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[0] != 1 or min(value.shape[1:]) < 1 or not value.is_floating_point():
        raise ValueError("Expected one floating point NCHW tensor.")
    if not torch.isfinite(value).all().item():
        raise ValueError("Tensor values must be finite.")

def split(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Reconstruct aligned Haar approximations and details.
    check(value)
    height, width = value.shape[-2:]
    padded = functional.pad(value.float(), (0, width % 2, 0, height % 2), mode="replicate")
    low = functional.avg_pool2d(padded, 2).repeat_interleave(2, -2).repeat_interleave(2, -1)[..., :height, :width]
    return low, value.float() - low

def gates(value: torch.Tensor, grid: int = 2) -> list[torch.Tensor]:
    # Keep gates constant within Haar blocks.
    check(value)
    height, width = value.shape[-2:]
    rows, columns = (height + 1) // 2, (width + 1) // 2
    if type(grid) is not int or not 1 <= grid <= min(rows, columns):
        raise ValueError("grid must fit the latent Haar lattice.")
    row = torch.arange(height, device=value.device) // 2 * grid // rows
    column = torch.arange(width, device=value.device) // 2 * grid // columns
    return [((row[:, None] == y) & (column[None, :] == x)).float()[None, None] for y in range(grid) for x in range(grid)]

@dataclass
class Atom:
    band: str
    region: int
    time: str
    value: torch.Tensor
    aliases: tuple[str, ...] = ()

    @property
    def group(self) -> str:
        return f"{self.region}:{self.band}"

    def metadata(self) -> dict:
        return {"band": self.band, "region": self.region, "time": self.time, "aliases": list(self.aliases)}

def sources(base: torch.Tensor, pool: dict[float, torch.Tensor]) -> dict[str, torch.Tensor]:
    check(base)
    result = {"base": base.float(), "zero": torch.zeros_like(base, dtype=torch.float32)}
    if not isinstance(pool, dict) or not pool:
        raise ValueError("At least one candidate time is required.")
    for time, value in sorted(pool.items()):
        if isinstance(time, bool) or not isinstance(time, (int, float)) or not math.isfinite(time) or not 0 < time <= 1:
            raise ValueError("Candidate times must be finite and in (0, 1].")
        check(value)
        if value.shape != base.shape or value.device != base.device:
            raise ValueError("Candidate fields must share the baseline shape and device.")
        result[str(float(time))] = value.float()
    return result

def dictionary(base: torch.Tensor, current: torch.Tensor, pool: dict[float, torch.Tensor], grid: int = 2, bands: str = "both", windows: list[torch.Tensor] | None = None) -> list[Atom]:
    # Subtract global fields before local replacement.
    check(current)
    if current.shape != base.shape or current.device != base.device or bands not in {"both", "low", "high"}:
        raise ValueError("Invalid dictionary shape or band selection.")
    windows = gates(base, grid) if windows is None else windows
    if not windows:
        return []
    total = torch.zeros_like(base[:, :1], dtype=torch.float32)
    for gate in windows:
        check(gate)
        if gate.shape != total.shape or gate.device != base.device or torch.any((gate != 0) & (gate != 1)).item() or float(split(gate)[1].abs().max()) > 1e-8:
            raise ValueError("Dictionary gates must be disjoint binary Haar-aligned blocks.")
        total += gate
    if torch.any(total > 1).item():
        raise ValueError("Dictionary gates cannot overlap.")
    entries = sources(base, pool)
    output = []
    for index, band in enumerate(("low", "high")):
        if bands not in {"both", band}:
            continue
        for region, gate in enumerate(windows):
            unique = []
            for time, candidate in entries.items():
                value = split(candidate - current)[index] * gate
                if float(value.abs().max()) <= 1e-8:
                    continue
                duplicate = next((atom for atom in unique if torch.allclose(value, atom.value, rtol=0, atol=1e-8)), None)
                if duplicate is not None:
                    duplicate.aliases = (*duplicate.aliases, time)
                    continue
                atom = Atom(band, region, time, value)
                unique.append(atom)
                output.append(atom)
    return output

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
