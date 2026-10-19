# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as functional

@dataclass
class FrequencyOutput:
    field: torch.Tensor
    mask: torch.Tensor
    target: torch.Tensor
    maps: dict[str, torch.Tensor]

class FrequencyRouter:
    def __init__(self, preserve: float = 0.5, couple: float = 0.25, ceiling: float = 1.5, epsilon: float = 1e-6) -> None:
        self.preserve = preserve
        self.couple = couple
        self.ceiling = ceiling
        self.epsilon = epsilon

    def low(self, value: torch.Tensor) -> torch.Tensor:
        height, width = value.shape[-2:]
        padded = functional.pad(value, (0, width % 2, 0, height % 2), mode="replicate")
        pooled = functional.avg_pool2d(padded, 2)
        return pooled.repeat_interleave(2, -2).repeat_interleave(2, -1)[..., :height, :width]

    def high(self, value: torch.Tensor) -> torch.Tensor:
        return value - self.low(value)

    def norm(self, value: torch.Tensor) -> torch.Tensor:
        return value.float().square().mean(dim=(1, 2, 3), keepdim=True).sqrt()

    def trust(self, change: torch.Tensor, source: torch.Tensor, anchor: torch.Tensor, drift: torch.Tensor) -> torch.Tensor:
        signal = self.norm(change)
        error = self.norm(source - anchor)
        uncertainty = self.norm(drift)
        return signal / (signal + error + 0.5 * uncertainty + self.epsilon)

    def activity(self, value: torch.Tensor) -> torch.Tensor:
        energy = self.high(value).square().mean(dim=1, keepdim=True)
        return functional.avg_pool2d(energy, 3, 1, 1).sqrt()

    def edge(self, value: torch.Tensor) -> torch.Tensor:
        smooth = functional.avg_pool2d(functional.pad(self.low(value), (1, 1, 1, 1), mode="replicate"), 3, 1)
        padded = functional.pad(smooth, (1, 1, 1, 1), mode="replicate")
        horizontal = 0.5 * (padded[..., 1:-1, 2:] - padded[..., 1:-1, :-2])
        vertical = 0.5 * (padded[..., 2:, 1:-1] - padded[..., :-2, 1:-1])
        return (horizontal.square() + vertical.square()).mean(dim=1, keepdim=True).sqrt()

    def cross(self, value: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        high = self.activity(value)
        low = self.edge(value)
        return torch.cat((high * self.edge(source), low * self.activity(source), high * low), dim=1)

    def solve(self, anchor: torch.Tensor, low: torch.Tensor, high: torch.Tensor, mask: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
        energy = (low.square() + high.square()).mean()
        if float(energy) <= self.epsilon ** 2:
            return torch.zeros(2, device=anchor.device, dtype=anchor.dtype)
        fields = (low, high)
        sourcecross = self.cross(anchor, anchor)
        targetcross = self.cross(anchor + low + high, anchor)
        firstcross = self.cross(anchor + low, anchor) - sourcecross
        secondcross = self.cross(anchor + high, anchor) - sourcecross
        desired = targetcross - sourcecross
        scale = (sourcecross.square().mean() + targetcross.square().mean()).sqrt().clamp_min(self.epsilon)
        components = (firstcross / scale, secondcross / scale)
        desired = desired / scale
        matrix = torch.zeros(2, 2, device=anchor.device, dtype=anchor.dtype)
        vector = torch.zeros(2, device=anchor.device, dtype=anchor.dtype)
        for row in range(2):
            fit = reliability[row] * fields[row].square().mean() / energy
            vector[row] = fit + self.couple * (components[row] * desired).mean()
            for column in range(2):
                protect = ((1.0 - mask).square() * fields[row] * fields[column]).mean() / energy
                relation = (components[row] * components[column]).mean()
                matrix[row, column] = self.preserve * protect + self.couple * relation
            matrix[row, row] += fit + 0.01
        weights = torch.ones(2, device=anchor.device, dtype=anchor.dtype)
        for _ in range(32):
            for row in range(2):
                other = 1 - row
                weights[row] = ((vector[row] - matrix[row, other] * weights[other]) / matrix[row, row].clamp_min(self.epsilon)).clamp(0.0, self.ceiling)
        return weights

    def __call__(self, anchor: torch.Tensor, current: torch.Tensor, previous: torch.Tensor, sourcecurrent: torch.Tensor, sourceprevious: torch.Tensor, first: float, second: float, mask: torch.Tensor | None = None) -> FrequencyOutput:
        tensors = (anchor, current, previous, sourcecurrent, sourceprevious)
        if anchor.ndim != 4 or any(value.shape != anchor.shape or value.device != anchor.device or not value.is_floating_point() for value in tensors):
            raise ValueError("Frequency inputs must share a floating point image shape and device.")
        if any(not torch.isfinite(value).all().item() for value in tensors):
            raise ValueError("Frequency inputs must be finite.")
        work = [value.float() for value in tensors]
        anchor, current, previous, sourcecurrent, sourceprevious = work
        drift = current - previous
        choices = ((self.low(current), self.low(previous), self.low(sourcecurrent), self.low(sourceprevious), self.low(drift)), (self.high(current), self.high(previous), self.high(sourcecurrent), self.high(sourceprevious), self.high(drift)))
        fields = []
        reliabilities = []
        times = []
        for early, late, sourceearly, sourcelate, change in choices:
            qualityearly = self.trust(early, sourceearly, self.low(anchor) if len(fields) == 0 else self.high(anchor), change)
            qualitylate = self.trust(late, sourcelate, self.low(anchor) if len(fields) == 0 else self.high(anchor), change)
            select = qualityearly >= qualitylate
            fields.append(torch.where(select, early, late))
            reliabilities.append(torch.where(select, qualityearly, qualitylate))
            times.append(torch.where(select, torch.full_like(qualityearly, first), torch.full_like(qualitylate, second)))
        low, high = fields
        magnitude = (low.square() + high.square()).sum(dim=1, keepdim=True).sqrt()
        cutoff = torch.quantile(magnitude.flatten(start_dim=2), 0.90, dim=2, keepdim=True).unsqueeze(-1).clamp_min(self.epsilon)
        explicit = mask is not None
        if explicit:
            if mask.shape != magnitude.shape or not torch.isfinite(mask).all() or torch.any((mask < 0) | (mask > 1)):
                raise ValueError("Edit masks must match the spatial shape and lie in [0, 1].")
            mask = mask.to(device=anchor.device, dtype=torch.float32)
        else:
            mask = functional.avg_pool2d(functional.pad((magnitude / cutoff).clamp(0.0, 1.0), (1, 1, 1, 1), mode="replicate"), 3, 1)
        weights = torch.stack([self.solve(anchor[index:index + 1], low[index:index + 1], high[index:index + 1], mask[index:index + 1], torch.stack([reliabilities[0][index, 0, 0, 0], reliabilities[1][index, 0, 0, 0]])) for index in range(anchor.shape[0])])
        field = weights[:, 0, None, None, None] * low + weights[:, 1, None, None, None] * high
        target = anchor + low + high
        maps = {"mask": mask, "lowfield": low, "highfield": high, "lowweight": weights[:, 0:1, None, None], "highweight": weights[:, 1:2, None, None], "lowtrust": reliabilities[0], "hightrust": reliabilities[1], "lowtime": times[0], "hightime": times[1]}
        maps.update({"earlylow": self.low(current), "earlyhigh": self.high(current), "latelow": self.low(previous), "latehigh": self.high(previous), "earlytime": torch.tensor(first, device=anchor.device), "latetime": torch.tensor(second, device=anchor.device), "explicit": torch.tensor(explicit, device=anchor.device)})
        return FrequencyOutput(field.to(dtype=tensors[0].dtype), mask.to(dtype=tensors[0].dtype), target, maps)

    def audit(self, anchor: torch.Tensor, result: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        anchor = anchor.float()
        result = result.float()
        target = target.float()
        higherror = self.high(result) - self.high(target)
        lowerror = self.low(result) - self.low(target)
        highscale = (self.high(target) - self.high(anchor)).square().mean(dim=1, keepdim=True).sqrt().mean(dim=(2, 3), keepdim=True).clamp_min(self.epsilon)
        lowscale = (self.low(target) - self.low(anchor)).square().mean(dim=1, keepdim=True).sqrt().mean(dim=(2, 3), keepdim=True).clamp_min(self.epsilon)
        sourcecross = self.cross(anchor, anchor)
        resultcross = self.cross(result, anchor)
        targetcross = self.cross(target, anchor)
        crossscale = (targetcross - sourcecross).abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(self.epsilon)
        hh = higherror.square().mean(dim=1, keepdim=True).sqrt() / highscale
        ll = lowerror.square().mean(dim=1, keepdim=True).sqrt() / lowscale
        hl = (resultcross - targetcross).abs().mean(dim=1, keepdim=True) / crossscale
        protected = (1.0 - mask.float()) * (result - anchor).square().mean(dim=1, keepdim=True).sqrt()
        return {"proxyhh": hh, "proxyhl": hl, "proxyll": ll, "protected": protected}

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
