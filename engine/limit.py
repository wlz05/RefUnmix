# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from dataclasses import dataclass, field
import math
from time import perf_counter

class Exhausted(RuntimeError):
    pass

@dataclass
class Budget:
    maximum: int = 64
    calls: int | None = None
    seconds: float | None = None
    started: float = field(default_factory=perf_counter)
    used: int = 0
    forwards: int = 0
    phase: str = "front"
    stages: dict = field(default_factory=dict)

    def __post_init__(self):
        if type(self.maximum) is not int or self.maximum < 1 or self.calls is not None and (type(self.calls) is not int or self.calls < 1):
            raise ValueError("Model budgets must be positive integers.")
        if self.seconds is not None and (isinstance(self.seconds, bool) or not math.isfinite(self.seconds) or self.seconds <= 0):
            raise ValueError("Wall budgets must be finite and positive.")

    def require(self, samples=0, calls=0):
        if type(samples) is not int or type(calls) is not int or min(samples, calls) < 0:
            raise ValueError("Query counts must be nonnegative integers.")
        if self.seconds is not None and perf_counter() - self.started >= self.seconds:
            raise Exhausted("The image wall budget was exhausted.")
        if self.used + samples > self.maximum or self.calls is not None and self.forwards + calls > self.calls:
            raise Exhausted("The shared model budget was exhausted.")

    def consume(self, samples, calls=1):
        # Count samples even when predictions are batched.
        self.require(samples, calls)
        self.used += samples
        self.forwards += calls
        row = self.stages.setdefault(self.phase, {"samples": 0, "calls": 0})
        row["samples"] += samples
        row["calls"] += calls

    def report(self):
        return {"maximum": self.maximum, "calllimit": self.calls, "walllimit": self.seconds, "samples": self.used, "calls": self.forwards, "remaining": self.maximum - self.used, "elapsed": perf_counter() - self.started, "stages": {key: dict(value) for key, value in self.stages.items()}}

# Author: Weiliang Zhu 2026
# Email: wlzchina05@gmail.com
