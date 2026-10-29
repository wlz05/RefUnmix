# -*- coding: utf-8 -*-
# RefUnmix: Endpoint-Aligned Fields for Protected Image Editing

from dataclasses import dataclass, field
import math
from numbers import Real
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
        self.check()

    def check(self):
        if type(self.maximum) is not int or self.maximum < 1 or self.calls is not None and (type(self.calls) is not int or self.calls < 1):
            raise ValueError("Model budgets must be positive integers.")
        try:
            if self.seconds is not None and (isinstance(self.seconds, bool) or not isinstance(self.seconds, Real) or not math.isfinite(self.seconds) or self.seconds <= 0):
                raise ValueError("Wall budgets must be finite and positive.")
            if isinstance(self.started, bool) or not isinstance(self.started, Real) or not math.isfinite(self.started):
                raise ValueError("Budget clocks must be finite numbers.")
        except OverflowError as error:
            raise ValueError("Budget clocks exceed numerical range.") from error
        if type(self.used) is not int or type(self.forwards) is not int or min(self.used, self.forwards) < 0 or self.used > self.maximum or self.calls is not None and self.forwards > self.calls:
            raise ValueError("Budget counters must be valid nonnegative integers.")
        if not isinstance(self.phase, str) or not self.phase or not self.phase.isascii() or not isinstance(self.stages, dict):
            raise ValueError("Budget stages require named counters.")
        for key, row in self.stages.items():
            if not isinstance(key, str) or not key or not key.isascii() or not isinstance(row, dict) or set(row) != {"samples", "calls"} or any(type(value) is not int or value < 0 for value in row.values()):
                raise ValueError("Budget stages require complete nonnegative counters.")

    def require(self, samples=0, calls=0):
        self.check()
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
        self.check()
        return {"maximum": self.maximum, "calllimit": self.calls, "walllimit": self.seconds, "samples": self.used, "calls": self.forwards, "remaining": self.maximum - self.used, "elapsed": perf_counter() - self.started, "stages": {key: dict(value) for key, value in self.stages.items()}}

# Author: Weiliang Zhu; Other main author: Yeming Dai
# Email: wlzchina05@gmail.com
