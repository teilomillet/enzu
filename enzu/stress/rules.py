"""
Built-in fault injection rules.

Rules are composable building blocks for scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Set

from enzu.stress.scenario import CallContext, Fault


@dataclass(frozen=True)
class LatencyRule:
    """
    Add latency with a given probability.

    Example:
        LatencyRule(p=0.3, delay_seconds=2.0)  # 30% chance of 2s delay
    """

    p: float
    delay_seconds: float

    def maybe_fault(self, ctx: CallContext) -> Optional[Fault]:
        if ctx.rng.random() < self.p:
            return Fault(delay_seconds=self.delay_seconds)
        return None


@dataclass(frozen=True)
class ErrorRateRule:
    """
    Raise an exception with a given probability.

    Example:
        ErrorRateRule(p=0.2, exc_factory=faults.rate_limit_429)
    """

    p: float
    exc_factory: Callable[[], Exception]

    def maybe_fault(self, ctx: CallContext) -> Optional[Fault]:
        if ctx.rng.random() < self.p:
            return Fault(exception=self.exc_factory())
        return None


@dataclass(frozen=True)
class BurstRule:
    """
    Raise exceptions for a contiguous window of calls.

    Useful for simulating provider outages.

    Example:
        BurstRule(start_call=5, length=10, exc_factory=faults.server_error_500)
        # Calls 5-14 will fail
    """

    start_call: int
    length: int
    exc_factory: Callable[[], Exception]

    def maybe_fault(self, ctx: CallContext) -> Optional[Fault]:
        if self.start_call <= ctx.call_index < self.start_call + self.length:
            return Fault(exception=self.exc_factory())
        return None


@dataclass(frozen=True)
class NthCallRule:
    """
    Fail at specific call indices (deterministic).

    Useful for reproducible testing.

    Example:
        NthCallRule(indices={1, 5, 10}, exc_factory=faults.rate_limit_429)
    """

    indices: Set[int]
    exc_factory: Callable[[], Exception]

    def maybe_fault(self, ctx: CallContext) -> Optional[Fault]:
        if ctx.call_index in self.indices:
            return Fault(exception=self.exc_factory())
        return None


@dataclass(frozen=True)
class LatencyWithErrorRule:
    """
    Add latency AND raise an error with given probability.

    Simulates slow failures (common in timeouts).
    """

    p: float
    delay_seconds: float
    exc_factory: Callable[[], Exception]

    def maybe_fault(self, ctx: CallContext) -> Optional[Fault]:
        if ctx.rng.random() < self.p:
            return Fault(delay_seconds=self.delay_seconds, exception=self.exc_factory())
        return None


def latency(p: float, delay_seconds: float) -> LatencyRule:
    """Convenience: add latency with probability p."""
    return LatencyRule(p=p, delay_seconds=delay_seconds)


def error_rate(p: float, exc_factory: Callable[[], Exception]) -> ErrorRateRule:
    """Convenience: raise errors with probability p."""
    return ErrorRateRule(p=p, exc_factory=exc_factory)


def burst(start: int, length: int, exc_factory: Callable[[], Exception]) -> BurstRule:
    """Convenience: fail calls in range [start, start+length)."""
    return BurstRule(start_call=start, length=length, exc_factory=exc_factory)


def nth_call(indices: Set[int], exc_factory: Callable[[], Exception]) -> NthCallRule:
    """Convenience: fail at specific call indices."""
    return NthCallRule(indices=indices, exc_factory=exc_factory)
