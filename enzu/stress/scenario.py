"""
Scenario and fault definitions for stress testing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Protocol


@dataclass(frozen=True)
class CallContext:
    """
    Context for each provider call.

    Attributes:
        call_index: Increments on every provider call (including retries).
        run_index: Which run this call belongs to.
        rng: Seeded random number generator for reproducibility.
    """

    call_index: int
    run_index: int
    rng: random.Random


@dataclass(frozen=True)
class Fault:
    """
    A fault to inject before a provider call.

    Attributes:
        delay_seconds: How long to delay before the call.
        exception: Exception to raise (if any).
    """

    delay_seconds: float = 0.0
    exception: Optional[Exception] = None


class FaultRule(Protocol):
    """Protocol for fault injection rules."""

    def maybe_fault(self, ctx: CallContext) -> Optional[Fault]:
        """
        Decide whether to inject a fault for this call.

        Args:
            ctx: Call context with index and RNG.

        Returns:
            Fault to inject, or None for no fault.
        """
        ...


@dataclass
class Scenario:
    """
    A named collection of fault rules.

    Rules are evaluated in order; first match wins.

    Attributes:
        name: Human-readable scenario name.
        rules: List of FaultRule to apply.
    """

    name: str
    rules: List[FaultRule] = field(default_factory=list)

    def decide(self, ctx: CallContext) -> Optional[Fault]:
        """
        Evaluate rules and return the first matching fault.

        Args:
            ctx: Call context.

        Returns:
            First matching Fault, or None.
        """
        for rule in self.rules:
            fault = rule.maybe_fault(ctx)
            if fault is not None:
                return fault
        return None

    def __add__(self, other: "Scenario") -> "Scenario":
        """Combine scenarios by concatenating rules."""
        return Scenario(
            name=f"{self.name}+{other.name}",
            rules=list(self.rules) + list(other.rules),
        )
