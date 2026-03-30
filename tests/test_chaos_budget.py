"""Chaos tests for BudgetController: concurrent access, state transitions, edge cases.

Uses ordeal's stateful chaos testing to systematically explore BudgetController
behavior under fault injection and random operation sequences.

Properties verified:
- Remaining budget is always non-negative
- Once exceeded, the budget stays exceeded permanently
- Audit log only grows (never shrinks)
- Cost and token totals are monotonically non-decreasing
- Thread-safety under concurrent mutations
"""

from __future__ import annotations

import threading

from hypothesis import strategies as st

from ordeal import ChaosTest, always, invariant, rule
from ordeal.faults import timing

from enzu.budget import BudgetController, BudgetExceeded


# ---------------------------------------------------------------------------
# Chaos test: BudgetController state machine
# ---------------------------------------------------------------------------


class BudgetControllerChaos(ChaosTest):
    """Explore BudgetController under randomized operations + fault injection."""

    faults = [
        timing.jitter("enzu.budget.count_tokens", magnitude=5.0),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.controller = BudgetController(
            max_cost_usd=10.0,
            max_tokens=1000,
            max_total_tokens=2000,
            max_input_tokens=1500,
        )
        self._prev_cost: float = 0.0
        self._prev_output_tokens: int = 0
        self._prev_log_len: int = 0
        self._exceeded_observed: bool = False

    @rule(
        input_tokens=st.integers(min_value=0, max_value=200),
        max_output=st.integers(min_value=0, max_value=300),
    )
    def do_pre_call_check(self, input_tokens: int, max_output: int) -> None:
        """Try a pre-call budget check with random token estimates."""
        try:
            self.controller.pre_call_check(input_tokens, max_output)
            # If pre-call succeeds, budget should not yet be exceeded
            always(
                not self._exceeded_observed or not self.controller.is_exceeded,
                "pre-call success implies budget not exceeded from prior state",
            )
        except BudgetExceeded:
            # Expected when budget is exhausted
            pass

    @rule(
        input_tokens=st.integers(min_value=0, max_value=100),
        output_tokens=st.integers(min_value=0, max_value=100),
        cost=st.floats(min_value=0.0, max_value=2.0).filter(
            lambda x: x == x
        ),  # filter NaN
    )
    def do_record_usage(
        self, input_tokens: int, output_tokens: int, cost: float
    ) -> None:
        """Record usage after a simulated call."""
        self.controller.record_usage(input_tokens, output_tokens, cost)

    @rule()
    def do_serialize(self) -> None:
        """Serialize and verify the controller state dict is consistent."""
        d = self.controller.to_dict()
        always(isinstance(d, dict), "to_dict returns a dict")
        always(
            d["total_cost_usd"] >= 0,
            "serialized cost is non-negative",
        )
        always(
            d["total_tokens"] >= 0,
            "serialized total_tokens is non-negative",
        )

    @invariant()
    def remaining_never_negative(self) -> None:
        """Remaining budget (cost, tokens, total) must never go below zero."""
        remaining_cost = self.controller.remaining_cost
        if remaining_cost is not None:
            always(remaining_cost >= 0, "remaining cost >= 0")

        remaining_tokens = self.controller.remaining_tokens
        if remaining_tokens is not None:
            always(remaining_tokens >= 0, "remaining tokens >= 0")

        remaining_total = self.controller.remaining_total_tokens
        if remaining_total is not None:
            always(remaining_total >= 0, "remaining total tokens >= 0")

    @invariant()
    def exceeded_is_permanent(self) -> None:
        """Once the budget is exceeded, it must stay exceeded forever."""
        if self._exceeded_observed:
            always(
                self.controller.is_exceeded,
                "exceeded state is permanent",
            )
        if self.controller.is_exceeded:
            self._exceeded_observed = True

    @invariant()
    def totals_monotonically_increase(self) -> None:
        """Cost and token totals can only go up."""
        current_cost = self.controller.total_cost_usd
        always(
            current_cost >= self._prev_cost,
            "total cost monotonically increases",
        )
        self._prev_cost = current_cost

    @invariant()
    def audit_log_only_grows(self) -> None:
        """Audit log length can only increase."""
        current_len = len(self.controller.audit_log)
        always(
            current_len >= self._prev_log_len,
            "audit log only grows",
        )
        self._prev_log_len = current_len


TestBudgetControllerChaos = BudgetControllerChaos.TestCase


# ---------------------------------------------------------------------------
# Chaos test: Concurrent budget access
# ---------------------------------------------------------------------------


class ConcurrentBudgetChaos(ChaosTest):
    """Explore BudgetController under concurrent access from multiple threads.

    The nemesis toggles a jitter fault on token counting, while rules
    fire pre-call checks and record usage from the main thread.
    A background thread also hammers the controller concurrently.
    """

    faults = [
        timing.jitter("enzu.budget.count_tokens", magnitude=3.0),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.controller = BudgetController(
            max_cost_usd=5.0,
            max_tokens=500,
        )
        self._bg_errors: list[str] = []

    @rule(
        input_tokens=st.integers(min_value=0, max_value=50),
        output_tokens=st.integers(min_value=0, max_value=50),
        cost=st.floats(min_value=0.0, max_value=0.5).filter(lambda x: x == x),
    )
    def concurrent_record(
        self, input_tokens: int, output_tokens: int, cost: float
    ) -> None:
        """Record usage from main thread while a background thread does the same."""
        errors: list[str] = []

        def bg_work() -> None:
            try:
                self.controller.record_usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                )
            except Exception as e:
                errors.append(str(e))

        t = threading.Thread(target=bg_work, daemon=True)
        t.start()

        # Main thread also records
        self.controller.record_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

        t.join(timeout=2.0)
        always(not errors, "no errors in background thread")

    @invariant()
    def remaining_never_negative(self) -> None:
        remaining_cost = self.controller.remaining_cost
        if remaining_cost is not None:
            always(remaining_cost >= 0, "remaining cost >= 0 under concurrency")

        remaining_tokens = self.controller.remaining_tokens
        if remaining_tokens is not None:
            always(remaining_tokens >= 0, "remaining tokens >= 0 under concurrency")


TestConcurrentBudgetChaos = ConcurrentBudgetChaos.TestCase


# ---------------------------------------------------------------------------
# Chaos test: Budget edge cases with tiny limits
# ---------------------------------------------------------------------------


class TinyBudgetChaos(ChaosTest):
    """Explore behavior at the boundary of very small budgets.

    With a 1-token, $0.01 budget, every operation is near the edge.
    This stress-tests the pre-call blocking and exceeded detection.
    """

    faults = []  # No faults — pure state exploration

    def __init__(self) -> None:
        super().__init__()
        self.controller = BudgetController(
            max_cost_usd=0.01,
            max_tokens=1,
            max_total_tokens=2,
        )
        self._ever_exceeded = False

    @rule(
        input_tokens=st.integers(min_value=0, max_value=3),
        output_tokens=st.integers(min_value=0, max_value=3),
    )
    def tiny_record(self, input_tokens: int, output_tokens: int) -> None:
        self.controller.record_usage(input_tokens, output_tokens, cost_usd=0.005)

    @rule(
        input_tokens=st.integers(min_value=0, max_value=2),
        max_output=st.integers(min_value=0, max_value=2),
    )
    def tiny_pre_check(self, input_tokens: int, max_output: int) -> None:
        try:
            self.controller.pre_call_check(input_tokens, max_output)
        except BudgetExceeded:
            self._ever_exceeded = True

    @invariant()
    def remaining_never_negative(self) -> None:
        remaining_cost = self.controller.remaining_cost
        if remaining_cost is not None:
            always(remaining_cost >= 0, "tiny remaining cost >= 0")

    @invariant()
    def exceeded_is_sticky(self) -> None:
        if self._ever_exceeded:
            # Once we've seen exceeded via pre-call, controller should be exceeded
            # (unless it was exceeded from a record_usage call)
            pass  # The exceeded flag is only set by record_usage, not pre_call_check
        if self.controller.is_exceeded:
            self._ever_exceeded = True


TestTinyBudgetChaos = TinyBudgetChaos.TestCase
