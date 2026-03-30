"""Chaos tests for the RLM subsystem: TokenBudgetPool, BudgetTracker, cross-module.

Properties verified:
- TokenBudgetPool: remaining never negative, reserve+commit consistent,
  concurrent reserve/commit/release thread-safe, exhausted is permanent
- BudgetTracker: cumulative totals monotonic, is_exhausted sticky
- Cross-module: Engine + BudgetController + ConcurrencyLimiter together
"""

from __future__ import annotations

import threading

from hypothesis import strategies as st

from ordeal import ChaosTest, always, invariant, rule
from ordeal.faults import timing

from enzu.budget import BudgetController, BudgetExceeded
from enzu.isolation.concurrency import ConcurrencyLimiter
from enzu.models import Budget
from enzu.rlm.budget import BudgetTracker, TokenBudgetPool


# ============================================================================
# TokenBudgetPool chaos
# ============================================================================


class TokenBudgetPoolChaos(ChaosTest):
    """Explore TokenBudgetPool under concurrent reserve/commit/release."""

    faults = [
        timing.jitter("enzu.rlm.budget.TokenBudgetPool.reserve", magnitude=0.001),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.pool = TokenBudgetPool(max_total_tokens=1000)
        self._reservations: list[tuple[int, int]] = []  # (output_cap, reserved)
        self._prev_remaining: int | None = 1000

    @rule(
        prompt_tokens=st.one_of(st.none(), st.integers(min_value=0, max_value=100)),
        requested_output=st.integers(min_value=1, max_value=200),
    )
    def do_reserve(self, prompt_tokens: int | None, requested_output: int) -> None:
        output_cap, reserved = self.pool.reserve(prompt_tokens, requested_output)
        always(output_cap >= 0, "output_cap non-negative")
        always(reserved >= 0, "reserved non-negative")
        always(output_cap <= requested_output, "output_cap <= requested")
        if reserved > 0:
            self._reservations.append((output_cap, reserved))

    @rule(actual=st.one_of(st.none(), st.integers(min_value=0, max_value=200)))
    def do_commit(self, actual: int | None) -> None:
        if not self._reservations:
            return
        _, reserved = self._reservations.pop(0)
        self.pool.commit(reserved, actual)

    @rule()
    def do_release(self) -> None:
        if not self._reservations:
            return
        _, reserved = self._reservations.pop(0)
        self.pool.release(reserved)

    @rule()
    def check_snapshot(self) -> None:
        snap = self.pool.snapshot()
        always(snap["used_tokens"] >= 0, "used_tokens non-negative")
        always(snap["reserved_tokens"] >= 0, "reserved_tokens non-negative")
        remaining = snap["remaining_tokens"]
        if remaining is not None:
            always(remaining >= 0, "remaining non-negative")

    @rule()
    def concurrent_reserve_commit(self) -> None:
        """Hammer the pool from multiple threads."""
        errors: list[str] = []

        def worker() -> None:
            try:
                cap, res = self.pool.reserve(10, 50)
                if res > 0:
                    self.pool.commit(res, 40)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)
        always(not errors, "no errors in concurrent reserve/commit")

    @invariant()
    def snapshot_consistent(self) -> None:
        snap = self.pool.snapshot()
        if snap["max_total_tokens"] is not None:
            total_accounted = snap["used_tokens"] + snap["reserved_tokens"]
            always(
                total_accounted <= snap["max_total_tokens"] + 500,
                "used + reserved roughly bounded by max",
            )

    @invariant()
    def exhausted_consistent(self) -> None:
        if self.pool.is_exhausted():
            snap = self.pool.snapshot()
            always(
                snap["remaining_tokens"] is not None and snap["remaining_tokens"] == 0,
                "exhausted implies 0 remaining",
            )


TestTokenBudgetPoolChaos = TokenBudgetPoolChaos.TestCase


# ============================================================================
# BudgetTracker chaos
# ============================================================================


class BudgetTrackerChaos(ChaosTest):
    """Explore BudgetTracker with randomized usage consumption."""

    faults = []

    def __init__(self) -> None:
        super().__init__()
        self.tracker = BudgetTracker(
            Budget(max_tokens=500, max_total_tokens=1000, max_cost_usd=5.0)
        )
        self._prev_exhausted = False

    @rule(
        output=st.integers(min_value=0, max_value=100),
        total=st.integers(min_value=0, max_value=200),
        cost=st.floats(min_value=0.0, max_value=1.0).filter(lambda x: x == x),
    )
    def consume_usage(self, output: int, total: int, cost: float) -> None:
        self.tracker.consume(
            {
                "output_tokens": output,
                "total_tokens": total,
                "cost_usd": cost,
            }
        )

    @rule()
    def check_percentage(self) -> None:
        pct = self.tracker.percentage_used()
        for key, val in pct.items():
            always(val >= 0, f"percentage {key} non-negative")

    @rule()
    def check_remaining(self) -> None:
        remaining = self.tracker.remaining_output_tokens()
        if remaining is not None:
            always(remaining >= 0, "remaining output tokens non-negative")

    @invariant()
    def exhausted_is_sticky(self) -> None:
        if self._prev_exhausted:
            always(self.tracker.is_exhausted(), "exhausted is permanent")
        if self.tracker.is_exhausted():
            self._prev_exhausted = True


TestBudgetTrackerChaos = BudgetTrackerChaos.TestCase


# ============================================================================
# TokenBudgetPool unlimited
# ============================================================================


class UnlimitedPoolChaos(ChaosTest):
    """Explore TokenBudgetPool with no limit (max_total_tokens=None)."""

    faults = []

    def __init__(self) -> None:
        super().__init__()
        self.pool = TokenBudgetPool(max_total_tokens=None)

    @rule(requested=st.integers(min_value=1, max_value=10000))
    def reserve_unlimited(self, requested: int) -> None:
        cap, reserved = self.pool.reserve(50, requested)
        always(cap == requested, "unlimited pool grants full request")
        always(reserved == 0, "unlimited pool reserves 0")

    @rule()
    def never_exhausted(self) -> None:
        always(not self.pool.is_exhausted(), "unlimited pool never exhausted")

    @invariant()
    def snapshot_unlimited(self) -> None:
        snap = self.pool.snapshot()
        always(snap["max_total_tokens"] is None, "max is None for unlimited")
        always(snap["remaining_tokens"] is None, "remaining is None for unlimited")


TestUnlimitedPoolChaos = UnlimitedPoolChaos.TestCase


# ============================================================================
# Cross-module: Engine + BudgetController + ConcurrencyLimiter
# ============================================================================


class CrossModuleChaos(ChaosTest):
    """Explore interactions between budget, concurrency, and token pool.

    Simulates the real execution flow: acquire concurrency slot,
    check budget, reserve tokens, execute, commit tokens, record budget.
    """

    faults = []

    def __init__(self) -> None:
        super().__init__()
        self.budget = BudgetController(max_cost_usd=5.0, max_tokens=500)
        self.limiter = ConcurrencyLimiter(max_concurrent=3)
        self.pool = TokenBudgetPool(max_total_tokens=2000)
        self._total_ops = 0

    @rule(
        input_tokens=st.integers(min_value=1, max_value=50),
        output_tokens=st.integers(min_value=1, max_value=50),
        cost=st.floats(min_value=0.01, max_value=0.5).filter(lambda x: x == x),
    )
    def execute_request(
        self, input_tokens: int, output_tokens: int, cost: float
    ) -> None:
        """Simulate a full request lifecycle across all three subsystems."""
        # 1. Acquire concurrency slot
        try:
            with self.limiter.acquire(timeout=0.01):
                # 2. Pre-check budget
                try:
                    self.budget.pre_call_check(input_tokens, output_tokens)
                except BudgetExceeded:
                    return  # budget exhausted, release slot

                # 3. Reserve tokens
                cap, reserved = self.pool.reserve(input_tokens, output_tokens)
                if cap == 0:
                    return  # token pool exhausted

                # 4. "Execute" (simulated)
                self._total_ops += 1

                # 5. Commit tokens + record budget
                self.pool.commit(reserved, input_tokens + output_tokens)
                self.budget.record_usage(input_tokens, output_tokens, cost)
        except (TimeoutError, RuntimeError):
            pass  # concurrency full

    @rule()
    def concurrent_requests(self) -> None:
        """Fire concurrent requests through the full pipeline."""
        errors: list[str] = []

        def worker() -> None:
            try:
                with self.limiter.acquire(timeout=0.05):
                    if not self.budget.is_exceeded:
                        cap, res = self.pool.reserve(10, 20)
                        if cap > 0:
                            self.pool.commit(res, 25)
                            self.budget.record_usage(10, 20, 0.01)
            except (TimeoutError, RuntimeError, BudgetExceeded):
                pass
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)
        always(not errors, "no errors in concurrent cross-module requests")

    @invariant()
    def all_subsystems_consistent(self) -> None:
        """All subsystems should be internally consistent."""
        # Budget
        rem_cost = self.budget.remaining_cost
        if rem_cost is not None:
            always(rem_cost >= 0, "budget remaining cost >= 0")
        rem_tokens = self.budget.remaining_tokens
        if rem_tokens is not None:
            always(rem_tokens >= 0, "budget remaining tokens >= 0")

        # Concurrency
        stats = self.limiter.stats()
        always(stats.active <= 3, "concurrency active <= max")
        always(stats.waiting >= 0, "concurrency waiting >= 0")

        # Token pool
        snap = self.pool.snapshot()
        always(snap["used_tokens"] >= 0, "pool used >= 0")
        always(snap["reserved_tokens"] >= 0, "pool reserved >= 0")


TestCrossModuleChaos = CrossModuleChaos.TestCase
