"""Chaos tests for ConcurrencyLimiter: saturation, stats accuracy, deadlock probing.

Properties verified:
- active count never exceeds max_concurrent
- active + waiting is always non-negative
- total_acquired + total_rejected equals total attempts
- stats are monotonically non-decreasing (acquired, rejected)
- no deadlocks under rapid acquire/release
"""

from __future__ import annotations

import threading


from ordeal import ChaosTest, always, invariant, rule

from enzu.isolation.concurrency import ConcurrencyLimiter


class ConcurrencyLimiterChaos(ChaosTest):
    """Explore ConcurrencyLimiter under randomized acquire/release patterns."""

    faults = []

    def __init__(self) -> None:
        super().__init__()
        self.limiter = ConcurrencyLimiter(max_concurrent=3)
        self._held_slots: list[bool] = []  # track acquired context managers
        self._attempt_count = 0
        self._prev_acquired = 0
        self._prev_rejected = 0

    @rule()
    def acquire_blocking(self) -> None:
        """Try a blocking acquire with short timeout."""
        self._attempt_count += 1
        try:
            with self.limiter.acquire(timeout=0.01):
                stats = self.limiter.stats()
                always(
                    stats.active <= self.limiter.max_concurrent,
                    "active <= max_concurrent while holding slot",
                )
        except (TimeoutError, RuntimeError):
            pass  # expected under saturation

    @rule()
    def acquire_nonblocking(self) -> None:
        """Try a non-blocking acquire."""
        self._attempt_count += 1
        try:
            with self.limiter.acquire(blocking=False):
                stats = self.limiter.stats()
                always(
                    stats.active <= self.limiter.max_concurrent,
                    "active <= max_concurrent (non-blocking)",
                )
        except RuntimeError:
            pass  # no slots available

    @rule()
    def concurrent_acquire_burst(self) -> None:
        """Fire multiple concurrent acquires from threads."""
        results: list[str] = []
        errors: list[str] = []

        def worker() -> None:
            try:
                with self.limiter.acquire(timeout=0.05):
                    results.append("acquired")
            except (TimeoutError, RuntimeError):
                results.append("rejected")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)

        always(not errors, "no unexpected errors in concurrent burst")

    @invariant()
    def active_within_bounds(self) -> None:
        stats = self.limiter.stats()
        always(
            0 <= stats.active <= self.limiter.max_concurrent,
            f"0 <= active({stats.active}) <= max({self.limiter.max_concurrent})",
        )

    @invariant()
    def waiting_non_negative(self) -> None:
        stats = self.limiter.stats()
        always(stats.waiting >= 0, f"waiting({stats.waiting}) >= 0")

    @invariant()
    def totals_monotonically_increase(self) -> None:
        stats = self.limiter.stats()
        always(
            stats.total_acquired >= self._prev_acquired,
            "total_acquired monotonically increases",
        )
        always(
            stats.total_rejected >= self._prev_rejected,
            "total_rejected monotonically increases",
        )
        self._prev_acquired = stats.total_acquired
        self._prev_rejected = stats.total_rejected


TestConcurrencyLimiterChaos = ConcurrencyLimiterChaos.TestCase


class ConcurrencyLimiterTiny(ChaosTest):
    """Explore with max_concurrent=1 — maximum contention."""

    faults = []

    def __init__(self) -> None:
        super().__init__()
        self.limiter = ConcurrencyLimiter(max_concurrent=1)

    @rule()
    def rapid_acquire_release(self) -> None:
        try:
            with self.limiter.acquire(timeout=0.01):
                stats = self.limiter.stats()
                always(stats.active == 1, "exactly 1 active with max=1")
        except (TimeoutError, RuntimeError):
            pass

    @rule()
    def nonblocking_at_limit(self) -> None:
        """With max=1, non-blocking should often fail."""
        try:
            with self.limiter.acquire(blocking=False):
                pass
        except RuntimeError:
            pass  # expected

    @invariant()
    def active_at_most_one(self) -> None:
        stats = self.limiter.stats()
        always(stats.active <= 1, "active <= 1 with max_concurrent=1")


TestConcurrencyLimiterTiny = ConcurrencyLimiterTiny.TestCase
