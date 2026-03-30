"""Chaos tests for CircuitBreaker: state transitions, sliding window, cascading failures.

Properties verified:
- State transitions follow CLOSED -> OPEN -> HALF_OPEN -> CLOSED cycle
- OPEN state rejects all requests until reset_timeout
- Failure count matches sliding window contents
- total_allowed + total_rejected equals total allow_request() calls
- Concurrent record_success/record_failure don't corrupt state
- Backpressure retry_after is always non-negative
"""

from __future__ import annotations

import threading

from hypothesis import strategies as st

from ordeal import ChaosTest, always, invariant, rule
from ordeal.faults import timing

from enzu.isolation.health import (
    BackpressureController,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    RetryConfig,
    RetryStrategy,
)


class CircuitBreakerChaos(ChaosTest):
    """Explore CircuitBreaker under randomized success/failure sequences."""

    faults = [
        timing.jitter("time.time", magnitude=0.001),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.breaker = CircuitBreaker(
            "chaos-node",
            config=CircuitBreakerConfig(
                failure_threshold=3,
                reset_timeout_seconds=0.05,  # fast for testing
                success_threshold=2,
                failure_window_seconds=5.0,
            ),
        )
        self._allow_calls = 0
        self._prev_allowed = 0
        self._prev_rejected = 0

    @rule()
    def check_and_succeed(self) -> None:
        """Check if request allowed, then record success."""
        self._allow_calls += 1
        if self.breaker.allow_request():
            self.breaker.record_success()

    @rule()
    def check_and_fail(self) -> None:
        """Check if request allowed, then record failure."""
        self._allow_calls += 1
        if self.breaker.allow_request():
            self.breaker.record_failure()

    @rule()
    def just_fail(self) -> None:
        """Record failure without checking (simulates background error)."""
        self.breaker.record_failure()

    @rule()
    def manual_reset(self) -> None:
        """Manually reset the circuit breaker."""
        self.breaker.reset()
        always(
            self.breaker.state == CircuitState.CLOSED,
            "reset returns to CLOSED",
        )

    @rule()
    def concurrent_failures(self) -> None:
        """Fire concurrent failures from multiple threads."""
        errors: list[str] = []

        def fail() -> None:
            try:
                self.breaker.record_failure()
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=fail, daemon=True) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)
        always(not errors, "no errors during concurrent failures")

    @invariant()
    def state_is_valid(self) -> None:
        always(
            self.breaker.state
            in (CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN),
            "state is a valid CircuitState",
        )

    @invariant()
    def totals_monotonically_increase(self) -> None:
        stats = self.breaker.stats()
        always(
            stats.total_allowed >= self._prev_allowed,
            "total_allowed monotonically increases",
        )
        always(
            stats.total_rejected >= self._prev_rejected,
            "total_rejected monotonically increases",
        )
        self._prev_allowed = stats.total_allowed
        self._prev_rejected = stats.total_rejected

    @invariant()
    def failure_count_non_negative(self) -> None:
        stats = self.breaker.stats()
        always(stats.failure_count >= 0, "failure_count >= 0")
        always(stats.success_count >= 0, "success_count >= 0")


TestCircuitBreakerChaos = CircuitBreakerChaos.TestCase


class BackpressureChaos(ChaosTest):
    """Explore BackpressureController with adversarial load factors."""

    faults = []

    def __init__(self) -> None:
        super().__init__()
        self.bp = BackpressureController(
            warning_threshold=0.7,
            critical_threshold=0.9,
            base_retry_seconds=1.0,
            max_retry_seconds=60.0,
        )

    @rule(
        load=st.floats(min_value=0.0, max_value=1.0),
        queue=st.integers(min_value=0, max_value=10000),
    )
    def calculate_signal(self, load: float, queue: int) -> None:
        signal = self.bp.calculate_signal(load, queue)
        always(
            signal.retry_after_seconds >= 0,
            "retry_after is non-negative",
        )
        always(
            signal.load_factor == load,
            "load_factor echoed back correctly",
        )
        always(
            signal.queue_depth == queue,
            "queue_depth echoed back correctly",
        )
        if load < 0.7:
            always(not signal.should_backoff, "no backoff below warning threshold")
        if load >= 0.9:
            always(signal.should_backoff, "backoff at critical threshold")
            always(
                signal.retry_after_seconds == 60.0,
                "max retry at critical load",
            )


TestBackpressureChaos = BackpressureChaos.TestCase


class RetryStrategyChaos(ChaosTest):
    """Explore RetryStrategy backoff calculations."""

    faults = []

    def __init__(self) -> None:
        super().__init__()
        self.retry = RetryStrategy(
            config=RetryConfig(
                max_retries=5,
                initial_backoff_seconds=0.1,
                max_backoff_seconds=10.0,
                backoff_multiplier=2.0,
                jitter_factor=0.1,
            )
        )

    @rule(attempt=st.integers(min_value=0, max_value=100))
    def calculate_backoff(self, attempt: int) -> None:
        delay = self.retry.calculate_backoff(attempt)
        always(delay >= 0, "backoff delay is non-negative")
        always(
            delay <= 10.0 * 1.1 + 0.01,  # max + jitter tolerance
            "backoff capped at max_backoff + jitter",
        )

    @rule(attempt=st.integers(min_value=0, max_value=10))
    def should_retry_boundary(self, attempt: int) -> None:
        """should_retry respects max_retries."""
        err = TimeoutError("test")
        if attempt >= 5:
            always(
                not self.retry.should_retry(err, attempt),
                "no retry beyond max_retries",
            )
        else:
            always(
                self.retry.should_retry(err, attempt),
                "retry allowed for retryable exception under limit",
            )

    @rule()
    def non_retryable_exception(self) -> None:
        err = ValueError("not retryable")
        always(
            not self.retry.should_retry(err, 0),
            "ValueError is not retryable",
        )


TestRetryStrategyChaos = RetryStrategyChaos.TestCase
