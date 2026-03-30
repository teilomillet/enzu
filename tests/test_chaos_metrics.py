"""Chaos tests for MetricsCollector: histogram consistency, concurrent updates.

Properties verified:
- Histogram percentiles are monotonic (p50 <= p90 <= p95 <= p99)
- Counter values never decrease
- Snapshot is self-consistent
- Prometheus format is valid text
- Concurrent record/snapshot doesn't crash
"""

from __future__ import annotations

import threading

from hypothesis import strategies as st

from ordeal import ChaosTest, always, invariant, rule

from enzu.isolation.metrics import (
    CounterValue,
    GaugeValue,
    HistogramValue,
    MetricsCollector,
)


class HistogramChaos(ChaosTest):
    """Explore HistogramValue under adversarial observations."""

    faults = []

    def __init__(self) -> None:
        super().__init__()
        self.hist = HistogramValue(buckets=(10, 50, 100, 500, 1000))
        self._observation_count = 0

    @rule(
        value=st.floats(min_value=0.0, max_value=50000.0).filter(lambda x: x == x),
    )
    def observe(self, value: float) -> None:
        self.hist.observe(value)
        self._observation_count += 1

    @rule()
    def check_percentiles_monotonic(self) -> None:
        if self._observation_count == 0:
            return
        p50 = self.hist.percentile(50)
        p90 = self.hist.percentile(90)
        p95 = self.hist.percentile(95)
        p99 = self.hist.percentile(99)
        always(p50 <= p90 + 0.01, f"p50({p50}) <= p90({p90})")
        always(p90 <= p95 + 0.01, f"p90({p90}) <= p95({p95})")
        always(p95 <= p99 + 0.01, f"p95({p95}) <= p99({p99})")

    @rule()
    def check_get_consistency(self) -> None:
        data = self.hist.get()
        always(data["count"] == self._observation_count, "count matches observations")
        always(data["sum"] >= 0, "sum is non-negative")

    @invariant()
    def percentile_non_negative(self) -> None:
        for p in (50, 90, 95, 99):
            val = self.hist.percentile(p)
            always(val >= 0, f"p{p} is non-negative")


TestHistogramChaos = HistogramChaos.TestCase


class MetricsCollectorChaos(ChaosTest):
    """Explore MetricsCollector under randomized operations."""

    faults = []

    def __init__(self) -> None:
        super().__init__()
        self.collector = MetricsCollector()
        self._total_requests = 0
        self._success_count = 0
        self._error_count = 0

    @rule(
        node=st.sampled_from(["node-0", "node-1", "node-2"]),
        duration=st.floats(min_value=0.1, max_value=30000.0).filter(lambda x: x == x),
        success=st.booleans(),
    )
    def record_request(self, node: str, duration: float, success: bool) -> None:
        error_type = None if success else "timeout"
        self.collector.record_request(node, duration, success, error_type)
        self._total_requests += 1
        if success:
            self._success_count += 1
        else:
            self._error_count += 1

    @rule(accepted=st.booleans())
    def record_admission(self, accepted: bool) -> None:
        self.collector.record_admission(accepted)

    @rule(
        node=st.sampled_from(["node-0", "node-1", "node-2"]),
        depth=st.integers(min_value=0, max_value=1000),
    )
    def set_queue_depth(self, node: str, depth: int) -> None:
        self.collector.set_queue_depth(node, depth)

    @rule(
        node=st.sampled_from(["node-0", "node-1", "node-2"]),
        state=st.sampled_from(["closed", "open", "half_open"]),
    )
    def set_circuit_state(self, node: str, state: str) -> None:
        self.collector.set_circuit_breaker_state(node, state)

    @rule()
    def take_snapshot(self) -> None:
        snap = self.collector.snapshot()
        always(snap.timestamp > 0, "snapshot has valid timestamp")

        total_from_snap = sum(snap.requests_total.values())
        always(
            total_from_snap == self._total_requests,
            f"snapshot total({total_from_snap}) == tracked({self._total_requests})",
        )

    @rule()
    def check_prometheus_format(self) -> None:
        text = self.collector.prometheus_format()
        always(isinstance(text, str), "prometheus returns string")
        always("enzu_requests_total" in text, "prometheus has requests_total")

    @rule()
    def concurrent_snapshot(self) -> None:
        """Take snapshots from multiple threads while recording."""
        snapshots: list = []
        errors: list[str] = []

        def worker() -> None:
            try:
                snapshots.append(self.collector.snapshot())
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)
        always(not errors, "no errors during concurrent snapshot")

    @invariant()
    def request_counts_consistent(self) -> None:
        snap = self.collector.snapshot()
        success = snap.requests_total.get("success", 0)
        error = snap.requests_total.get("error", 0)
        always(success >= 0, "success count >= 0")
        always(error >= 0, "error count >= 0")
        always(
            success + error == self._total_requests,
            "success + error == total",
        )

    @invariant()
    def latency_percentiles_monotonic(self) -> None:
        if self._total_requests == 0:
            return
        percs = self.collector.latency_percentiles()
        always(
            percs["p50"] <= percs["p90"] + 0.01,
            "p50 <= p90",
        )
        always(
            percs["p90"] <= percs["p95"] + 0.01,
            "p90 <= p95",
        )
        always(
            percs["p95"] <= percs["p99"] + 0.01,
            "p95 <= p99",
        )


TestMetricsCollectorChaos = MetricsCollectorChaos.TestCase


class CounterGaugeChaos(ChaosTest):
    """Explore CounterValue and GaugeValue thread-safety."""

    faults = []

    def __init__(self) -> None:
        super().__init__()
        self.counter = CounterValue()
        self.gauge = GaugeValue()
        self._counter_total = 0.0

    @rule(amount=st.floats(min_value=0.0, max_value=100.0).filter(lambda x: x == x))
    def inc_counter(self, amount: float) -> None:
        self.counter.inc(amount)
        self._counter_total += amount

    @rule(value=st.floats(min_value=-1000.0, max_value=1000.0).filter(lambda x: x == x))
    def set_gauge(self, value: float) -> None:
        self.gauge.set(value)

    @rule()
    def concurrent_counter_inc(self) -> None:
        errors: list[str] = []

        def inc() -> None:
            try:
                self.counter.inc(1.0)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=inc, daemon=True) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)
        self._counter_total += 10.0
        always(not errors, "no errors during concurrent inc")

    @invariant()
    def counter_matches_tracking(self) -> None:
        val = self.counter.get()
        always(
            abs(val - self._counter_total) < 0.01,
            f"counter({val}) matches tracked({self._counter_total})",
        )

    @invariant()
    def counter_non_negative(self) -> None:
        always(self.counter.get() >= 0, "counter is non-negative")


TestCounterGaugeChaos = CounterGaugeChaos.TestCase
