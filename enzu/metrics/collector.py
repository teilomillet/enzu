"""
RunMetricsCollector: Aggregates run metrics for p50/p95 and terminal state distributions.

Thread-safe, in-memory collection with Prometheus-compatible export.
Reuses histogram primitives from enzu.isolation.metrics.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

from enzu.metrics.run_event import RunEvent

COST_BUCKETS_USD: Tuple[float, ...] = (
    0.0001,
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.10,
    0.25,
    0.50,
    1.0,
    2.0,
    5.0,
    10.0,
)

SECONDS_BUCKETS: Tuple[float, ...] = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    600.0,
)

TOKENS_BUCKETS: Tuple[float, ...] = (
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000,
    10000,
    25000,
    50000,
    100000,
    250000,
)


class _HistogramValue:
    """Thread-safe histogram with configurable buckets."""

    def __init__(self, buckets: Tuple[float, ...]) -> None:
        self._buckets = tuple(sorted(buckets)) + (float("inf"),)
        self._counts = [0] * len(self._buckets)
        self._sum = 0.0
        self._count = 0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._sum += value
            self._count += 1
            for i, bound in enumerate(self._buckets):
                if value <= bound:
                    self._counts[i] += 1

    def get(self) -> Dict[str, Any]:
        with self._lock:
            bucket_data = []
            cumulative = 0
            for i, bound in enumerate(self._buckets):
                cumulative += self._counts[i]
                if bound != float("inf"):
                    bucket_data.append((bound, cumulative))
            return {
                "buckets": bucket_data,
                "sum": self._sum,
                "count": self._count,
            }

    def percentile(self, p: float) -> Optional[float]:
        with self._lock:
            if self._count == 0:
                return None

            target = self._count * (p / 100.0)
            cumulative = 0

            for i, bound in enumerate(self._buckets):
                cumulative += self._counts[i]
                if cumulative >= target:
                    if i == 0:
                        return bound if bound != float("inf") else None
                    prev_cumulative = cumulative - self._counts[i]
                    prev_bound = 0 if i == 0 else self._buckets[i - 1]
                    if self._counts[i] == 0:
                        return prev_bound
                    ratio = (target - prev_cumulative) / self._counts[i]
                    result = prev_bound + ratio * (bound - prev_bound)
                    return result if result != float("inf") else None

            return self._buckets[-2] if len(self._buckets) > 1 else None

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def sum(self) -> float:
        with self._lock:
            return self._sum


class _CounterValue:
    """Thread-safe counter."""

    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def inc(self, amount: int = 1) -> None:
        with self._lock:
            self._value += amount

    def get(self) -> int:
        with self._lock:
            return self._value


class RunMetricsCollector:
    """
    Collects run-level metrics for p50/p95 analysis and terminal state distributions.

    Thread-safe, in-memory. Call observe() after each run completes.

    Metrics tracked:
    - elapsed_seconds: Wall-clock time per run (histogram)
    - cost_usd: Cost per run in USD (histogram, when known)
    - total_tokens: Total tokens per run (histogram, when known)
    - input_tokens: Input tokens per run (histogram, when known)
    - output_tokens: Output tokens per run (histogram, when known)
    - outcome distribution: Count by outcome + partial flag
    - retries: Retry count per run (histogram)

    Example:
        collector = RunMetricsCollector()
        collector.observe(run_event)
        stats = collector.snapshot()
        print(stats["percentiles"]["cost_usd"]["p95"])
    """

    def __init__(self) -> None:
        self._elapsed = _HistogramValue(SECONDS_BUCKETS)
        self._cost = _HistogramValue(COST_BUCKETS_USD)
        self._total_tokens = _HistogramValue(TOKENS_BUCKETS)
        self._input_tokens = _HistogramValue(TOKENS_BUCKETS)
        self._output_tokens = _HistogramValue(TOKENS_BUCKETS)
        self._retries = _HistogramValue((0, 1, 2, 3, 5, 10, 25, 50, 100))

        self._runs_by_outcome: Dict[str, _CounterValue] = defaultdict(_CounterValue)
        self._cost_unknown = _CounterValue()
        self._total_runs = _CounterValue()

        self._lock = threading.Lock()

    def observe(self, event: RunEvent) -> None:
        """
        Record a completed run.

        Args:
            event: RunEvent from a completed run.
        """
        with self._lock:
            self._total_runs.inc()

            outcome_key = f"{event.outcome.value}|partial={int(event.partial)}"
            self._runs_by_outcome[outcome_key].inc()

            self._elapsed.observe(event.elapsed_seconds)
            self._retries.observe(float(event.retries))

            if event.total_tokens is not None:
                self._total_tokens.observe(float(event.total_tokens))
            if event.input_tokens is not None:
                self._input_tokens.observe(float(event.input_tokens))
            if event.output_tokens is not None:
                self._output_tokens.observe(float(event.output_tokens))

            if event.cost_usd is not None:
                self._cost.observe(event.cost_usd)
            else:
                self._cost_unknown.inc()

    def percentiles(self) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Get p50/p95 for all tracked metrics.

        Returns:
            Dict with metric names as keys, each containing p50/p95 values.
            None values indicate insufficient data.
        """
        return {
            "elapsed_seconds": {
                "p50": self._elapsed.percentile(50),
                "p95": self._elapsed.percentile(95),
                "p99": self._elapsed.percentile(99),
            },
            "cost_usd": {
                "p50": self._cost.percentile(50),
                "p95": self._cost.percentile(95),
                "p99": self._cost.percentile(99),
            },
            "total_tokens": {
                "p50": self._total_tokens.percentile(50),
                "p95": self._total_tokens.percentile(95),
                "p99": self._total_tokens.percentile(99),
            },
            "input_tokens": {
                "p50": self._input_tokens.percentile(50),
                "p95": self._input_tokens.percentile(95),
                "p99": self._input_tokens.percentile(99),
            },
            "output_tokens": {
                "p50": self._output_tokens.percentile(50),
                "p95": self._output_tokens.percentile(95),
                "p99": self._output_tokens.percentile(99),
            },
            "retries": {
                "p50": self._retries.percentile(50),
                "p95": self._retries.percentile(95),
                "p99": self._retries.percentile(99),
            },
        }

    def outcome_distribution(self) -> Dict[str, int]:
        """
        Get terminal state distribution.

        Returns:
            Dict mapping "outcome|partial=0/1" to count.
        """
        with self._lock:
            return {k: v.get() for k, v in self._runs_by_outcome.items()}

    def snapshot(self) -> Dict[str, Any]:
        """
        Get complete metrics snapshot.

        Returns:
            Dict containing:
            - total_runs: Total number of runs observed
            - cost_known_runs: Runs with known cost (for p95 accuracy)
            - cost_unknown_runs: Runs without cost data
            - percentiles: p50/p95/p99 for all metrics
            - outcome_distribution: Terminal state counts
            - histograms: Raw histogram data for custom analysis
        """
        cost_known = self._cost.count
        cost_unknown = self._cost_unknown.get()

        return {
            "total_runs": self._total_runs.get(),
            "cost_known_runs": cost_known,
            "cost_unknown_runs": cost_unknown,
            "cost_coverage": cost_known / max(1, cost_known + cost_unknown),
            "percentiles": self.percentiles(),
            "outcome_distribution": self.outcome_distribution(),
            "histograms": {
                "elapsed_seconds": self._elapsed.get(),
                "cost_usd": self._cost.get(),
                "total_tokens": self._total_tokens.get(),
                "input_tokens": self._input_tokens.get(),
                "output_tokens": self._output_tokens.get(),
                "retries": self._retries.get(),
            },
            "averages": {
                "elapsed_seconds": (
                    self._elapsed.sum / self._elapsed.count
                    if self._elapsed.count > 0
                    else None
                ),
                "cost_usd": (
                    self._cost.sum / self._cost.count if self._cost.count > 0 else None
                ),
                "total_tokens": (
                    self._total_tokens.sum / self._total_tokens.count
                    if self._total_tokens.count > 0
                    else None
                ),
            },
        }

    def prometheus_format(self) -> str:
        """
        Export metrics in Prometheus text exposition format.

        Returns:
            String suitable for /metrics endpoint.
        """
        lines = []

        lines.append("# HELP enzu_runs_total Total number of runs by outcome")
        lines.append("# TYPE enzu_runs_total counter")
        for key, counter in self._runs_by_outcome.items():
            outcome, partial = key.split("|")
            partial_val = partial.split("=")[1]
            lines.append(
                f'enzu_runs_total{{outcome="{outcome}",partial="{partial_val}"}} {counter.get()}'
            )

        lines.append("")
        lines.append("# HELP enzu_runs_cost_unknown_total Runs without cost data")
        lines.append("# TYPE enzu_runs_cost_unknown_total counter")
        lines.append(f"enzu_runs_cost_unknown_total {self._cost_unknown.get()}")

        for name, histogram, unit in [
            ("enzu_run_elapsed_seconds", self._elapsed, "seconds"),
            ("enzu_run_cost_usd", self._cost, "USD"),
            ("enzu_run_total_tokens", self._total_tokens, "tokens"),
            ("enzu_run_input_tokens", self._input_tokens, "tokens"),
            ("enzu_run_output_tokens", self._output_tokens, "tokens"),
            ("enzu_run_retries", self._retries, "count"),
        ]:
            data = histogram.get()
            lines.append("")
            lines.append(f"# HELP {name} Distribution of {unit} per run")
            lines.append(f"# TYPE {name} histogram")

            for le, count in data["buckets"]:
                lines.append(f'{name}_bucket{{le="{le}"}} {count}')
            lines.append(f'{name}_bucket{{le="+Inf"}} {data["count"]}')
            lines.append(f"{name}_sum {data['sum']}")
            lines.append(f"{name}_count {data['count']}")

        return "\n".join(lines)


_global_collector: Optional[RunMetricsCollector] = None
_global_lock = threading.Lock()


def get_run_metrics() -> RunMetricsCollector:
    """
    Get the global RunMetricsCollector singleton.

    Returns:
        The shared RunMetricsCollector instance.
    """
    global _global_collector
    with _global_lock:
        if _global_collector is None:
            _global_collector = RunMetricsCollector()
        return _global_collector


def reset_run_metrics() -> None:
    """Reset the global collector (mainly for testing)."""
    global _global_collector
    with _global_lock:
        _global_collector = RunMetricsCollector()
