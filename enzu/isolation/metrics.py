"""
Metrics collection for production observability (Phase 5).

Provides Prometheus-compatible metrics without requiring Prometheus library:
- Counter: Monotonically increasing values (requests, errors)
- Gauge: Point-in-time values (queue depth, active workers)
- Histogram: Distribution tracking (latency percentiles)

Metrics are collected in-memory and exposed via:
- snapshot(): Returns current metrics as dict (for JSON export)
- prometheus_format(): Returns Prometheus text exposition format

Integration:
    collector = get_metrics_collector()
    collector.record_request(node_id, duration_ms, success=True)

    # Expose via HTTP endpoint
    @app.get("/metrics")
    def metrics():
        return collector.prometheus_format()

"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from collections import defaultdict

# Histogram buckets for latency (milliseconds)
DEFAULT_LATENCY_BUCKETS = (
    10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000
)


@dataclass
class CounterValue:
    """Thread-safe counter."""
    value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self.value += amount

    def get(self) -> float:
        with self._lock:
            return self.value


@dataclass
class GaugeValue:
    """Thread-safe gauge."""
    value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set(self, value: float) -> None:
        with self._lock:
            self.value = value

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self.value += amount

    def dec(self, amount: float = 1.0) -> None:
        with self._lock:
            self.value -= amount

    def get(self) -> float:
        with self._lock:
            return self.value


class HistogramValue:
    """
    Thread-safe histogram with configurable buckets.

    Tracks:
    - Count per bucket
    - Total sum
    - Total count
    """

    def __init__(self, buckets: Tuple[float, ...] = DEFAULT_LATENCY_BUCKETS) -> None:
        self._buckets = tuple(sorted(buckets)) + (float('inf'),)
        self._counts = [0] * len(self._buckets)
        self._sum = 0.0
        self._count = 0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        """Record an observation."""
        with self._lock:
            self._sum += value
            self._count += 1
            for i, bound in enumerate(self._buckets):
                if value <= bound:
                    self._counts[i] += 1

    def get(self) -> Dict[str, Any]:
        """Get histogram data."""
        with self._lock:
            bucket_data = []
            cumulative = 0
            for i, bound in enumerate(self._buckets):
                cumulative += self._counts[i]
                if bound != float('inf'):
                    bucket_data.append((bound, cumulative))

            return {
                "buckets": bucket_data,
                "sum": self._sum,
                "count": self._count,
            }

    def percentile(self, p: float) -> float:
        """
        Estimate percentile from histogram buckets.

        Args:
            p: Percentile (0-100)

        Returns:
            Estimated value at percentile (linear interpolation)
        """
        with self._lock:
            if self._count == 0:
                return 0.0

            target = self._count * (p / 100.0)
            cumulative = 0

            for i, bound in enumerate(self._buckets):
                cumulative += self._counts[i]
                if cumulative >= target:
                    if i == 0:
                        return bound
                    # Linear interpolation
                    prev_cumulative = cumulative - self._counts[i]
                    prev_bound = 0 if i == 0 else self._buckets[i - 1]
                    ratio = (target - prev_cumulative) / max(1, self._counts[i])
                    return prev_bound + ratio * (bound - prev_bound)

            return self._buckets[-2] if len(self._buckets) > 1 else 0.0


@dataclass
class MetricSnapshot:
    """Snapshot of all metrics at a point in time."""
    timestamp: float

    # Request metrics
    requests_total: Dict[str, float]  # {status: count}
    requests_by_node: Dict[str, float]  # {node_id: count}

    # Latency histogram data
    request_duration_ms: Dict[str, Any]

    # Queue metrics
    queue_depth: Dict[str, float]  # {node_id: depth}
    active_workers: Dict[str, float]  # {node_id: count}

    # Concurrency metrics
    concurrency_active: float
    concurrency_waiting: float
    concurrency_limit: float

    # Circuit breaker metrics
    circuit_breaker_state: Dict[str, str]  # {node_id: state}
    circuit_breaker_failures: Dict[str, float]  # {node_id: count}

    # Error metrics
    errors_by_type: Dict[str, float]  # {error_type: count}

    # Admission metrics
    admission_accepted: float
    admission_rejected: float


class MetricsCollector:
    """
    Central metrics collector for production observability.

    Collects:
    - Request throughput and latency
    - Queue depth and worker utilization
    - Concurrency limiter state
    - Circuit breaker status
    - Error rates by type

    Thread-safe for concurrent updates.

    Usage:
        collector = MetricsCollector()

        # Record request
        collector.record_request(
            node_id="node-1",
            duration_ms=150.5,
            success=True,
        )

        # Record queue state
        collector.set_queue_depth("node-1", 50)
        collector.set_active_workers("node-1", 25)

        # Get snapshot
        snapshot = collector.snapshot()

        # Export Prometheus format
        prom_text = collector.prometheus_format()
    """

    def __init__(
        self,
        latency_buckets: Tuple[float, ...] = DEFAULT_LATENCY_BUCKETS,
    ) -> None:
        self._latency_buckets = latency_buckets

        # Request counters by status
        self._requests_total: Dict[str, CounterValue] = defaultdict(CounterValue)
        self._requests_by_node: Dict[str, CounterValue] = defaultdict(CounterValue)

        # Latency histograms by node
        self._latency_histograms: Dict[str, HistogramValue] = {}
        self._global_latency = HistogramValue(latency_buckets)

        # Gauges for queue/workers
        self._queue_depth: Dict[str, GaugeValue] = defaultdict(GaugeValue)
        self._active_workers: Dict[str, GaugeValue] = defaultdict(GaugeValue)

        # Concurrency gauges
        self._concurrency_active = GaugeValue()
        self._concurrency_waiting = GaugeValue()
        self._concurrency_limit = GaugeValue()

        # Circuit breaker state
        self._circuit_state: Dict[str, str] = {}
        self._circuit_failures: Dict[str, CounterValue] = defaultdict(CounterValue)

        # Error counters
        self._errors_by_type: Dict[str, CounterValue] = defaultdict(CounterValue)

        # Admission counters
        self._admission_accepted = CounterValue()
        self._admission_rejected = CounterValue()

        self._lock = threading.Lock()
        self._start_time = time.time()

    def record_request(
        self,
        node_id: str,
        duration_ms: float,
        success: bool,
        error_type: Optional[str] = None,
    ) -> None:
        """
        Record a completed request.

        Args:
            node_id: Node that processed the request
            duration_ms: Request duration in milliseconds
            success: Whether request succeeded
            error_type: Error category if failed
        """
        status = "success" if success else "error"
        self._requests_total[status].inc()
        self._requests_by_node[node_id].inc()

        # Record latency
        self._global_latency.observe(duration_ms)
        with self._lock:
            if node_id not in self._latency_histograms:
                self._latency_histograms[node_id] = HistogramValue(self._latency_buckets)
            self._latency_histograms[node_id].observe(duration_ms)

        # Record error type if failed
        if not success and error_type:
            self._errors_by_type[error_type].inc()

    def record_admission(self, accepted: bool) -> None:
        """Record admission control decision."""
        if accepted:
            self._admission_accepted.inc()
        else:
            self._admission_rejected.inc()

    def set_queue_depth(self, node_id: str, depth: int) -> None:
        """Update queue depth for a node."""
        self._queue_depth[node_id].set(depth)

    def set_active_workers(self, node_id: str, count: int) -> None:
        """Update active worker count for a node."""
        self._active_workers[node_id].set(count)

    def set_concurrency(self, active: int, waiting: int, limit: int) -> None:
        """Update concurrency limiter state."""
        self._concurrency_active.set(active)
        self._concurrency_waiting.set(waiting)
        self._concurrency_limit.set(limit)

    def set_circuit_breaker_state(self, node_id: str, state: str) -> None:
        """Update circuit breaker state for a node."""
        with self._lock:
            self._circuit_state[node_id] = state

    def record_circuit_breaker_failure(self, node_id: str) -> None:
        """Record circuit breaker failure."""
        self._circuit_failures[node_id].inc()

    def snapshot(self) -> MetricSnapshot:
        """Get snapshot of all current metrics."""
        with self._lock:
            circuit_state = dict(self._circuit_state)

        return MetricSnapshot(
            timestamp=time.time(),
            requests_total={k: v.get() for k, v in self._requests_total.items()},
            requests_by_node={k: v.get() for k, v in self._requests_by_node.items()},
            request_duration_ms=self._global_latency.get(),
            queue_depth={k: v.get() for k, v in self._queue_depth.items()},
            active_workers={k: v.get() for k, v in self._active_workers.items()},
            concurrency_active=self._concurrency_active.get(),
            concurrency_waiting=self._concurrency_waiting.get(),
            concurrency_limit=self._concurrency_limit.get(),
            circuit_breaker_state=circuit_state,
            circuit_breaker_failures={k: v.get() for k, v in self._circuit_failures.items()},
            errors_by_type={k: v.get() for k, v in self._errors_by_type.items()},
            admission_accepted=self._admission_accepted.get(),
            admission_rejected=self._admission_rejected.get(),
        )

    def latency_percentiles(self) -> Dict[str, float]:
        """Get latency percentiles (p50, p90, p95, p99)."""
        return {
            "p50": self._global_latency.percentile(50),
            "p90": self._global_latency.percentile(90),
            "p95": self._global_latency.percentile(95),
            "p99": self._global_latency.percentile(99),
        }

    def prometheus_format(self) -> str:
        """
        Export metrics in Prometheus text exposition format.

        Returns:
            String in Prometheus format for /metrics endpoint
        """
        lines = []
        snap = self.snapshot()

        # Request counters
        lines.append("# HELP enzu_requests_total Total requests by status")
        lines.append("# TYPE enzu_requests_total counter")
        for status, count in snap.requests_total.items():
            lines.append(f'enzu_requests_total{{status="{status}"}} {count}')

        lines.append("")
        lines.append("# HELP enzu_requests_by_node Total requests by node")
        lines.append("# TYPE enzu_requests_by_node counter")
        for node_id, count in snap.requests_by_node.items():
            lines.append(f'enzu_requests_by_node{{node_id="{node_id}"}} {count}')

        # Latency histogram
        lines.append("")
        lines.append("# HELP enzu_request_duration_ms Request duration in milliseconds")
        lines.append("# TYPE enzu_request_duration_ms histogram")
        hist = snap.request_duration_ms
        for bound, count in hist.get("buckets", []):
            lines.append(f'enzu_request_duration_ms_bucket{{le="{bound}"}} {count}')
        lines.append(f'enzu_request_duration_ms_bucket{{le="+Inf"}} {hist.get("count", 0)}')
        lines.append(f'enzu_request_duration_ms_sum {hist.get("sum", 0)}')
        lines.append(f'enzu_request_duration_ms_count {hist.get("count", 0)}')

        # Queue depth
        lines.append("")
        lines.append("# HELP enzu_queue_depth Current queue depth by node")
        lines.append("# TYPE enzu_queue_depth gauge")
        for node_id, depth in snap.queue_depth.items():
            lines.append(f'enzu_queue_depth{{node_id="{node_id}"}} {depth}')

        # Active workers
        lines.append("")
        lines.append("# HELP enzu_active_workers Active workers by node")
        lines.append("# TYPE enzu_active_workers gauge")
        for node_id, count in snap.active_workers.items():
            lines.append(f'enzu_active_workers{{node_id="{node_id}"}} {count}')

        # Concurrency
        lines.append("")
        lines.append("# HELP enzu_concurrency_active Active LLM calls")
        lines.append("# TYPE enzu_concurrency_active gauge")
        lines.append(f"enzu_concurrency_active {snap.concurrency_active}")

        lines.append("")
        lines.append("# HELP enzu_concurrency_waiting Waiting for LLM slot")
        lines.append("# TYPE enzu_concurrency_waiting gauge")
        lines.append(f"enzu_concurrency_waiting {snap.concurrency_waiting}")

        lines.append("")
        lines.append("# HELP enzu_concurrency_limit Maximum concurrent LLM calls")
        lines.append("# TYPE enzu_concurrency_limit gauge")
        lines.append(f"enzu_concurrency_limit {snap.concurrency_limit}")

        # Circuit breaker
        lines.append("")
        lines.append("# HELP enzu_circuit_breaker_state Circuit breaker state (0=closed, 1=open, 2=half_open)")
        lines.append("# TYPE enzu_circuit_breaker_state gauge")
        state_map = {"closed": 0, "open": 1, "half_open": 2}
        for node_id, state in snap.circuit_breaker_state.items():
            state_val = state_map.get(state, -1)
            lines.append(f'enzu_circuit_breaker_state{{node_id="{node_id}"}} {state_val}')

        # Errors
        lines.append("")
        lines.append("# HELP enzu_errors_total Errors by type")
        lines.append("# TYPE enzu_errors_total counter")
        for error_type, count in snap.errors_by_type.items():
            lines.append(f'enzu_errors_total{{type="{error_type}"}} {count}')

        # Admission
        lines.append("")
        lines.append("# HELP enzu_admission_accepted Accepted requests")
        lines.append("# TYPE enzu_admission_accepted counter")
        lines.append(f"enzu_admission_accepted {snap.admission_accepted}")

        lines.append("")
        lines.append("# HELP enzu_admission_rejected Rejected requests")
        lines.append("# TYPE enzu_admission_rejected counter")
        lines.append(f"enzu_admission_rejected {snap.admission_rejected}")

        lines.append("")
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics. For testing only."""
        with self._lock:
            self._requests_total.clear()
            self._requests_by_node.clear()
            self._latency_histograms.clear()
            self._global_latency = HistogramValue(self._latency_buckets)
            self._queue_depth.clear()
            self._active_workers.clear()
            self._concurrency_active = GaugeValue()
            self._concurrency_waiting = GaugeValue()
            self._concurrency_limit = GaugeValue()
            self._circuit_state.clear()
            self._circuit_failures.clear()
            self._errors_by_type.clear()
            self._admission_accepted = CounterValue()
            self._admission_rejected = CounterValue()


# Global singleton
_global_collector: Optional[MetricsCollector] = None
_global_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """
    Get the global metrics collector.

    Creates with default settings if not configured.
    Thread-safe singleton.
    """
    global _global_collector
    if _global_collector is None:
        with _global_lock:
            if _global_collector is None:
                _global_collector = MetricsCollector()
    assert _global_collector is not None
    return _global_collector


def configure_metrics_collector(
    latency_buckets: Tuple[float, ...] = DEFAULT_LATENCY_BUCKETS,
) -> MetricsCollector:
    """
    Configure the global metrics collector.

    Call at startup before processing requests.

    Args:
        latency_buckets: Custom latency histogram buckets

    Returns:
        The configured metrics collector
    """
    global _global_collector

    with _global_lock:
        _global_collector = MetricsCollector(latency_buckets=latency_buckets)
        return _global_collector


def reset_metrics_collector() -> None:
    """Reset global collector. For testing only."""
    global _global_collector
    with _global_lock:
        if _global_collector:
            _global_collector.reset()
        _global_collector = None
