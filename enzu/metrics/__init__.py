"""
First-class run metrics for production observability.

Provides:
- RunEvent: Canonical run summary schema (cost/time/tokens/terminal state)
- RunMetricsCollector: Aggregates p50/p95, histograms, terminal state distributions
- Prometheus/JSON export for external integration

Usage:
    from enzu.metrics import RunEvent, RunMetricsCollector, get_run_metrics

    collector = get_run_metrics()
    event = RunEvent.from_execution_report(run_id="...", report=report, ...)
    collector.observe(event)

    # Get p50/p95 and distributions
    stats = collector.snapshot()
    print(stats["percentiles"]["cost_usd"]["p95"])

    # Prometheus export
    print(collector.prometheus_format())
"""

from enzu.metrics.run_event import RunEvent
from enzu.metrics.collector import (
    RunMetricsCollector,
    get_run_metrics,
    reset_run_metrics,
)

__all__ = [
    "RunEvent",
    "RunMetricsCollector",
    "get_run_metrics",
    "reset_run_metrics",
]
