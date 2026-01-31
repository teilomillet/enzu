"""Prometheus metrics for the API service."""

import time
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RequestMetric:
    """Single request metric."""

    customer_id: str
    endpoint: str
    status_code: int
    latency_ms: float
    tokens: int = 0
    cost_usd: float = 0.0
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """
    Collect and export Prometheus-style metrics.

    Tracks:
    - Request latency (histogram)
    - Request count by status/endpoint
    - Token usage
    - Cost by customer
    """

    def __init__(self):
        self._requests: List[RequestMetric] = []
        self._start_time = time.time()

    def record_request(
        self,
        customer_id: str,
        endpoint: str,
        status_code: int,
        latency_ms: float,
        tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Record a request metric."""
        self._requests.append(RequestMetric(
            customer_id=customer_id,
            endpoint=endpoint,
            status_code=status_code,
            latency_ms=latency_ms,
            tokens=tokens,
            cost_usd=cost_usd,
        ))

    def get_summary(self) -> dict:
        """Get metrics summary."""
        if not self._requests:
            return {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "avg_latency_ms": 0.0,
                "success_rate": 0.0,
                "uptime_seconds": time.time() - self._start_time,
            }

        total = len(self._requests)
        success = sum(1 for r in self._requests if 200 <= r.status_code < 300)
        latencies = [r.latency_ms for r in self._requests]

        return {
            "total_requests": total,
            "total_tokens": sum(r.tokens for r in self._requests),
            "total_cost_usd": sum(r.cost_usd for r in self._requests),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "success_rate": success / total if total > 0 else 0.0,
            "uptime_seconds": time.time() - self._start_time,
        }

    def get_by_customer(self) -> Dict[str, dict]:
        """Get metrics grouped by customer."""
        customers: Dict[str, dict] = {}
        for r in self._requests:
            if r.customer_id not in customers:
                customers[r.customer_id] = {
                    "requests": 0,
                    "tokens": 0,
                    "cost_usd": 0.0,
                    "errors": 0,
                }
            customers[r.customer_id]["requests"] += 1
            customers[r.customer_id]["tokens"] += r.tokens
            customers[r.customer_id]["cost_usd"] += r.cost_usd
            if r.status_code >= 400:
                customers[r.customer_id]["errors"] += 1
        return customers

    def prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        summary = self.get_summary()

        # Counter metrics
        lines.append("# HELP enzu_api_requests_total Total API requests")
        lines.append("# TYPE enzu_api_requests_total counter")
        lines.append(f"enzu_api_requests_total {summary['total_requests']}")

        lines.append("# HELP enzu_api_tokens_total Total tokens used")
        lines.append("# TYPE enzu_api_tokens_total counter")
        lines.append(f"enzu_api_tokens_total {summary['total_tokens']}")

        lines.append("# HELP enzu_api_cost_usd_total Total cost in USD")
        lines.append("# TYPE enzu_api_cost_usd_total counter")
        lines.append(f"enzu_api_cost_usd_total {summary['total_cost_usd']:.6f}")

        # Gauge metrics
        lines.append("# HELP enzu_api_latency_avg_ms Average latency in milliseconds")
        lines.append("# TYPE enzu_api_latency_avg_ms gauge")
        lines.append(f"enzu_api_latency_avg_ms {summary['avg_latency_ms']:.2f}")

        lines.append("# HELP enzu_api_success_rate Request success rate")
        lines.append("# TYPE enzu_api_success_rate gauge")
        lines.append(f"enzu_api_success_rate {summary['success_rate']:.4f}")

        lines.append("# HELP enzu_api_uptime_seconds Service uptime in seconds")
        lines.append("# TYPE enzu_api_uptime_seconds gauge")
        lines.append(f"enzu_api_uptime_seconds {summary['uptime_seconds']:.0f}")

        # Per-customer metrics
        by_customer = self.get_by_customer()
        if by_customer:
            lines.append("# HELP enzu_api_customer_requests Customer request count")
            lines.append("# TYPE enzu_api_customer_requests counter")
            for customer_id, data in by_customer.items():
                lines.append(
                    f'enzu_api_customer_requests{{customer="{customer_id}"}} {data["requests"]}'
                )

            lines.append("# HELP enzu_api_customer_cost_usd Customer cost in USD")
            lines.append("# TYPE enzu_api_customer_cost_usd counter")
            for customer_id, data in by_customer.items():
                lines.append(
                    f'enzu_api_customer_cost_usd{{customer="{customer_id}"}} {data["cost_usd"]:.6f}'
                )

        return "\n".join(lines)


# Global metrics instance
_metrics: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics
