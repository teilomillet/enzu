# Run Metrics: p95 Cost/Run & Terminal State Distributions

Production systems need visibility into **tail behavior**, not just averages. This guide explains enzu's first-class run metrics system.

## The Problem

Teams typically track:
- Total tokens consumed
- Total cost spent

But what breaks in production is:
- **p95 cost per run** (the expensive outliers)
- **Terminal state distribution** (what fraction hits TIMEOUT vs BUDGET_EXCEEDED?)
- **Correlated retries** (429s and transient errors that blow budgets)

## Quick Start

```python
from enzu import Enzu
from enzu.metrics import RunEvent, get_run_metrics
from datetime import datetime, timezone
import uuid

client = Enzu()
collector = get_run_metrics()

# Run a task
started_at = datetime.now(timezone.utc)
report = client.run("Analyze this document", tokens=500, return_report=True)
finished_at = datetime.now(timezone.utc)

# Record the run
event = RunEvent.from_execution_report(
    run_id=str(uuid.uuid4()),
    report=report,
    started_at=started_at,
    finished_at=finished_at,
)
collector.observe(event)

# Get p50/p95/p99
stats = collector.snapshot()
print(f"p95 cost/run: ${stats['percentiles']['cost_usd']['p95']:.4f}")
print(f"p95 time/run: {stats['percentiles']['elapsed_seconds']['p95']:.2f}s")
```

## RunEvent Schema

`RunEvent` is the canonical run summary—emitted once per run at completion:

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | str | Unique identifier for correlation |
| `task_id` | str? | Task identifier from TaskSpec |
| `provider` | str? | LLM provider (low-cardinality) |
| `model` | str? | Model used (low-cardinality) |
| `outcome` | Outcome | Terminal state (SUCCESS, BUDGET_EXCEEDED, etc.) |
| `success` | bool | Whether the run succeeded |
| `partial` | bool | True if result is incomplete |
| `elapsed_seconds` | float | Wall-clock time |
| `input_tokens` | int? | Input tokens consumed |
| `output_tokens` | int? | Output tokens consumed |
| `total_tokens` | int? | Total tokens consumed |
| `cost_usd` | float? | Cost in USD (OpenRouter only) |
| `limits_exceeded` | list[str] | Which budget limits were hit |
| `retries` | int | Number of retries |
| `attributes` | dict | High-cardinality data (logs only) |

## Metrics Collected

### Histograms (per-run distributions)

| Metric | Buckets | Use Case |
|--------|---------|----------|
| `elapsed_seconds` | 0.05s → 600s | Time per run |
| `cost_usd` | $0.0001 → $10 | Cost per run |
| `total_tokens` | 50 → 250k | Token consumption |
| `input_tokens` | 50 → 250k | Prompt size |
| `output_tokens` | 50 → 250k | Generation size |
| `retries` | 0 → 100 | Retry frequency |
| `retry_backoff_seconds` | — | Time spent in backoff |

### Counters (terminal states)

| Metric | Labels | Description |
|--------|--------|-------------|
| `runs_total` | outcome, partial | Count by terminal state |
| `runs_cost_unknown_total` | — | Runs without cost data |
| `retries_total` | reason | Total retries by reason |
| `budget_exceeded_during_retry_total` | — | Budget hits during retries |

## Getting p50/p95/p99

```python
stats = collector.snapshot()

# Percentiles for all metrics
for metric in ["elapsed_seconds", "cost_usd", "total_tokens"]:
    p = stats["percentiles"][metric]
    print(f"{metric}: p50={p['p50']}, p95={p['p95']}, p99={p['p99']}")

# Terminal state distribution
for outcome, count in stats["outcome_distribution"].items():
    print(f"{outcome}: {count}")

# Cost coverage (what fraction of runs have cost data)
print(f"Cost coverage: {stats['cost_coverage']:.1%}")
```

## Prometheus Export

Expose metrics at `/metrics`:

```python
from enzu.metrics import get_run_metrics

@app.get("/metrics")
def metrics():
    return get_run_metrics().prometheus_format()
```

Output format:

```prometheus
# HELP enzu_runs_total Total number of runs by outcome
# TYPE enzu_runs_total counter
enzu_runs_total{outcome="success",partial="0"} 142
enzu_runs_total{outcome="budget_exceeded",partial="1"} 8

# HELP enzu_run_cost_usd Distribution of USD per run
# TYPE enzu_run_cost_usd histogram
enzu_run_cost_usd_bucket{le="0.001"} 45
enzu_run_cost_usd_bucket{le="0.01"} 120
enzu_run_cost_usd_bucket{le="0.1"} 148
enzu_run_cost_usd_bucket{le="+Inf"} 150
enzu_run_cost_usd_sum 0.8234
enzu_run_cost_usd_count 150
```

## JSON Log Integration

```python
import logging
import json

logger = logging.getLogger("enzu.runs")

# After each run
event = RunEvent.from_execution_report(...)
logger.info("run.summary", extra={"run_event": event.to_log_dict()})
```

Output:
```json
{
  "type": "enzu.run_event.v1",
  "run_id": "a1b2c3d4",
  "outcome": "success",
  "elapsed_seconds": 1.234,
  "cost_usd": 0.0023,
  "total_tokens": 456
}
```

## OpenTelemetry Integration

For OTel, emit `RunEvent` fields as span attributes or metrics:

```python
from opentelemetry import metrics

meter = metrics.get_meter("enzu")
cost_histogram = meter.create_histogram("enzu.run.cost_usd")
token_histogram = meter.create_histogram("enzu.run.total_tokens")

def emit_otel_metrics(event: RunEvent):
    labels = {
        "provider": event.provider or "unknown",
        "model": event.model or "unknown",
        "outcome": event.outcome.value,
    }
    if event.cost_usd is not None:
        cost_histogram.record(event.cost_usd, labels)
    if event.total_tokens is not None:
        token_histogram.record(event.total_tokens, labels)
```

## Why p95, Not Average?

| Metric | Average | p95 | Insight |
|--------|---------|-----|---------|
| cost/run | $0.002 | $0.05 | 5% of runs cost 25x average |
| time/run | 1.2s | 12s | 5% of runs take 10x longer |

**Averages hide tail behavior.** Production SLOs and budgets should be set against p95/p99.

## Best Practices

1. **Track cost/run, not just total cost**: A few expensive runs can dominate spend.

2. **Monitor terminal state distribution**: A rising `budget_exceeded` rate signals prompt bloat or model issues.

3. **Set alerts on p95**: Alert when `p95_cost_usd > threshold` or `p95_elapsed_seconds > SLO`.

4. **Use low-cardinality labels**: Only use provider/model/outcome as Prometheus labels. Never use run_id, task_id, or user_id.

5. **Track cost coverage**: If `cost_coverage < 50%`, your p95 cost estimates are unreliable (consider using OpenRouter for cost tracking).

## Retry Tracking

Retries are a hidden cost multiplier. enzu tracks them as first-class budget signals.

### Using retry_tracking_context

```python
from enzu import Enzu
from enzu.retries import retry_tracking_context
from enzu.metrics import RunEvent, get_run_metrics

client = Enzu()
collector = get_run_metrics()

with retry_tracking_context() as tracker:
    started = datetime.now(timezone.utc)
    report = client.run("Do something", tokens=500, return_report=True)
    finished = datetime.now(timezone.utc)

    # Tracker captures retries that happened during the run
    print(f"Retries: {tracker.total_retries}")
    print(f"By reason: {tracker.to_dict()}")
    print(f"Backoff time: {tracker.backoff_seconds_total}s")

    # Create event with retry data
    event = RunEvent.from_report_with_tracker(
        run_id="...",
        report=report,
        started_at=started,
        finished_at=finished,
        tracker=tracker,
    )
    collector.observe(event)
```

### Retry Reasons

| Reason | Cause |
|--------|-------|
| `rate_limit` | 429 Too Many Requests |
| `timeout` | Request timed out |
| `server_error` | 5xx server errors |
| `connection_error` | Network/connection failures |
| `unknown` | Other retryable errors |

### Budget Attribution

When budget is exceeded and retries occurred, `budget_exceeded_during_retry=True`:

```python
stats = collector.snapshot()

# Rate of budget exhaustion during retry storms
print(f"Budget exceeded during retry: {stats['budget_exceeded_during_retry_rate']:.1%}")

# Retry reason breakdown
for reason, count in stats["retry_reason_distribution"].items():
    print(f"  {reason}: {count} retries")
```

### Mitigating Retry Storms

When you see high `budget_exceeded_during_retry_rate`:

1. **Add jitter**: Randomize request timing to avoid thundering herd
2. **Use concurrency limits**: `configure_global_limiter(max_concurrent=10)`
3. **Enable circuit breakers**: Stop retrying when provider is degraded
4. **Queue with backpressure**: Don't spawn new requests during storms

## Examples

See:
- [examples/run_metrics_demo.py](../examples/run_metrics_demo.py) - Basic metrics
- [examples/retry_tracking_demo.py](../examples/retry_tracking_demo.py) - Retry tracking
