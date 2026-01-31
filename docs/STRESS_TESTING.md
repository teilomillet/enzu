# Stress Testing: Test p95 Behavior Before Production

Production systems face degraded conditions that are rarely tested during development. This guide explains enzu's stress testing harness for simulating "bad day" scenarios.

## The Problem

Teams typically test the **happy path**:
- Fast provider responses
- No rate limits
- Low concurrency
- Baseline latency

But what breaks in production is:
- **Rate limit storms** (429s during traffic spikes)
- **Provider degradation** (slow responses, timeouts, intermittent 5xx errors)
- **Retry amplification** (retries that blow through budgets)
- **Tail latency** (p95/p99 cost and time)

## Quick Start

```python
from enzu import Enzu
from enzu.stress import Scenario, run_scenarios, rules, faults
from enzu.retries import retry_tracking_context
from enzu.metrics import RunEvent
from datetime import datetime, timezone

client = Enzu()

# Define your task function
def my_task(injector):
    with retry_tracking_context() as tracker:
        started = datetime.now(timezone.utc)

        # Inject fault before provider call
        injector.inject()

        report = client.run("Your prompt", tokens=500, return_report=True)
        finished = datetime.now(timezone.utc)

        return RunEvent.from_report_with_tracker(
            run_id=f"run_{injector.call_count}",
            report=report,
            started_at=started,
            finished_at=finished,
            tracker=tracker,
        )

# Define scenarios
scenarios = [
    Scenario("baseline", []),
    Scenario("high_429s", [rules.error_rate(0.3, faults.rate_limit_429)]),
    Scenario("slow_provider", [rules.latency(0.5, 2.0)]),
]

# Run and get reports
reports = run_scenarios(task=my_task, scenarios=scenarios, runs_per_scenario=50)

# Analyze results
from enzu.stress import format_report
print(format_report(reports))
```

## Scenarios

A **Scenario** is a named collection of fault injection rules:

```python
from enzu.stress import Scenario, rules, faults

# No faults (baseline)
baseline = Scenario("baseline", [])

# 30% chance of 429 on each call
high_rate_limits = Scenario("high_429s", [
    rules.error_rate(0.3, faults.rate_limit_429)
])

# 50% chance of 2-second delay
slow_provider = Scenario("slow_provider", [
    rules.latency(0.5, 2.0)
])

# Fail calls 5-14 (simulates provider outage)
outage_burst = Scenario("outage", [
    rules.burst(start=5, length=10, exc_factory=faults.server_error_503)
])

# Combine multiple rules (first match wins)
mixed = Scenario("mixed", [
    rules.error_rate(0.2, faults.rate_limit_429),
    rules.latency(0.3, 1.5),
])
```

## Fault Rules

### Error Rate

Raise exceptions with probability `p`:

```python
rules.error_rate(0.3, faults.rate_limit_429)  # 30% chance of 429
rules.error_rate(0.1, faults.timeout)         # 10% chance of timeout
rules.error_rate(0.05, faults.server_error_500)  # 5% chance of 500
```

### Latency

Add delay with probability `p`:

```python
rules.latency(0.5, 2.0)  # 50% chance of 2-second delay
rules.latency(1.0, 5.0)  # Always delay by 5 seconds
```

### Burst

Fail a contiguous window of calls (useful for simulating outages):

```python
rules.burst(start=5, length=10, exc_factory=faults.server_error_503)
# Calls 5-14 will fail with 503
```

### Nth Call

Fail at specific call indices (deterministic):

```python
rules.nth_call({1, 5, 10}, faults.rate_limit_429)
# Calls 1, 5, and 10 will fail with 429
```

### Latency + Error

Add delay AND raise an error:

```python
from enzu.stress.rules import LatencyWithErrorRule

rule = LatencyWithErrorRule(p=0.2, delay_seconds=3.0, exc_factory=faults.timeout)
# 20% chance of 3-second delay followed by timeout
```

## Fault Types

All faults are OpenAI-compatible exceptions that trigger `@with_retry`:

| Fault | Description | Retry Reason |
|-------|-------------|--------------|
| `faults.rate_limit_429()` | 429 Too Many Requests | `RATE_LIMIT` |
| `faults.timeout()` | Request timeout | `TIMEOUT` |
| `faults.connection_error()` | Connection failure | `CONNECTION_ERROR` |
| `faults.server_error_500()` | 500 Internal Server Error | `SERVER_ERROR` |
| `faults.server_error_503()` | 503 Service Unavailable | `SERVER_ERROR` |

## Running Scenarios

```python
reports = run_scenarios(
    task=my_task_function,
    scenarios=[baseline, high_429s, slow_provider],
    runs_per_scenario=50,  # How many runs per scenario
    seed=42,               # For reproducibility
)
```

The task function receives a `FailureInjector`. Call `injector.inject()` before each provider call:

```python
def my_task(injector):
    with retry_tracking_context() as tracker:
        started = utc_now()

        # Inject fault (may delay or raise exception)
        injector.inject()

        report = client.run("...", tokens=500, return_report=True)
        finished = utc_now()

        return RunEvent.from_report_with_tracker(...)
```

## Interpreting Results

### Terminal State Distribution

```
--- Outcomes ---
  success: 45 (90.0%)
  budget_exceeded: 5 (10.0%)
```

**Key insight**: What fraction of runs hit each terminal state?

- Rising `budget_exceeded` → prompt bloat or retry storms
- High `timeout` → provider degradation or insufficient time budgets

### Retry Metrics

```
--- Retry Metrics ---
  Retries/run: p50=2.0 p95=8.0
  By reason:
    rate_limit: 120
    timeout: 15
  Budget exceeded during retry: 5 (10.0%)
```

**Key insights**:
- **p95 retries**: 5% of runs needed 8+ retries
- **Retry reason distribution**: Rate limits dominate (vs timeouts)
- **Budget exceeded during retry**: 10% of runs hit budget caps due to retry storms

### Latency

```
--- Latency ---
  p50=1.234s p95=5.678s p99=12.345s
```

**Key insight**: Tail latency (p95/p99) often 5-10x higher than p50.

### Cost

```
--- Cost ---
  p50=$0.0023 p95=$0.0089 p99=$0.0234
```

**Key insight**: p95 cost per run can be 4-10x the median due to retries and longer runs.

## Scenario Comparison

Compare baseline vs. degraded conditions:

| Scenario | Success Rate | p95 Retries | p95 Cost | p95 Time |
|----------|--------------|-------------|----------|----------|
| baseline | 100% | 0 | $0.002 | 1.2s |
| high_429s | 92% | 8 | $0.008 | 5.6s |
| slow_provider | 98% | 1 | $0.003 | 8.2s |

**Insights**:
- `high_429s` scenario → 4x cost increase due to retries
- `slow_provider` scenario → 7x latency increase
- Both scenarios still have >90% success rate

## Best Practices

### 1. Test Before Production

Run stress scenarios during CI/CD to catch budget/latency regressions:

```bash
python examples/stress_testing_demo.py > stress_report.txt
```

### 2. Set SLOs Based on p95/p99

Don't use averages:

```python
# Bad
assert avg_cost < 0.01

# Good
assert p95_cost < 0.05
assert p99_latency < 10.0
```

### 3. Identify Retry Amplification

If `budget_exceeded_during_retry_rate > 0.1`, you have retry storms:

**Mitigations**:
- Add jitter to retry backoff
- Use concurrency limits
- Implement circuit breakers
- Queue with backpressure

### 4. Test Your Actual Workload

Replace the demo task with your real prompts and token limits:

```python
def my_production_task(injector):
    # Your actual task logic
    injector.inject()
    result = my_agent.run(...)
    return RunEvent.from_report_with_tracker(...)
```

### 5. Use Reproducible Seeds

The `seed` parameter ensures reproducible test runs:

```python
reports = run_scenarios(..., seed=42)
```

Changing the seed will produce different random fault patterns.

## Advanced: Composing Scenarios

Combine scenarios with `+`:

```python
latency_scenario = Scenario("latency", [rules.latency(0.5, 2.0)])
error_scenario = Scenario("errors", [rules.error_rate(0.2, faults.rate_limit_429)])

# Combine both
combined = latency_scenario + error_scenario
# Name: "latency+errors"
# Rules: both latency and errors applied
```

## Advanced: Custom Rules

Implement the `FaultRule` protocol:

```python
from enzu.stress.scenario import FaultRule, CallContext, Fault
from typing import Optional

class CustomRule:
    def maybe_fault(self, ctx: CallContext) -> Optional[Fault]:
        # ctx.call_index: increments on every call (including retries)
        # ctx.run_index: which run this call belongs to
        # ctx.rng: seeded random number generator

        if ctx.call_index % 10 == 0:
            return Fault(exception=faults.timeout())
        return None

scenario = Scenario("custom", [CustomRule()])
```

## Integration with Run Metrics

Stress testing is built on top of `RunMetricsCollector`:

```python
from enzu.metrics import get_run_metrics

collector = get_run_metrics()

# Run scenarios (events are automatically collected)
reports = run_scenarios(...)

# Access global metrics
global_stats = collector.snapshot()
print(f"Total runs: {global_stats['total_runs']}")
```

## Examples

See:
- [examples/stress_testing_demo.py](../examples/stress_testing_demo.py) - Full runnable example
- [examples/run_metrics_demo.py](../examples/run_metrics_demo.py) - Run metrics basics
- [examples/retry_tracking_demo.py](../examples/retry_tracking_demo.py) - Retry tracking

## Why This Matters

| What You Test | What Actually Happens |
|---------------|----------------------|
| Baseline (fast provider) | Provider degradation during peak hours |
| Low concurrency | Traffic spikes trigger rate limits |
| Average cost | p95 runs cost 10x more |
| Success rate | 5% of runs hit budget caps |

**Stress testing makes tail behavior visible during development, not in production.**
