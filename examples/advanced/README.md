# Advanced

Deep dives into RLM patterns, observability, and resilience testing.

## Recursive Language Models (RLM)

### RLM with Context
[`rlm_with_context.py`](rlm_with_context.py) - Basic RLM with context documents

Features:
- Multi-step reasoning chains
- Context-aware task execution
- `RLMEngine`, `TaskSpec`, `SuccessCriteria`

### RLM Context Optimization
[`rlm_context_optimization.py`](rlm_context_optimization.py) - Advanced context strategies

Based on academic research on efficient context passing:
- **Symbolic context**: Reference documents by ID
- **Inline context**: Embed full text
- Cost comparison and optimization guidance
- `ContextBreakdown` metrics

**Key insight:** Selective symbolic references can dramatically reduce token usage.

## Observability

### Run Metrics
[`run_metrics_demo.py`](run_metrics_demo.py) - Comprehensive metrics collection

Features:
- p50/p95/p99 percentile tracking
- Outcome distribution analysis
- Cost aggregation across runs
- Prometheus format export
- JSON logging

### Retry Tracking
[`retry_tracking_demo.py`](retry_tracking_demo.py) - Retry behavior analysis

Features:
- Retry reason distribution
- Budget-during-retry detection
- Correlation analysis (retries vs budget exhaustion)
- `retry_tracking_context` integration

**Key insight:** Identifies retry storms that cause budget exhaustion.

## Resilience Testing

### Stress Testing
[`stress_testing_demo.py`](stress_testing_demo.py) - Failure injection harness

Scenarios:
- **Baseline**: Normal operation
- **429 Rate Limits**: 50% rate limit responses
- **High Latency**: 2-5 second delays
- **Burst Traffic**: 10 concurrent requests
- **Mixed Chaos**: Combined failures

Features:
- `FailureInjector` with fault rules
- p95 metrics under degraded conditions
- Pre-production resilience validation

## When to Use These

| Scenario | Example |
|----------|---------|
| Reducing context costs | `rlm_context_optimization.py` |
| Setting up monitoring | `run_metrics_demo.py` |
| Debugging retry issues | `retry_tracking_demo.py` |
| Pre-launch testing | `stress_testing_demo.py` |

## Prerequisites

These examples require understanding of:
- Core budget concepts (see [`../concepts/`](../concepts/))
- Production patterns (see [`../production/`](../production/))
- Enzu's internal APIs (`Engine`, `TaskSpec`, `Provider`)
