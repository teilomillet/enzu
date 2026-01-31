# Invoice Processing Pipeline Architecture

## Overview

A batch processing system that extracts structured data from invoice documents with strict per-item budget control.

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Invoice Pipeline                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │   Invoice    │    │   Thread Pool    │    │    Results       │  │
│  │   Directory  │───▶│   Executor       │───▶│    Collector     │  │
│  │              │    │                  │    │                  │  │
│  │ *.txt files  │    │ parallelism=3    │    │ results.json     │  │
│  └──────────────┘    └────────┬─────────┘    │ metrics.json     │  │
│                               │              └──────────────────┘  │
│                               │                                     │
│                    ┌──────────▼─────────┐                          │
│                    │   Per-Invoice      │                          │
│                    │   Processing       │                          │
│                    │                    │                          │
│                    │ ┌────────────────┐ │                          │
│                    │ │ Budget Guard   │ │                          │
│                    │ │ $0.02 / 300tok │ │                          │
│                    │ └───────┬────────┘ │                          │
│                    │         │          │                          │
│                    │ ┌───────▼────────┐ │                          │
│                    │ │ Retry Logic    │ │                          │
│                    │ │ max=2, backoff │ │                          │
│                    │ └───────┬────────┘ │                          │
│                    │         │          │                          │
│                    │ ┌───────▼────────┐ │                          │
│                    │ │ LLM Extract    │◀┼──── Enzu Client          │
│                    │ │ JSON parsing   │ │                          │
│                    │ └───────┬────────┘ │                          │
│                    │         │          │                          │
│                    │ ┌───────▼────────┐ │                          │
│                    │ │ Metrics        │◀┼──── RunMetricsCollector  │
│                    │ │ p50/p95 costs  │ │                          │
│                    │ └────────────────┘ │                          │
│                    └────────────────────┘                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
1. Discovery
   sample_invoices/*.txt ──▶ List[Path]

2. Parallel Distribution
   List[Path] ──▶ ThreadPoolExecutor(workers=3)
                   ├─▶ Worker 1: invoice_001.txt
                   ├─▶ Worker 2: invoice_002.txt
                   └─▶ Worker 3: invoice_003.txt

3. Per-Invoice Processing
   invoice.txt ──▶ _process_single()
                    │
                    ├─▶ Read content
                    │
                    ├─▶ _extract_invoice() [with retry loop]
                    │    │
                    │    ├─▶ Build prompt
                    │    │
                    │    ├─▶ Enzu.run(tokens=300, cost=$0.02)
                    │    │    │
                    │    │    └─▶ Budget enforcement
                    │    │
                    │    ├─▶ Parse JSON response
                    │    │
                    │    └─▶ Record metrics
                    │
                    └─▶ ProcessingResult

4. Aggregation
   List[ProcessingResult] ──▶ metrics.json + results.json
```

## Budget Enforcement Points

```
┌─────────────────────────────────────────┐
│           Budget Checkpoints            │
├─────────────────────────────────────────┤
│                                         │
│  1. Pre-call Check                      │
│     ┌─────────────────────────┐         │
│     │ tokens=300, cost=$0.02  │         │
│     │ Set hard limits         │         │
│     └───────────┬─────────────┘         │
│                 │                       │
│  2. During Generation                   │
│     ┌───────────▼─────────────┐         │
│     │ Token counter running   │         │
│     │ Stop at 300 tokens      │         │
│     └───────────┬─────────────┘         │
│                 │                       │
│  3. Post-call Accounting                │
│     ┌───────────▼─────────────┐         │
│     │ Record actual usage     │         │
│     │ Update metrics          │         │
│     └─────────────────────────┘         │
│                                         │
└─────────────────────────────────────────┘
```

## Retry Strategy

```
Attempt 1 ──▶ FAIL ──▶ wait 0.5s ──▶ Attempt 2 ──▶ FAIL ──▶ wait 1.0s ──▶ Attempt 3
                                                                              │
                                                                              ▼
                                                                    RETRY_EXHAUSTED
```

- **Max retries**: 2 (3 attempts total)
- **Backoff**: Exponential (0.5s, 1.0s)
- **Budget per retry**: Same limits apply to each attempt
- **Retry on**: API errors, timeouts
- **No retry on**: Budget exceeded (deterministic)

## Graceful Degradation

When budget is exceeded, the pipeline attempts to salvage partial data:

```
Budget Exceeded
      │
      ▼
┌─────────────────────────────────┐
│ Check for partial response      │
│                                 │
│ if response.output:             │
│   try:                          │
│     parse JSON                  │──▶ Partial InvoiceData
│   except:                       │
│     return raw response         │──▶ InvoiceData(raw_response=...)
│ else:                           │
│   return None                   │──▶ Failed extraction
└─────────────────────────────────┘
```

## Metrics Collection

```
Per-Invoice Metrics (RunEvent):
  - run_id: "invoice_001-0" (filename-retry)
  - outcome: "success" | "budget_exceeded" | ...
  - elapsed_seconds: 1.23
  - cost_usd: 0.0012
  - output_tokens: 180
  - total_tokens: 450

Aggregated Metrics:
  - p50/p95/p99 for cost, tokens, time
  - outcome distribution
  - success/failure rates
  - total cost tracking
```

## Key Design Decisions

### 1. ThreadPoolExecutor vs AsyncIO

**Chosen**: ThreadPoolExecutor

**Why**:
- Simpler mental model
- Each thread gets its own Enzu client instance
- Easier error isolation
- Good enough for I/O-bound LLM calls

**Alternative**: asyncio with `aiohttp` for higher concurrency

### 2. Per-Invoice Budget vs Total Budget

**Chosen**: Per-invoice budget ($0.02 each)

**Why**:
- Predictable per-item cost
- One expensive invoice can't starve others
- Easy to reason about scaling

**Alternative**: Total budget with proportional allocation

### 3. JSON Extraction Prompt

**Chosen**: Single-shot JSON extraction

**Why**:
- Simpler than multi-step
- Lower latency
- Works well with structured invoices

**Alternative**: Chain-of-thought with validation step

### 4. Retry Budget

**Chosen**: Same budget per retry attempt

**Why**:
- Simple accounting
- Retries typically for transient errors
- Budget exceeded doesn't retry anyway

**Alternative**: Decreasing budget per retry

## Failure Modes

| Scenario | Detection | Handling |
|----------|-----------|----------|
| Invalid JSON | JSONDecodeError | Store raw response |
| Budget exceeded | Outcome.BUDGET_EXCEEDED | Try partial extraction |
| API timeout | Exception | Retry with backoff |
| Rate limit | 429 response | Retry with backoff |
| Auth failure | 401 response | Fail immediately |
| Parse failure | Missing fields | Use defaults |

## Scaling Considerations

### For 1,000 invoices

```python
PARALLELISM = 10          # More concurrent workers
# Enable connection pooling in Enzu client
client = Enzu(provider=PROVIDER, model=MODEL, use_pool=True)
```

### For 100,000 invoices

1. Use `enzu.queue` for distributed processing
2. Add checkpointing every 100 invoices
3. Consider chunking into batches of 1,000
4. Add dead letter queue for failed items

### Cost Projection

| Invoices | Budget/Item | Max Total | Typical |
|----------|-------------|-----------|---------|
| 100 | $0.02 | $2.00 | ~$0.12 |
| 1,000 | $0.02 | $20.00 | ~$1.20 |
| 10,000 | $0.02 | $200.00 | ~$12.00 |

*Typical assumes 60% of budget used per invoice*
