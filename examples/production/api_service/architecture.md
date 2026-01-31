# API Service Architecture

## Overview

A multi-tenant document analysis API built on FastAPI with per-customer budget enforcement.

## System Diagram

```
                                    ┌─────────────────┐
                                    │   Prometheus    │
                                    │   (optional)    │
                                    └────────▲────────┘
                                             │ scrape /metrics
┌──────────────┐                             │
│   Client     │                    ┌────────┴────────┐
│              │  POST /analyze     │                 │
│  Customer A  │───────────────────▶│   FastAPI App   │
│  Customer B  │◀───────────────────│                 │
│  Customer C  │  Job ID            │  Port 8000      │
└──────────────┘                    └────────┬────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
           ┌────────▼────────┐      ┌────────▼────────┐      ┌────────▼────────┐
           │ Budget          │      │ Job Queue       │      │ Metrics         │
           │ Controller      │      │                 │      │ Collector       │
           │                 │      │ ┌─────────────┐ │      │                 │
           │ Per-customer    │      │ │ Worker 1    │ │      │ Latency         │
           │ limits &        │◀────▶│ │ Worker 2    │ │      │ Tokens          │
           │ tracking        │      │ │ ...         │ │      │ Cost            │
           │                 │      │ └──────┬──────┘ │      │ By customer     │
           └─────────────────┘      └────────┼────────┘      └─────────────────┘
                                             │
                                    ┌────────▼────────┐
                                    │     Enzu        │
                                    │                 │
                                    │  LLM execution  │
                                    │  with budgets   │
                                    └─────────────────┘
```

## Request Flow

```
1. Request Received
   POST /analyze
   Header: X-Customer-ID: customer-123
   Body: {document, task, max_tokens}
           │
           ▼
2. Authentication
   ┌─────────────────────────┐
   │ Extract X-Customer-ID   │
   │ (401 if missing)        │
   └───────────┬─────────────┘
               │
               ▼
3. Budget Check
   ┌─────────────────────────┐
   │ BudgetController        │
   │ .check_budget()         │
   │ (402 if exceeded)       │
   └───────────┬─────────────┘
               │
               ▼
4. Job Submission
   ┌─────────────────────────┐
   │ JobQueue.submit()       │
   │ Returns job_id          │
   │ Status: PENDING         │
   └───────────┬─────────────┘
               │
               ▼
5. Response (202 Accepted)
   {job_id, status: pending}
               │
               │ (async)
               ▼
6. Background Processing
   ┌─────────────────────────┐
   │ Worker picks up job     │
   │ Status: RUNNING         │
   │                         │
   │ enzu.run(               │
   │   task,                 │
   │   data=document,        │
   │   tokens=max_tokens     │
   │ )                       │
   │                         │
   │ Record cost to budget   │
   │ Status: COMPLETED       │
   └─────────────────────────┘
```

## Component Details

### Budget Controller

Thread-safe per-customer budget tracking.

```
┌──────────────────────────────────────────┐
│           Budget Controller              │
├──────────────────────────────────────────┤
│                                          │
│  _budgets: Dict[customer_id, BudgetState]│
│  _lock: RLock (thread safety)            │
│                                          │
│  Methods:                                │
│  ├─ get_or_create(customer_id)           │
│  ├─ check_budget(customer_id, est_cost)  │
│  ├─ record_usage(customer_id, cost)      │
│  └─ get_status(customer_id)              │
│                                          │
│  Auto-reset: Monthly (configurable)      │
│                                          │
└──────────────────────────────────────────┘
```

### Job Queue

In-memory queue with background workers.

```
┌──────────────────────────────────────────┐
│              Job Queue                   │
├──────────────────────────────────────────┤
│                                          │
│  _queue: Queue[Job]                      │
│  _jobs: Dict[job_id, Job]                │
│  _workers: List[Thread]                  │
│                                          │
│  State Machine:                          │
│  PENDING ──▶ RUNNING ──▶ COMPLETED       │
│                    └───▶ FAILED          │
│                    └───▶ BUDGET_EXCEEDED │
│                                          │
└──────────────────────────────────────────┘
```

### Metrics Collector

Prometheus-compatible metrics.

```
Counters:
  enzu_api_requests_total
  enzu_api_tokens_total
  enzu_api_cost_usd_total
  enzu_jobs_completed_total
  enzu_jobs_failed_total

Gauges:
  enzu_api_latency_avg_ms
  enzu_api_success_rate
  enzu_api_uptime_seconds
  enzu_jobs_pending

Labels:
  customer (for per-customer metrics)
```

## Multi-Tenant Isolation

### Data Isolation

```
Customer A                    Customer B
    │                             │
    ▼                             ▼
┌─────────────┐             ┌─────────────┐
│ Budget: $10 │             │ Budget: $50 │
│ Used: $2.50 │             │ Used: $15   │
│ Jobs: 42    │             │ Jobs: 156   │
└─────────────┘             └─────────────┘
        │                         │
        └─────────┬───────────────┘
                  │
                  ▼
           ┌─────────────┐
           │ Shared      │
           │ Job Queue   │
           │ & Workers   │
           └─────────────┘
```

### Enforcement Points

| Point | Check | Action |
|-------|-------|--------|
| POST /analyze | Budget available | 402 if exceeded |
| Worker start | Budget re-check | BUDGET_EXCEEDED |
| GET /jobs/{id} | Customer owns job | 404 if not owner |

## Scaling Considerations

### Vertical Scaling

```python
# Increase workers
NUM_WORKERS=10
```

### Horizontal Scaling

```
                    ┌─────────────┐
                    │ Load        │
                    │ Balancer    │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
    ┌─────▼─────┐    ┌─────▼─────┐    ┌─────▼─────┐
    │ Instance 1│    │ Instance 2│    │ Instance 3│
    └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
          │                │                │
          └────────────────┼────────────────┘
                           │
                    ┌──────▼──────┐
                    │ Redis       │
                    │ (shared     │
                    │  state)     │
                    └─────────────┘
```

For horizontal scaling, replace in-memory stores with Redis:
- Budget state → Redis hash
- Job queue → Redis list + pub/sub
- Metrics → Redis or Prometheus remote write

## Error Handling

```
┌─────────────────────────────────────────────────┐
│                Error Hierarchy                  │
├─────────────────────────────────────────────────┤
│                                                 │
│  401 Unauthorized                               │
│  └── Missing X-Customer-ID header               │
│                                                 │
│  402 Payment Required                           │
│  └── Customer budget exceeded                   │
│                                                 │
│  404 Not Found                                  │
│  ├── Job ID doesn't exist                       │
│  └── Job belongs to different customer          │
│                                                 │
│  500 Internal Server Error                      │
│  ├── Enzu provider error                        │
│  └── Unhandled exception                        │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Security Considerations

### Current (Demo)

- Customer ID via header (no auth)
- In-memory state (no persistence)

### Production Upgrades

1. **Authentication**: JWT tokens or API keys
2. **Authorization**: Role-based access control
3. **Rate limiting**: Per-customer request limits
4. **Encryption**: TLS for all traffic
5. **Audit logging**: All requests logged with customer ID
