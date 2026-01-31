# Multi-Tenant API Service

Production SaaS example: FastAPI + enzu with per-customer budgets.

## Features

- **FastAPI integration** - REST endpoints with async job processing
- **Multi-tenant isolation** - Per-customer budget tracking and enforcement
- **Job queue pattern** - Background workers for async document analysis
- **Prometheus metrics** - Request latency, token usage, cost tracking
- **Docker deployment** - Ready for production with docker-compose

## Quick Start

### Local Development

```bash
# Install dependencies
pip install fastapi uvicorn

# Set your API key
export OPENAI_API_KEY=sk-...

# Run the server
cd examples/production/api_service
uvicorn main:app --reload
```

### With Docker

```bash
# Set your API key
export OPENAI_API_KEY=sk-...

# Run with docker-compose
cd examples/production/api_service
docker-compose up

# With Prometheus/Grafana monitoring
docker-compose --profile monitoring up
```

## API Endpoints

### POST /analyze

Submit a document for analysis.

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -H "X-Customer-ID: customer-123" \
  -d '{
    "document": "This agreement is between Party A and Party B...",
    "task": "Extract key terms and obligations",
    "max_tokens": 500
  }'
```

Response:
```json
{
  "job_id": "job-abc123def456",
  "status": "pending",
  "customer_id": "customer-123",
  "created_at": "2024-01-15T10:30:00Z",
  "message": "Job submitted. Poll GET /jobs/{job_id} for status."
}
```

### GET /jobs/{job_id}

Get job status and result.

```bash
curl http://localhost:8000/jobs/job-abc123def456 \
  -H "X-Customer-ID: customer-123"
```

Response (completed):
```json
{
  "job_id": "job-abc123def456",
  "status": "completed",
  "customer_id": "customer-123",
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:30:05Z",
  "result": "Key terms extracted: 1. Payment terms...",
  "tokens_used": 342,
  "cost_usd": 0.0023
}
```

### GET /budget

Get customer budget status.

```bash
curl http://localhost:8000/budget \
  -H "X-Customer-ID: customer-123"
```

Response:
```json
{
  "customer_id": "customer-123",
  "budget_limit_usd": 10.0,
  "budget_used_usd": 0.15,
  "budget_remaining_usd": 9.85,
  "requests_count": 42,
  "period_start": "2024-01-01T00:00:00Z",
  "period_end": "2024-01-31T00:00:00Z"
}
```

### GET /metrics

Prometheus metrics.

```bash
curl http://localhost:8000/metrics
```

### GET /health

Health check.

```bash
curl http://localhost:8000/health
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key |
| `OPENROUTER_API_KEY` | - | OpenRouter API key (optional) |
| `DEFAULT_BUDGET_USD` | 10.0 | Default monthly budget per customer |
| `NUM_WORKERS` | 2 | Background worker threads |

## Architecture

See [architecture.md](architecture.md) for detailed design.

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI App                            │
├─────────────────────────────────────────────────────────────┤
│  Endpoints      │  Budget Controller  │  Metrics Collector  │
│  /analyze       │  Per-customer       │  Prometheus format  │
│  /jobs/{id}     │  limits & tracking  │  Latency, cost,     │
│  /budget        │                     │  tokens             │
│  /metrics       │                     │                     │
│  /health        │                     │                     │
├─────────────────┼─────────────────────┼─────────────────────┤
│                      Job Queue                              │
│  Thread-safe queue + N worker threads                       │
│  Each worker uses enzu for document analysis                │
└─────────────────────────────────────────────────────────────┘
```

## Multi-Tenant Isolation

Each customer is identified by `X-Customer-ID` header:

1. **Budget isolation** - Each customer has their own budget
2. **Job isolation** - Customers can only see their own jobs
3. **Metrics isolation** - Per-customer metrics available

## Error Handling

| Status | Meaning |
|--------|---------|
| 200 | Success |
| 202 | Job accepted (async) |
| 401 | Missing X-Customer-ID |
| 402 | Budget exceeded |
| 404 | Job not found |
| 500 | Internal error |

## Testing

```bash
# Run tests
pytest examples/production/api_service/tests/ -v

# Test with curl
./test_api.sh
```

## Production Considerations

1. **Authentication**: Replace X-Customer-ID with proper auth (JWT, API keys)
2. **Database**: Use Redis/PostgreSQL for job persistence
3. **Scaling**: Run multiple instances behind load balancer
4. **Monitoring**: Connect Prometheus to Grafana for dashboards
5. **Rate limiting**: Add rate limits per customer tier
