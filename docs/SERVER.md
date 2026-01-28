# HTTP API Server

enzu includes a FastAPI server for running tasks over HTTP.

## Install

```bash
uv pip install "enzu[server]"
# or: pip install "enzu[server]"
```

## Configure

```bash
export OPENAI_API_KEY=sk-...
# Optional auth (requires X-API-Key on requests):
# export ENZU_API_KEY=your-secret

# Optional defaults:
# export ENZU_DEFAULT_MODEL=gpt-4o
# export ENZU_DEFAULT_PROVIDER=openai
```

## Run

```bash
uvicorn enzu.server:app --host 0.0.0.0 --port 8000
```

## Run a task

```bash
curl http://localhost:8000/v1/run \
  -H "Content-Type: application/json" \
  -d '{"task":"Say hello","model":"gpt-4o","provider":"openai"}'
```

Response shape:

```json
{
  "answer": "...",
  "request_id": "...",
  "model": "...",
  "usage": {
    "total_tokens": 123,
    "prompt_tokens": 45,
    "completion_tokens": 78,
    "cost_usd": 0.0123
  }
}
```

## Sessions (multi-turn)

Create a session:

```bash
curl http://localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","provider":"openai","max_cost_usd":1.0}'
```

Run inside the session:

```bash
curl http://localhost:8000/v1/sessions/<session_id>/run \
  -H "Content-Type: application/json" \
  -d '{"task":"What did I just say?"}'
```

Fetch session state:

```bash
curl http://localhost:8000/v1/sessions/<session_id>
```

## Auth

If `ENZU_API_KEY` is set, include `X-API-Key` on every request.

## Notes

- Sessions are in-memory and scoped to a single server process.
- Use `ENZU_DEFAULT_MODEL` and `ENZU_DEFAULT_PROVIDER` to avoid sending model/provider each request.
- `GET /health` is available for load balancers.
