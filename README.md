![enzu](github-32.jpeg)

Budgeted LLM tasks that scale beyond context.

enzu is a Python-first toolkit for AI engineers and builders who need reliable, budgeted LLM runs. It enforces hard limits (tokens, time, cost), switches to RLM when context is large, and works across OpenAI-compatible providers. Use it from Python, the CLI, or the HTTP API.

## Quickstart (Python)

```bash
uv add enzu
# or: pip install enzu
```

```bash
export OPENAI_API_KEY=sk-...
```

```python
from enzu import Enzu, ask

print(ask("What is 2+2?"))

client = Enzu()  # Auto-detects from env
answer = client.run(
    "Summarize the key points",
    data="...long document...",
    tokens=400,
)
print(answer)
```

Tip: Set `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, or another provider key. You can always pass `model=` and `provider=` explicitly.

## HTTP API (server)

```bash
uv pip install "enzu[server]"
uvicorn enzu.server:app --host 0.0.0.0 --port 8000
```

```bash
curl http://localhost:8000/v1/run \
  -H "Content-Type: application/json" \
  -d '{"task":"Say hello","model":"gpt-4o","provider":"openai"}'
```

If you set `ENZU_API_KEY`, pass `X-API-Key` on every request.

## CLI worker

```bash
cat <<'JSON' | enzu
{
  "provider": "openai",
  "task": {
    "task_id": "hello-1",
    "input_text": "Say hello in one sentence.",
    "model": "gpt-4o"
  }
}
JSON
```

## Docs

- `docs/README.md` - Start here
- `docs/QUICKREF.md` - Providers, env vars, model formats
- `docs/DEPLOYMENT_QUICKSTART.md` - CLI + integration patterns
- `docs/SERVER.md` - HTTP API
- `docs/PYTHON_API_REFERENCE.md` - Full Python API
- `docs/COOKBOOK.md` - Patterns and recipes

## Contributing

See `CONTRIBUTING.md`.

## Requirements

Python 3.9+
