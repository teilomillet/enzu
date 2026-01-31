![enzu](github-32.jpeg)

# enzu

Budgeted LLM tasks that scale beyond context.

![PyPI](https://img.shields.io/pypi/v/enzu)
![Python](https://img.shields.io/pypi/pyversions/enzu)
![License](https://img.shields.io/github/license/teilomillet/enzu)

enzu is a Python-first toolkit for AI engineers and builders who need reliable, budgeted LLM runs. It enforces hard limits (tokens, time, cost), switches to RLM when context is large, and works across OpenAI-compatible providers. Use it from Python, the CLI, or the HTTP API.

## 30-second quickstart

```bash
uv add enzu
export OPENAI_API_KEY=sk-...
python -c "from enzu import ask; print(ask('Say hello in one sentence.'))"
```

## What enzu is (and isn’t)

- **Enzu is** a *budget + reliability layer* for LLM work: caps that actually stop execution when you hit token/time/cost limits.
- **Enzu isn’t** a giant agent framework. It’s meant to stay small, composable, and easy to drop into existing code.

## Why enzu

- **Hard budgets by default**: tokens, time, and cost caps that actually stop work
- **RLM mode for long context**: recursive subcalls when prompts are too large
- **Provider-agnostic**: OpenAI-compatible APIs and bring-your-own model
- **Production-ready surfaces**: Python SDK, CLI worker, and HTTP API

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

## Examples

- `examples/python_quickstart.py` - Minimal Python usage
- `examples/python_budget_guardrails.py` - Hard budget limits
- `examples/budget_cap_total_tokens.py` - Tiny total-token cap (hard stop)
- `examples/budget_cap_seconds.py` - Tiny time cap (hard stop)
- `examples/budget_cap_cost_openrouter.py` - Tiny cost cap (OpenRouter only)
- `examples/http_quickstart.sh` - HTTP API run
- `examples/chat_with_budget.py` - TaskSpec + budgets + success criteria
- `examples/rlm_with_context.py` - RLM run over longer context
- `examples/research_with_exa.py` - Research tool + synthesis
- `examples/file_chatbot.py` - File-based chat loop
- `examples/file_researcher.py` - Session-based research loop

## Contributing

See `CONTRIBUTING.md`.

## Requirements

Python 3.9+
