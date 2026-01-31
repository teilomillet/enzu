![enzu](github-32.jpeg)

# enzu

Budgeted LLM tasks that scale beyond context.

![PyPI](https://img.shields.io/pypi/v/enzu)
![Python](https://img.shields.io/pypi/pyversions/enzu)
![License](https://img.shields.io/github/license/teilomillet/enzu)

enzu is a Python-first toolkit for AI engineers and builders who need reliable, budgeted LLM runs. It enforces hard limits (tokens, time, cost), switches to RLM when context is large, and works across OpenAI-compatible providers. Use it from Python, the CLI, or the HTTP API.

## Why enzu

- **Hard budgets by default**: tokens, time, and cost caps that actually stop work
- **RLM mode for long context**: recursive subcalls when prompts are too large
- **Provider-agnostic**: OpenAI-compatible APIs and bring-your-own model
- **Production-ready surfaces**: Python SDK, CLI worker, and HTTP API

## What enzu is / isn't

| enzu is | enzu is not |
|---------|-------------|
| A budget-first execution engine | A prompt library or template system |
| Hard stops when limits are hit | Best-effort throttling |
| RLM for tasks that exceed context | A vector DB or RAG framework |
| Provider-agnostic (OpenAI-compatible) | Tied to one vendor |
| Lightweight (~2k LOC core) | A full agent framework |

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

## Budget hard-stop (killer feature)

enzu enforces budgets as physics, not policy. When you set a limit, the system **will** stop:

```python
from enzu import Enzu

client = Enzu()

# Ask for 500 words but cap at 50 tokens - enzu stops deterministically
result = client.run(
    "Write a 500-word essay on climate change.",
    data="...long research document...",
    tokens=50,  # Hard cap: output stops here
)
# Result: "[PARTIAL - budget exhausted]..." - work stopped, no runaway costs
```

See [examples/budget_hardstop_demo.py](examples/budget_hardstop_demo.py) for the full demo.

## Typed outcomes (predictable handling)

Every run returns a typed `Outcome` for deterministic error handling:

```python
from enzu import Enzu, Outcome

client = Enzu()
result = client.run("Analyze this", data=doc, tokens=100, return_report=True)

if result.outcome == Outcome.SUCCESS:
    print(result.answer)
elif result.outcome == Outcome.BUDGET_EXCEEDED:
    print(f"Partial result: {result.answer}" if result.partial else "Budget hit")
elif result.outcome == Outcome.TIMEOUT:
    handle_timeout()
# Also: PROVIDER_ERROR, TOOL_ERROR, VERIFICATION_FAILED, CANCELLED, INVALID_REQUEST
```

See [examples/typed_outcomes_demo.py](examples/typed_outcomes_demo.py) for the full demo.

## RLM mode (reasoning over long context)

When your input exceeds context limits, enzu automatically switches to RLM (Reasoning Language Model) mode—recursive subcalls that break the problem into manageable pieces:

```python
from enzu import Enzu

client = Enzu()

# Pass a large document - enzu auto-detects and uses RLM
answer = client.run(
    "Who is credited with the first algorithm?",
    data=open("large_research_paper.txt").read(),  # 100k+ tokens
    tokens=500,
)
```

RLM mode provides progress callbacks, step-by-step reasoning, and budget enforcement across all subcalls.

## Use cases

**1. Cost-controlled batch processing**
```python
# Process 1000 documents with a $10 budget cap
client = Enzu(cost=10.0)
for doc in documents:
    result = client.run("Extract key entities", data=doc)
```

**2. Research assistant with guardrails**
```python
# Research task with time and token limits
answer = client.run(
    "Research recent AI safety papers and summarize",
    seconds=60,   # Max 1 minute
    tokens=1000,  # Max 1000 output tokens
)
```

**3. Long document analysis**
```python
# Analyze a document too large for context window
summary = client.run(
    "Summarize the main arguments and conclusions",
    data=open("100_page_report.pdf.txt").read(),
    tokens=500,
)
```

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

- `examples/budget_hardstop_demo.py` - **Killer demo**: budget cap stops work deterministically
- `examples/typed_outcomes_demo.py` - Typed outcomes for predictable error handling
- `examples/python_quickstart.py` - Minimal Python usage
- `examples/python_budget_guardrails.py` - Hard budget limits
- `examples/rlm_with_context.py` - RLM run over longer context
- `examples/chat_with_budget.py` - TaskSpec + budgets + success criteria
- `examples/http_quickstart.sh` - HTTP API run
- `examples/research_with_exa.py` - Research tool + synthesis
- `examples/file_chatbot.py` - File-based chat loop
- `examples/file_researcher.py` - Session-based research loop

## Contributing

See `CONTRIBUTING.md`.

## Requirements

Python 3.9+
