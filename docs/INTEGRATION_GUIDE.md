# enzu Integration Guide

This guide is for two audiences:
- Models/tools that need a stable contract to call enzu.
- Engineers who need to deploy enzu quickly across services.

For a step-by-step rollout, see `docs/DEPLOYMENT_QUICKSTART.md`.

## Why enzu

- Single JSON contract for CLI and Python, with explicit schemas.
- Built-in budget limits and output verification for predictable runs.
- Provider-agnostic OpenAI-compatible routing (OpenRouter, OpenAI, Mistral, etc.).
- RLM mode for large contexts without stuffing prompts.
- CLI is stdin in, stdout out: easy to wrap in any stack.

## Model/tool usage

Goal: build a valid payload, run the CLI, parse the JSON report.

1. Review the schema bundle in `docs/schema/bundle.json` (or run `enzu --print-schema`).
2. Fetch schemas with `enzu --print-schema` or use `docs/schema/bundle.json`.
3. Build a payload that matches `run_payload`.
4. Execute CLI and parse JSON output.

Schema bundle notes:
- `meta.defaults` lists injected defaults for missing `budget` and `success_criteria`.
- `meta.mode_requirements` lists mode-specific required fields.

Minimal payload (chat mode):

```json
{
  "mode": "chat",
  "provider": "openrouter",
  "task": {
    "task_id": "example-task",
    "input_text": "Summarize this text.",
    "model": "openrouter/auto",
    "budget": { "max_output_tokens": 200 },
    "success_criteria": { "min_word_count": 1 }
  }
}
```

RLM payload (context required):

```json
{
  "mode": "rlm",
  "provider": "openrouter",
  "task": {
    "task_id": "rlm-task",
    "input_text": "Answer using the context.",
    "model": "openrouter/auto",
    "budget": { "max_output_tokens": 400 },
    "success_criteria": { "required_substrings": ["Answer"] }
  },
  "context": "large text or preloaded context"
}
```

Interpreting results:
- `success=false` means verification failed or budget limits were exceeded.
- `errors` and `verification.reasons` explain why.

Exit codes:
- `0` on success (report printed to stdout).
- `1` on failure (error printed to stderr).

Minimal chat report shape:

```json
{
  "success": true,
  "task_id": "example-task",
  "provider": "openrouter",
  "model": "openrouter/auto",
  "output_text": "..."
}
```

Full report shapes are in `docs/schema/report.json`.

Programmatic CLI call (language-agnostic pattern):

```python
import json
import subprocess

payload = {"mode": "chat", "provider": "openrouter", "task": {"task_id": "t1", "input_text": "Ping", "model": "openrouter/auto"}}
proc = subprocess.run(
    ["enzu"],
    input=json.dumps(payload),
    text=True,
    capture_output=True,
)
if proc.returncode != 0:
    raise RuntimeError(proc.stderr.strip())
report = json.loads(proc.stdout)
```

## Engineer deployment

### Install

```
uv pip install enzu
# or: pip install enzu

# HTTP API server:
# uv pip install "enzu[server]"
```

### Provider configuration

Set env vars:
- `OPENROUTER_API_KEY` (OpenRouter)
- `OPENAI_API_KEY`, `OPENAI_ORG`, `OPENAI_PROJECT` (OpenAI)
- `{PROVIDER}_API_KEY` for others (e.g., `MISTRAL_API_KEY`)

Supported provider names come from the registry:

```python
from enzu import list_providers

print(list_providers())
```

Custom providers can be registered with `register_provider(...)` when they are OpenAI-compatible.

### CLI deployment pattern

The CLI is a stable JSON interface. Use it as a worker:

```bash
cat payload.json | enzu --provider openrouter --model "openrouter/auto"
```

This returns a JSON report on stdout. Use `--print-schema` at startup to validate payloads.
Each CLI invocation runs a single task and exits.

### Background orchestration

For background worker patterns (CLI worker, in‑process, Session, pipelines), see
`docs/BACKGROUND_ORCHESTRATION.md`.

### Python deployment pattern

```python
from enzu import run

report = run(
    "Summarize this text.",
    provider="openrouter",
    model="openrouter/auto",
    return_report=True,
)
```
Python `run()` defaults to `mode="auto"` (see `enzu/api.py`), which selects chat or RLM based on inputs. Use `mode="chat"` or `mode="rlm"` to force a choice.

### Python API notes

<!-- Source: enzu/api.py, enzu/session.py -->
- `run()` switches to RLM when any of: `data`, `cost`, `seconds`, `goal`, or prompt+data > ~256k chars.
- Session history is prepended to `data`, which triggers RLM in auto mode once history exists.

Session example:

```python
from enzu import Session, SessionBudgetExceeded

session = Session(model="openrouter/auto", provider="openrouter", max_cost_usd=5.00)
try:
    session.run("Find the bug.", data=logs, cost=1.00)
    session.run("Fix it.")
except SessionBudgetExceeded as exc:
    print(exc)
```

See `docs/PYTHON_API_REFERENCE.md` for a full module map and usage patterns.

If you receive JSON payloads and want the same defaulting logic as the CLI:

```python
from enzu import task_spec_from_payload

spec = task_spec_from_payload(payload, model_override="openrouter/auto")
```

### Operational tips

- Use `budget` to cap cost and time.
- Use `success_criteria` to enforce deterministic checks.
- Use `mode="rlm"` + `context` for large inputs.
- Use `max_steps` to bound RLM iteration count.
- Enable telemetry with Logfire if you need traces (`ENZU_LOGFIRE=1`).
- Parse stderr for progress events when `--progress` is enabled.

## Getting the most value

- Reuse the schema and payload across services to reduce integration time.
- Prefer CLI when you need language-agnostic integration.
- Prefer Python when you need tight in-process control and callbacks.
- Use RLM mode for large context, with structured checks to avoid rework.
 - Validate payloads against `bundle.json` or `enzu --print-schema` before execution.

## Multi-service rollout checklist

- Pin the enzu version per service.
- Store provider keys in a secrets manager.
- Validate payloads against `enzu --print-schema`.
- Capture the JSON report for audit and debugging.

## Notes

- RLM/automode execute Python in-process. It is not a security boundary.
- `automode` is CLI-only and requires `fs_root`.
