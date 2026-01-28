# enzu Python API reference

This document maps the Python modules to their public entry points and shows the
expected usage patterns from the codebase.

## Quickstart: run()

<!-- Source: enzu/api.py run() -->
```python
from enzu import run

text = run(
    "Summarize this text.",
    provider="openrouter",
    model="openrouter/auto",
)
```

Return a full report (ExecutionReport or RLMExecutionReport):

```python
from enzu import run

report = run(
    "Summarize this text.",
    provider="openrouter",
    model="openrouter/auto",
    return_report=True,
)
```

Open Responses API passthrough:

```python
from enzu import run

answer = run(
    "",
    provider="openai",
    model="gpt-4o",
    responses={
        "instructions": "You are concise.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Summarize this text."}],
            }
        ],
    },
)
```

## Mode selection (Python)

<!-- Source: enzu/api.py run() -->
`run()` accepts `mode="auto"` (default), `mode="chat"`, or `mode="rlm"`.
Auto mode switches to RLM when any of:
- `data` provided (including empty string)
- `cost` or `seconds` provided
- `goal` provided
- prompt + data size exceeds ~256k chars (~64k tokens)

Force a mode explicitly:

```python
from enzu import run

run("Write a haiku.", model="openrouter/auto", mode="chat")
run("Investigate root cause.", model="openrouter/auto", mode="rlm", data=logs)
```

## TaskSpec + defaults

<!-- Source: enzu/models.py, enzu/contract.py -->
If you already have JSON shaped like the CLI payload, use the same defaults and validation:

```python
from enzu import task_spec_from_payload

spec = task_spec_from_payload(payload, model_override="openrouter/auto")
```

Defaults injected when missing:
- `budget.max_tokens = 256`
- `success_criteria.min_word_count = 1`

## Reports

<!-- Source: enzu/models.py -->
- `ExecutionReport` (chat): `output_text`, `verification`, `budget_usage`, `progress_events`, `trajectory`, `errors`.
- `RLMExecutionReport` (rlm): `answer`, `steps`, `budget_usage`, `errors`.

## Open Responses schema

```python
from enzu import openresponses_openapi_schema

schema = openresponses_openapi_schema()
```

## Session API (conversation state)

<!-- Source: enzu/session.py -->
Sessions keep conversation history and prepend it to `data` on every call.

```python
from enzu import Session, SessionBudgetExceeded

session = Session(
    model="openrouter/auto",
    provider="openrouter",
    max_cost_usd=5.00,
    max_tokens=20000,
)

try:
    answer = session.run("Find the bug.", data=logs, cost=1.00)
    follow_up = session.run("Fix it.")
except SessionBudgetExceeded as exc:
    print(exc)

session.save("debug_session.json")
session = Session.load("debug_session.json")
```

Notes:
- History is capped by `history_max_chars` (default 10,000).
- History is passed via `data`, so auto mode resolves to RLM once history exists.
- Use `raise_cost_cap()` / `raise_token_cap()` to increase session caps.
- Use `clear()` to reset history.

For end‑to‑end background worker patterns, see `docs/BACKGROUND_ORCHESTRATION.md`.
For file‑based flows, see `docs/FILE_BASED_CHATBOT.md` and `docs/FILE_BASED_RESEARCHER.md`.

## Providers and registry

<!-- Source: enzu/api.py, enzu/providers/registry.py, enzu/providers/openai_compat.py -->
Resolve a provider by name (uses the registry and OpenAI‑compatible base URLs):

```python
from enzu import resolve_provider

provider = resolve_provider("openrouter")
```

Register a custom provider:

```python
from enzu import register_provider, resolve_provider

register_provider("myapi", base_url="https://api.example.com/v1", supports_responses=True)
provider = resolve_provider("myapi")
```

Provider env vars used by `resolve_provider()`:
- `OPENROUTER_API_KEY`, `OPENROUTER_REFERER`, `OPENROUTER_APP_NAME`
- `OPENAI_API_KEY`, `OPENAI_ORG`, `OPENAI_PROJECT`
- `{PROVIDER}_API_KEY` for other names (e.g., `MISTRAL_API_KEY`)

## Engine and RLMEngine (direct use)

<!-- Source: enzu/engine.py, enzu/rlm/engine.py -->
Use these when you already have a validated `TaskSpec` and a provider instance.

```python
from enzu import Engine, TaskSpec, Budget, SuccessCriteria

spec = TaskSpec(
    task_id="t1",
    input_text="Say hello.",
    model="openrouter/auto",
    budget=Budget(max_tokens=64),
    success_criteria=SuccessCriteria(min_word_count=1),
)

engine = Engine()
report = engine.run(spec, provider)
```

RLM (requires `data`):

```python
from enzu import RLMEngine

rlm = RLMEngine()
report = rlm.run(spec, provider, data=context_text)
```

## Schemas and contract

<!-- Source: enzu/schema.py -->
Use JSON schemas to validate payloads:

```python
from enzu import schema_bundle

bundle = schema_bundle()
```

CLI prints the same bundle:

```
enzu --print-schema
```

## RLM tools (for sandbox use)

<!-- Source: enzu/tools/*.py -->
These helpers are designed for the RLM sandbox:

- `enzu.tools.exa`: web search (`exa_search`, `exa_news`, `exa_papers`, `exa_similar`). Requires `EXA_API_KEY`.
- `enzu.tools.research`: high‑level research that auto‑accumulates into the context store.
- `enzu.tools.context`: context store (`ctx_add`, `ctx_get`, `ctx_stats`, `ctx_sources`, `ctx_save`, `ctx_load`).
- `enzu.tools.filesystem`: filesystem helpers for automode (`build_fs_helpers`, `FS_TOOL_GUIDANCE`).

## Telemetry

<!-- Source: enzu/telemetry.py -->
Logfire tracing is enabled when Logfire is installed. Env controls:
- `ENZU_LOGFIRE` (enable/disable)
- `ENZU_LOGFIRE_CONSOLE` (console output)
- `ENZU_LOGFIRE_STREAM` (token stream logs)
- `ENZU_TELEMETRY_STDERR` (fallback stderr logging)
- `ENZU_LOGFIRE_INSTRUMENT_OPENAI` (OpenAI instrumentation)

## Module map

<!-- Source: enzu/* -->
- `enzu/__init__.py`: public exports (run, Session, Engine, models, schema helpers).
- `enzu/api.py`: `run()` entry point, mode resolution, provider resolution, `generate()` (deprecated).
- `enzu/cli.py`: CLI entry point, guided mode, schema printing, mode handling.
- `enzu/contract.py`: task defaulting and `task_spec_from_payload()`.
- `enzu/engine.py`: chat engine (verification, budgets, progress events).
- `enzu/models.py`: Pydantic models for task input, budgets, reports, verification.
- `enzu/schema.py`: JSON schema generation and bundle.
- `enzu/session.py`: Session persistence and budget caps.
- `enzu/bench.py`: JSONL benchmark runner used by `enzu-bench`.
- `enzu/retry.py`: retry wrapper for provider calls.
- `enzu/telemetry.py`: Logfire instrumentation and logging helpers.
- `enzu/providers/base.py`: BaseProvider interface.
- `enzu/providers/openai_compat.py`: OpenAI‑compatible provider implementation.
- `enzu/providers/registry.py`: provider registry + custom registration.
- `enzu/rlm/engine.py`: RLM engine and guardrails.
- `enzu/rlm/__init__.py`: RLM exports.
- `enzu/repl/sandbox.py`: RLM Python sandbox implementation.
- `enzu/repl/safe.py`: safe helper utilities for sandbox code.
- `enzu/tools/context.py`: context store for RLM research accumulation.
- `enzu/tools/exa.py`: Exa search client and helpers.
- `enzu/tools/research.py`: research helper (auto-accumulates context).
- `enzu/tools/filesystem.py`: filesystem helpers for automode.
- `enzu/tools/__init__.py`: tool exports.
