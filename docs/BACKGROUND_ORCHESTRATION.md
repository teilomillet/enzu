# enzu background orchestration

This guide describes how to use enzu as a background agent orchestration layer.
It focuses on stable integration points in the codebase.

## File-based recipes

If you prefer file-first flows (no server), see:
- `docs/FILE_BASED_CHATBOT.md`
- `docs/FILE_BASED_RESEARCHER.md`

## Pattern 1: CLI worker (process-per-task)

<!-- Source: enzu/cli.py -->
The CLI is a stable JSON worker. Each invocation runs one task and exits.
This is the simplest background pattern because it isolates runs per process.

Example worker loop (queue → CLI → JSON report):

```python
import json
import subprocess

def run_job(payload: dict) -> dict:
    proc = subprocess.run(
        ["enzu"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    return json.loads(proc.stdout)

payload = {
    "mode": "chat",
    "provider": "openrouter",
    "task": {
        "task_id": "job-1",
        "input_text": "Summarize this text.",
        "model": "openrouter/auto",
    },
}
report = run_job(payload)
```

Operational notes:
- Use `enzu --print-schema` at startup to validate payloads. (`docs/schema/bundle.json` matches it.)
- Use `--progress` to stream progress events to stderr for logs/metrics.
- Exit code is `0` on success, `1` on error.

## Pattern 2: In-process worker (Python API)

<!-- Source: enzu/api.py -->
Use `run()` for in-process orchestration when you already host a Python worker.

```python
from enzu import run

report = run(
    "Summarize this text.",
    provider="openrouter",
    model="openrouter/auto",
    return_report=True,
)
```

Mode selection (Python):
- `mode="auto"` (default) chooses chat or RLM.
- Auto switches to RLM when any of: `data`, `cost`, `seconds`, `goal`, or prompt+data > ~256k chars.
- Force a mode with `mode="chat"` or `mode="rlm"` when determinism is required.

## Pattern 3: Session-backed agent (stateful)

<!-- Source: enzu/session.py -->
Use `Session` when you need a stateful, multi-step agent that accumulates context.
Session prepends history to `data` on every call.

```python
from enzu import Session

session = Session(
    model="openrouter/auto",
    provider="openrouter",
    max_cost_usd=5.00,
    max_tokens=20000,
)

session.run("Find the bug.", data=logs, cost=1.00)
session.run("Fix it.")
session.save("debug_session.json")
```

Notes:
- Session history is passed via `data`, so auto mode resolves to RLM once history exists.
- Use `raise_cost_cap()` / `raise_token_cap()` when you need to extend caps.

## Pattern 4: Multi-stage pipelines (agent orchestration)

Build pipelines by chaining explicit stages and persisting artifacts between them
(JSON on disk, object storage, or a DB). Each stage is a normal enzu task with
clear budgets and success criteria, which makes retries and fallbacks easy.

Sketch:

```python
from enzu import run

def stage(name: str, task: str, data: str) -> str:
    return run(
        task,
        data=data,
        model="openrouter/auto",
        tokens=600,
        cost=0.10,
    )

research = stage("research", "Find key points", data=source)
draft = stage("draft", "Write a draft", data=research)
final = stage("final", "Edit and tighten", data=draft)
```

Pipeline examples will be published separately.

## Budgets and verification

<!-- Source: enzu/models.py, enzu/engine.py -->
Every task is constrained by `budget` and validated by `success_criteria`.
Use these to make background jobs deterministic and fail fast:

```python
from enzu import Budget, SuccessCriteria, TaskSpec, Engine

spec = TaskSpec(
    task_id="job-2",
    input_text="Extract the year.",
    model="openrouter/auto",
    budget=Budget(max_tokens=64),
    success_criteria=SuccessCriteria(required_regex=[r"\\b\\d{4}\\b"]),
)
```

## Observability

<!-- Source: enzu/telemetry.py -->
Telemetry is optional but useful in background workers:
- `ENZU_LOGFIRE=1` enables Logfire spans when installed.
- `ENZU_LOGFIRE_STREAM=1` logs token stream events.
- `ENZU_TELEMETRY_STDERR=1` keeps fallback stderr logging on.

## When to choose each pattern

- **CLI worker**: easiest to scale horizontally; process isolation; language‑agnostic.
- **In‑process**: lowest latency; shared memory; easier callbacks.
- **Session**: stateful agent flows; multi‑turn context.
- **Pipeline scripts**: template for multi‑stage orchestration with artifacts.
