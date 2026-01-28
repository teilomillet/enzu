# enzu Deployment Quickstart

This quickstart is for deployment engineers integrating enzu into services.

## 1) Install

```
uv pip install enzu
# or: pip install enzu

# HTTP API server:
# uv pip install "enzu[server]"
```

## 2) Configure provider credentials

Set the provider env var:

- `OPENROUTER_API_KEY`
- `OPENAI_API_KEY`
- `{PROVIDER}_API_KEY` for other providers

## 3) Validate the contract

```
enzu --print-schema > /tmp/enzu-schema.json
```

Use `/tmp/enzu-schema.json` to validate payloads in your service.

## 4) Run a smoke test (chat)

```
cat <<'JSON' | enzu
{
  "mode": "chat",
  "provider": "openrouter",
  "task": {
    "task_id": "smoke-1",
    "input_text": "Say hello in one sentence.",
    "model": "openrouter/auto"
  }
}
JSON
```

Expected: a JSON report printed to stdout with `success=true`.

## 5) Run a smoke test (rlm)

```
cat <<'JSON' | enzu
{
  "mode": "rlm",
  "provider": "openrouter",
  "task": {
    "task_id": "smoke-rlm-1",
    "input_text": "Extract the year.",
    "model": "openrouter/auto"
  },
  "context": "Alice joined in 2022."
}
JSON
```

## 6) Service wrapper pattern

The CLI is a stable JSON worker. Call it from any service:

```python
import json
import subprocess

payload = {
    "mode": "chat",
    "provider": "openrouter",
    "task": {
        "task_id": "svc-1",
        "input_text": "Summarize this text.",
        "model": "openrouter/auto"
    }
}

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

## 7) Guided local run

Run with no args to enter guided mode:

```
enzu
```

Guided mode prompts for provider, mode, model, and task, then runs a single task.

## Operational notes

- Exit code `0` on success, `1` on error.
- JSON report is always printed to stdout.
- Use `--progress` to send progress events to stderr.
