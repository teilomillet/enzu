# Basics

Minimal examples to get started with enzu in under 5 minutes.

## Examples

### Python Quickstart
[`python_quickstart.py`](python_quickstart.py) - Your first enzu call in 2 lines

```python
from enzu import ask
print(ask("What is 2+2?"))
```

### Budget Guardrails
[`python_budget_guardrails.py`](python_budget_guardrails.py) - Set spending limits

```python
from enzu import run, Limits
result = run("Write a poem", limits=Limits(tokens=100, seconds=30, cost_usd=0.01))
```

### HTTP API
[`http_quickstart.sh`](http_quickstart.sh) - Call enzu via REST API

```bash
curl -X POST http://localhost:8080/v1/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Summarize this text"}'
```

### CLI Usage
[`enzu.md`](enzu.md) - Command-line interface documentation

## What's Next?

Once you've run these examples, move to [`../concepts/`](../concepts/) to learn about budget enforcement and typed outcomes.
