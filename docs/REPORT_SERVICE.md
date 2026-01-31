# Report Service Pattern

## What it does

The report service pattern uses enzu to transform a corpus of documents into a structured report with citations and cross-document connections, all within strict budget constraints. It demonstrates enzu's core value proposition: **goal + corpus + hard budgets = bounded, auditable output**.

## Inputs / Outputs

**Inputs:**
- Task: Natural language description of desired report structure
- Corpus: Collection of documents (text files, research papers, logs)
- Budgets: Token limit, time limit, cost limit

**Outputs:**
- `report.md`: Structured report with citations to source documents
- `trace.json`: Execution trace with accounting (tokens used, time elapsed, cost)

## Budget Example

```python
from enzu import Enzu

client = Enzu()
report = client.run(
    "Analyze these documents and produce a report with citations",
    data=corpus,
    tokens=1200,    # Max 1200 output tokens
    seconds=120,    # Max 2 minutes
    cost=0.50,      # Max $0.50
    return_report=True,
)

# Check what happened
print(f"Outcome: {report.outcome}")      # success / budget_exceeded
print(f"Tokens: {report.budget_usage.output_tokens}")
print(f"Cost: ${report.budget_usage.cost_usd:.4f}")
```

## Failure Modes Prevented

| Failure Mode | Without enzu | With enzu |
|--------------|--------------|-----------|
| Runaway token consumption | Model produces unbounded output | Hard stop at token limit |
| Unbounded API costs | Recursive calls accumulate cost | Cost cap enforced |
| Infinite loops | Agent loops indefinitely | Time limit terminates |
| Silent overruns | No visibility into actual usage | Full accounting in trace |

## Why enzu matters

Hand-rolled wrappers typically implement "best-effort" limits: they check after each call whether limits are exceeded. This means:
- The final call can still overrun
- Recursive/multi-step execution accumulates unpredictably
- No structured outcome reporting

enzu enforces limits **during** execution:
- Token limits are passed to the API and enforced by the provider
- Time limits terminate execution mid-stream
- Every run returns a typed `Outcome` and `BudgetUsage`

## Run the Demo

```bash
export OPENAI_API_KEY=sk-...
uv run examples/report_service/demo.py
```

Output:
```
Corpus: 4 docs, 6325 chars
Budget: 1200 tokens, 120s, $0.5

Outcome: success, Partial: False
Elapsed: 11.2s, Tokens: 762/4320
Cost: $0.0163

Wrote: report.md
Wrote: trace.json
```

See [examples/report_service/](../examples/report_service/) for the full implementation.
