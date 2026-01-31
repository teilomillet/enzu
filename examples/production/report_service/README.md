# Report Service Demo

Demonstrates the core enzu promise: **goal + corpus + hard budgets = bounded output**.

## Run

```bash
export OPENAI_API_KEY=sk-...
uv run examples/report_service/demo.py
```

## What it does

1. Loads 4 sample documents about AI safety
2. Submits a report generation task with budget constraints
3. Produces `report.md` with citations and cross-doc connections
4. Writes `trace.json` with accounting (tokens, time, cost)

## Budget constraints

- **Tokens**: 1200 output tokens max
- **Time**: 120 seconds max
- **Cost**: $0.50 max

## Output

- `report.md`: Generated report with:
  - Executive summary
  - Key findings with document citations
  - Connections map linking documents
  - Recommendations

- `trace.json`: Execution trace with:
  - Outcome (success/budget_exceeded/etc.)
  - Actual token usage
  - Elapsed time
  - Cost

## Why this matters

Without budget enforcement, LLM tasks can:
- Run indefinitely on complex inputs
- Consume unbounded tokens
- Accumulate unpredictable costs

enzu makes these runs **predictable and auditable**.
