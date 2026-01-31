# Case Study: Report Service

## Before enzu

We needed to generate structured reports from document corpora. Initial attempts with raw LLM calls faced:

- **Runaway costs**: Recursive analysis calls accumulated unpredictably, sometimes exceeding $5 for a single report
- **Timeout failures**: Long documents caused 10+ minute runs with no visibility into progress
- **No accountability**: When reports failed, we couldn't trace what went wrong or how much was spent

## With enzu

We wrapped the same logic in enzu with explicit budgets:

```python
report = client.run(task, data=corpus, tokens=1200, cost=0.50, return_report=True)
```

Results:
- **Capped spend at $0.50** per report (previously unbounded)
- **Stopped runaway loops** with 120s time limit
- **Processed 4 documents in 11 seconds** with full accounting

## What it produced

- Structured report with 4 citations and 3 cross-document connections
- `trace.json` with exact token counts (762 output, 4320 total)
- Typed outcome (`success`) enabling automated retry/escalation logic

## Links

- Demo: [examples/report_service/](../examples/report_service/)
- Architecture: [docs/REPORT_SERVICE.md](REPORT_SERVICE.md)
