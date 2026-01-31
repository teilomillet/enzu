# Report Service Architecture

## Overview

This example demonstrates bounded document analysis: given a corpus of documents and a budget, produce a structured report without exceeding resource limits.

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Report Service                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   Corpus     │───▶│    Enzu      │───▶│    Output        │  │
│  │   Loader     │    │   Engine     │    │   Writer         │  │
│  │              │    │              │    │                  │  │
│  │ docs/*.txt   │    │ Budget:      │    │ report.md        │  │
│  │              │    │ - tokens     │    │ trace.json       │  │
│  └──────────────┘    │ - seconds    │    └──────────────────┘  │
│                      │ - cost_usd   │                          │
│                      └──────┬───────┘                          │
│                             │                                  │
│                      ┌──────▼───────┐                          │
│                      │   Provider   │                          │
│                      │  (OpenAI/    │                          │
│                      │  OpenRouter) │                          │
│                      └──────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
1. Load Corpus
   docs/*.txt ──▶ load_corpus() ──▶ concatenated string

2. Execute Task
   corpus + task ──▶ Enzu.run() ──▶ ExecutionReport
                      │
                      ├─ Budget enforced at every LLM call
                      ├─ Stops if tokens/seconds/cost exceeded
                      └─ Returns partial result if budget hit

3. Write Output
   report.answer ──▶ report.md
   report.usage  ──▶ trace.json (accounting)
```

## Budget Enforcement Points

| Checkpoint | What's Checked | Action on Exceed |
|------------|---------------|------------------|
| Pre-call | Remaining budget vs estimated need | Skip call, return partial |
| Mid-stream | Tokens generated so far | Truncate output |
| Post-call | Actual usage | Update accounting, check limits |

### Token Budget (`BUDGET_TOKENS`)
- Controls maximum output tokens
- Directly limits response length
- Most predictable budget type

### Time Budget (`BUDGET_SECONDS`)
- Wall-clock limit for entire operation
- Includes network latency
- Good for SLAs

### Cost Budget (`BUDGET_COST`)
- USD limit (OpenRouter only)
- Real-time cost tracking
- Direct spend control

## Key Design Decisions

### Why concatenate all documents?
**Trade-off**: Single-shot vs multi-step

- **Single-shot** (chosen): Send all docs in one call
  - Pros: Simpler, model sees everything at once
  - Cons: Large context, higher cost per call

- **Multi-step alternative**: Process docs iteratively
  - Pros: Smaller context per call
  - Cons: More complex, harder to cross-reference

For this demo, single-shot shows budget enforcement clearly.

### Why structured task prompt?
The prompt uses markdown headers (`## EXECUTIVE SUMMARY`, etc.) to:
1. Guide model output structure
2. Make validation easier
3. Ensure consistent report format

### Why write trace.json?
Accounting transparency:
- Audit trail for cost/token usage
- Debug information for failures
- Input for monitoring systems

## Failure Modes

| Scenario | Outcome | Partial Result? |
|----------|---------|-----------------|
| Normal completion | `SUCCESS` | No |
| Token budget hit | `BUDGET_EXCEEDED` | Yes, truncated |
| Time budget hit | `TIMEOUT` | Yes, if any output |
| Provider error | `PROVIDER_ERROR` | No |

## When to Use This Pattern

**Good fit:**
- Document summarization with cost control
- Batch report generation
- Any "analyze these docs" task

**Not ideal for:**
- Real-time streaming requirements
- Interactive Q&A
- Tasks requiring tool use

## Extending This Example

1. **Add document chunking**: Process large corpora in batches
2. **Add caching**: Skip unchanged documents
3. **Add validation**: Check report structure before writing
4. **Add retry logic**: Handle transient provider errors
