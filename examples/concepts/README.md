# Concepts

Single-feature demos that explain core enzu behaviors.

## The Core Promise

Enzu's key differentiator: **budgets are enforced, not suggested**. When you set a token limit, enzu guarantees you won't exceed it.

## Budget Demos

### Flagship Demo
[`budget_hardstop_demo.py`](budget_hardstop_demo.py) - The definitive budget enforcement demo

Asks for a 500-word essay with only 50 tokens of budget. Shows how enzu:
- Enforces the limit absolutely
- Returns partial results when budget exhausted
- Provides detailed accounting breakdown

### Minimal Demos
Quick proofs of budget behavior:

| Example | What It Shows |
|---------|---------------|
| [`minimal_budget_clamp.py`](minimal_budget_clamp.py) | Output gets clamped at budget limit |
| [`minimal_budget_exceeded.py`](minimal_budget_exceeded.py) | `BUDGET_EXCEEDED` outcome with partial results |
| [`minimal_p95_logging.py`](minimal_p95_logging.py) | Collect metrics across runs |

### Budget Types

Different ways to cap costs:

| Example | Budget Type | Use Case |
|---------|-------------|----------|
| [`budget_cap_total_tokens.py`](budget_cap_total_tokens.py) | Total tokens | Predictable token accounting |
| [`budget_cap_seconds.py`](budget_cap_seconds.py) | Elapsed time | Response time SLAs |
| [`budget_cap_cost_openrouter.py`](budget_cap_cost_openrouter.py) | USD cost | Direct cost control |

## Outcome Handling

### Typed Outcomes
[`typed_outcomes_demo.py`](typed_outcomes_demo.py) - All possible outcomes and how to handle them

Enzu returns deterministic outcomes:
- `SUCCESS` - Task completed within budget
- `BUDGET_EXCEEDED` - Hit budget limit, may have partial result
- `TIMEOUT` - Time limit exceeded
- `PROVIDER_ERROR` - API failure
- `VERIFICATION_FAILED` - Output didn't meet criteria

### Chat with Budget
[`chat_with_budget.py`](chat_with_budget.py) - Chat mode with budget constraints

Shows `SuccessCriteria` with `required_substrings` for validation.

## What's Next?

Ready for production patterns? See [`../production/`](../production/) for sessions, async jobs, and real-world applications.
