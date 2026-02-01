# Budget Contracts

> Decide what happens at the limit *before* you ship, not from the invoice.

## The Problem

Most teams model LLM costs as "tokens × price" and hope for the best. When limits hit in production, behavior is undefined—timeouts, silent failures, infinite loops, or runaway bills.

**Budget contracts** make the limit behavior explicit and testable.

---

## The Contract

A budget contract answers one question:

> **When we hit [token / time / cost] limit, what happens?**

There are exactly four valid answers:

| Mode | Behavior | When to Use |
|------|----------|-------------|
| `partial` | Return best-effort result | Summaries, drafts, exploratory tasks |
| `fallback` | Switch to cheaper/faster model | Latency-sensitive with quality floor |
| `fail` | Hard error, no output | Critical tasks where partial = wrong |
| `degrade` | Reduce capabilities (fewer tools, shorter context) | Keep working, accept lower quality |

---

## Defining Contracts in Enzu

```python
from enzu import Run

# Contract: summarization task, partial is acceptable
summary_run = Run(
    budget_tokens=10_000,
    on_limit="partial",
    partial_message="[Summary truncated due to length]"
)

# Contract: customer-facing, must complete or fail cleanly
support_run = Run(
    budget_tokens=50_000,
    budget_time=30.0,
    on_limit="fail",
    fail_message="Unable to process request. Please try again."
)

# Contract: research task, degrade gracefully
research_run = Run(
    budget_tokens=100_000,
    on_limit="degrade",
    degrade_steps=[
        {"at": 0.7, "action": "disable_web_search"},
        {"at": 0.9, "action": "shorten_context"},
    ]
)
```

---

## Contract Design Checklist

Before deploying any agent:

- [ ] What's the token budget? (hard number, not "about")
- [ ] What's the time budget? (user patience ≠ infinite)
- [ ] What happens at 70% budget? 90%? 100%?
- [ ] Is partial output acceptable? To whom?
- [ ] What's the fallback? Is it tested?
- [ ] Who gets alerted when contracts trigger?

---

## Testing Contracts

Contracts aren't real until they're tested:

```python
from enzu.testing import stress_budget

# Simulate hitting limits
results = stress_budget(
    run=my_run,
    scenarios=[
        {"tokens": 0.5, "expect": "normal"},
        {"tokens": 0.9, "expect": "degraded"},
        {"tokens": 1.1, "expect": "partial"},
    ]
)
```

See [STRESS_TESTING.md](./STRESS_TESTING.md) for full testing patterns.

---

## Anti-Patterns

❌ **"We'll handle it later"** — You won't. The invoice will.

❌ **Silent truncation** — User gets garbage, doesn't know why.

❌ **Infinite retry loops** — Multiplies cost, never recovers.

❌ **No fallback model** — All eggs, one basket.

❌ **Budget = target** — Budgets are *contracts*, not aspirations.

---

## The Contract Principle

> If you can't write down what happens at the limit, you don't have a budget—you have a hope.

Enzu enforces contracts by default. No undefined behavior at limits.
