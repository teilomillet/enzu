# Production Cost Multipliers

> Extracted from real production discussions. These are the hidden costs that blow up LLM bills.

## The Core Formula

```
Total Cost ≈ Fixed Overhead + (Per-Turn Variable) × Multipliers
```

Where "tokens per request" is just one variable—**multipliers** are what explode the bill.

---

## The 10 High-Signal Multipliers

### 1. Tool Fanout
Every tool call is another LLM invocation. Structure tools to minimize total calls per task.

**Policy:** Set `max_tool_calls` per run. Degrade to single-tool fallback at limit.

### 2. Retries / Rate Limits (429s)
Timeouts and rate limits cause re-runs. Each retry multiplies base cost.

**Policy:** Exponential backoff + circuit breaker. After N retries, return partial or fail fast.

### 3. Context Growth (P95)
Long conversations accumulate context. P95 requests can cost 10× median.

**Policy:** Hard `max_context_tokens`. Summarize or truncate at threshold.

### 4. Safety/Guardrail Passes
Content filters, PII detection, output validation—each is an extra call or embedding lookup.

**Policy:** Run guardrails *once* at intake, not per-turn. Cache decisions where safe.

### 5. UUID Bloat
Serializing data structures with UUIDs into prompts burns tokens on noise.

**Policy:** Map UUIDs → short unique integers in-memory before prompt injection.

### 6. Single Monolithic History
Keeping everything in one chat history means paying for irrelevant context.

**Policy:** Use sub-agents with isolated, disposable histories. Discard after task completion.

### 7. System Prompt Churn
Variable system prompt parts at the start break prefix caching.

**Policy:** Structure system prompts with static parts first, variable parts at end.

### 8. Burst Traffic
Spikes hit rate limits → retries → compounding costs.

**Policy:** Queue + smooth traffic. Accept latency over cost explosion.

### 9. Long-Tail Pathological Queries
A handful of 10× queries dominate the bill.

**Policy:** Hard `max_tokens_per_request`. Log P90/P95 separately. Alert on outliers.

### 10. Fanout × Retries × Context (Compound)
These three multiply together, not add. 2× fanout + 2× retries + 2× context = 8× cost, not 6×.

**Policy:** Budget as hard constraint. Degrade *before* compounding kicks in.

---

## The 4 Metrics to Log in Pilot

Before prod, instrument these. They'll predict 80% of your bill:

| Metric | What to Track |
|--------|---------------|
| `tokens/run` | Input + output, P50 and P95 |
| `tool_calls/run` | Count + fanout depth |
| `retry_rate` | Timeouts + 429s per run |
| `context_length` | Per-turn growth, P50/P95 |

---

## Enzu's Built-in Controls

```python
from enzu import Run

run = Run(
    budget_tokens=50_000,      # Hard token cap
    budget_time=30.0,          # Hard time cap (seconds)
    on_limit="partial",        # Return what you have at limit
    max_tool_calls=5,          # Tool fanout cap
    max_context_tokens=8192,   # Context growth cap
)
```

See [BUDGETS_AS_PHYSICS.md](./BUDGETS_AS_PHYSICS.md) for the full mental model.
