#!/usr/bin/env python3
"""Ultra-short: p95 cost/run logging skeleton (18 lines)."""
from enzu import Enzu

client = Enzu()
runs = []

# Run multiple tasks and collect metrics
for task in ["Hello", "What is 2+2?", "Name 3 colors"]:
    report = client.run(task, tokens=100, return_report=True)
    runs.append({
        "outcome": report.outcome.value,
        "elapsed": report.budget_usage.elapsed_seconds,
        "tokens": report.budget_usage.total_tokens,
        "cost": report.budget_usage.cost_usd or 0,
    })

# Compute p50/p95 (no external deps)
costs = sorted([r["cost"] for r in runs])
p50, p95 = costs[len(costs)//2], costs[int(len(costs)*0.95)]
print(f"p50 cost: ${p50:.6f}, p95 cost: ${p95:.6f}")
print(f"Outcomes: {[r['outcome'] for r in runs]}")
