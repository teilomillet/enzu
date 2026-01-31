"""Minimal demo: cost budget is a hard cap (OpenRouter only).

Run:
  export OPENROUTER_API_KEY=...
  python examples/budget_cap_cost_openrouter.py
"""

import os

from enzu import Enzu

if not os.getenv("OPENROUTER_API_KEY"):
    raise SystemExit("Set OPENROUTER_API_KEY")

client = Enzu(provider="openrouter", model=os.getenv("OPENROUTER_MODEL") or "openai/gpt-4o")

report = client.run(
    "Write a long, multi-section report on the history of navigation.",
    cost=0.001,  # intentionally tiny
    return_report=True,
)

print("success:", report.success)
print("limits_exceeded:", report.budget_usage.limits_exceeded)
print("cost_usd:", report.budget_usage.cost_usd)
