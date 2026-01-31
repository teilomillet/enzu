"""Minimal demo: time budget is a hard cap.

Tip: Use a slow model/provider, or a large prompt, to trigger the limit.

Run:
  export OPENAI_API_KEY=...
  python examples/budget_cap_seconds.py
"""

from enzu import Enzu

client = Enzu()

report = client.run(
    "Summarize this text in detail.",
    data="A" * 200_000,  # big context to slow things down
    seconds=1.5,
    return_report=True,
)

print("success:", report.success)
print("limits_exceeded:", report.budget_usage.limits_exceeded)
print("elapsed_seconds:", round(report.budget_usage.elapsed_seconds, 2))
