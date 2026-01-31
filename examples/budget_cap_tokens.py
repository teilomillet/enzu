"""Minimal demo: token budget is a hard cap.

Run:
  export OPENAI_API_KEY=...
  python examples/budget_cap_tokens.py
"""

from enzu import Enzu

client = Enzu()

report = client.run(
    "Write a 200-word story about a lighthouse.",
    tokens=20,  # absurdly small on purpose
    return_report=True,
)

print("success:", report.success)
print("limits_exceeded:", report.budget_usage.limits_exceeded)
print("output:", (report.output_text or "").strip())
