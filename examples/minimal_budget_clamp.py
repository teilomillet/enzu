#!/usr/bin/env python3
"""Ultra-short: Budget clamps output tokens (10 lines)."""
from enzu import Enzu

client = Enzu()

# Budget enforces hard limit on output tokens
report = client.run(
    "Write a 500-word essay on AI.",  # Asks for long output
    tokens=50,  # But budget clamps to 50 tokens max
    return_report=True,
)

print(f"Output tokens: {report.budget_usage.output_tokens}")
print(f"Clamped at: 50 tokens (budget enforced)")
print(f"Outcome: {report.outcome.value}")
