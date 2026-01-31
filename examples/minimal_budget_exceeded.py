#!/usr/bin/env python3
"""Ultra-short: BUDGET_EXCEEDED outcome with partial result (12 lines)."""
from enzu import Enzu, Outcome

client = Enzu()

# Intentionally tiny budget triggers BUDGET_EXCEEDED
report = client.run(
    "Write a comprehensive analysis of machine learning.",
    tokens=20,  # Too small - will hit limit
    return_report=True,
)

print(f"Outcome: {report.outcome.value}")
print(f"Is BUDGET_EXCEEDED: {report.outcome == Outcome.BUDGET_EXCEEDED}")
print(f"Partial result: {report.partial}")
print(f"Success: {report.success}")
