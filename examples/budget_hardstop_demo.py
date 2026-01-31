#!/usr/bin/env python3
"""
Killer demo: Budget cap stops work deterministically.

This example shows enzu's core promise: when you set a budget,
it's a hard limit that actually stops execution—not a suggestion.

Run:
    export OPENAI_API_KEY=sk-...
    uv run examples/budget_hardstop_demo.py
"""

from enzu import Enzu
from enzu.budget import BudgetExceeded

client = Enzu()

print("=" * 60)
print("DEMO: Budget hard-stop in action")
print("=" * 60)

long_document = (
    """
Climate change represents one of the most significant challenges facing humanity.
Rising global temperatures are causing widespread environmental disruption,
including melting ice caps, rising sea levels, and more frequent extreme weather.
Scientists have documented unprecedented rates of species extinction,
coral reef bleaching, and ecosystem collapse across the globe.
Mitigation strategies include transitioning to renewable energy sources,
improving energy efficiency, carbon capture technologies, and reforestation.
Adaptation measures focus on building resilient infrastructure,
developing drought-resistant crops, and improving early warning systems.
International cooperation through agreements like the Paris Accord
aims to limit warming to 1.5 degrees Celsius above pre-industrial levels.
"""
    * 10
)

print(f"\nDocument length: ~{len(long_document.split())} words")
print("Budget: 50 output tokens (intentionally small)")
print("-" * 60)

try:
    result = client.run(
        "Write a comprehensive 500-word analysis of this document.",
        data=long_document,
        tokens=50,  # Hard cap: 50 output tokens max
    )
    print(f"\nResult (truncated by budget): {result[:200]}...")
    print("\n✓ Output was clamped to token budget.")
except BudgetExceeded as e:
    print("\n✓ Budget enforced! Execution stopped.")
    print(f"  Limit type: {e.limit_type}")
    print(f"  Limit value: {e.limit_value}")
    print(f"  Current value: {e.current_value}")

print("\n" + "=" * 60)
print("This is enzu's guarantee: budgets are physics, not policy.")
print("=" * 60)
