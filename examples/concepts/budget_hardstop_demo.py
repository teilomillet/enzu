#!/usr/bin/env python3
"""
Killer demo: Budget cap stops work deterministically.

This example shows enzu's core promise: when you set a budget,
it's a hard limit that actually stops execution—not a suggestion.

Run:
    export OPENAI_API_KEY=sk-...
    uv run examples/budget_hardstop_demo.py
"""

from enzu import Enzu, Outcome

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
print("Task: Write a comprehensive 500-word analysis (impossible under budget)")
print("-" * 60)

# Run with return_report=True to get full structured result
report = client.run(
    "Write a comprehensive 500-word analysis of this document.",
    data=long_document,
    tokens=50,  # Hard cap: 50 output tokens max
    return_report=True,
)

# Show structured result
print("\n📊 STRUCTURED RESULT:")
print(f"  Outcome:        {report.outcome.value}")
print(f"  Partial:        {report.partial}")
print(f"  Success:        {report.success}")

print("\n📈 ACCOUNTING:")
print(f"  Output tokens:  {report.budget_usage.output_tokens}")
print(f"  Input tokens:   {report.budget_usage.input_tokens}")
print(f"  Total tokens:   {report.budget_usage.total_tokens}")
print(f"  Elapsed time:   {report.budget_usage.elapsed_seconds:.2f}s")
if report.budget_usage.cost_usd:
    print(f"  Estimated cost: ${report.budget_usage.cost_usd:.4f}")
if report.budget_usage.limits_exceeded:
    print(f"  Limits hit:     {report.budget_usage.limits_exceeded}")

# Show outcome handling
print("\n🎯 OUTCOME HANDLING:")
if report.outcome == Outcome.SUCCESS:
    print("  ✓ Task completed successfully")
elif report.outcome == Outcome.BUDGET_EXCEEDED:
    print("  ⚠ Budget exceeded - execution stopped deterministically")
    if report.partial:
        output = getattr(report, "answer", None) or getattr(report, "output_text", None)
        if output:
            print(f"  Partial output available ({len(output)} chars)")
else:
    print(f"  ✗ Other outcome: {report.outcome.value}")

print("\n" + "=" * 60)
print("This is enzu's guarantee: budgets are physics, not policy.")
print("=" * 60)
