#!/usr/bin/env python3
"""
Demo: Typed outcomes for predictable execution.

Shows how enzu returns structured outcomes (not just success/failure)
for deterministic error handling in production systems.

Run:
    export OPENAI_API_KEY=sk-...
    uv run examples/typed_outcomes_demo.py
"""

from enzu import Enzu, Outcome

client = Enzu()

print("=" * 60)
print("DEMO: Typed outcomes for predictable handling")
print("=" * 60)

# Scenario: Run with intentionally tiny budget to trigger budget_exceeded
result = client.run(
    "Write a detailed 1000-word essay on machine learning.",
    data="Machine learning is a subset of artificial intelligence...",
    tokens=30,  # Too small - will exceed budget
    return_report=True,
)

print(f"\nOutcome: {result.outcome}")
print(f"Outcome value: {result.outcome.value}")
print(f"Success: {result.success}")
print(f"Partial: {result.partial}")
print(f"Output tokens used: {result.budget_usage.output_tokens}")

# Handle outcome (Python 3.9 compatible)
if result.outcome == Outcome.SUCCESS:
    print("\n✓ Task completed successfully")
elif result.outcome == Outcome.BUDGET_EXCEEDED:
    msg = (
        "\n⚠ Budget exceeded - partial result available"
        if result.partial
        else "\n✗ Budget exceeded"
    )
    print(msg)
elif result.outcome == Outcome.TIMEOUT:
    print("\n⚠ Task timed out")
elif result.outcome == Outcome.PROVIDER_ERROR:
    print(f"\n✗ Provider error: {result.errors}")
elif result.outcome == Outcome.VERIFICATION_FAILED:
    print("\n✗ Output did not pass verification")
else:
    print(f"\n? Unknown outcome: {result.outcome}")

output = getattr(result, "answer", None) or getattr(result, "output_text", None)
if result.partial and output:
    print(f"\nPartial output ({len(output)} chars):")
    print(f"  {output[:100]}...")

print("\n" + "=" * 60)
print("Typed outcomes enable: match/case, logging, retry policies")
print("=" * 60)
