#!/usr/bin/env python3
"""
Email Writer: Generate professional emails with tone control.

Simple use case: you need to write an email but don't want to
spend time crafting it. Give enzu the context, get a polished email.

Run:
    export OPENAI_API_KEY=sk-...
    python examples/usecases/email_writer.py
"""

from enzu import Enzu

client = Enzu()

print("=" * 60)
print("EMAIL WRITER")
print("=" * 60)

# Example 1: Follow-up email
print("\n📧 Example 1: Project follow-up\n")

result = client.run(
    """Write a professional email:
    
    Context: I met Sarah at a conference last week. She mentioned 
    her company might need help with their Python backend.
    
    Goal: Follow up, remind her who I am, suggest a call.
    Tone: Friendly but professional. Keep it short.""",
    tokens=200,
    return_report=True,
)

output = getattr(result, "answer", None) or getattr(result, "output_text", None)
print(output)
print(f"\n   Tokens: {result.budget_usage.total_tokens}")
if result.budget_usage.cost_usd:
    print(f"   Cost: ${result.budget_usage.cost_usd:.4f}")

# Example 2: Declining a meeting
print("\n" + "-" * 60)
print("\n📧 Example 2: Declining politely\n")

result = client.run(
    """Write a short email declining a meeting invitation.
    
    Context: I'm invited to a weekly status meeting but I'm too 
    busy and it's not relevant to my work.
    
    Tone: Polite, suggest they loop me in via email instead.""",
    tokens=150,
    return_report=True,
)

output = getattr(result, "answer", None) or getattr(result, "output_text", None)
print(output)
print(f"\n   Tokens: {result.budget_usage.total_tokens}")
if result.budget_usage.cost_usd:
    print(f"   Cost: ${result.budget_usage.cost_usd:.4f}")

print("\n" + "=" * 60)
