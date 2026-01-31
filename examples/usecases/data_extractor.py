#!/usr/bin/env python3
"""
Data Extractor: Pull structured data from unstructured text.

Simple use case: you have messy text (emails, forms, notes) and need
to extract specific fields. Get clean, structured output.

Run:
    export OPENAI_API_KEY=sk-...
    python examples/usecases/data_extractor.py
"""

from enzu import Enzu

client = Enzu()

print("=" * 60)
print("DATA EXTRACTOR")
print("=" * 60)

# Example: Extract from an email
email = """
Hi there,

Thanks for reaching out! I'm John Smith, the CTO at TechStartup Inc. 
We're definitely interested in your proposal.

You can reach me at john.smith@techstartup.io or call my cell: 
(555) 123-4567. Our office is at 123 Innovation Drive, San Francisco.

Looking forward to discussing further.

Best,
John
"""

print(f"\n📧 Input email:\n{email}")
print("-" * 60)
print("\n📊 Extracted data:\n")

result = client.run(
    f"""Extract the following from this email. Format as JSON:
- name
- title
- company
- email
- phone
- address

{email}""",
    tokens=150,
    return_report=True,
)

output = getattr(result, "answer", None) or getattr(result, "output_text", None)
print(output)

print("\n" + "-" * 60)
print(f"   Tokens: {result.budget_usage.total_tokens}")
if result.budget_usage.cost_usd:
    print(f"   Cost: ${result.budget_usage.cost_usd:.4f}")
print("=" * 60)
