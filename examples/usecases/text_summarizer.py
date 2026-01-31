#!/usr/bin/env python3
"""
Text Summarizer: Condense articles or documents into key points.

Simple use case: you have an article or document but only need
the key points. Get a summary with controlled length.

Run:
    export OPENAI_API_KEY=sk-...
    python examples/usecases/text_summarizer.py
"""

from enzu import Enzu

client = Enzu()

print("=" * 60)
print("TEXT SUMMARIZER")
print("=" * 60)

# Example article
article = """
Remote work has fundamentally changed how companies operate. A recent study 
of 500 companies found that 73% now offer some form of remote work, up from 
just 30% in 2019. The shift has had mixed effects on productivity.

Companies report that individual contributor work has improved, with 
developers and writers producing more output when working from home. However, 
collaborative work has suffered. Brainstorming sessions and spontaneous 
conversations are harder to replicate virtually.

The biggest challenge is onboarding. New employees take 50% longer to become 
productive when starting remotely. They miss the informal learning that 
happens by sitting near experienced colleagues.

Companies are experimenting with hybrid models: 2-3 days in office for 
collaboration, remaining days at home for focused work. Early data suggests 
this balances productivity with team cohesion.

The long-term effects on company culture remain unclear. Some predict a 
return to offices as economic conditions tighten, while others believe 
remote work is here to stay.
"""

print(f"\n📄 Article ({len(article.split())} words)")
print("-" * 60)
print("\n📌 Summary:\n")

result = client.run(
    f"""Summarize this article in 3 bullet points. Be concise.

{article}""",
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
