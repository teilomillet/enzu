#!/usr/bin/env python3
"""
Code Reviewer: Get quick feedback on code snippets.

Simple use case: you wrote some code and want a second opinion.
Paste it, get actionable feedback on bugs, style, and improvements.

Run:
    export OPENAI_API_KEY=sk-...
    python examples/usecases/code_reviewer.py
"""

from enzu import Enzu

client = Enzu()

print("=" * 60)
print("CODE REVIEWER")
print("=" * 60)

# Example: Review a function
code = """
def get_user(user_id):
    users = db.query("SELECT * FROM users WHERE id = " + user_id)
    if len(users) > 0:
        return users[0]
    else:
        return None
"""

print(f"\n📝 Code to review:\n{code}")
print("-" * 60)
print("\n🔍 Review:\n")

result = client.run(
    f"""Review this Python function. List:
1. Any bugs or security issues
2. Style improvements
3. A corrected version

```python
{code}
```""",
    tokens=400,
    return_report=True,
)

output = getattr(result, "answer", None) or getattr(result, "output_text", None)
print(output)

print("\n" + "-" * 60)
print(f"   Tokens: {result.budget_usage.total_tokens}")
if result.budget_usage.cost_usd:
    print(f"   Cost: ${result.budget_usage.cost_usd:.4f}")
print("=" * 60)
