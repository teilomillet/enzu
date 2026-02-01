#!/usr/bin/env python3
"""
hello_budget.py - Your first budgeted LLM run

This is the minimal example: set a budget, run a task, handle the limit.
"""

from enzu import Run

# Create a run with hard budget constraints
run = Run(
    budget_tokens=1000,       # Max 1000 tokens (input + output)
    budget_time=10.0,         # Max 10 seconds
    on_limit="partial",       # Return what we have if limit hit
)

# Execute a simple task
result = run.execute(
    prompt="Explain quantum computing in simple terms.",
    model="gpt-4o-mini",      # Or any OpenAI-compatible model
)

# Check what happened
print(f"Status: {result.status}")           # "complete" or "partial" or "failed"
print(f"Tokens used: {result.tokens_used}") # Actual consumption
print(f"Time taken: {result.time_seconds:.2f}s")
print(f"\nOutput:\n{result.output}")

# The budget contract in action:
# - If we finished under budget: status = "complete"
# - If we hit token limit: status = "partial", output = best effort
# - If we hit time limit: status = "partial", output = what we got

# This is the core idea: budgets are contracts, not hopes.
# Decide what happens at the limit BEFORE you ship.
