"""
Token terminology definitions for enzu.

This module centralizes token-related terminology to ensure consistency
across the codebase. Import these docstrings where needed.

## Key Distinction

There are TWO different concepts that sound similar:

1. **Budget.max_tokens** (cumulative output tokens)
   - Total output tokens allowed across ALL API calls in a task/session
   - Primary billing metric - this is what users want to control
   - Enforced by BudgetController

2. **TaskSpec.max_output_tokens** (per-call limit)
   - Maximum output tokens for a SINGLE API call
   - Sent directly to the LLM provider in the request
   - Controls response length for individual completions

## Examples

```python
# Budget: allow up to 5000 output tokens total across all calls
budget = Budget(max_tokens=5000)

# TaskSpec: each API call can generate up to 1000 tokens
task = TaskSpec(..., budget=budget, max_output_tokens=1000)

# If the task makes 5 calls each generating 1000 tokens = 5000 total
# The budget is exhausted after the 5th call
```

## Other Token Limits

- **Budget.max_total_tokens**: Cumulative input + output tokens (advanced)
- **BudgetController.max_input_tokens**: Cumulative input tokens (advanced)
"""

BUDGET_MAX_TOKENS_DOC = """
Maximum cumulative OUTPUT tokens across all API calls.

This is the primary budget limit since output tokens are the most expensive
and the main billing metric. When this limit is reached, no more calls are allowed.

Note: This is different from TaskSpec.max_output_tokens, which is the per-call
limit sent to the API.
"""

BUDGET_MAX_TOTAL_TOKENS_DOC = """
Maximum cumulative TOTAL tokens (input + output) across all API calls.

Advanced option for users who need to control total context window usage.
Most users should use max_tokens (output only) instead.
"""

TASKSPEC_MAX_OUTPUT_TOKENS_DOC = """
Maximum output tokens for a SINGLE API call.

This value is sent directly to the LLM provider in the API request.
It controls the response length for individual completions.

Note: This is different from Budget.max_tokens, which is the cumulative
output token limit across all calls.
"""

SESSION_MAX_TOKENS_DOC = """
Maximum cumulative OUTPUT tokens across all run() calls in this session.

When this limit is reached, SessionBudgetExceeded is raised.
Call raise_token_cap() to increase the limit and continue.
"""

SESSION_TOTAL_TOKENS_DOC = """
Cumulative OUTPUT tokens consumed across all run() calls.

This tracks the primary billing metric (output tokens), not total tokens.
"""
