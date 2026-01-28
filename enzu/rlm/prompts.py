"""
RLM system prompt constants.

Extracted from engine.py for modularity. These are injected into
the system prompt based on context size and available tools.
"""
from __future__ import annotations


from enzu.contract import DEFAULT_MIN_WORD_COUNT
from enzu.models import SuccessCriteria


SYSTEM_PROMPT_GUARDRAILS = """
## Code-First Patterns

The prompt lives in `context`. Write code to manipulate it:

1. Probe: print(context[0]) for the query; print(context[1][:500]) for data if present
2. Filter in code: relevant = [x for x in chunks if 'keyword' in x]
3. Delegate to sub-LLMs: use llm_query or llm_batch (see below)
4. Aggregate in code: counts = {k: v for k, v in results}

## Sub-LLM Calls: llm_query vs llm_batch

**llm_query(prompt)** - Sequential single call
- Use for: one-off queries, dependent operations
- Example: `result = llm_query("Summarize this text")`

**llm_batch(prompts)** - Parallel batch calls (MUCH FASTER)
- Use for: independent queries that can run simultaneously
- Returns list in same order as inputs
- Example:
  ```python
  prompts = [f"Classify: {chunk}" for chunk in filtered_chunks]
  results = llm_batch(prompts)  # All run in parallel
  ```

CRITICAL: When you have multiple independent sub-queries, ALWAYS use llm_batch instead of loops with llm_query. This reduces latency by N× where N is the number of queries.

## When to Use llm_query/llm_batch

USE for: semantic classification, summarization, meaning extraction
DO NOT use for: counting, filtering, sorting, formatting, string ops

## Safe Helpers (available in namespace)

- safe_get(d, key, default): Dict access that never crashes
- safe_rows(context): Extract list from any structure
- safe_sort(context, key): Sort without type errors
- budget_status(): Inspect current token budget and remaining capacity

## Rules

- ONE code block per response. Multiple blocks = only first executes.
- Filter before delegating. Don't pass full context to sub-LLMs.
- Use llm_batch for parallel queries, not loops with llm_query.
"""

SEARCH_TOOLS_GUIDANCE = """
## Web Research Tools

### RECOMMENDED: High-Level Functions (use these first)

**research(topic)** - All-in-one research function
```python
# Simple usage - does search, filter, and format for you
result = research("AI agents")
print(result["stats"])  # {'total_found': 15, 'kept': 8, 'context_length': 12000}

# The context is ready for llm_query
summary = llm_query(f"Write a newsletter about:\\n{result['context']}")
FINAL(summary)
```

**research() options:**
```python
result = research(
    "AI agents",
    num_results=10,        # Results per search
    min_score=0.5,         # Quality filter (0-1)
    max_characters=None,   # None = no per-result limit
    max_chars_per_source=None,  # None = no per-source truncation
    include_news=True,     # Add news articles
    include_papers=True,   # Add research papers
    days_back=7,           # Recent content only
)
```
Note: research() auto-adds results to the context store when available.
Note: Context formatting includes Published dates when present.

**explore(url)** - Find similar content from a good source
```python
result = explore("https://good-article.com/...")
summary = llm_query(f"Summarize related content:\\n{result['context']}")
```
Note: explore() auto-adds results to the context store when available.
Note: Context formatting includes Published dates when present.

**format_sources(sources)** - Create citation list
```python
refs = format_sources(result["sources"])
FINAL(summary + "\\n\\n" + refs)
```

### Low-Level Functions (for fine control)

**exa_search(query)** - Direct search
```python
results = exa_search("AI", num_results=5, category="news", days_back=7)
# Returns: [{"url", "title", "text", "score", "highlights", "published_date"}, ...]
```

**exa_news(query)** / **exa_papers(query)** - Category shortcuts
**exa_similar(url)** - Find related pages
**exa_contents(urls)** - Fetch specific URLs
Note: exa_* functions auto-add results to the context store when available.

### Categories: "news", "research paper", "github", "company", "pdf", "tweet"

## Workflow Example

```python
# Step 1: Research the topic (defaults to last 7 days)
topic = "AI agents"
result = research(topic, include_news=True)
print(f"Found {result['stats']['kept']} quality sources")

# Step 2: If not enough, search more
if result['stats']['kept'] < 3:
    more = research(topic, min_score=0.3, num_results=20)

# Step 3: Get accumulated context and synthesize
context = ctx_get(max_chars=40000, max_chars_per_source=None)
newsletter = llm_query(f\'\'\'
Write a newsletter section about {topic}.

Sources:
{context}

Include:
- Key insights and trends
- Notable quotes with attribution
- 3-5 bullet point takeaways
\'\'\')

# Step 4: Finish with citations
FINAL(newsletter + "\\n\\n" + format_sources(ctx_sources()))
```

## Context Management (for accumulating research)

**ctx_add(sources, query)** - Add sources to persistent store (deduped)
**ctx_get(max_chars=50000, max_chars_per_source=None)** - Get accumulated context for llm_query (includes Published date)
**ctx_stats()** - Check what's accumulated
**ctx_sources()** - Get list of all sources
**ctx_has_query(q)** - Check if query was already done (avoid re-searching)
**ctx_save(path)** - Save for later (actu.me persistence)
**ctx_load(path)** - Load previous research

**exa_cost()** - Check search API costs for this session

## Context Bootstrapping

- If ctx_stats()["num_sources"] == 0, run research() or exa_search() first.
- Persist context with ctx_save(path) after gathering sources.
"""

PIP_INSTALL_GUIDANCE = """
## Dynamic Package Installation

**pip_install(*packages)** - Install any pip package during execution

Examples:
```python
# Install single package
pip_install("numpy")
import numpy as np

# Install multiple packages at once
pip_install("pandas", "scipy", "matplotlib")
import pandas as pd

# Standard library always available (no install needed)
import re, math, json, datetime
```

The RLM tracks which packages are installed. You can install any package from PyPI as needed for your task.

BEST PRACTICE: Install packages before using them. Check if you need specialized libraries (numpy for math, pandas for data, beautifulsoup4 for HTML parsing, etc.) and install them first.
"""

STRATEGY_HINTS = """
## Strategy by Context Size
- Small (<10K chars): Direct analysis, few llm_query calls
- Medium (10K-100K): Chunk by structure (lines, paragraphs, ###)
- Large (>100K): Probe first, filter by keywords, batch llm_query

## Efficient Patterns
- Probe: print(context[0]) to read the query; print(context[1][:500]) to sample data if present
- Filter in code: relevant = [x for x in chunks if 'keyword' in x]
- Batch llm_query: result = llm_query(f"Process all:\\n{chunk1}\\n{chunk2}")
- Aggregate in code: counts = Counter(results)

## Anti-patterns (costly)
- llm_query per line (N calls) -> batch into ~5-10 calls max
- llm_query(context) without filtering -> slice or filter first
- llm_query for counting/sorting -> use Python
"""


def format_success_criteria(criteria: SuccessCriteria) -> str:
    """
    Render success criteria for the system prompt.

    Two modes:
    1. Goal-based: tell model the goal, trust its FINAL() judgment
    2. Mechanical: list specific checks (substrings, regex, word count)
    """
    if criteria.goal:
        return (
            f"Goal: {criteria.goal}\n\n"
            "Work toward this goal. Call FINAL(answer) when you believe the goal is achieved.\n"
            "You are the judge of success. Continue until you are confident or budget runs out.\n\n"
        )

    lines: list[str] = []
    if criteria.required_substrings:
        lines.append("Required substrings: " + ", ".join(criteria.required_substrings))
    if criteria.required_regex:
        lines.append("Required regex: " + ", ".join(criteria.required_regex))
    if criteria.min_word_count:
        lines.append(f"Minimum word count: {criteria.min_word_count}")
    if criteria.case_insensitive:
        lines.append("Case-insensitive checks: true")
    if not lines:
        return ""
    bullets = "\n- ".join(lines)
    return (
        "Success criteria (stop only when met):\n"
        "- " + bullets + "\n"
        "If criteria are not met, continue and try again.\n\n"
    )


def has_strong_success_criteria(criteria: SuccessCriteria) -> bool:
    """
    Return True when criteria set a concrete stop condition.

    Strong criteria:
    - Mechanical: required_substrings, required_regex, or min_word_count > default
    - Goal-based: goal field is set
    """
    if criteria.required_substrings or criteria.required_regex:
        return True
    if criteria.min_word_count and criteria.min_word_count > DEFAULT_MIN_WORD_COUNT:
        return True
    if criteria.goal:
        return True
    return False
