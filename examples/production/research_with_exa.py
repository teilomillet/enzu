#!/usr/bin/env python3
"""
Research task with Exa search integration.

This example demonstrates external tool integration:
- Uses Exa API for live web search
- RLM mode for goal-oriented research
- Budget-controlled output

Run:
    export OPENAI_API_KEY=sk-...
    export EXA_API_KEY=...
    python examples/production/research_with_exa.py

Requires: pip install exa
"""

import os

from enzu import run

TOPIC = "efficient inference for large language models"
MODEL = os.getenv("OPENROUTER_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

if not os.getenv("EXA_API_KEY"):
    raise SystemExit("Set EXA_API_KEY to use research()")

provider = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "openai"
if provider == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
    raise SystemExit("Set OPENROUTER_API_KEY or OPENAI_API_KEY")
if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("Set OPENROUTER_API_KEY or OPENAI_API_KEY")

TASK = """
Research this topic and produce a short brief.

Use research() to gather sources, then synthesize.
"""

result = run(
    TASK,
    model=MODEL,
    provider=provider,
    data=TOPIC,
    tokens=1200,
)

print(result)
