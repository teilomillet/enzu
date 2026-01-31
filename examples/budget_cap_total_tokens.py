"""Minimal demo: total-token budget is a hard cap (reliable).

Run:
  (have OPENROUTER_API_KEY in .env)
  python3 examples/budget_cap_total_tokens.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from enzu import Budget, Engine, OpenAICompatProvider, SuccessCriteria, TaskSpec

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise SystemExit("Set OPENROUTER_API_KEY (e.g. in .env)")

provider = OpenAICompatProvider(
    name="openrouter",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

# Intentionally huge input to exceed a tiny total-token budget.
BIG = "A" * 200_000

task = TaskSpec(
    task_id="budget-total-demo",
    input_text=f"Summarize this in one sentence:\n\n{BIG}",
    model="z-ai/glm-4.7",
    budget=Budget(max_total_tokens=100),
    success_criteria=SuccessCriteria(min_word_count=1),
    max_output_tokens=200,
)

report = Engine().run(task, provider)
print("success:", report.success)
print("limits_exceeded:", report.budget_usage.limits_exceeded)
