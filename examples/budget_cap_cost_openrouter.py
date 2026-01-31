"""Minimal demo: cost budget is a hard cap (OpenRouter only).

Run:
  export OPENROUTER_API_KEY=...
  python examples/budget_cap_cost_openrouter.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Allows running from a git checkout without installing enzu.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from enzu import Enzu  # noqa: E402

if not os.getenv("OPENROUTER_API_KEY"):
    raise SystemExit("Set OPENROUTER_API_KEY (e.g. in .env)")

client = Enzu(provider="openrouter", model="z-ai/glm-4.7")

report = client.run(
    "Write a long, multi-section report on the history of navigation.",
    mode="chat",
    cost=0.000001,  # intentionally absurd: should always exceed
    return_report=True,
)

print("success:", report.success)
print("limits_exceeded:", report.budget_usage.limits_exceeded)
print("cost_usd:", report.budget_usage.cost_usd)
