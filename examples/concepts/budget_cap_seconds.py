"""Minimal demo: time budget is a hard cap.

Tip: Use a slow model/provider, or a large prompt, to trigger the limit.

Run:
  export OPENAI_API_KEY=...
  python examples/budget_cap_seconds.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from enzu import Enzu  # noqa: E402

if not os.getenv("OPENROUTER_API_KEY"):
    raise SystemExit("Set OPENROUTER_API_KEY (e.g. in .env)")

# Force OpenRouter + a consistent test model.
client = Enzu(provider="openrouter", model="z-ai/glm-4.7")

report = client.run(
    "Say 'ok' and nothing else.",
    mode="chat",
    seconds=0.01,  # intentionally tiny: should always exceed wall time
    return_report=True,
)

print("success:", report.success)
print("limits_exceeded:", report.budget_usage.limits_exceeded)
print("elapsed_seconds:", round(report.budget_usage.elapsed_seconds, 3))
