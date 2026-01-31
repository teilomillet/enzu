from __future__ import annotations

import os
from pathlib import Path

import pytest

from enzu import Budget, OpenAICompatProvider, SuccessCriteria, TaskSpec
from enzu.rlm import RLMEngine
from tests.providers.mock_provider import MockProvider


def test_rlm_example1_multihop() -> None:
    data_path = Path(__file__).parent / "fixtures" / "example1" / "context.txt"
    input_data = data_path.read_text(encoding="utf-8")
    model_output = """
```python
import re

docs = [doc.strip() for doc in data.split("###") if doc.strip()]
matches = []
for doc in docs:
    if re.search(r"Nobel|Physics|president|crisis|Cuba", doc, re.I):
        matches.append(doc)
print(len(matches))

facts = []
for doc in matches:
    title_line = doc.splitlines()[0].strip()
    facts.append(llm_query(f"SUBCALL:extract:{title_line}"))

summary = llm_query("SUBCALL:synthesize")
FINAL(summary)
```
""".strip()

    subcalls = {
        "extract:doc_1": "name=Albert Einstein; year=1921; crisis=Cuban Missile Crisis",
        "extract:doc_2": "crisis=Cuban Missile Crisis; island=Cuba",
        "extract:doc_3": "award=Nobel Prize in Physics",
        "synthesize*": "Albert Einstein, 1921, Cuban Missile Crisis",
    }
    # Fallback output in case budget is exhausted during step 1
    fallback_output = (
        '```python\nFINAL("Albert Einstein, 1921, Cuban Missile Crisis")\n```'
    )
    provider = MockProvider(
        main_outputs=[model_output, fallback_output], subcall_responses=subcalls
    )
    budget = Budget(max_tokens=120, max_total_tokens=300)
    criteria = SuccessCriteria(required_substrings=["Albert Einstein"])
    task = TaskSpec(
        task_id="example1",
        input_text=("Find the scientist, Nobel year, and crisis name from the data."),
        model="mock-model",
        budget=budget,
        success_criteria=criteria,
    )
    engine = RLMEngine(max_steps=2, allowed_imports=["re"])
    report = engine.run(task, provider, data=input_data)

    assert report.success
    assert report.answer == "Albert Einstein, 1921, Cuban Missile Crisis"
    assert report.budget_usage.limits_exceeded == []


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY env var is missing",
)
def test_rlm_example1_integration() -> None:
    data_path = Path(__file__).parent / "fixtures" / "example1" / "context.txt"
    input_data = data_path.read_text(encoding="utf-8")
    budget = Budget(max_tokens=400, max_seconds=120)
    criteria = SuccessCriteria(required_substrings=["Albert Einstein"])
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    task = TaskSpec(
        task_id="example1-live",
        input_text=("Find the scientist, Nobel year, and crisis name from the data."),
        model=model,
        budget=budget,
        success_criteria=criteria,
    )
    provider = OpenAICompatProvider(
        name="openrouter",
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    engine = RLMEngine()
    report = engine.run(task, provider, data=input_data)

    assert report.answer is not None
