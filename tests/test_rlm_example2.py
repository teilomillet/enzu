from __future__ import annotations

import os
from pathlib import Path

import pytest

from enzu import Budget, OpenAICompatProvider, SuccessCriteria, TaskSpec
from enzu.rlm import RLMEngine
from tests.providers.mock_provider import MockProvider


def test_rlm_example2_codebase_understanding() -> None:
    data_path = Path(__file__).parent / "fixtures" / "example2" / "context.txt"
    input_data = data_path.read_text(encoding="utf-8")
    model_output = """
```python
import ast

entries = []
for chunk in data.split("FILE:"):
    chunk = chunk.strip()
    if not chunk:
        continue
    lines = chunk.splitlines()
    path = lines[0].strip()
    code = "\\n".join(lines[1:]).strip()
    entries.append((path, code))

def complexity(func_node):
    score = 1
    for node in ast.walk(func_node):
        if isinstance(
            node,
            (
                ast.If,
                ast.For,
                ast.While,
                ast.With,
                ast.Try,
                ast.BoolOp,
            ),
        ):
            score += 1
    return score

best = {"file": None, "function": None, "complexity": 0}
for path, code in entries:
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            score = complexity(node)
            if score > best["complexity"]:
                best = {"file": path, "function": node.name, "complexity": score}

result = best
FINAL_VAR("result")
```
""".strip()

    provider = MockProvider(main_outputs=[model_output])
    budget = Budget(max_tokens=150, max_total_tokens=300)
    criteria = SuccessCriteria(required_substrings=["beta"])
    task = TaskSpec(
        task_id="example2",
        input_text=(
            "Find the function with the highest cyclomatic complexity."
        ),
        model="mock-model",
        budget=budget,
        success_criteria=criteria,
    )
    engine = RLMEngine(max_steps=2, allowed_imports=["ast"])
    report = engine.run(task, provider, data=input_data)

    assert report.success
    assert "app/beta.py" in (report.answer or "")
    assert "beta" in (report.answer or "")
    assert "5" in (report.answer or "")


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY env var is missing",
)
def test_rlm_example2_integration() -> None:
    data_path = Path(__file__).parent / "fixtures" / "example2" / "context.txt"
    input_data = data_path.read_text(encoding="utf-8")
    budget = Budget(max_tokens=600, max_seconds=180)
    criteria = SuccessCriteria(required_substrings=["complexity"])
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    task = TaskSpec(
        task_id="example2-live",
        input_text=(
            "Find the function with the highest cyclomatic complexity."
        ),
        model=model,
        budget=budget,
        success_criteria=criteria,
    )
    provider = OpenAICompatProvider(
        name="openrouter",
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    engine = RLMEngine(allowed_imports=["ast"])
    report = engine.run(task, provider, data=input_data)

    assert report.answer is not None
