from __future__ import annotations

import json
from pathlib import Path

from enzu import Budget, SuccessCriteria, TaskSpec
from enzu.rlm import RLMEngine
from enzu.tools.context import ctx_add, ctx_clear, ctx_save
from tests.providers.mock_provider import MockProvider


def _seed_artifact(path: Path) -> None:
    ctx_clear()
    ctx_add(
        [
            {
                "url": "https://example.com/seed",
                "title": "Seed",
                "text": "seed text",
                "score": 0.7,
                "published_date": "2025-01-01",
            }
        ],
        query="seed",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ctx_save(str(path))
    ctx_clear()


def test_context_artifact_grows_and_persists(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifacts" / "context.json"
    _seed_artifact(artifact_path)

    model_output = """
```python
ctx_add([{
    "url": "https://example.com/new",
    "title": "New",
    "text": "new text",
    "score": 0.9,
    "published_date": "2025-01-02",
}], query="new")
FINAL("ok")
```
""".strip()

    task = TaskSpec(
        task_id="artifact-grow",
        input_text="Add a new source.",
        model="mock-model",
        budget=Budget(max_tokens=50, max_total_tokens=100),
        success_criteria=SuccessCriteria(required_substrings=["ok"]),
        metadata={"context_path": str(artifact_path)},
    )
    provider = MockProvider(main_outputs=[model_output])
    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="context", namespace={"ctx_add": ctx_add})

    assert report.success
    data = json.loads(artifact_path.read_text())
    assert len(data["sources"]) == 2


def test_context_artifact_not_rewritten_without_growth(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifacts" / "context.json"
    _seed_artifact(artifact_path)
    original = artifact_path.read_text()

    model_output = """
```python
FINAL("ok")
```
""".strip()

    task = TaskSpec(
        task_id="artifact-no-growth",
        input_text="Return ok.",
        model="mock-model",
        budget=Budget(max_tokens=50, max_total_tokens=100),
        success_criteria=SuccessCriteria(required_substrings=["ok"]),
        metadata={"context_path": str(artifact_path)},
    )
    provider = MockProvider(main_outputs=[model_output])
    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="context")

    assert report.success
    assert artifact_path.read_text() == original
