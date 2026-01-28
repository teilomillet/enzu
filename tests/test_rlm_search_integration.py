from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from enzu import Budget, SuccessCriteria, TaskSpec
from enzu.rlm import RLMEngine
from enzu.tools import exa
from enzu.tools.context import ctx_clear, ctx_get
from enzu.tools.exa import ExaResult, exa_search
from tests.providers.mock_provider import MockProvider


@dataclass
class _FakeExaClient:
    def search(self, *args: Any, **kwargs: Any) -> List[ExaResult]:
        return [
            ExaResult(
                url="https://example.com/a",
                title="Example A",
                text="alpha",
                score=0.9,
                published_date="2025-01-01",
            )
        ]


def test_rlm_exa_search_populates_ctx_get(monkeypatch) -> None:
    ctx_clear()
    monkeypatch.setattr(exa, "_get_client", lambda: _FakeExaClient())

    model_output = """
```python
exa_search("topic", num_results=1)
context = ctx_get(max_chars=2000, max_chars_per_source=None)
FINAL(context)
```
""".strip()

    provider = MockProvider(main_outputs=[model_output])
    task = TaskSpec(
        task_id="rlm-search",
        input_text="Use exa_search and ctx_get, then return the context.",
        model="mock-model",
        budget=Budget(max_tokens=200, max_total_tokens=400),
        success_criteria=SuccessCriteria(required_substrings=["Published: 2025-01-01"]),
    )

    # Wire exa_search/ctx_get into the sandbox to exercise the RLM path.
    engine = RLMEngine(max_steps=1)
    report = engine.run(
        task,
        provider,
        data="context",
        namespace={"exa_search": exa_search, "ctx_get": ctx_get},
    )

    assert report.success
    assert "https://example.com/a" in (report.answer or "")

