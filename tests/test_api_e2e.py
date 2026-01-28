from __future__ import annotations

from pathlib import Path

from enzu import Budget, Check, Limits, SuccessCriteria, TaskSpec, run
from tests.providers.mock_provider import MockProvider


def test_run_chat_mode() -> None:
    """Test the simple chat mode (no context)."""
    provider = MockProvider(main_outputs=["The answer is 42."])
    
    result = run(
        "What is the answer?",
        model="mock-model",
        provider=provider,
        tokens=100,
    )
    
    assert result == "The answer is 42."


def test_run_with_data_triggers_rlm() -> None:
    """Test that providing data triggers RLM mode automatically."""
    data_path = Path(__file__).parent / "fixtures" / "example1" / "context.txt"
    input_data = data_path.read_text(encoding="utf-8")
    
    model_output = """
```python
import re
docs = [doc.strip() for doc in data.split("###") if doc.strip()]
for doc in docs[:2]:
    print(doc[:100])
answer = llm_query("SUBCALL:extract:doc_1")
FINAL(answer)
```
""".strip()

    subcalls = {
        "extract:doc_1": "Albert Einstein, 1921",
    }
    provider = MockProvider(main_outputs=[model_output], subcall_responses=subcalls)
    
    # data present = RLM mode (no mode flag needed)
    result = run(
        "Find the scientist and year.",
        model="mock-model",
        provider=provider,
        data=input_data,
        tokens=200,
        contains=["Einstein"],
    )
    
    assert "Einstein" in result
    assert "1921" in result


def test_run_with_limits_object() -> None:
    """Test using Limits object for complex constraints."""
    provider = MockProvider(main_outputs=["Response within limits."])
    
    result = run(
        "Test query",
        model="mock-model",
        provider=provider,
        limits=Limits(tokens=500, seconds=60, cost=0.10),
    )
    
    assert result == "Response within limits."


def test_run_with_check_object() -> None:
    """Test using Check object for output verification."""
    provider = MockProvider(main_outputs=["The value is 123 units."])
    
    result = run(
        "What is the value?",
        model="mock-model",
        provider=provider,
        tokens=100,
        check=Check(contains=["123"], matches=[r"\d+ units"]),
    )
    
    assert "123" in result


def test_run_accepts_task_spec() -> None:
    provider = MockProvider(main_outputs=["Spec ok."])
    task = TaskSpec(
        task_id="spec-task",
        input_text="Say ok.",
        model="mock-model",
        budget=Budget(max_tokens=32),
        success_criteria=SuccessCriteria(min_word_count=1),
    )

    result = run(task, provider=provider)

    assert result == "Spec ok."


def test_run_accepts_task_dict() -> None:
    provider = MockProvider(main_outputs=["Dict ok."])
    task_payload = {
        "task": {
            "task_id": "dict-task",
            "input_text": "Say ok.",
            "budget": {"max_tokens": 32},
            "success_criteria": {"min_word_count": 1},
        }
    }

    result = run(task_payload, model="mock-model", provider=provider)

    assert result == "Dict ok."


def test_run_return_report() -> None:
    provider = MockProvider(main_outputs=["Report ok."])

    report = run(
        "Return the report.",
        model="mock-model",
        provider=provider,
        return_report=True,
    )

    assert hasattr(report, "output_text") and report.output_text == "Report ok."
