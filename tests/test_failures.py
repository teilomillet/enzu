from __future__ import annotations

from typing import Dict

from enzu import (
    Budget,
    Engine,
    ProviderResult,
    RLMEngine,
    SuccessCriteria,
    TaskSpec,
)
from enzu.providers.base import BaseProvider
from tests.providers.mock_provider import MockProvider


class _StaticProvider(BaseProvider):
    def __init__(self, output_text: str, usage: Dict[str, int] | None = None) -> None:
        self._output_text = output_text
        self._usage = usage or {"output_tokens": 1, "total_tokens": 1}

    def generate(self, task: TaskSpec) -> ProviderResult:
        # Deterministic provider output for Engine verification tests.
        return ProviderResult(
            output_text=self._output_text,
            raw={"static": True},
            usage=dict(self._usage),
            provider=self.name,
            model=task.model,
        )


class _ExplodingProvider(BaseProvider):
    def generate(self, task: TaskSpec) -> ProviderResult:
        # Simulates provider failure to validate Engine error handling.
        raise RuntimeError("provider boom")


class _ShouldNotRunProvider(BaseProvider):
    def stream(self, task: TaskSpec, on_progress=None) -> ProviderResult:
        # Preflight should stop before provider stream is called.
        raise RuntimeError("preflight failed")


def test_engine_budget_preflight_rejects() -> None:
    engine = Engine()
    task = TaskSpec(
        task_id="preflight",
        input_text="hello",
        model="mock",
        budget=Budget(max_tokens=5),
        success_criteria=SuccessCriteria(required_substrings=["hello"]),
        max_output_tokens=10,
    )
    report = engine.run(task, _ShouldNotRunProvider())

    assert report.success is False
    assert "task.max_output_tokens exceeds budget.max_tokens" in report.errors


def test_rlm_budget_preflight_rejects() -> None:
    engine = RLMEngine(max_steps=1)
    task = TaskSpec(
        task_id="rlm-preflight",
        input_text="hello",
        model="mock",
        budget=Budget(max_tokens=5),
        success_criteria=SuccessCriteria(required_substrings=["hello"]),
        max_output_tokens=10,
    )
    report = engine.run(task, _ShouldNotRunProvider(), data="context")

    assert report.success is False
    assert "task.max_output_tokens exceeds budget.max_tokens" in report.errors


def test_rlm_requires_strong_success_criteria() -> None:
    engine = RLMEngine(max_steps=1)
    task = TaskSpec(
        task_id="rlm-weak-criteria",
        input_text="hello",
        model="mock",
        budget=Budget(max_tokens=5),
        success_criteria=SuccessCriteria(min_word_count=1),
    )
    report = engine.run(task, _ShouldNotRunProvider(), data="context")

    assert report.success is False
    assert "success_criteria_missing_or_weak" in report.errors


def test_engine_provider_error_is_captured() -> None:
    engine = Engine()
    task = TaskSpec(
        task_id="provider-error",
        input_text="hello",
        model="mock",
        budget=Budget(max_tokens=5),
        success_criteria=SuccessCriteria(required_substrings=["hello"]),
    )
    report = engine.run(task, _ExplodingProvider())

    assert report.success is False
    assert any("provider boom" in err for err in report.errors)


def test_engine_verification_failure() -> None:
    engine = Engine()
    task = TaskSpec(
        task_id="verification-fail",
        input_text="hello",
        model="mock",
        budget=Budget(max_tokens=5),
        success_criteria=SuccessCriteria(required_substrings=["must-have"]),
    )
    report = engine.run(task, _StaticProvider("hello"))

    assert report.success is False
    assert report.verification.passed is False
    assert "missing_substring:must-have" in report.verification.reasons


def test_rlm_invalid_code_records_error() -> None:
    model_output = """
```python
raise ValueError("boom")
```
""".strip()
    provider = MockProvider(main_outputs=[model_output])
    task = TaskSpec(
        task_id="rlm-invalid",
        input_text="trigger invalid code",
        model="mock-model",
        budget=Budget(max_tokens=20, max_total_tokens=40),
        success_criteria=SuccessCriteria(required_substrings=["unused"]),
    )
    engine = RLMEngine(max_steps=1)
    # RLMEngine.run expects data for the sandbox context input.
    report = engine.run(task, provider, data="context")

    assert report.success is False
    assert report.steps[0].error is not None
    assert "boom" in report.steps[0].error
    # Salvage recovers content, so we have an answer (but verification fails)
    # The "never lose work" philosophy means salvage always tries
    assert report.answer is not None


def test_rlm_verifies_sandbox_answer() -> None:
    model_output = """
```python
FINAL("answer from sandbox")
```
""".strip()
    provider = MockProvider(main_outputs=[model_output])
    task = TaskSpec(
        task_id="rlm-verify-answer",
        input_text="use final answer",
        model="mock-model",
        budget=Budget(max_tokens=20, max_total_tokens=40),
        success_criteria=SuccessCriteria(required_substrings=["answer from sandbox"]),
    )
    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="context")

    assert report.success is True
    assert report.answer == "answer from sandbox"


def test_rlm_no_final_hits_max_steps() -> None:
    model_output = """
```python
print("working")
```
""".strip()
    provider = MockProvider(main_outputs=[model_output])
    task = TaskSpec(
        task_id="rlm-no-final",
        input_text="do work",
        model="mock-model",
        budget=Budget(max_tokens=20, max_total_tokens=40),
        success_criteria=SuccessCriteria(required_substrings=["unused"]),
    )
    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="context")

    assert report.success is False
    # Salvage recovers stdout/output when max_steps reached without FINAL
    # The "never lose work" philosophy means we always try to recover
    assert report.answer is not None
    assert "working" in report.answer


def test_rlm_budget_exhausted_mid_task() -> None:
    model_output = """
```python
first = llm_query("SUBCALL:first")
print(first)
second = llm_query("SUBCALL:second")
print(second)
```
""".strip()
    provider = MockProvider(
        main_outputs=[model_output],
        subcall_responses={"first": "ok"},
        usage={"output_tokens": 6, "total_tokens": 6},
    )
    task = TaskSpec(
        task_id="rlm-budget",
        input_text="force budget exhaustion",
        model="mock-model",
        budget=Budget(max_total_tokens=12),
        success_criteria=SuccessCriteria(required_substrings=["unused"]),
    )
    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="context")

    assert report.success is False
    assert report.steps[0].error is not None
    assert "budget_exhausted" in report.steps[0].error
    # Exhaustion happens before the second subcall is sent.
    assert "SUBCALL:second" not in provider.calls


def test_rlm_budget_exhausted_with_completion_tokens() -> None:
    model_output = """
```python
first = llm_query("SUBCALL:first")
print(first)
second = llm_query("SUBCALL:second")
print(second)
```
""".strip()
    provider = MockProvider(
        main_outputs=[model_output],
        subcall_responses={"first": "ok"},
        usage={"completion_tokens": 6, "total_tokens": 6},
    )
    task = TaskSpec(
        task_id="rlm-budget-completion",
        input_text="force budget exhaustion",
        model="mock-model",
        budget=Budget(max_tokens=5),
        success_criteria=SuccessCriteria(required_substrings=["unused"]),
    )
    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="context")

    assert report.success is False
    assert report.steps[0].error is not None
    assert (
        "critical budget threshold" in report.steps[0].error
        or "budget_exhausted" in report.steps[0].error
    )
    assert "SUBCALL:second" not in provider.calls


# Goal-based success tests: model self-judges completion


def test_rlm_goal_based_accepts_final() -> None:
    """Goal-based: model's FINAL() is trusted as success judgment."""
    model_output = """
```python
FINAL("I found the root cause: null pointer in line 42")
```
""".strip()
    provider = MockProvider(main_outputs=[model_output])
    task = TaskSpec(
        task_id="rlm-goal-based",
        input_text="find the bug",
        model="mock-model",
        budget=Budget(max_tokens=100),
        # Goal-based: no mechanical checks, model self-judges.
        success_criteria=SuccessCriteria(goal="Find the root cause of the bug"),
    )
    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="error logs here")

    assert report.success is True
    assert report.answer == "I found the root cause: null pointer in line 42"
    assert not report.errors


def test_rlm_goal_based_no_final_fails() -> None:
    """Goal-based: no FINAL() means model didn't finish, but salvage recovers."""
    model_output = """
```python
print("still working on it")
```
""".strip()
    provider = MockProvider(main_outputs=[model_output])
    task = TaskSpec(
        task_id="rlm-goal-no-final",
        input_text="find the bug",
        model="mock-model",
        budget=Budget(max_tokens=100),
        success_criteria=SuccessCriteria(goal="Find the root cause of the bug"),
    )
    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="error logs here")

    # With goal-based criteria, salvaged answer is trusted if it exists
    # The "never lose work" philosophy recovers stdout as the answer
    assert report.answer is not None
    # Salvage captures the output from the step
    assert "working" in report.answer or "INCOMPLETE" in report.answer


def test_rlm_goal_counts_as_strong_criteria() -> None:
    """Goal is treated as strong criteria, allowing RLM to run."""
    model_output = """
```python
FINAL("done")
```
""".strip()
    provider = MockProvider(main_outputs=[model_output])
    task = TaskSpec(
        task_id="rlm-goal-strong",
        input_text="do something",
        model="mock-model",
        budget=Budget(max_tokens=100),
        # Only goal, no mechanical checks. Should still run.
        success_criteria=SuccessCriteria(goal="Complete the task"),
    )
    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="context")

    # Should succeed without "success_criteria_missing_or_weak" error.
    assert report.success is True
    assert "success_criteria_missing_or_weak" not in report.errors


def test_rlm_prompt_is_goal_when_no_criteria() -> None:
    """RLM mode auto-infers goal from prompt when no explicit criteria given."""
    from enzu import run

    # Mock the provider to avoid actual API calls
    mock_provider = MockProvider(
        main_outputs=['```python\nFINAL("Found the issue")\n```']
    )

    # No explicit success criteria - prompt should become the goal
    report = run(
        "Find the root cause of the crash",
        model="mock",
        provider=mock_provider,
        data="error logs here",
        tokens=100,
        return_report=True,
    )

    # Should succeed because prompt was used as goal, model called FINAL()
    assert hasattr(report, "success") and report.success is True
    assert hasattr(report, "answer") and report.answer == "Found the issue"


def test_rlm_budget_triggers_goal_mode_without_data() -> None:
    """cost/seconds triggers RLM mode even without explicit data."""
    from enzu import run
    from enzu.models import RLMExecutionReport

    mock_provider = MockProvider(
        main_outputs=['```python\nFINAL("Done analyzing the embedded data")\n```']
    )

    # No data, but cost triggers RLM mode. Prompt contains everything.
    report = run(
        "Here's my log:\nERROR at line 42\n\nFind the bug.",
        model="mock",
        provider=mock_provider,
        cost=5.00,  # This triggers RLM mode
        return_report=True,
    )

    # Should be RLM report (not ExecutionReport)
    assert isinstance(report, RLMExecutionReport)
    assert report.success is True
    assert report.answer == "Done analyzing the embedded data"
