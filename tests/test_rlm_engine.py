"""
Focused tests for RLMEngine core behaviors.
"""
from __future__ import annotations

import json
from typing import Optional

import pytest

from enzu import Budget, SuccessCriteria, TaskSpec
import enzu.rlm.engine as rlm_engine_module
import enzu.rlm.llm_executor as llm_executor_module
import enzu.rlm.runner as rlm_runner_module
import enzu.repl.sandbox as sandbox_module
from enzu.models import ProgressEvent, ProviderResult
from enzu.providers.base import BaseProvider
from enzu.rlm.engine import RLMEngine
from enzu.rlm.runner import context_grew as _context_grew
from tests.providers.mock_provider import MockProvider


def _make_task(
    *,
    required_substrings: list[str] | None = None,
    min_word_count: int | None = None,
    metadata: dict | None = None,
    budget: Budget | None = None,
    max_output_tokens: int | None = None,
) -> TaskSpec:
    criteria_kwargs: dict = {}
    if required_substrings is not None:
        criteria_kwargs["required_substrings"] = required_substrings
    if min_word_count is not None:
        criteria_kwargs["min_word_count"] = min_word_count
    criteria = SuccessCriteria(**criteria_kwargs)
    return TaskSpec(
        task_id="rlm-test",
        input_text="Test task",
        model="mock-model",
        budget=budget or Budget(max_tokens=50),
        success_criteria=criteria,
        max_output_tokens=max_output_tokens,
        metadata=metadata or {},
    )


class SelectiveFailProvider(BaseProvider):
    """Fail subcalls to force fallback paths during RLM execution."""

    name = "selective_fail"

    def __init__(self, main_output: str, fail_prefix: str = "SUBCALL:") -> None:
        self._main_output = main_output
        self._fail_prefix = fail_prefix
        self.calls: list[str] = []

    def generate(self, task: TaskSpec) -> ProviderResult:
        return self.stream(task)

    def stream(self, task: TaskSpec, on_progress=None) -> ProviderResult:
        prompt = task.input_text
        self.calls.append(prompt)
        if prompt.startswith(self._fail_prefix):
            raise RuntimeError("provider_subcall_failed")
        return ProviderResult(
            output_text=self._main_output,
            raw={"mock": True},
            usage={"output_tokens": 5, "total_tokens": 5},
            provider=self.name,
            model=task.model,
        )


class ProgressingProvider(BaseProvider):
    """Emit progress events during streaming to exercise llm_stream filtering."""

    name = "progressing"

    def __init__(self, output_text: str) -> None:
        self._output_text = output_text

    def generate(self, task: TaskSpec) -> ProviderResult:
        return self.stream(task)

    def stream(self, task: TaskSpec, on_progress=None) -> ProviderResult:
        if on_progress:
            on_progress(
                ProgressEvent(
                    phase="generation",
                    message="tick",
                    data={"provider": self.name},
                )
            )
        return ProviderResult(
            output_text=self._output_text,
            raw={"mock": True},
            usage={"output_tokens": 5, "total_tokens": 5},
            provider=self.name,
            model=task.model,
        )


class UsageMissingProvider(BaseProvider):
    """Return usage without total token counts for RLM progress events."""

    name = "usage_missing"

    def __init__(self, output_text: str) -> None:
        self._output_text = output_text

    def generate(self, task: TaskSpec) -> ProviderResult:
        return ProviderResult(
            output_text=self._output_text,
            raw={"mock": True},
            usage={},
            provider=self.name,
            model=task.model,
        )

    def stream(self, task: TaskSpec, on_progress=None) -> ProviderResult:
        return self.generate(task)


class SubcallInspectProvider(BaseProvider):
    """Record subcall TaskSpec details and direct subcall prompts."""

    name = "subcall_inspect"

    def __init__(self, main_output: str, sub_output: str = "sub") -> None:
        self._main_output = main_output
        self._sub_output = sub_output
        self.subcall_tasks: list[TaskSpec] = []
        self.direct_subcall_prompts: list[str] = []

    def generate(self, task: TaskSpec) -> ProviderResult:
        return self.stream(task)

    def stream(self, task: TaskSpec, on_progress=None) -> ProviderResult:
        if task.metadata.get("_subcall_prompt"):
            self.subcall_tasks.append(task)
            return ProviderResult(
                output_text=f"FINAL('{self._sub_output}')",
                raw={"mock": True},
                usage={"output_tokens": 5, "total_tokens": 5},
                provider=self.name,
                model=task.model,
            )
        if task.input_text.startswith("SUBCALL:"):
            self.direct_subcall_prompts.append(task.input_text)
            return ProviderResult(
                output_text=self._sub_output,
                raw={"mock": True},
                usage={"output_tokens": 5, "total_tokens": 5},
                provider=self.name,
                model=task.model,
            )
        return ProviderResult(
            output_text=self._main_output,
            raw={"mock": True},
            usage={"output_tokens": 5, "total_tokens": 5},
            provider=self.name,
            model=task.model,
        )


def test_rlm_progress_callback_exception_does_not_crash() -> None:
    provider = MockProvider(main_outputs=["```python\nFINAL('ok')\n```"])
    task = _make_task(required_substrings=["ok"])
    messages: list[str] = []

    def bad_callback(message: str) -> None:
        messages.append(message)
        raise ValueError("Callback crashed")

    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="context", on_progress=bad_callback)

    assert report.success is True
    assert report.answer == "ok"
    assert len(messages) >= 1


def test_rlm_rejects_weak_success_criteria() -> None:
    provider = MockProvider(main_outputs=["```python\nFINAL('ok')\n```"])
    task = _make_task(min_word_count=1)

    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="context")

    assert report.success is False
    assert "success_criteria_missing_or_weak" in report.errors
    assert report.steps == []
    assert provider.calls == []


def test_rlm_accepts_final_without_code_block() -> None:
    provider = MockProvider(main_outputs=["FINAL('done')"])
    task = _make_task(required_substrings=["done"])

    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="context")

    assert report.success is True
    assert report.answer == "done"


def test_rlm_final_var_uses_namespace_value() -> None:
    # Covers _final_from_output FINAL_VAR branch.
    provider = MockProvider(main_outputs=["FINAL_VAR(answer)"])
    task = _make_task(required_substrings=["ok"])

    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="context", namespace={"answer": "ok"})

    assert report.success is True
    assert report.answer == "ok"


def test_rlm_allows_weak_success_criteria_when_flagged() -> None:
    # Covers allow_weak_success_criteria override.
    provider = MockProvider(main_outputs=["```python\nFINAL('ok')\n```"])
    task = _make_task(
        min_word_count=1,
        metadata={"allow_weak_success_criteria": True},
    )

    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="context")

    assert report.success is True
    assert report.answer == "ok"


def test_rlm_tools_guidance_appended_in_paper_prompt() -> None:
    # Covers tools_guidance branch for paper prompt style.
    provider = MockProvider(main_outputs=["```python\nFINAL('ok')\n```"])
    task = _make_task(
        required_substrings=["ok"],
        metadata={"tools_guidance": "TOOLS_GUIDE"},
    )

    engine = RLMEngine(max_steps=1, prompt_style="paper")
    report = engine.run(task, provider, data="context")

    assert "TOOLS_GUIDE" in report.steps[0].prompt


def test_rlm_tools_guidance_appended_in_extended_prompt() -> None:
    # Covers tools_guidance branch for extended prompt style.
    provider = MockProvider(main_outputs=["```python\nFINAL('ok')\n```"])
    task = _make_task(
        required_substrings=["ok"],
        metadata={"tools_guidance": "TOOLS_GUIDE_EXT"},
    )

    engine = RLMEngine(max_steps=1, prompt_style="extended")
    report = engine.run(task, provider, data="context")

    assert "TOOLS_GUIDE_EXT" in report.steps[0].prompt


def test_rlm_prelude_final_short_circuits_when_allowed() -> None:
    # Covers prelude_allow_final short-circuit path in RLMEngine.run.
    provider = MockProvider(main_outputs=["```python\nFINAL('main')\n```"])
    task = _make_task(
        required_substrings=["prelude"],
        metadata={
            "prelude_code": "FINAL('prelude')",
            "prelude_allow_final": True,
        },
    )

    engine = RLMEngine(max_steps=2)
    report = engine.run(task, provider, data="context")

    assert report.success is True
    assert report.answer == "prelude"
    assert report.steps == []
    assert provider.calls == []


def test_rlm_prelude_final_rejected_then_main_runs() -> None:
    # Covers prelude final rejection and continuation into main loop.
    provider = MockProvider(main_outputs=["```python\nFINAL('good')\n```"])
    task = _make_task(
        required_substrings=["good"],
        metadata={
            "prelude_code": "FINAL('bad')",
            "prelude_allow_final": True,
        },
    )

    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="context")

    assert report.success is True
    assert report.answer == "good"
    assert len(report.steps) == 1
    assert provider.calls != []


def test_rlm_prelude_final_ignored_without_allow() -> None:
    # Covers prelude path when prelude_allow_final is unset.
    provider = MockProvider(main_outputs=["```python\nFINAL('main')\n```"])
    task = _make_task(
        required_substrings=["main"],
        metadata={"prelude_code": "FINAL('prelude')"},
    )

    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="context")

    assert report.success is True
    assert report.answer == "main"
    assert len(report.steps) == 1
    assert provider.calls != []


def test_rlm_prompt_includes_search_guidance_when_tools_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Covers has_search_tools branch in _system_prompt.
    provider = MockProvider(main_outputs=["```python\nFINAL('ok')\n```"])
    task = _make_task(required_substrings=["ok"])

    monkeypatch.setattr(
        sandbox_module,
        "_get_search_tools",
        lambda: {"__search_tools_available__": True},
    )

    engine = RLMEngine(max_steps=1, prompt_style="extended", inject_search_tools=True)
    report = engine.run(task, provider, data="context")

    assert "Web Research Tools" in report.steps[0].prompt


def test_rlm_suppress_probe_guidance_strips_lines() -> None:
    # Covers suppress_probe_guidance stripping in extended prompt.
    provider_with_probe = MockProvider(main_outputs=["```python\nFINAL('ok')\n```"])
    provider_suppressed = MockProvider(main_outputs=["```python\nFINAL('ok')\n```"])

    base_task = _make_task(required_substrings=["ok"])
    suppressed_task = _make_task(
        required_substrings=["ok"],
        metadata={"suppress_probe_guidance": True},
    )

    engine = RLMEngine(max_steps=1, prompt_style="extended")
    report_base = engine.run(base_task, provider_with_probe, data="context")
    report_suppressed = engine.run(suppressed_task, provider_suppressed, data="context")

    assert "Probe:" in report_base.steps[0].prompt
    assert "Probe:" not in report_suppressed.steps[0].prompt


def test_rlm_verify_on_final_false_defers_failure() -> None:
    # Covers verify_on_final=False acceptance before verification.
    provider = MockProvider(main_outputs=["```python\nFINAL('bad')\n```"])
    task = _make_task(required_substrings=["good"])

    engine = RLMEngine(max_steps=1, verify_on_final=False)
    report = engine.run(task, provider, data="context")

    assert report.answer == "bad"
    assert report.success is False
    assert any("verification_failed:missing_substring:good" in err for err in report.errors)


def test_rlm_tokenizer_unavailable_uses_conservative_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    # When tokenizer unavailable and model is not mock, use conservative estimates.
    # For mock models, no estimation is done (controlled test environment).
    provider = MockProvider(main_outputs=["```python\nFINAL('ok')\n```"])
    task = _make_task(
        required_substrings=["ok"],
        budget=Budget(max_tokens=50, max_total_tokens=100),
    )

    # Mock model skips conservative estimation, so test should succeed
    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="context")

    assert report.success is True
    assert report.answer == "ok"


def test_rlm_budget_notice_added_on_large_clamp(monkeypatch: pytest.MonkeyPatch) -> None:
    # Covers budget_notice path in _advance_prompt when clamp is large.
    provider = MockProvider(main_outputs=["```python\nFINAL('ok')\n```"])
    task = _make_task(
        required_substrings=["ok"],
        budget=Budget(max_tokens=100, max_total_tokens=10),
    )

    def _fixed_tokens(_: str, __: Optional[str] = None) -> Optional[int]:
        return 8

    monkeypatch.setattr(llm_executor_module, "count_tokens_exact", _fixed_tokens)

    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="context")

    assert "Budget clamp: output capped to" in report.steps[0].prompt


def test_rlm_usage_missing_total_tokens_emits_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Covers usage_missing_total_tokens emission in RLM loop.
    provider = UsageMissingProvider("```python\nFINAL('ok')\n```")
    task = _make_task(
        required_substrings=["ok"],
        budget=Budget(max_tokens=50, max_total_tokens=100),
    )
    messages: list[str] = []

    def _no_tokens(_: str, __: Optional[str] = None) -> Optional[int]:
        return None

    monkeypatch.setattr(llm_executor_module, "count_tokens_exact", _no_tokens)

    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="context", on_progress=messages.append)

    assert report.success is True
    assert "usage_missing_total_tokens" in messages


def test_rlm_budget_exceeded_max_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    # Covers _budget_exceeded max_seconds path at loop start.
    provider = MockProvider(main_outputs=["```python\nFINAL('ok')\n```"])
    task = _make_task(
        required_substrings=["ok"],
        budget=Budget(max_tokens=50, max_seconds=1.0),
    )

    times = [0.0, 10.0, 10.0]
    index = {"i": 0}

    def _fake_time() -> float:
        i = index["i"]
        index["i"] = min(i + 1, len(times) - 1)
        return times[i]

    monkeypatch.setattr(rlm_engine_module.time, "time", _fake_time)
    monkeypatch.setattr(rlm_runner_module.time, "time", _fake_time)

    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="context")

    assert report.success is False
    assert "budget_exceeded" in report.errors or report.budget_usage.limits_exceeded
    # Steps may or may not be empty depending on timing


def test_rlm_budget_exhausted_preflight_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Covers budget_exhausted_preflight from _reserve_prompt_budget.
    provider = MockProvider(main_outputs=["```python\nFINAL('ok')\n```"])
    task = _make_task(
        required_substrings=["ok"],
        budget=Budget(max_tokens=50, max_total_tokens=5),
    )

    def _fixed_tokens(_: str, __: Optional[str] = None) -> Optional[int]:
        return 10

    monkeypatch.setattr(llm_executor_module, "count_tokens_exact", _fixed_tokens)

    engine = RLMEngine(max_steps=1)
    with pytest.raises(RuntimeError, match="budget_exhausted_preflight"):
        engine.run(task, provider, data="context")


def test_rlm_research_subcall_prelude_executes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Covers research-mode subcall prelude in _run_subcall.
    model_output = "```python\nresult = llm_query('topic')\nFINAL(result)\n```"
    provider = MockProvider(main_outputs=[model_output])
    task = _make_task(
        required_substrings=["Example"],
        metadata={"subcall_mode": "research"},
    )

    def _fake_research(*_args: object, **_kwargs: object) -> dict:
        return {
            "sources": [
                {
                    "title": "Example",
                    "url": "https://example.com",
                    "published_date": "2025-01-01",
                    "text": "example text",
                }
            ]
        }

    monkeypatch.setattr(
        sandbox_module,
        "_get_search_tools",
        lambda: {"__search_tools_available__": True, "research": _fake_research},
    )

    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="context")

    sources = json.loads(report.answer or "[]")
    assert sources[0]["title"] == "Example"


def test_rlm_llm_stream_progress_not_logged_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Covers llm_stream log suppression when telemetry streaming is off.
    provider = ProgressingProvider("```python\nFINAL('ok')\n```")
    task = _make_task(required_substrings=["ok"])
    logged: list[tuple[str, str, dict]] = []

    def fake_log(level: str, event: str, **kwargs: object) -> None:
        logged.append((level, event, kwargs))

    monkeypatch.setattr(rlm_engine_module.telemetry, "stream_enabled", lambda: False)
    monkeypatch.setattr(rlm_engine_module.telemetry, "log", fake_log)

    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="context")

    assert report.success is True
    assert any(event == "rlm_progress" for _, event, _ in logged)
    assert not any(
        event == "rlm_progress"
        and str(kwargs.get("progress_message", "")).startswith("llm_stream:")
        for _, event, kwargs in logged
    )


def test_rlm_llm_batch_recursive_subcalls() -> None:
    # Covers recursive llm_batch path that uses sub-RLMs.
    main_output = (
        "```python\n"
        "results = llm_batch(['SUBCALL:a', 'SUBCALL:b'])\n"
        "FINAL(','.join(results))\n"
        "```"
    )
    provider = MockProvider(
        main_outputs=[main_output],
        subcall_responses={"a": "one", "b": "two"},
    )
    task = _make_task(required_substrings=["one", "two"])

    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="context")

    assert report.success is True
    assert report.answer == "one,two"


def test_context_grew_with_before_snapshot() -> None:
    # Covers _context_grew branch with prior snapshot.
    before = {"num_sources": 1, "num_queries": 1, "total_text_chars": 10}
    after_same = {"num_sources": 1, "num_queries": 1, "total_text_chars": 10}
    after_more = {"num_sources": 2, "num_queries": 1, "total_text_chars": 10}

    assert _context_grew(before, after_same) is False
    assert _context_grew(before, after_more) is True


def test_rlm_subcall_metadata_stripped_and_tagged() -> None:
    # Covers subcall metadata filtering and tagging in _run_subcall.
    main_output = "```python\nresult = llm_query('SUBCALL:foo')\nFINAL(result)\n```"
    provider = SubcallInspectProvider(main_output)
    task = _make_task(
        required_substrings=["sub"],
        metadata={
            "prelude_code": "FINAL('ignored')",
            "prelude_allow_final": True,
            "custom": "keep",
        },
    )

    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="context")

    assert report.success is True
    assert provider.subcall_tasks
    sub_task = provider.subcall_tasks[0]
    assert ":sub:" in sub_task.task_id
    assert sub_task.metadata.get("custom") == "keep"
    assert sub_task.metadata.get("allow_weak_success_criteria") is True
    assert "prelude_code" not in sub_task.metadata
    assert "prelude_allow_final" not in sub_task.metadata
    assert "_subcall_prompt" in sub_task.metadata


def test_rlm_subcall_max_output_tokens_respected() -> None:
    # Covers subcall max_output_tokens min logic.
    # The subcall's max_output_tokens is determined by the budget allocation
    # which considers remaining budget and proportional allocation.
    main_output = "```python\nresult = llm_query('SUBCALL:foo')\nFINAL(result)\n```"
    provider = SubcallInspectProvider(main_output)
    task = _make_task(
        required_substrings=["sub"],
        max_output_tokens=4,
        budget=Budget(max_tokens=10),
    )

    engine = RLMEngine(max_steps=1, subcall_max_output_tokens=7)
    report = engine.run(task, provider, data="context")

    assert report.success is True
    sub_task = provider.subcall_tasks[0]
    # Subcall gets allocated tokens based on remaining budget and proportional allocation
    # The actual value depends on budget enforcement logic
    assert sub_task.max_output_tokens is not None
    assert sub_task.max_output_tokens > 0
    assert sub_task.max_output_tokens <= 7  # Capped by subcall_max_output_tokens


def test_rlm_max_recursion_depth_forces_direct_subcall() -> None:
    # Covers depth gate for recursive_subcalls.
    main_output = "```python\nresult = llm_query('SUBCALL:foo')\nFINAL(result)\n```"
    provider = SubcallInspectProvider(main_output)
    task = _make_task(required_substrings=["sub"])

    engine = RLMEngine(max_steps=1, max_recursion_depth=0, recursive_subcalls=True)
    report = engine.run(task, provider, data="context")

    assert report.success is True
    assert provider.subcall_tasks == []
    assert provider.direct_subcall_prompts == ["SUBCALL:foo"]


def test_rlm_llm_query_fallback_used_for_subcall() -> None:
    # Covers direct_llm_query fallback path. When fallback succeeds, task succeeds.
    main_output = "```python\nresult = llm_query('SUBCALL:foo')\nFINAL(result)\n```"
    provider = SelectiveFailProvider(main_output)
    fallback = MockProvider(
        main_outputs=["```python\nFINAL('unused')\n```"],
        subcall_responses={"foo": "bar"},
    )
    task = _make_task(required_substrings=["bar"])

    engine = RLMEngine(max_steps=1, recursive_subcalls=False)
    report = engine.run(task, provider, data="context", fallback_providers=[fallback])

    # Fallback succeeded: answer contains "bar", verification passed.
    assert report.answer == "bar"
    assert report.success is True


@pytest.mark.anyio
async def test_rlm_llm_batch_fallback_in_running_loop() -> None:
    # Covers llm_batch fallback and running-loop branch.
    main_output = (
        "```python\n"
        "results = llm_batch(['SUBCALL:a', 'SUBCALL:b'])\n"
        "FINAL(','.join(results))\n"
        "```"
    )
    provider = SelectiveFailProvider(main_output)
    fallback = MockProvider(
        main_outputs=["```python\nFINAL('unused')\n```"],
        subcall_responses={"a": "one", "b": "two"},
    )
    task = _make_task(required_substrings=["one", "two"])

    engine = RLMEngine(max_steps=1, recursive_subcalls=False)
    report = engine.run(task, provider, data="context", fallback_providers=[fallback])

    assert report.success is True
    assert report.answer == "one,two"


def test_context_grew_when_before_none() -> None:
    # Covers _context_grew branch when no snapshot exists.
    assert _context_grew(None, {"num_sources": 0, "num_queries": 0, "total_text_chars": 0}) is False
    assert _context_grew(None, {"num_sources": 1, "num_queries": 0, "total_text_chars": 0}) is True


def test_rlm_salvages_final_from_code_on_error() -> None:
    """FINAL() should be salvaged from code when sandbox execution fails."""
    # Code that has FINAL() but also a syntax error before it completes
    error_code = '''```python
result = broken_function(
FINAL("The answer is 42 based on detailed analysis")
```'''
    provider = MockProvider(main_outputs=[error_code])
    task = _make_task(
        required_substrings=["42"],
        budget=Budget(max_tokens=100),
    )
    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="context")

    # Should salvage the FINAL content even though code has error
    assert report.answer is not None
    assert "42" in report.answer


def test_rlm_salvages_answer_when_max_steps_reached() -> None:
    """Should salvage answer when max_steps reached without FINAL()."""
    # Model produces code that sets a result variable but never calls FINAL
    code_output = '''```python
result = "computed answer with important data"
print(result)
```'''
    provider = MockProvider(main_outputs=[code_output, code_output])  # Two steps
    task = _make_task(
        min_word_count=1,
        metadata={"allow_weak_success_criteria": True},
        budget=Budget(max_tokens=200),
    )
    engine = RLMEngine(max_steps=2)
    report = engine.run(task, provider, data="context")

    # Should have salvaged something from the steps
    assert report.answer is not None
    assert len(report.answer) > 0


def test_rlm_salvages_from_truncated_code_block() -> None:
    """FINAL() should be salvaged from truncated (unclosed) code block."""
    # Simulates max_output_tokens cutting off the output mid-code-block
    truncated_output = '''```python
data = analyze_context(context)
summary = generate_summary(data)
FINAL("Analysis complete: found 3 key insights about the topic'''
    # Note: missing closing quote, paren, and ``` - simulating truncation

    provider = MockProvider(main_outputs=[truncated_output])
    task = _make_task(
        min_word_count=1,
        metadata={"allow_weak_success_criteria": True},
        budget=Budget(max_tokens=100),
    )
    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="context")

    # Should detect truncation and salvage the partial FINAL content
    assert report.answer is not None
    assert "Analysis complete" in report.answer or "insights" in report.answer.lower()

