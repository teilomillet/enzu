"""
Tests for RLM guardrails: error recovery and code pattern guidance.

Tests cover:
1. Code extraction with block counting
2. Error classification to hints
3. Code pattern analysis
4. Feedback building and formatting
5. Safe helper injection
6. Multiple block detection
"""

from __future__ import annotations


from enzu.models import StepFeedback, SuccessCriteria, TaskSpec
from enzu.repl import (
    PythonSandbox,
    exec_code,
    build_namespace,
    safe_get,
    safe_rows,
    safe_sort,
)
from enzu.models import Budget

# Feedback functions extracted to rlm/feedback.py for modularity
from enzu.rlm.feedback import (
    analyze_code_patterns,
    build_feedback,
    classify_error,
    extract_code,
    format_feedback,
)

# Prompt constants extracted to rlm/prompts.py
from enzu.rlm.engine import RLMEngine
from enzu.rlm.budget import BudgetTracker as _BudgetTracker
from enzu.rlm.runner import StepRunner
from tests.providers.mock_provider import MockProvider


class TestExtractCode:
    """Test code extraction from model output."""

    def test_single_block(self) -> None:
        output = """
Here is the code:
```python
print("hello")
```
"""
        code, count = extract_code(output)
        assert code == 'print("hello")'
        assert count == 1

    def test_multiple_blocks_returns_first(self) -> None:
        """Multiple blocks: first is returned (contains setup code)."""
        output = """
First block:
```python
x = 1
```
Second block:
```python
y = 2
```
Third block:
```python
z = 3
```
"""
        code, count = extract_code(output)
        assert code == "x = 1"
        assert count == 3

    def test_no_code_block(self) -> None:
        output = "No code here, just text."
        code, count = extract_code(output)
        assert code is None
        assert count == 0

    def test_empty_block(self) -> None:
        output = """
```python
```
"""
        code, count = extract_code(output)
        assert code == ""
        assert count == 1

    def test_multiline_code(self) -> None:
        output = """
```python
def foo():
    x = 1
    return x * 2

result = foo()
print(result)
```
"""
        code, count = extract_code(output)
        assert code is not None and "def foo():" in code
        assert code is not None and "result = foo()" in code
        assert count == 1


class TestClassifyError:
    """Test error classification to actionable hints."""

    def test_key_error(self) -> None:
        error = "KeyError: 'missing_key'"
        hint = classify_error(error)
        assert hint is not None
        assert "safe_get" in hint

    def test_none_subscript(self) -> None:
        error = "TypeError: 'NoneType' object is not subscriptable"
        hint = classify_error(error)
        assert hint is not None
        assert "safe_get" in hint

    def test_none_iterable(self) -> None:
        error = "TypeError: 'NoneType' object is not iterable"
        hint = classify_error(error)
        assert hint is not None
        assert "safe_rows" in hint

    def test_attribute_error(self) -> None:
        error = "AttributeError: object has no attribute 'foo'"
        hint = classify_error(error)
        assert hint is not None
        assert "safe_rows" in hint

    def test_comparison_error(self) -> None:
        error = "TypeError: '<' not supported between instances of 'str' and 'int'"
        hint = classify_error(error)
        assert hint is not None
        assert "safe_sort" in hint

    def test_unknown_error(self) -> None:
        error = "SomeUnknownError: weird stuff"
        hint = classify_error(error)
        assert hint is None

    def test_none_input(self) -> None:
        hint = classify_error(None)
        assert hint is None

    def test_search_tools_missing_hint(self) -> None:
        error = "Search tools unavailable: set EXA_API_KEY to enable Exa tools."
        hint = classify_error(error)
        assert hint is not None
        assert "EXA_API_KEY" in hint


class TestAnalyzeCodePatterns:
    """Test code pattern analysis for anti-patterns."""

    def test_llm_query_in_loop(self) -> None:
        code = """
for item in items:
    result = llm_query(f"Process {item}")
    results.append(result)
another_call = llm_query("more")
"""
        warnings = analyze_code_patterns(code)
        assert len(warnings) >= 1
        assert any(
            "llm_query" in w and ("loop" in w.lower() or "runtime" in w.lower())
            for w in warnings
        )

    def test_full_data_to_llm_query(self) -> None:
        code = 'result = llm_query(f"Analyze: {data}")'
        warnings = analyze_code_patterns(code)
        assert len(warnings) >= 1
        assert any("context" in w.lower() and "filter" in w.lower() for w in warnings)

    def test_code_doable_task_delegated(self) -> None:
        code = 'result = llm_query("count the items in this list")'
        warnings = analyze_code_patterns(code)
        assert len(warnings) >= 1
        assert any("count" in w.lower() for w in warnings)

    def test_clean_code_no_warnings(self) -> None:
        code = """
chunks = [data[i:i+1000] for i in range(0, len(data), 1000)]
relevant = [c for c in chunks if 'keyword' in c]
result = llm_query(f"Summarize: {relevant[0]}")
print(result)
"""
        warnings = analyze_code_patterns(code)
        # May have some warnings, but not for over-delegation
        for w in warnings:
            assert "loop" not in w.lower() or "llm_query" not in w

    def test_empty_code(self) -> None:
        warnings = analyze_code_patterns("")
        assert warnings == []


class TestBuildFeedback:
    """Test feedback building from execution results."""

    def test_successful_execution(self) -> None:
        feedback = build_feedback(
            stdout="output text",
            error=None,
            code='print("hello")',
            block_count=1,
        )
        assert feedback.stdout == "output text"
        assert feedback.error is None
        assert feedback.violation is None
        assert feedback.hint is None

    def test_error_with_hint(self) -> None:
        feedback = build_feedback(
            stdout="",
            error="KeyError: 'foo'",
            code="x = d['foo']",
            block_count=1,
        )
        assert feedback.error == "KeyError: 'foo'"
        assert feedback.hint is not None
        assert "safe_get" in feedback.hint

    def test_multiple_blocks_violation(self) -> None:
        feedback = build_feedback(
            stdout="",
            error=None,
            code="x = 1",
            block_count=3,
        )
        assert feedback.violation == "multiple_blocks:3"

    def test_pattern_warnings_included(self) -> None:
        code = """
for i in range(10):
    r = llm_query(f"Process {i}")
    llm_query(f"Another {i}")
    llm_query(f"Third {i}")
"""
        feedback = build_feedback(
            stdout="",
            error=None,
            code=code,
            block_count=1,
        )
        assert len(feedback.pattern_warnings) >= 1

    def test_available_helpers(self) -> None:
        feedback = build_feedback(
            stdout="",
            error=None,
            code="x = 1",
            block_count=1,
        )
        assert "safe_get" in feedback.available_helpers
        assert "safe_rows" in feedback.available_helpers
        assert "safe_sort" in feedback.available_helpers


class TestFormatFeedback:
    """Test feedback formatting for prompt."""

    def test_output_only(self) -> None:
        feedback = StepFeedback(stdout="result: 42")
        text = format_feedback(feedback)
        assert "[OUTPUT]" in text
        assert "result: 42" in text

    def test_error_with_hint(self) -> None:
        feedback = StepFeedback(
            stdout="",
            error="KeyError: 'x'",
            hint="Use safe_get(d, key) for dict access",
        )
        text = format_feedback(feedback)
        assert "[ERROR]" in text
        assert "KeyError" in text
        assert "[HINT]" in text
        assert "safe_get" in text

    def test_multiple_blocks_violation(self) -> None:
        feedback = StepFeedback(
            violation="multiple_blocks:3",
            stdout="",
        )
        text = format_feedback(feedback)
        assert "[VIOLATION]" in text
        assert "3 code blocks" in text
        assert "first was executed" in text.lower()

    def test_pattern_warnings(self) -> None:
        feedback = StepFeedback(
            stdout="",
            pattern_warnings=["llm_query in loop", "passing full data"],
        )
        text = format_feedback(feedback)
        assert text.count("[PATTERN]") == 2
        assert "llm_query in loop" in text
        assert "passing full data" in text

    def test_rejection_reasons(self) -> None:
        feedback = StepFeedback(
            rejection_reasons=["missing_substring:alpha", "min_word_count:10"]
        )
        text = format_feedback(feedback)
        assert "[REJECTED]" in text
        assert "missing_substring:alpha" in text
        assert "min_word_count:10" in text


class TestStopHook:
    """Test stop-hook behavior for FINAL verification."""

    def test_rejects_then_retries(self) -> None:
        outputs = [
            """
```python
FINAL("bad answer")
```
""".strip(),
            """
```python
FINAL("good answer")
```
""".strip(),
        ]
        provider = MockProvider(main_outputs=outputs)
        budget = Budget(max_tokens=120, max_total_tokens=300)
        criteria = SuccessCriteria(required_substrings=["good"])
        task = TaskSpec(
            task_id="stop-hook",
            input_text="Return a good answer.",
            model="mock-model",
            budget=budget,
            success_criteria=criteria,
        )
        engine = RLMEngine(max_steps=3)
        report = engine.run(task, provider, data="context")

        assert report.success
        assert report.answer == "good answer"
        assert len(report.steps) == 2
        assert report.errors == []
        assert "[REJECTED]" in report.steps[0].prompt


class TestSandboxSafeHelpers:
    """Test safe helpers injection into sandbox."""

    def test_safe_helpers_injected_by_default(self) -> None:
        sandbox = PythonSandbox(
            data="test",
            llm_query=lambda x: x,
        )
        # Safe helpers should be in namespace
        assert sandbox.get_global("safe_get") is not None
        assert sandbox.get_global("safe_rows") is not None
        assert sandbox.get_global("safe_sort") is not None

    def test_safe_helpers_can_be_disabled(self) -> None:
        sandbox = PythonSandbox(
            data="test",
            llm_query=lambda x: x,
            inject_safe_helpers=False,
        )
        assert sandbox.get_global("safe_get") is None
        assert sandbox.get_global("safe_rows") is None
        assert sandbox.get_global("safe_sort") is None

    def test_safe_get_works_in_sandbox(self) -> None:
        sandbox = PythonSandbox(
            data={"key": "value"},
            llm_query=lambda x: x,
        )
        result = sandbox.exec("""
result = safe_get(data, 'key', 'default')
missing = safe_get(data, 'missing', 'default')
none_case = safe_get(None, 'key', 'default')
print(f"{result},{missing},{none_case}")
""")
        assert result.error is None
        assert "value,default,default" in result.stdout

    def test_safe_rows_works_in_sandbox(self) -> None:
        sandbox = PythonSandbox(
            data={"rows": [1, 2, 3]},
            llm_query=lambda x: x,
        )
        result = sandbox.exec("""
rows = safe_rows(data)
print(len(rows))
""")
        assert result.error is None
        assert "3" in result.stdout

    def test_safe_sort_works_in_sandbox(self) -> None:
        sandbox = PythonSandbox(
            data={"rows": [{"v": 2}, {"v": 1}, {"v": 3}]},
            llm_query=lambda x: x,
        )
        result = sandbox.exec("""
sorted_data = safe_sort(data, key='v', reverse=False)
print([x['v'] for x in sorted_data])
""")
        assert result.error is None
        assert "[1, 2, 3]" in result.stdout


class TestExecCodeFunctional:
    """Test functional core of sandbox execution."""

    def test_exec_code_returns_tuple(self) -> None:
        namespace = build_namespace(
            data="test",
            llm_query=lambda x: x,
            allowed_imports=set(),
        )
        result, updated_ns = exec_code(
            code="x = 42\nprint(x)",
            namespace=namespace,
            output_limit=1000,
            timeout_seconds=None,
        )
        assert result.stdout.strip() == "42"
        assert result.error is None
        assert updated_ns["x"] == 42

    def test_exec_code_captures_error(self) -> None:
        namespace = build_namespace(
            data="test",
            llm_query=lambda x: x,
            allowed_imports=set(),
        )
        result, _ = exec_code(
            code="x = 1 / 0",
            namespace=namespace,
            output_limit=1000,
            timeout_seconds=None,
        )
        assert result.error is not None
        assert "division" in result.error.lower()

    def test_namespace_persists_across_calls(self) -> None:
        namespace = build_namespace(
            data="test",
            llm_query=lambda x: x,
            allowed_imports=set(),
        )
        # First call
        _, namespace = exec_code(
            code="counter = 1",
            namespace=namespace,
            output_limit=1000,
            timeout_seconds=None,
        )
        # Second call
        result, namespace = exec_code(
            code="counter += 1\nprint(counter)",
            namespace=namespace,
            output_limit=1000,
            timeout_seconds=None,
        )
        assert result.stdout.strip() == "2"


class TestSafeHelperFunctions:
    """Test safe helper functions directly."""

    def test_safe_get_normal(self) -> None:
        d = {"a": 1, "b": 2}
        assert safe_get(d, "a") == 1
        assert safe_get(d, "c", "default") == "default"

    def test_safe_get_none_input(self) -> None:
        assert safe_get(None, "key", "default") == "default"

    def test_safe_get_non_dict(self) -> None:
        assert safe_get("string", "key", "default") == "default"
        assert safe_get([1, 2, 3], "key", "default") == "default"

    def test_safe_rows_list(self) -> None:
        assert safe_rows([1, 2, 3]) == [1, 2, 3]

    def test_safe_rows_dict_with_rows(self) -> None:
        assert safe_rows({"rows": [1, 2]}) == [1, 2]
        assert safe_rows({"items": [3, 4]}) == [3, 4]
        assert safe_rows({"data": [5, 6]}) == [5, 6]
        assert safe_rows({"results": [7, 8]}) == [7, 8]

    def test_safe_rows_none(self) -> None:
        assert safe_rows(None) == []

    def test_safe_rows_non_container(self) -> None:
        assert safe_rows("string") == []
        assert safe_rows(123) == []

    def test_safe_sort_normal(self) -> None:
        data = {"rows": [{"v": 2}, {"v": 1}, {"v": 3}]}
        result = safe_sort(data, key="v", reverse=False)
        assert [x["v"] for x in result] == [1, 2, 3]

    def test_safe_sort_reverse(self) -> None:
        data = {"rows": [{"v": 2}, {"v": 1}, {"v": 3}]}
        result = safe_sort(data, key="v", reverse=True)
        assert [x["v"] for x in result] == [3, 2, 1]

    def test_safe_sort_missing_key(self) -> None:
        data = {"rows": [{"v": 2}, {"x": 1}, {"v": 3}]}
        # Should not crash, treats missing as 0
        result = safe_sort(data, key="v", reverse=True)
        assert len(result) == 3

    def test_safe_sort_none_input(self) -> None:
        assert safe_sort(None, key="v") == []

    def test_safe_sort_no_key(self) -> None:
        data = [1, 2, 3]
        assert safe_sort(data) == [1, 2, 3]  # Returns as-is


class TestBudgetPercentage:
    """Test budget percentage tracking in _BudgetTracker."""

    def test_percentage_output_tokens(self) -> None:
        budget = Budget(max_tokens=1000)
        tracker = _BudgetTracker(budget)
        tracker.consume({"output_tokens": 250})
        pct = tracker.percentage_used()
        assert pct["output_tokens"] == 25

    def test_percentage_total_tokens(self) -> None:
        budget = Budget(max_total_tokens=2000)
        tracker = _BudgetTracker(budget)
        tracker.consume({"total_tokens": 1000})
        pct = tracker.percentage_used()
        assert pct["total_tokens"] == 50

    def test_percentage_cost_usd(self) -> None:
        budget = Budget(max_cost_usd=1.0)
        tracker = _BudgetTracker(budget)
        tracker.consume({"cost_usd": 0.8})
        pct = tracker.percentage_used()
        assert pct["cost_usd"] == 80

    def test_percentage_multiple_dimensions(self) -> None:
        budget = Budget(max_tokens=1000, max_total_tokens=2000)
        tracker = _BudgetTracker(budget)
        tracker.consume({"output_tokens": 500, "total_tokens": 800})
        pct = tracker.percentage_used()
        assert pct["output_tokens"] == 50
        assert pct["total_tokens"] == 40

    def test_percentage_empty_when_no_limits(self) -> None:
        budget = Budget(max_seconds=60.0)
        tracker = _BudgetTracker(budget)
        tracker.consume({"output_tokens": 500})
        pct = tracker.percentage_used()
        assert pct == {}

    def test_percentage_accumulates_across_calls(self) -> None:
        budget = Budget(max_tokens=1000)
        tracker = _BudgetTracker(budget)
        tracker.consume({"output_tokens": 100})
        tracker.consume({"output_tokens": 200})
        tracker.consume({"output_tokens": 300})
        pct = tracker.percentage_used()
        assert pct["output_tokens"] == 60


class TestStrategyHints:
    """Test strategy hints injection based on context size."""

    def test_small_data_no_strategy_hints(self) -> None:
        """Context < 10K chars should not include strategy hints."""
        from enzu import Budget, SuccessCriteria, TaskSpec

        task = TaskSpec(
            task_id="test",
            input_text="Test task",
            model="test-model",
            budget=Budget(max_tokens=100),
            success_criteria=SuccessCriteria(required_substrings=["test"]),
        )
        engine = RLMEngine(prompt_style="extended")
        prompt = engine._system_prompt(task, data_len=5000)
        assert "Strategy by Context Size" not in prompt
        assert "Anti-patterns (costly)" not in prompt

    def test_large_data_includes_strategy_hints(self) -> None:
        """Context >= 10K chars should include strategy hints."""
        from enzu import Budget, SuccessCriteria, TaskSpec

        task = TaskSpec(
            task_id="test",
            input_text="Test task",
            model="test-model",
            budget=Budget(max_tokens=100),
            success_criteria=SuccessCriteria(required_substrings=["test"]),
        )
        engine = RLMEngine(prompt_style="extended")
        prompt = engine._system_prompt(task, data_len=15000)
        assert "Strategy by Context Size" in prompt
        assert "Anti-patterns (costly)" in prompt

    def test_boundary_10k_includes_hints(self) -> None:
        """Exactly 10K chars should include strategy hints."""
        from enzu import Budget, SuccessCriteria, TaskSpec

        task = TaskSpec(
            task_id="test",
            input_text="Test task",
            model="test-model",
            budget=Budget(max_tokens=100),
            success_criteria=SuccessCriteria(required_substrings=["test"]),
        )
        engine = RLMEngine(prompt_style="extended")
        prompt = engine._system_prompt(task, data_len=10000)
        assert "Strategy by Context Size" in prompt


class TestBudgetAwarenessInPrompt:
    """Test budget percentage in prompt during RLM loop."""

    def test_advance_prompt_includes_budget_percentage(self) -> None:
        feedback = StepFeedback(stdout="output")
        budget_pct = {"output_tokens": 45}

        prompt = StepRunner._advance_prompt(
            prompt="initial",
            model_output="some code",
            feedback=feedback,
            budget_pct=budget_pct,
        )
        assert "45% of the token budget has been consumed" in prompt

    def test_advance_prompt_80_percent_wrap_up_warning(self) -> None:
        feedback = StepFeedback(stdout="output")
        budget_pct = {"output_tokens": 85}

        prompt = StepRunner._advance_prompt(
            prompt="initial",
            model_output="some code",
            feedback=feedback,
            budget_pct=budget_pct,
        )
        assert "85% of the token budget has been consumed" in prompt
        assert "WRAP UP SOON" in prompt

    def test_advance_prompt_50_percent_efficient_warning(self) -> None:
        feedback = StepFeedback(stdout="output")
        budget_pct = {"output_tokens": 55}

        prompt = StepRunner._advance_prompt(
            prompt="initial",
            model_output="some code",
            feedback=feedback,
            budget_pct=budget_pct,
        )
        assert "55% of the token budget has been consumed" in prompt
        assert "be efficient" in prompt
        assert "WRAP UP SOON" not in prompt

    def test_advance_prompt_low_budget_no_warning(self) -> None:
        feedback = StepFeedback(stdout="output")
        budget_pct = {"output_tokens": 25}

        prompt = StepRunner._advance_prompt(
            prompt="initial",
            model_output="some code",
            feedback=feedback,
            budget_pct=budget_pct,
        )
        assert "25% of the token budget has been consumed" in prompt
        assert "WRAP UP SOON" not in prompt
        assert "be efficient" not in prompt

    def test_advance_prompt_empty_budget_no_line(self) -> None:
        feedback = StepFeedback(stdout="output")

        prompt = StepRunner._advance_prompt(
            prompt="initial",
            model_output="some code",
            feedback=feedback,
            budget_pct={},
        )
        assert "Budget used:" not in prompt

    def test_advance_prompt_multiple_dimensions_uses_max(self) -> None:
        feedback = StepFeedback(stdout="output")
        budget_pct = {"output_tokens": 30, "total_tokens": 70, "cost_usd": 50}

        prompt = StepRunner._advance_prompt(
            prompt="initial",
            model_output="some code",
            feedback=feedback,
            budget_pct=budget_pct,
        )
        assert "70% of the token budget has been consumed" in prompt
        assert "be efficient" in prompt
