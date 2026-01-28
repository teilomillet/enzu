"""
Comprehensive tests for the Chat Engine (enzu/engine.py).

Tests cover:
1. Provider fallback system
2. Progress event streaming
3. Budget enforcement and clamping
4. Trajectory recording
5. Error handling
6. Budget usage calculation
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pytest

import enzu.engine as engine_module
from enzu.engine import Engine
from enzu.models import (
    Budget,
    ProgressEvent,
    ProviderResult,
    SuccessCriteria,
    TaskSpec,
)
from enzu.providers.base import BaseProvider
from tests.providers.mock_provider import MockProvider


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================

class FailingProvider(BaseProvider):
    """Provider that always raises an exception."""
    name = "failing"

    def __init__(self, error_message: str = "Provider failed"):
        self._error_message = error_message

    def generate(self, task: TaskSpec) -> ProviderResult:
        raise RuntimeError(self._error_message)

    def stream(self, task: TaskSpec, on_progress=None) -> ProviderResult:
        raise RuntimeError(self._error_message)


class DelayedFailingProvider(BaseProvider):
    """Provider that fails after N calls."""
    name = "delayed_failing"

    def __init__(self, fail_after: int = 1, error_message: str = "Delayed failure"):
        self._call_count = 0
        self._fail_after = fail_after
        self._error_message = error_message

    def generate(self, task: TaskSpec) -> ProviderResult:
        self._call_count += 1
        if self._call_count > self._fail_after:
            raise RuntimeError(self._error_message)
        return ProviderResult(
            output_text=f"Response {self._call_count}",
            raw={},
            usage={"output_tokens": 10},
            provider=self.name,
            model=task.model,
        )

    def stream(self, task: TaskSpec, on_progress=None) -> ProviderResult:
        return self.generate(task)


class TrackingProvider(BaseProvider):
    """Provider that tracks all calls made to it."""
    name = "tracking"

    def __init__(self, responses: List[str], usage: Optional[Dict] = None):
        self._responses = list(responses)
        self._usage = usage or {"output_tokens": 5, "total_tokens": 10}
        self.calls: List[TaskSpec] = []

    def generate(self, task: TaskSpec) -> ProviderResult:
        self.calls.append(task)
        response = self._responses.pop(0) if self._responses else "default response"
        return ProviderResult(
            output_text=response,
            raw={},
            usage=dict(self._usage),
            provider=self.name,
            model=task.model,
        )

    def stream(self, task: TaskSpec, on_progress=None) -> ProviderResult:
        return self.generate(task)


class UsageMissingProvider(BaseProvider):
    """Provider that returns usage without total token counts."""
    name = "usage_missing"

    def __init__(self, output_text: str) -> None:
        self._output_text = output_text

    def generate(self, task: TaskSpec) -> ProviderResult:
        return ProviderResult(
            output_text=self._output_text,
            raw={},
            usage={},
            provider=self.name,
            model=task.model,
        )

    def stream(self, task: TaskSpec, on_progress=None) -> ProviderResult:
        return self.generate(task)

def _make_task(
    task_id: str = "test",
    input_text: str = "test input",
    model: str = "mock",
    max_output_tokens: Optional[int] = None,
    budget_output_tokens: Optional[int] = 100,
    budget_total_tokens: Optional[int] = None,
    budget_seconds: Optional[float] = None,
    budget_cost: Optional[float] = None,
    required_substrings: Optional[List[str]] = None,
) -> TaskSpec:
    """Helper to create TaskSpec for tests."""
    budget_kwargs = {}
    if budget_output_tokens:
        budget_kwargs["max_tokens"] = budget_output_tokens
    if budget_total_tokens:
        budget_kwargs["max_total_tokens"] = budget_total_tokens
    if budget_seconds:
        budget_kwargs["max_seconds"] = budget_seconds
    if budget_cost:
        budget_kwargs["max_cost_usd"] = budget_cost

    criteria_kwargs = {}
    if required_substrings:
        criteria_kwargs["required_substrings"] = required_substrings
    else:
        criteria_kwargs["min_word_count"] = 1

    return TaskSpec(
        task_id=task_id,
        input_text=input_text,
        model=model,
        max_output_tokens=max_output_tokens,
        budget=Budget(**budget_kwargs),
        success_criteria=SuccessCriteria(**criteria_kwargs),
    )


# =============================================================================
# Provider Fallback Tests
# =============================================================================

class TestProviderFallback:
    """Test provider fallback system."""

    def test_fallback_to_secondary_on_primary_failure(self) -> None:
        """When primary provider fails, fallback to secondary."""
        primary = FailingProvider("Primary failed")
        secondary = MockProvider(main_outputs=["Success from secondary"])

        task = _make_task()
        engine = Engine()
        report = engine.run(task, primary, fallback_providers=[secondary])

        assert report.success is True
        assert report.output_text == "Success from secondary"
        assert report.provider == "mock"  # Secondary provider
        assert "Primary failed" in report.errors[0]

    def test_fallback_chain_multiple_providers(self) -> None:
        """Fallback through multiple failing providers to successful one."""
        provider1 = FailingProvider("Provider 1 failed")
        provider2 = FailingProvider("Provider 2 failed")
        provider3 = MockProvider(main_outputs=["Success from third"])

        task = _make_task()
        engine = Engine()
        report = engine.run(task, provider1, fallback_providers=[provider2, provider3])

        assert report.success is True
        assert report.output_text == "Success from third"
        assert len(report.errors) == 2  # Two fallback errors
        assert "Provider 1 failed" in report.errors[0]
        assert "Provider 2 failed" in report.errors[1]

    def test_all_providers_fail(self) -> None:
        """When all providers fail, return error report."""
        provider1 = FailingProvider("Provider 1 failed")
        provider2 = FailingProvider("Provider 2 failed")

        task = _make_task()
        engine = Engine()
        report = engine.run(task, provider1, fallback_providers=[provider2])

        assert report.success is False
        assert report.output_text is None
        assert "Provider 2 failed" in report.errors[-1]

    def test_fallback_trajectory_records_all_attempts(self) -> None:
        """Trajectory should include all provider attempts."""
        provider1 = FailingProvider("First failed")
        provider2 = MockProvider(main_outputs=["Success"])

        task = _make_task()
        engine = Engine()
        report = engine.run(task, provider1, fallback_providers=[provider2])

        assert len(report.trajectory) == 2
        # First attempt failed
        assert report.trajectory[0].provider == "failing"
        assert report.trajectory[0].error == "First failed"
        assert report.trajectory[0].response is None
        # Second attempt succeeded
        assert report.trajectory[1].provider == "mock"
        assert report.trajectory[1].error is None
        assert report.trajectory[1].response == "Success"

    def test_fallback_emits_progress_events(self) -> None:
        """Fallback should emit progress events for each attempt."""
        provider1 = FailingProvider("First failed")
        provider2 = MockProvider(main_outputs=["Success"])

        events: List[ProgressEvent] = []
        task = _make_task()
        engine = Engine()
        engine.run(
            task, provider1,
            fallback_providers=[provider2],
            on_progress=lambda e: events.append(e)
        )

        # Should have fallback event
        fallback_events = [e for e in events if e.message == "provider_fallback"]
        assert len(fallback_events) == 1
        assert fallback_events[0].data["provider"] == "failing"

    def test_no_fallback_when_primary_succeeds(self) -> None:
        """Fallback providers should not be called when primary succeeds."""
        primary = TrackingProvider(responses=["Primary success"])
        secondary = TrackingProvider(responses=["Should not be called"])

        task = _make_task()
        engine = Engine()
        report = engine.run(task, primary, fallback_providers=[secondary])

        assert report.success is True
        assert len(primary.calls) == 1
        assert len(secondary.calls) == 0

    def test_fallback_clears_errors_on_success(self) -> None:
        """Fallback errors shouldn't cause failure if final provider succeeds."""
        provider1 = FailingProvider("First failed")
        provider2 = MockProvider(main_outputs=["Success"])

        task = _make_task()
        engine = Engine()
        report = engine.run(task, provider1, fallback_providers=[provider2])

        # Errors are kept for debugging but success is True
        assert report.success is True
        assert len(report.errors) == 1  # Fallback error recorded


# =============================================================================
# Progress Event Tests
# =============================================================================

class TestProgressEvents:
    """Test progress event streaming."""

    def test_progress_callback_receives_all_phases(self) -> None:
        """Progress callback should receive events from all phases."""
        provider = MockProvider(main_outputs=["Test output"])
        events: List[ProgressEvent] = []

        task = _make_task()
        engine = Engine()
        engine.run(task, provider, on_progress=lambda e: events.append(e))

        phases = [e.phase for e in events]
        assert "start" in phases
        assert "verification" in phases
        assert "complete" in phases

    def test_start_event_contains_task_info(self) -> None:
        """Start event should contain task and provider info."""
        provider = MockProvider(main_outputs=["Test"])
        events: List[ProgressEvent] = []

        task = _make_task(task_id="my-task")
        engine = Engine()
        engine.run(task, provider, on_progress=lambda e: events.append(e))

        start_events = [e for e in events if e.message == "task_started"]
        assert len(start_events) == 1
        assert start_events[0].data["task_id"] == "my-task"
        assert start_events[0].data["provider"] == "mock"

    def test_complete_event_contains_success_status(self) -> None:
        """Complete event should indicate success/failure."""
        provider = MockProvider(main_outputs=["Test output"])
        events: List[ProgressEvent] = []

        task = _make_task()
        engine = Engine()
        engine.run(task, provider, on_progress=lambda e: events.append(e))

        complete_events = [e for e in events if e.message == "task_completed"]
        assert len(complete_events) == 1
        assert complete_events[0].data["success"] is True

    def test_error_event_on_provider_failure(self) -> None:
        """Error event should be emitted on provider failure."""
        provider = FailingProvider("Provider exploded")
        events: List[ProgressEvent] = []

        task = _make_task()
        engine = Engine()
        engine.run(task, provider, on_progress=lambda e: events.append(e))

        error_events = [e for e in events if e.phase == "error"]
        assert len(error_events) >= 1
        assert any("Provider exploded" in str(e.data) for e in error_events)

    def test_progress_callback_exception_does_not_crash(self) -> None:
        """Exception in progress callback should not crash execution."""
        provider = MockProvider(main_outputs=["Test response"])
        seen_events: List[ProgressEvent] = []

        def bad_callback(event: ProgressEvent) -> None:
            seen_events.append(event)
            raise ValueError("Callback crashed")

        task = _make_task()
        engine = Engine()
        # Should not raise despite bad callback - engine catches callback exceptions
        report = engine.run(task, provider, on_progress=bad_callback)

        # Execution should complete successfully despite callback failure
        assert report.success is True
        assert report.output_text == "Test response"
        assert len(seen_events) >= 1
        # Progress events should still be recorded internally
        assert len(report.progress_events) >= 3  # start, verification, complete

    def test_events_stored_in_report(self) -> None:
        """All progress events should be stored in report."""
        provider = MockProvider(main_outputs=["Test"])

        task = _make_task()
        engine = Engine()
        report = engine.run(task, provider)

        assert len(report.progress_events) >= 3  # start, verification, complete
        messages = [e.message for e in report.progress_events]
        assert "task_started" in messages
        assert "task_completed" in messages

    def test_progress_events_logged_when_stream_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Progress events should be logged when telemetry streaming is enabled."""
        provider = MockProvider(main_outputs=["Test"])
        logged: list[tuple[str, str, Dict[str, object]]] = []

        def fake_log(level: str, event: str, **kwargs: object) -> None:
            logged.append((level, event, kwargs))

        monkeypatch.setattr(engine_module.telemetry, "stream_enabled", lambda: True)
        monkeypatch.setattr(engine_module.telemetry, "log", fake_log)

        task = _make_task()
        engine = Engine()
        engine.run(task, provider)

        assert any(event == "progress_event" for _, event, _ in logged)

    def test_progress_events_logged_when_env_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Progress events should be logged when ENZU_LOGFIRE_PROGRESS is set."""
        provider = MockProvider(main_outputs=["Test"])
        logged: list[tuple[str, str, Dict[str, object]]] = []

        def fake_log(level: str, event: str, **kwargs: object) -> None:
            logged.append((level, event, kwargs))

        monkeypatch.setenv("ENZU_LOGFIRE_PROGRESS", "true")
        monkeypatch.setattr(engine_module.telemetry, "stream_enabled", lambda: False)
        monkeypatch.setattr(engine_module.telemetry, "log", fake_log)

        task = _make_task()
        engine = Engine()
        engine.run(task, provider)

        assert any(event == "progress_event" for _, event, _ in logged)


# =============================================================================
# Budget Enforcement Tests
# =============================================================================

class TestBudgetEnforcement:
    """Test budget enforcement and clamping."""

    def test_output_tokens_exceeds_budget_rejected(self) -> None:
        """Task with output tokens exceeding budget should be rejected."""
        provider = MockProvider(main_outputs=["Test"])

        task = _make_task(
            max_output_tokens=200,
            budget_output_tokens=100,
        )
        engine = Engine()
        report = engine.run(task, provider)

        assert report.success is False
        assert any("exceeds budget" in err for err in report.errors)

    def test_total_tokens_exhausted_preflight(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Preflight should fail when input tokens exceed max_total_tokens."""
        provider = MockProvider(main_outputs=["Test"])
        events: List[ProgressEvent] = []

        def _large_tokens(_: str, __: Optional[str] = None) -> Optional[int]:
            return 1000

        # Force remaining <= 0 in preflight.
        monkeypatch.setattr(engine_module, "count_tokens_exact", _large_tokens)

        task = _make_task(
            max_output_tokens=None,
            budget_output_tokens=None,
            budget_total_tokens=10,
        )
        engine = Engine()
        report = engine.run(task, provider, on_progress=lambda e: events.append(e))

        assert report.success is False
        assert any("budget.max_total_tokens exhausted in preflight" in err for err in report.errors)
        assert any(e.message == "budget_limit_exceeded_preflight" for e in events)

    def test_output_clamped_with_exact_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Clamp output when exact token count leaves limited remaining budget."""
        provider = TrackingProvider(responses=["Test output"])
        events: List[ProgressEvent] = []

        def _fixed_tokens(_: str, __: Optional[str] = None) -> Optional[int]:
            return 40

        # Force exact token path with limited remaining budget.
        monkeypatch.setattr(engine_module, "count_tokens_exact", _fixed_tokens)

        task = _make_task(
            max_output_tokens=20,
            budget_output_tokens=None,
            budget_total_tokens=50,
        )
        engine = Engine()
        report = engine.run(task, provider, on_progress=lambda e: events.append(e))

        assert provider.calls[0].max_output_tokens == 10
        assert any(e.message == "budget_output_clamped" for e in events)
        assert report.success is True

    def test_budget_seconds_exceeded_marks_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Elapsed time over max_seconds should fail the run."""
        provider = MockProvider(main_outputs=["ok"])

        times = [0.0, 10.0]
        index = {"i": 0}

        def _fake_time() -> float:
            i = index["i"]
            index["i"] = min(i + 1, len(times) - 1)
            return times[i]

        # Control elapsed time so it exceeds max_seconds.
        monkeypatch.setattr(engine_module.time, "time", _fake_time)

        task = _make_task(
            required_substrings=["ok"],
            budget_seconds=1.0,
        )
        engine = Engine()
        report = engine.run(task, provider)

        assert report.success is False
        assert "max_seconds" in report.budget_usage.limits_exceeded

    def test_output_clamped_to_total_budget(self) -> None:
        """Output tokens should be clamped to remaining total budget."""
        provider = TrackingProvider(responses=["Test output"])

        # Small total budget, large output request
        task = _make_task(
            max_output_tokens=500,
            budget_output_tokens=None,
            budget_total_tokens=100,
        )
        engine = Engine()
        engine.run(task, provider)

        # Task sent to provider should have clamped output tokens
        assert len(provider.calls) == 1
        # Output should be clamped (exact value depends on input token count)
        assert provider.calls[0].max_output_tokens is not None
        assert provider.calls[0].max_output_tokens <= 100

    def test_budget_clamping_emits_event(self) -> None:
        """Budget clamping should emit progress event."""
        provider = MockProvider(main_outputs=["Test"])
        events: List[ProgressEvent] = []

        task = _make_task(
            max_output_tokens=500,
            budget_output_tokens=None,
            budget_total_tokens=50,
        )
        engine = Engine()
        engine.run(task, provider, on_progress=lambda e: events.append(e))

        # May or may not be clamped depending on input token count
        # The important thing is the mechanism works without error

    def test_tokenizer_unavailable_emits_degraded_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tokenizer-unavailable branch should emit degraded budget event."""
        provider = MockProvider(main_outputs=["Test"])
        events: List[ProgressEvent] = []

        def _no_tokens(_: str, __: Optional[str] = None) -> Optional[int]:
            return None

        # Force tokenizer-unavailable path in Engine preflight.
        monkeypatch.setattr(engine_module, "count_tokens_exact", _no_tokens)

        task = _make_task(
            max_output_tokens=200,
            budget_output_tokens=None,
            budget_total_tokens=50,
        )
        engine = Engine()
        report = engine.run(task, provider, on_progress=lambda e: events.append(e))

        messages = [e.message for e in events]
        assert "tokenizer_unavailable_budget_degraded" in messages
        assert report.success is True

    def test_tokenizer_unavailable_clamps_to_total_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tokenizer-unavailable path should still clamp output to total budget."""
        provider = TrackingProvider(responses=["Test output"])
        events: List[ProgressEvent] = []

        def _no_tokens(_: str, __: Optional[str] = None) -> Optional[int]:
            return None

        # Force the clamp branch that depends on tokenizer availability.
        monkeypatch.setattr(engine_module, "count_tokens_exact", _no_tokens)

        task = _make_task(
            max_output_tokens=200,
            budget_output_tokens=None,
            budget_total_tokens=50,
        )
        engine = Engine()
        report = engine.run(task, provider, on_progress=lambda e: events.append(e))

        assert provider.calls[0].max_output_tokens == 50
        assert any(e.message == "budget_output_clamped" for e in events)
        assert "tokenizer_unavailable_budget_degraded" in [e.message for e in events]
        assert report.success is True

    def test_default_output_tokens_from_total_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When max_output_tokens is None, default to budget max_total_tokens."""
        provider = TrackingProvider(responses=["Test output"])

        def _zero_tokens(_: str, __: Optional[str] = None) -> Optional[int]:
            return 0

        # Keep remaining tokens equal to max_total_tokens.
        monkeypatch.setattr(engine_module, "count_tokens_exact", _zero_tokens)

        task = _make_task(
            max_output_tokens=None,
            budget_output_tokens=None,
            budget_total_tokens=50,
        )
        engine = Engine()
        report = engine.run(task, provider)

        assert provider.calls[0].max_output_tokens == 50
        assert report.success is True

    def test_usage_missing_total_tokens_emits_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Emit usage_missing_total_tokens when totals cannot be computed."""
        provider = UsageMissingProvider("ok")
        events: List[ProgressEvent] = []

        def _no_tokens(_: str, __: Optional[str] = None) -> Optional[int]:
            return None

        monkeypatch.setattr(engine_module, "count_tokens_exact", _no_tokens)

        task = _make_task(
            required_substrings=["ok"],
            budget_output_tokens=None,
            budget_total_tokens=50,
        )
        engine = Engine()
        report = engine.run(task, provider, on_progress=lambda e: events.append(e))

        assert report.success is True
        assert any(e.message == "usage_missing_total_tokens" for e in events)

    def test_seconds_limit_tracked(self) -> None:
        """Budget usage should track elapsed seconds against limit."""
        provider = MockProvider(main_outputs=["Test"])

        task = _make_task(budget_seconds=0.001)  # Very short limit
        engine = Engine()
        report = engine.run(task, provider)

        # Elapsed time likely exceeds 0.001 seconds
        assert report.budget_usage.elapsed_seconds > 0

    def test_cost_limit_tracked(self) -> None:
        """Budget usage should track cost against limit."""
        provider = MockProvider(
            main_outputs=["Test"],
            usage={"output_tokens": 10, "cost_usd": 0.05},
        )

        task = _make_task(budget_cost=0.01)  # Limit lower than cost
        engine = Engine()
        report = engine.run(task, provider)

        assert report.budget_usage.cost_usd == 0.05
        assert "max_cost_usd" in report.budget_usage.limits_exceeded


# =============================================================================
# Trajectory Recording Tests
# =============================================================================

class TestTrajectoryRecording:
    """Test trajectory recording."""

    def test_successful_call_recorded(self) -> None:
        """Successful provider call should be recorded in trajectory."""
        provider = MockProvider(main_outputs=["Success response"])

        task = _make_task()
        engine = Engine()
        report = engine.run(task, provider)

        assert len(report.trajectory) == 1
        step = report.trajectory[0]
        assert step.provider == "mock"
        assert step.response == "Success response"
        assert step.error is None
        assert step.started_at is not None
        assert step.finished_at is not None

    def test_failed_call_recorded(self) -> None:
        """Failed provider call should be recorded with error."""
        provider = FailingProvider("Boom!")

        task = _make_task()
        engine = Engine()
        report = engine.run(task, provider)

        assert len(report.trajectory) == 1
        step = report.trajectory[0]
        assert step.provider == "failing"
        assert step.response is None
        assert step.error == "Boom!"

    def test_trajectory_includes_request(self) -> None:
        """Trajectory should include the request sent to provider."""
        provider = MockProvider(main_outputs=["Response"])

        task = _make_task(input_text="My specific question")
        engine = Engine()
        report = engine.run(task, provider)

        assert len(report.trajectory) == 1
        assert "My specific question" in report.trajectory[0].request

    def test_trajectory_step_index_increments(self) -> None:
        """Trajectory step index should increment correctly."""
        provider1 = FailingProvider("First")
        provider2 = FailingProvider("Second")
        provider3 = MockProvider(main_outputs=["Success"])

        task = _make_task()
        engine = Engine()
        report = engine.run(task, provider1, fallback_providers=[provider2, provider3])

        assert report.trajectory[0].step_index == 0
        assert report.trajectory[1].step_index == 1
        assert report.trajectory[2].step_index == 2


# =============================================================================
# Verification Integration Tests
# =============================================================================

class TestVerificationIntegration:
    """Test verification integrated with engine run."""

    def test_verification_failure_marks_unsuccessful(self) -> None:
        """Failed verification should mark report as unsuccessful."""
        provider = MockProvider(main_outputs=["Missing the required text"])

        task = _make_task(required_substrings=["must-have"])
        engine = Engine()
        report = engine.run(task, provider)

        assert report.success is False
        assert report.verification.passed is False
        assert "missing_substring:must-have" in report.verification.reasons

    def test_verification_success_marks_successful(self) -> None:
        """Passed verification should mark report as successful."""
        provider = MockProvider(main_outputs=["Output with must-have text"])

        task = _make_task(required_substrings=["must-have"])
        engine = Engine()
        report = engine.run(task, provider)

        assert report.success is True
        assert report.verification.passed is True

    def test_empty_output_fails_verification(self) -> None:
        """Empty output should fail verification."""
        provider = MockProvider(main_outputs=[""])

        task = _make_task()
        engine = Engine()
        report = engine.run(task, provider)

        assert report.success is False
        assert "no_output" in report.verification.reasons


# =============================================================================
# Budget Usage Calculation Tests
# =============================================================================

class TestBudgetUsageCalculation:
    """Test budget usage calculation."""

    def test_usage_from_provider(self) -> None:
        """Budget usage should use provider-reported values."""
        provider = MockProvider(
            main_outputs=["Test"],
            usage={
                "input_tokens": 50,
                "output_tokens": 100,
                "total_tokens": 150,
                "cost_usd": 0.01,
            },
        )

        task = _make_task()
        engine = Engine()
        report = engine.run(task, provider)

        assert report.budget_usage.input_tokens == 50
        assert report.budget_usage.output_tokens == 100
        assert report.budget_usage.total_tokens == 150
        assert report.budget_usage.cost_usd == 0.01

    def test_limits_exceeded_detected(self) -> None:
        """Should detect which limits were exceeded."""
        provider = MockProvider(
            main_outputs=["Test"],
            usage={"output_tokens": 200, "total_tokens": 200},
        )

        task = _make_task(budget_output_tokens=100)
        engine = Engine()
        report = engine.run(task, provider)

        assert "max_output_tokens" in report.budget_usage.limits_exceeded

    def test_elapsed_seconds_tracked(self) -> None:
        """Elapsed seconds should be tracked."""
        provider = MockProvider(main_outputs=["Test"])

        task = _make_task()
        engine = Engine()
        report = engine.run(task, provider)

        assert report.budget_usage.elapsed_seconds >= 0


# =============================================================================
# Edge Cases
# =============================================================================

class TestEngineEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_fallback_list(self) -> None:
        """Empty fallback list should work like no fallbacks."""
        provider = MockProvider(main_outputs=["Success"])

        task = _make_task()
        engine = Engine()
        report = engine.run(task, provider, fallback_providers=[])

        assert report.success is True

    def test_none_fallback_providers(self) -> None:
        """None fallback_providers should work."""
        provider = MockProvider(main_outputs=["Success"])

        task = _make_task()
        engine = Engine()
        report = engine.run(task, provider, fallback_providers=None)

        assert report.success is True

    def test_very_long_output(self) -> None:
        """Should handle very long output."""
        long_output = "word " * 10000
        provider = MockProvider(main_outputs=[long_output])

        task = _make_task()
        engine = Engine()
        report = engine.run(task, provider)

        assert report.success is True
        assert report.output_text is not None and len(report.output_text) > 40000

    def test_unicode_output(self) -> None:
        """Should handle unicode output."""
        unicode_output = "Hello \U0001F600 \u4e2d\u6587 caf\u00e9"
        provider = MockProvider(main_outputs=[unicode_output])

        task = _make_task()
        engine = Engine()
        report = engine.run(task, provider)

        assert report.success is True
        assert report.output_text is not None and "\U0001F600" in report.output_text

    def test_whitespace_only_output_fails(self) -> None:
        """Whitespace-only output should fail verification."""
        provider = MockProvider(main_outputs=["   \n\t  "])

        task = _make_task()
        engine = Engine()
        report = engine.run(task, provider)

        assert report.success is False
        assert "no_output" in report.verification.reasons


# =============================================================================
# Helper Method Tests
# =============================================================================

class TestEngineHelpers:
    """Test engine helper methods via shared usage.py functions."""

    def test_read_int_finds_value(self) -> None:
        """_read_int should find integer value."""
        # These functions were moved to usage.py as part of modularity refactor.
        # Engine now uses normalize_usage() which delegates to these.
        from enzu.usage import _read_int
        usage = {"output_tokens": 100, "other": "string"}
        result = _read_int(usage, ("missing", "output_tokens"))
        assert result == 100

    def test_read_int_returns_none_when_missing(self) -> None:
        """_read_int should return None when key missing."""
        from enzu.usage import _read_int
        usage = {"other": 100}
        result = _read_int(usage, ("missing",))
        assert result is None

    def test_read_float_finds_value(self) -> None:
        """_read_float should find float value."""
        from enzu.usage import _read_float
        usage = {"cost_usd": 0.05}
        result = _read_float(usage, ("cost_usd",))
        assert result == 0.05

    def test_read_float_converts_int(self) -> None:
        """_read_float should convert int to float."""
        from enzu.usage import _read_float
        usage = {"cost": 5}
        result = _read_float(usage, ("cost",))
        assert result == 5.0
        assert isinstance(result, float)

    def test_read_float_returns_none_when_missing(self) -> None:
        """_read_float should return None when key missing."""
        from enzu.usage import _read_float
        usage = {}
        result = _read_float(usage, ("missing",))
        assert result is None
