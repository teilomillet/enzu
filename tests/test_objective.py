"""Tests for Objective goal decomposition via RLM.

Covers:
- _parse_budget: dict → Budget model (cost, hours, tokens, combined, edge cases)
- _build_planner_prompt: goal + constraints + history + budget status
- Phase dataclass: serialization round-trip
- Objective.step(): phase recording, budget accumulation, done detection
- Objective.run(): fire-and-forget loop
- Objective.save() / Objective.load(): idle/wake round-trip, step-save-load-step
- Objective.done: via success, budget exhaustion, flag
- Objective.result: explicit vs best-partial
- objective() function: delegates to enzu.api.run() correctly
- Integration: MockProvider through real RLM engine
- Client API: Enzu.objective() and module-level enzu.objective()
- Internal helpers: _remaining_budget, _format_history, _format_budget_status
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from enzu.models import Budget, BudgetUsage, Outcome, RLMExecutionReport, TaskSpec
from enzu.objective import (
    Objective,
    Phase,
    _build_planner_prompt,
    _parse_budget,
    objective,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_mock_report(
    answer: str = "Phase result",
    success: bool = True,
    outcome: Outcome = Outcome.SUCCESS,
    cost_usd: float = 1.0,
    output_tokens: int = 500,
    elapsed_seconds: float = 30.0,
) -> RLMExecutionReport:
    return RLMExecutionReport(
        success=success,
        outcome=outcome,
        task_id="test-task",
        provider="mock",
        model="mock",
        answer=answer,
        budget_usage=BudgetUsage(
            elapsed_seconds=elapsed_seconds,
            output_tokens=output_tokens,
            total_tokens=output_tokens * 2,
            cost_usd=cost_usd,
        ),
    )


# ============================================================================
# Budget parsing
# ============================================================================

class TestParseBudget:

    def test_none_gives_default(self) -> None:
        b = _parse_budget(None)
        assert b.max_tokens == 200_000
        assert b.max_cost_usd is None
        assert b.max_seconds is None

    def test_passthrough_budget_object(self) -> None:
        b = Budget(max_tokens=1000)
        assert _parse_budget(b) is b

    def test_cost_only(self) -> None:
        b = _parse_budget({"cost": 50})
        assert b.max_cost_usd == 50.0
        assert b.max_tokens is None

    def test_hours(self) -> None:
        b = _parse_budget({"hours": 2})
        assert b.max_seconds == 7200.0

    def test_minutes(self) -> None:
        b = _parse_budget({"minutes": 30})
        assert b.max_seconds == 1800.0

    def test_seconds(self) -> None:
        b = _parse_budget({"seconds": 120})
        assert b.max_seconds == 120.0

    def test_combined_time(self) -> None:
        """hours + minutes accumulate into max_seconds."""
        b = _parse_budget({"hours": 1, "minutes": 30})
        assert b.max_seconds == 5400.0

    def test_all_time_units(self) -> None:
        """hours + minutes + seconds all accumulate."""
        b = _parse_budget({"hours": 1, "minutes": 30, "seconds": 45})
        assert b.max_seconds == 5445.0

    def test_tokens(self) -> None:
        b = _parse_budget({"tokens": 50000})
        assert b.max_tokens == 50000

    def test_combined(self) -> None:
        b = _parse_budget({"cost": 10, "hours": 2, "tokens": 100000})
        assert b.max_cost_usd == 10.0
        assert b.max_seconds == 7200.0
        assert b.max_tokens == 100000

    def test_empty_dict_gives_default(self) -> None:
        b = _parse_budget({})
        assert b.max_tokens == 200_000

    def test_unrecognized_keys_ignored(self) -> None:
        """Unknown keys don't crash — only recognized keys used."""
        b = _parse_budget({"widgets": 99})
        assert b.max_tokens == 200_000  # falls back to default


# ============================================================================
# Planner prompt
# ============================================================================

class TestPlannerPrompt:

    def test_basic_goal(self) -> None:
        prompt = _build_planner_prompt("Analyze data")
        assert "Analyze data" in prompt
        assert "OBJECTIVE MODE" in prompt
        assert "PLAN" in prompt
        assert "EXECUTE" in prompt
        assert "TRACK" in prompt
        assert "ADAPT" in prompt
        assert "SYNTHESIZE" in prompt

    def test_with_constraints(self) -> None:
        prompt = _build_planner_prompt(
            "Analyze data", constraints=["Use Python only", "No external APIs"]
        )
        assert "Use Python only" in prompt
        assert "No external APIs" in prompt
        assert "Constraints:" in prompt

    def test_with_history(self) -> None:
        prompt = _build_planner_prompt(
            "Analyze data",
            phase_number=3,
            history="- Phase 1 (success): baseline\n- Phase 2 (success): comparison",
        )
        assert "Phase 1" in prompt
        assert "Phase 2" in prompt
        assert "Current phase: 3" in prompt
        assert "Previous Phases" in prompt

    def test_with_budget_status(self) -> None:
        prompt = _build_planner_prompt(
            "Analyze data",
            budget_status="$34.30 remaining of $50.00 / 5.6h remaining",
        )
        assert "$34.30 remaining" in prompt
        assert "Budget status:" in prompt

    def test_no_history_no_block(self) -> None:
        prompt = _build_planner_prompt("Goal")
        assert "Previous Phases" not in prompt

    def test_no_constraints_no_block(self) -> None:
        prompt = _build_planner_prompt("Goal")
        assert "Constraints:" not in prompt

    def test_empty_constraints_no_block(self) -> None:
        prompt = _build_planner_prompt("Goal", constraints=[])
        assert "Constraints:" not in prompt

    def test_all_fields_combined(self) -> None:
        prompt = _build_planner_prompt(
            "Run experiments",
            constraints=["Stay under budget"],
            phase_number=2,
            history="- Phase 1: done",
            budget_status="$40.00 remaining",
        )
        assert "Run experiments" in prompt
        assert "Stay under budget" in prompt
        assert "Phase 1: done" in prompt
        assert "Current phase: 2" in prompt
        assert "$40.00 remaining" in prompt


# ============================================================================
# Phase dataclass
# ============================================================================

class TestPhase:

    def test_round_trip(self) -> None:
        phase = Phase(
            number=1,
            answer="Result here",
            budget_usage={"cost_usd": 1.5, "output_tokens": 500},
            outcome="success",
        )
        d = phase.to_dict()
        loaded = Phase.from_dict(d)
        assert loaded.number == 1
        assert loaded.answer == "Result here"
        assert loaded.outcome == "success"
        assert loaded.budget_usage["cost_usd"] == 1.5

    def test_round_trip_with_none_answer(self) -> None:
        phase = Phase(number=2, answer=None, budget_usage={}, outcome="budget_exceeded")
        d = phase.to_dict()
        loaded = Phase.from_dict(d)
        assert loaded.answer is None
        assert loaded.outcome == "budget_exceeded"

    def test_from_dict_defaults(self) -> None:
        """Minimal dict still loads."""
        loaded = Phase.from_dict({"number": 1})
        assert loaded.number == 1
        assert loaded.answer is None
        assert loaded.outcome == "unknown"
        assert loaded.budget_usage == {}

    def test_timestamp_preserved(self) -> None:
        phase = Phase(
            number=1, answer="x", budget_usage={},
            outcome="success", timestamp="2025-01-01T00:00:00",
        )
        d = phase.to_dict()
        loaded = Phase.from_dict(d)
        assert loaded.timestamp == "2025-01-01T00:00:00"


# ============================================================================
# Objective.step() with mock (unit-level, patches enzu.api.run)
# ============================================================================

class TestObjectiveStep:

    @patch("enzu.api.run")
    def test_step_records_phase(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_mock_report("Phase 1 done", success=False)

        obj = Objective(goal="Test", budget={"tokens": 50000}, model="mock")
        result = obj.step()

        assert result == "Phase 1 done"
        assert len(obj._phases) == 1
        assert obj._phases[0].number == 1
        assert obj._phases[0].answer == "Phase 1 done"
        assert obj._phases[0].outcome == "success"

    @patch("enzu.api.run")
    def test_step_accumulates_budget(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_mock_report(
            cost_usd=5.0, output_tokens=1000, elapsed_seconds=60.0, success=False,
        )

        obj = Objective(goal="Test", budget={"cost": 50}, model="mock")
        obj.step()

        assert obj._budget_used["cost_usd"] == 5.0
        assert obj._budget_used["output_tokens"] == 1000
        assert obj._budget_used["elapsed_seconds"] == 60.0

    @patch("enzu.api.run")
    def test_step_marks_done_on_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_mock_report("Final answer", success=True)

        obj = Objective(goal="Test", budget={"tokens": 50000}, model="mock")
        obj.step()

        assert obj.done is True
        assert obj.result == "Final answer"

    @patch("enzu.api.run")
    def test_step_marks_done_on_budget_exceeded(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_mock_report(
            "Partial", success=False, outcome=Outcome.BUDGET_EXCEEDED,
        )

        obj = Objective(goal="Test", budget={"tokens": 50000}, model="mock")
        obj.step()

        assert obj.done is True
        assert obj._result == "Partial"

    @patch("enzu.api.run")
    def test_multiple_phases(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = [
            _make_mock_report("Phase 1", success=False),
            _make_mock_report("Phase 2", success=True),
        ]

        obj = Objective(goal="Test", budget={"tokens": 50000}, model="mock")
        obj.step()
        assert not obj._done

        obj.step()
        assert obj.done is True
        assert obj.result == "Phase 2"
        assert len(obj._phases) == 2

    @patch("enzu.api.run")
    def test_step_noop_when_done(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_mock_report("Done", success=True)

        obj = Objective(goal="Test", budget={"tokens": 50000}, model="mock")
        obj.step()
        assert obj.done

        result = obj.step()
        assert result == "Done"
        assert mock_run.call_count == 1  # not called again

    @patch("enzu.api.run")
    def test_step_passes_correct_args(self, mock_run: MagicMock) -> None:
        """Verify the TaskSpec + kwargs passed to enzu.api.run."""
        mock_run.return_value = _make_mock_report("ok", success=True)

        obj = Objective(
            goal="My goal",
            budget={"tokens": 10000},
            model="gpt-4o",
            provider="openai",
            context="Background info",
            constraints=["Be concise"],
            max_steps_per_phase=3,
            isolation="subprocess",
        )
        obj.step()

        call_kwargs = mock_run.call_args.kwargs
        spec = call_kwargs["task"]

        # TaskSpec checks
        assert isinstance(spec, TaskSpec)
        assert spec.input_text == "My goal"
        assert spec.model == "gpt-4o"
        assert "tools_guidance" in spec.metadata
        assert "OBJECTIVE MODE" in spec.metadata["tools_guidance"]
        assert "Be concise" in spec.metadata["tools_guidance"]
        assert spec.success_criteria.goal == "My goal"

        # Run kwargs checks
        assert call_kwargs["mode"] == "rlm"
        assert call_kwargs["max_steps"] == 3
        assert call_kwargs["data"] == "Background info"
        assert call_kwargs["return_report"] is True
        assert call_kwargs["provider"] == "openai"
        assert call_kwargs["isolation"] == "subprocess"

    @patch("enzu.api.run")
    def test_step_budget_decreases_between_phases(self, mock_run: MagicMock) -> None:
        """Each phase gets a smaller remaining budget."""
        mock_run.side_effect = [
            _make_mock_report("P1", success=False, output_tokens=3000),
            _make_mock_report("P2", success=True, output_tokens=2000),
        ]

        obj = Objective(goal="Test", budget={"tokens": 10000}, model="mock")
        obj.step()

        # Check remaining budget for phase 2
        remaining = obj._remaining_budget()
        assert remaining.max_tokens == 7000

        obj.step()
        remaining = obj._remaining_budget()
        assert remaining.max_tokens == 5000

    @patch("enzu.api.run")
    def test_step_includes_history_in_prompt(self, mock_run: MagicMock) -> None:
        """Phase 2 prompt should contain Phase 1 results."""
        mock_run.side_effect = [
            _make_mock_report("Baseline results", success=False),
            _make_mock_report("Comparison results", success=True),
        ]

        obj = Objective(goal="Test", budget={"tokens": 50000}, model="mock")
        obj.step()
        obj.step()

        # Second call should have history in tools_guidance
        second_spec = mock_run.call_args_list[1].kwargs["task"]
        guidance = second_spec.metadata["tools_guidance"]
        assert "Phase 1" in guidance
        assert "Baseline results" in guidance
        assert "Current phase: 2" in guidance


# ============================================================================
# Objective.run() — fire-and-forget via class
# ============================================================================

class TestObjectiveRun:

    @patch("enzu.api.run")
    def test_run_loops_until_done(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = [
            _make_mock_report("P1", success=False),
            _make_mock_report("P2", success=False),
            _make_mock_report("Final", success=True),
        ]

        obj = Objective(goal="Test", budget={"tokens": 50000}, model="mock")
        result = obj.run()

        assert result == "Final"
        assert len(obj._phases) == 3
        assert mock_run.call_count == 3

    @patch("enzu.api.run")
    def test_run_stops_on_budget_exhaustion(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_mock_report(
            "Partial", success=False, output_tokens=600,
        )

        obj = Objective(goal="Test", budget={"tokens": 1000}, model="mock")
        result = obj.run()

        # After ~2 calls, tokens exhausted (600 + 600 >= 1000)
        assert obj.done is True
        assert len(obj._phases) >= 1
        assert result is not None


# ============================================================================
# Objective.done via budget exhaustion
# ============================================================================

class TestObjectiveDone:

    def test_done_when_tokens_exhausted(self) -> None:
        obj = Objective(goal="Test", budget={"tokens": 1000}, model="mock")
        obj._budget_used["output_tokens"] = 1000
        assert obj.done is True

    def test_done_when_cost_exhausted(self) -> None:
        obj = Objective(goal="Test", budget={"cost": 10}, model="mock")
        obj._budget_used["cost_usd"] = 10.0
        assert obj.done is True

    def test_done_when_seconds_exhausted(self) -> None:
        obj = Objective(goal="Test", budget={"seconds": 60}, model="mock")
        obj._budget_used["elapsed_seconds"] = 60.0
        assert obj.done is True

    def test_not_done_with_remaining_budget(self) -> None:
        obj = Objective(goal="Test", budget={"tokens": 1000}, model="mock")
        obj._budget_used["output_tokens"] = 500
        assert obj.done is False

    def test_done_flag_takes_precedence(self) -> None:
        """_done flag means done regardless of budget."""
        obj = Objective(goal="Test", budget={"tokens": 100000}, model="mock")
        obj._done = True
        assert obj.done is True

    def test_done_with_combined_budget(self) -> None:
        """Any one limit exhausted → done."""
        obj = Objective(
            goal="Test", budget={"cost": 10, "tokens": 100000}, model="mock",
        )
        obj._budget_used["cost_usd"] = 10.0
        assert obj.done is True


# ============================================================================
# Objective.result
# ============================================================================

class TestObjectiveResult:

    def test_result_returns_best_partial(self) -> None:
        obj = Objective(goal="Test", budget={"tokens": 50000}, model="mock")
        obj._phases.append(Phase(
            number=1, answer="Partial result",
            budget_usage={}, outcome="success",
        ))
        assert obj.result == "Partial result"

    def test_result_none_when_no_phases(self) -> None:
        obj = Objective(goal="Test", budget={"tokens": 50000}, model="mock")
        assert obj.result is None

    def test_result_prefers_explicit_result(self) -> None:
        obj = Objective(goal="Test", budget={"tokens": 50000}, model="mock")
        obj._result = "Final answer"
        obj._phases.append(Phase(
            number=1, answer="Phase answer",
            budget_usage={}, outcome="success",
        ))
        assert obj.result == "Final answer"

    def test_best_partial_returns_last_phase(self) -> None:
        """When multiple phases, returns the most recent."""
        obj = Objective(goal="Test", budget={"tokens": 50000}, model="mock")
        obj._phases.append(Phase(number=1, answer="Old", budget_usage={}, outcome="success"))
        obj._phases.append(Phase(number=2, answer="New", budget_usage={}, outcome="success"))
        assert obj.result == "New"


# ============================================================================
# Save / Load round-trip
# ============================================================================

class TestObjectivePersistence:

    def test_save_load_round_trip(self) -> None:
        obj = Objective(
            goal="Test goal",
            budget={"cost": 50, "hours": 8},
            model="gpt-4o",
            provider="openai",
            constraints=["Use Python"],
            context="Background info",
            max_steps_per_phase=3,
            isolation="subprocess",
        )
        obj._phases.append(Phase(
            number=1,
            answer="Phase 1 result",
            budget_usage={"cost_usd": 5.0, "output_tokens": 1000, "elapsed_seconds": 60.0},
            outcome="success",
        ))
        obj._budget_used = {"cost_usd": 5.0, "elapsed_seconds": 60.0, "output_tokens": 1000}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            saved_path = obj.save(path)
            assert saved_path == path

            loaded = Objective.load(path)
            assert loaded.goal == "Test goal"
            assert loaded.model == "gpt-4o"
            assert loaded.provider == "openai"
            assert loaded.constraints == ["Use Python"]
            assert loaded.context == "Background info"
            assert loaded.max_steps_per_phase == 3
            assert loaded.isolation == "subprocess"
            assert len(loaded._phases) == 1
            assert loaded._phases[0].answer == "Phase 1 result"
            assert loaded._phases[0].number == 1
            assert loaded._budget_used["cost_usd"] == 5.0
            assert loaded._budget_used["elapsed_seconds"] == 60.0
            assert loaded._budget_used["output_tokens"] == 1000
            assert loaded._done is False
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_auto_generates_path(self) -> None:
        obj = Objective(goal="Test", budget={"tokens": 1000}, model="mock")
        path = obj.save()
        try:
            assert Path(path).exists()
            assert path.startswith("objective-")
            assert path.endswith(".json")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_preserves_done_state(self) -> None:
        obj = Objective(goal="Test", budget={"tokens": 1000}, model="mock")
        obj._done = True
        obj._result = "Final answer"

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            obj.save(path)
            loaded = Objective.load(path)
            assert loaded._done is True
            assert loaded._result == "Final answer"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_load_json_is_valid(self) -> None:
        """Saved JSON is well-formed and human-readable."""
        obj = Objective(goal="Test", budget={"tokens": 1000}, model="mock")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            obj.save(path)
            raw = Path(path).read_text()
            data = json.loads(raw)
            assert data["goal"] == "Test"
            assert data["model"] == "mock"
            assert isinstance(data["phases"], list)
            assert isinstance(data["budget_used"], dict)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_with_budget_object(self) -> None:
        """Budget object serializes to dict in JSON."""
        budget = Budget(max_tokens=5000, max_cost_usd=10.0)
        obj = Objective(goal="Test", budget=budget, model="mock")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            obj.save(path)
            data = json.loads(Path(path).read_text())
            assert data["budget"]["max_tokens"] == 5000
            assert data["budget"]["max_cost_usd"] == 10.0
        finally:
            Path(path).unlink(missing_ok=True)


# ============================================================================
# Idle / Wake round-trip (step → save → load → step)
# ============================================================================

class TestIdleWake:

    @patch("enzu.api.run")
    def test_step_save_load_step(self, mock_run: MagicMock) -> None:
        """Simulate: step → save → load → step (the core idle/wake pattern)."""
        mock_run.side_effect = [
            _make_mock_report(
                "Phase 1 result", success=False,
                cost_usd=5.0, output_tokens=1000, elapsed_seconds=60.0,
            ),
            _make_mock_report("Phase 2 final", success=True),
        ]

        # Phase 1
        obj = Objective(
            goal="Run experiments",
            budget={"cost": 50, "hours": 8},
            model="mock",
        )
        obj.step()
        assert len(obj._phases) == 1
        assert obj._phases[0].answer == "Phase 1 result"
        assert not obj.done

        # Save (go dormant)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            obj.save(path)

            # Load (wake up)
            restored = Objective.load(path)
            assert restored.goal == "Run experiments"
            assert len(restored._phases) == 1
            assert restored._budget_used["cost_usd"] == 5.0
            assert not restored.done

            # Phase 2
            restored.step()
            assert len(restored._phases) == 2
            assert restored.done is True
            assert restored.result == "Phase 2 final"
        finally:
            Path(path).unlink(missing_ok=True)

    @patch("enzu.api.run")
    def test_loaded_objective_has_correct_remaining_budget(self, mock_run: MagicMock) -> None:
        """After load, remaining budget reflects prior usage."""
        mock_run.return_value = _make_mock_report(
            "P1", success=False, cost_usd=20.0, output_tokens=5000,
        )

        obj = Objective(goal="Test", budget={"cost": 50, "tokens": 20000}, model="mock")
        obj.step()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            obj.save(path)
            loaded = Objective.load(path)

            remaining = loaded._remaining_budget()
            assert remaining.max_cost_usd == pytest.approx(30.0, abs=0.01)
            assert remaining.max_tokens == 15000
        finally:
            Path(path).unlink(missing_ok=True)


# ============================================================================
# objective() fire-and-forget function
# ============================================================================

class TestObjectiveFunction:

    @patch("enzu.api.run")
    def test_delegates_to_api_run(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_mock_report("The answer")

        result = objective("Test goal", budget={"tokens": 50000}, model="mock")

        assert result == "The answer"
        mock_run.assert_called_once()

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["mode"] == "rlm"
        assert call_kwargs["max_steps"] == 25
        assert call_kwargs["return_report"] is True

    @patch("enzu.api.run")
    def test_passes_task_spec_with_tools_guidance(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_mock_report("Result")

        objective("Analyze data", budget={"tokens": 10000}, model="mock")

        spec = mock_run.call_args.kwargs["task"]
        assert isinstance(spec, TaskSpec)
        assert "tools_guidance" in spec.metadata
        assert "OBJECTIVE MODE" in spec.metadata["tools_guidance"]
        assert "Analyze data" in spec.metadata["tools_guidance"]

    @patch("enzu.api.run")
    def test_custom_max_steps(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_mock_report("Result")

        objective("Goal", budget={"tokens": 10000}, model="mock", max_steps=10)

        assert mock_run.call_args.kwargs["max_steps"] == 10

    @patch("enzu.api.run")
    def test_with_constraints(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_mock_report("Result")

        objective(
            "Goal", budget={"tokens": 10000}, model="mock",
            constraints=["Use Python only"],
        )

        spec = mock_run.call_args.kwargs["task"]
        assert "Use Python only" in spec.metadata["tools_guidance"]

    @patch("enzu.api.run")
    def test_with_context(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_mock_report("Result")

        objective(
            "Goal", budget={"tokens": 10000}, model="mock",
            context="Background context here",
        )

        assert mock_run.call_args.kwargs["data"] == "Background context here"

    @patch("enzu.api.run")
    def test_budget_passed_to_spec(self, mock_run: MagicMock) -> None:
        """Budget dict is parsed and used in TaskSpec."""
        mock_run.return_value = _make_mock_report("Result")

        objective("Goal", budget={"cost": 25, "hours": 4}, model="mock")

        spec = mock_run.call_args.kwargs["task"]
        assert spec.budget.max_cost_usd == 25.0
        assert spec.budget.max_seconds == 14400.0

    @patch("enzu.api.run")
    def test_returns_empty_string_on_no_answer(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_mock_report(answer=None, success=True)

        # answer is None → should return ""
        result = objective("Goal", budget={"tokens": 10000}, model="mock")
        assert result == ""


# ============================================================================
# Integration: through real RLM engine with MockProvider
# ============================================================================

class TestIntegration:

    def test_objective_function_through_rlm(self) -> None:
        """objective() → enzu.api.run → RLMEngine → MockProvider → answer."""
        from tests.providers.mock_provider import MockProvider

        mock_provider = MockProvider(
            main_outputs=['```python\nFINAL("Python is great for CLI tools")\n```'],
        )

        with patch("enzu.runtime.local.resolve_provider", return_value=mock_provider):
            result = objective(
                "List pros and cons of Python for CLI tools",
                budget={"tokens": 50000},
                model="mock",
            )

        assert "Python is great for CLI tools" in result

    def test_objective_step_through_rlm(self) -> None:
        """Objective.step() through the real RLM engine."""
        from tests.providers.mock_provider import MockProvider

        mock_provider = MockProvider(
            main_outputs=['```python\nFINAL("Step 1 complete")\n```'],
        )

        with patch("enzu.runtime.local.resolve_provider", return_value=mock_provider):
            obj = Objective(
                goal="Run an experiment",
                budget={"tokens": 50000},
                model="mock",
            )
            result = obj.step()

        assert "Step 1 complete" in result
        assert obj.done is True
        assert len(obj._phases) == 1

    def test_objective_multiphase_through_rlm(self) -> None:
        """Multiple phases through real RLM engine."""
        from tests.providers.mock_provider import MockProvider

        mock_provider = MockProvider(
            main_outputs=[
                '```python\nFINAL("Phase 1: planned experiments")\n```',
                '```python\nFINAL("Phase 2: ran baseline")\n```',
            ],
        )

        with patch("enzu.runtime.local.resolve_provider", return_value=mock_provider):
            obj = Objective(
                goal="Run multi-phase experiment",
                budget={"tokens": 50000},
                model="mock",
            )
            obj.step()
            assert obj.done is True  # FINAL succeeds
            assert "Phase 1" in (obj.result or "")


# ============================================================================
# Client API: Enzu.objective() and module-level enzu.objective()
# ============================================================================

class TestClientAPI:

    @patch("enzu.api.run")
    def test_enzu_class_objective(self, mock_run: MagicMock) -> None:
        """Enzu().objective() delegates correctly."""
        from enzu import Enzu

        mock_run.return_value = _make_mock_report("Client answer")

        client = Enzu("mock-model", provider="mock")
        result = client.objective(
            "Analyze something",
            budget={"tokens": 10000},
        )

        assert result == "Client answer"
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["model"] == "mock-model"
        assert call_kwargs["provider"] == "mock"

    @patch("enzu.api.run")
    def test_module_level_objective(self, mock_run: MagicMock) -> None:
        """enzu.objective() module-level function delegates correctly."""
        import enzu

        mock_run.return_value = _make_mock_report("Module answer")

        result = enzu.objective(
            "Analyze something",
            budget={"tokens": 10000},
            model="mock",
        )

        assert result == "Module answer"

    def test_objective_in_all(self) -> None:
        """objective and Objective are in __all__."""
        import enzu

        assert "objective" in enzu.__all__
        assert "Objective" in enzu.__all__

    def test_imports_from_enzu(self) -> None:
        """Both can be imported directly from enzu."""
        from enzu import objective as obj_fn, Objective as ObjCls

        assert callable(obj_fn)
        assert callable(ObjCls)


# ============================================================================
# Internal helpers
# ============================================================================

class TestInternalHelpers:

    def test_remaining_budget_subtracts_usage(self) -> None:
        obj = Objective(goal="Test", budget={"cost": 50, "tokens": 10000}, model="mock")
        obj._budget_used = {"cost_usd": 10.0, "output_tokens": 3000, "elapsed_seconds": 0}

        remaining = obj._remaining_budget()
        assert remaining.max_cost_usd == pytest.approx(40.0, abs=0.01)
        assert remaining.max_tokens == 7000

    def test_remaining_budget_floors_at_minimum(self) -> None:
        """Budget never goes below floor values."""
        obj = Objective(goal="Test", budget={"cost": 10, "tokens": 100}, model="mock")
        obj._budget_used = {"cost_usd": 999, "output_tokens": 999, "elapsed_seconds": 0}

        remaining = obj._remaining_budget()
        assert remaining.max_cost_usd >= 0.01
        assert remaining.max_tokens >= 1

    def test_format_history_empty(self) -> None:
        obj = Objective(goal="Test", budget={"tokens": 1000}, model="mock")
        assert obj._format_history() == ""

    def test_format_history_with_phases(self) -> None:
        obj = Objective(goal="Test", budget={"tokens": 1000}, model="mock")
        obj._phases.append(Phase(
            number=1, answer="Baseline results here",
            budget_usage={"cost_usd": 5.0, "elapsed_seconds": 120.0},
            outcome="success",
        ))

        history = obj._format_history()
        assert "Phase 1" in history
        assert "success" in history
        assert "$5.00" in history
        assert "Baseline results here" in history

    def test_format_budget_status_cost(self) -> None:
        obj = Objective(goal="Test", budget={"cost": 50}, model="mock")
        obj._budget_used["cost_usd"] = 15.0

        status = obj._format_budget_status()
        assert "$35.00 remaining" in status

    def test_format_budget_status_time(self) -> None:
        obj = Objective(goal="Test", budget={"hours": 2}, model="mock")
        obj._budget_used["elapsed_seconds"] = 3600.0

        status = obj._format_budget_status()
        assert "1.0h remaining" in status

    def test_format_budget_status_tokens(self) -> None:
        obj = Objective(goal="Test", budget={"tokens": 10000}, model="mock")
        obj._budget_used["output_tokens"] = 3000

        status = obj._format_budget_status()
        assert "7,000 tokens remaining" in status

    def test_format_budget_status_combined(self) -> None:
        obj = Objective(goal="Test", budget={"cost": 50, "hours": 2, "tokens": 10000}, model="mock")
        status = obj._format_budget_status()
        assert "$50.00" in status
        assert "2.0h" in status
        assert "10,000 tokens" in status

    def test_accumulate_budget_ignores_none(self) -> None:
        """Accumulation skips None/0 values gracefully."""
        obj = Objective(goal="Test", budget={"tokens": 1000}, model="mock")
        obj._accumulate_budget({"cost_usd": None, "output_tokens": 0, "elapsed_seconds": 0})
        assert obj._budget_used["cost_usd"] == 0.0
        assert obj._budget_used["output_tokens"] == 0


# ============================================================================
# Repr
# ============================================================================

def test_repr() -> None:
    obj = Objective(goal="Test goal", budget={"tokens": 1000}, model="mock")
    r = repr(obj)
    assert "Test goal" in r
    assert "phases=0" in r
    assert "done=False" in r


def test_repr_with_phases() -> None:
    obj = Objective(goal="G", budget={"tokens": 1000}, model="mock")
    obj._phases.append(Phase(number=1, answer="x", budget_usage={}, outcome="success"))
    r = repr(obj)
    assert "phases=1" in r


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:

    def test_objective_with_budget_object(self) -> None:
        """Objective accepts a Budget object directly."""
        budget = Budget(max_tokens=5000)
        obj = Objective(goal="Test", budget=budget, model="mock")
        assert obj._parsed_budget.max_tokens == 5000

    @patch("enzu.api.run")
    def test_step_with_none_answer_in_report(self, mock_run: MagicMock) -> None:
        """Phase records None answer gracefully."""
        report = _make_mock_report(answer=None, success=False)
        # Override the answer field after creation to avoid type issues
        report.answer = None
        mock_run.return_value = report

        obj = Objective(goal="Test", budget={"tokens": 50000}, model="mock")
        result = obj.step()

        assert result == ""
        assert obj._phases[0].answer is None

    def test_phases_property_returns_copy(self) -> None:
        """phases property returns a copy, not the internal list."""
        obj = Objective(goal="Test", budget={"tokens": 1000}, model="mock")
        obj._phases.append(Phase(number=1, answer="x", budget_usage={}, outcome="ok"))

        phases = obj.phases
        phases.clear()  # mutate the copy
        assert len(obj._phases) == 1  # internal unaffected

    @patch("enzu.client._detect_provider_and_model", return_value=("openrouter", "gpt-4o"))
    @patch("enzu.api.run")
    def test_auto_detect_model_when_none(self, mock_run: MagicMock, mock_detect: MagicMock) -> None:
        """When model=None, auto-detect is used."""
        mock_run.return_value = _make_mock_report("Result")

        obj = Objective(goal="Test", budget={"tokens": 1000})
        obj.step()

        spec = mock_run.call_args.kwargs["task"]
        assert spec.model == "gpt-4o"

    @patch("enzu.client._detect_provider_and_model", return_value=("openrouter", "gpt-4o"))
    @patch("enzu.api.run")
    def test_objective_function_auto_detect(self, mock_run: MagicMock, mock_detect: MagicMock) -> None:
        """objective() function auto-detects model when None."""
        mock_run.return_value = _make_mock_report("Result")

        objective("Goal", budget={"tokens": 1000})

        spec = mock_run.call_args.kwargs["task"]
        assert spec.model == "gpt-4o"
