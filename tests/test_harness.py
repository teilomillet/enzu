"""Tests for harness mode — expanded sandbox capabilities with guardrails.

Covers:
- HARNESS_PROFILE: registration, pip, timeout, expanded imports
- RuntimeOptions: new harness fields default to None, can be set
- Planner prompt: harness vs non-harness template selection
- Objective defaults: max_steps_per_phase override, explicit preservation
- Integration: full code path through real RLM engine with MockProvider
- Persistence: save/load round-trip, backward compat
- Convenience: harness() function, exports, importability
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from enzu.models import Budget, BudgetUsage, Outcome, RLMExecutionReport
from enzu.objective import (
    Objective,
    _build_planner_prompt,
    harness,
    objective,
)
from enzu.runtime.protocol import RuntimeOptions
from enzu.security import HARNESS_PROFILE, get_security_profile


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
# HARNESS_PROFILE tests
# ============================================================================

class TestHarnessProfile:

    def test_profile_registered(self) -> None:
        profile = get_security_profile("harness")
        assert profile.name == "harness"

    def test_pip_enabled(self) -> None:
        assert HARNESS_PROFILE.enable_pip is True

    def test_timeout_one_hour(self) -> None:
        assert HARNESS_PROFILE.timeout_seconds == 3600.0

    def test_expanded_imports(self) -> None:
        for mod in ("subprocess", "os", "pathlib", "sys"):
            assert mod in HARNESS_PROFILE.allowed_imports, f"{mod} missing"

    def test_output_char_limit(self) -> None:
        assert HARNESS_PROFILE.output_char_limit == 65536

    def test_network_enabled(self) -> None:
        assert HARNESS_PROFILE.enable_network is True

    def test_filesystem_disabled(self) -> None:
        assert HARNESS_PROFILE.enable_filesystem is False


# ============================================================================
# RuntimeOptions tests
# ============================================================================

class TestRuntimeOptions:

    def test_defaults_are_none(self) -> None:
        opts = RuntimeOptions()
        assert opts.enable_pip is None
        assert opts.allowed_imports is None
        assert opts.output_char_limit is None
        assert opts.prompt_style is None
        assert opts.inject_search_tools is None

    def test_harness_fields_set(self) -> None:
        opts = RuntimeOptions(
            enable_pip=True,
            allowed_imports=["os", "sys"],
            output_char_limit=65536,
            prompt_style="extended",
            inject_search_tools=True,
        )
        assert opts.enable_pip is True
        assert opts.allowed_imports == ["os", "sys"]
        assert opts.output_char_limit == 65536
        assert opts.prompt_style == "extended"
        assert opts.inject_search_tools is True


# ============================================================================
# Planner prompt tests
# ============================================================================

class TestPlannerPrompt:

    def test_harness_prompt_header(self) -> None:
        prompt = _build_planner_prompt("test goal", harness=True)
        assert "HARNESS MODE" in prompt

    def test_harness_prompt_pip_mention(self) -> None:
        prompt = _build_planner_prompt("test goal", harness=True)
        assert "pip_install" in prompt

    def test_non_harness_prompt_unchanged(self) -> None:
        prompt = _build_planner_prompt("test goal", harness=False)
        assert "OBJECTIVE MODE" in prompt
        assert "HARNESS MODE" not in prompt

    def test_harness_prompt_contains_goal(self) -> None:
        prompt = _build_planner_prompt("my specific goal", harness=True)
        assert "my specific goal" in prompt


# ============================================================================
# Objective defaults tests
# ============================================================================

class TestObjectiveDefaults:

    def test_max_steps_defaults_to_50(self) -> None:
        obj = Objective(goal="test", harness=True)
        assert obj.max_steps_per_phase == 50

    def test_explicit_max_steps_not_overridden(self) -> None:
        obj = Objective(goal="test", harness=True, max_steps_per_phase=10)
        assert obj.max_steps_per_phase == 10

    def test_non_harness_stays_5(self) -> None:
        obj = Objective(goal="test")
        assert obj.max_steps_per_phase == 5

    def test_harness_flag_defaults_false(self) -> None:
        obj = Objective(goal="test")
        assert obj.harness is False


# ============================================================================
# Integration tests (real code path, MockProvider for LLM only)
# ============================================================================

class TestHarnessIntegration:

    def test_harness_step_through_real_rlm(self) -> None:
        """Objective(harness=True).step() → real RLMEngine → sandbox → FINAL()."""
        from tests.providers.mock_provider import MockProvider

        mock_provider = MockProvider(
            main_outputs=['```python\nFINAL("harness step done")\n```'],
        )

        with patch("enzu.runtime.local.resolve_provider", return_value=mock_provider):
            obj = Objective(
                goal="Run harness experiment",
                budget={"tokens": 50000},
                model="mock",
                harness=True,
            )
            result = obj.step()

        assert "harness step done" in result
        assert obj.done is True
        assert len(obj._phases) == 1

    def test_harness_sandbox_has_pip_install(self) -> None:
        """In harness mode, pip_install should be available in the sandbox."""
        from tests.providers.mock_provider import MockProvider

        mock_provider = MockProvider(
            main_outputs=[
                '```python\nresult = "pip_install" in dir()\nFINAL(str(result))\n```'
            ],
        )

        with patch("enzu.runtime.local.resolve_provider", return_value=mock_provider):
            obj = Objective(
                goal="Check pip_install availability",
                budget={"tokens": 50000},
                model="mock",
                harness=True,
            )
            result = obj.step()

        assert "True" in result

    def test_harness_sandbox_expanded_imports(self) -> None:
        """Harness mode should allow importing pathlib and subprocess."""
        from tests.providers.mock_provider import MockProvider

        mock_provider = MockProvider(
            main_outputs=[
                '```python\nimport pathlib\nimport subprocess\nFINAL("imports ok")\n```'
            ],
        )

        with patch("enzu.runtime.local.resolve_provider", return_value=mock_provider):
            obj = Objective(
                goal="Test expanded imports",
                budget={"tokens": 50000},
                model="mock",
                harness=True,
            )
            result = obj.step()

        assert "imports ok" in result

    def test_non_harness_blocks_subprocess_import(self) -> None:
        """Without harness, import subprocess should be blocked by sandbox."""
        from tests.providers.mock_provider import MockProvider

        # First attempt tries subprocess (sandbox blocks it),
        # second attempt reports the block via FINAL.
        mock_provider = MockProvider(
            main_outputs=[
                '```python\nimport subprocess\nFINAL("allowed")\n```',
                '```python\nFINAL("blocked")\n```',
            ],
        )

        with patch("enzu.runtime.local.resolve_provider", return_value=mock_provider):
            obj = Objective(
                goal="Test import restriction",
                budget={"tokens": 50000},
                model="mock",
                harness=False,
            )
            result = obj.step()

        assert "blocked" in result


# ============================================================================
# Persistence tests
# ============================================================================

class TestHarnessPersistence:

    def test_save_load_preserves_harness_true(self) -> None:
        obj = Objective(goal="persist test", harness=True)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            obj.save(path)
            loaded = Objective.load(path)
            assert loaded.harness is True
            assert loaded.max_steps_per_phase == 50
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_load_preserves_harness_false(self) -> None:
        obj = Objective(goal="persist test", harness=False)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            obj.save(path)
            loaded = Objective.load(path)
            assert loaded.harness is False
        finally:
            Path(path).unlink(missing_ok=True)

    def test_old_checkpoint_defaults_false(self) -> None:
        """A checkpoint without the 'harness' key should default to False."""
        data = {
            "goal": "old checkpoint",
            "budget": None,
            "model": None,
            "provider": None,
            "constraints": None,
            "context": None,
            "max_steps_per_phase": 5,
            "isolation": None,
            "phases": [],
            "budget_used": {"cost_usd": 0.0, "elapsed_seconds": 0.0, "output_tokens": 0},
            "done": False,
            "result": None,
            "created_at": "2024-01-01T00:00:00",
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name
        try:
            loaded = Objective.load(path)
            assert loaded.harness is False
        finally:
            Path(path).unlink(missing_ok=True)


# ============================================================================
# Convenience function tests
# ============================================================================

class TestConvenienceFunctions:

    def test_harness_function_runs(self) -> None:
        """harness() should delegate to objective(harness=True)."""
        from tests.providers.mock_provider import MockProvider

        mock_provider = MockProvider(
            main_outputs=['```python\nFINAL("harness result")\n```'],
        )

        with patch("enzu.runtime.local.resolve_provider", return_value=mock_provider):
            result = harness(
                "Run a harness goal",
                budget={"tokens": 50000},
                model="mock",
            )

        assert "harness result" in result

    def test_harness_in_enzu_all(self) -> None:
        import enzu

        assert "harness" in enzu.__all__

    def test_harness_importable(self) -> None:
        from enzu import harness as h

        assert callable(h)

    def test_client_harness_method(self) -> None:
        """Enzu().harness() delegates correctly."""
        from tests.providers.mock_provider import MockProvider

        mock_provider = MockProvider(
            main_outputs=['```python\nFINAL("client harness")\n```'],
        )

        with patch("enzu.runtime.local.resolve_provider", return_value=mock_provider):
            from enzu import Enzu

            client = Enzu("mock-model", provider="mock")
            result = client.harness(
                "Run harness via client",
                budget={"tokens": 10000},
            )

        assert "client harness" in result

    def test_module_level_harness(self) -> None:
        """Module-level enzu.harness() works."""
        from tests.providers.mock_provider import MockProvider

        mock_provider = MockProvider(
            main_outputs=['```python\nFINAL("module harness")\n```'],
        )

        with patch("enzu.runtime.local.resolve_provider", return_value=mock_provider):
            from enzu.client import harness as client_harness

            result = client_harness(
                "Run harness at module level",
                budget={"tokens": 10000},
                model="mock",
            )

        assert "module harness" in result
