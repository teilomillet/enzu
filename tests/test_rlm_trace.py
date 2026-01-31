"""Tests for scripts/rlm_trace.py to prevent regressions."""

from enzu.models import RLMExecutionReport, BudgetUsage


class TestRLMExecutionReportFields:
    """Ensure trace script assumptions about RLMExecutionReport hold."""

    def test_report_has_required_fields(self) -> None:
        """Verify RLMExecutionReport has fields used by rlm_trace.py."""
        report = RLMExecutionReport(
            success=True,
            task_id="test-task",
            provider="test-provider",
            model="test-model",
            answer="test answer",
            steps=[],
            budget_usage=BudgetUsage(
                elapsed_seconds=1.0,
                output_tokens=100,
                total_tokens=200,
                cost_usd=0.01,
                limits_exceeded=[],
            ),
            errors=[],
        )

        # Fields accessed by rlm_trace.py
        assert hasattr(report, "success")
        assert hasattr(report, "answer")
        assert hasattr(report, "errors")
        assert hasattr(report, "budget_usage")
        assert hasattr(report, "steps")

    def test_report_does_not_have_verification_passed(self) -> None:
        """Confirm verification_passed is not a field (must be derived from errors)."""
        report = RLMExecutionReport(
            success=True,
            task_id="test",
            provider="test",
            model="test",
            answer="ok",
            steps=[],
            budget_usage=BudgetUsage(
                elapsed_seconds=1.0,
                output_tokens=10,
                total_tokens=20,
                cost_usd=0.001,
            ),
            errors=[],
        )
        assert not hasattr(report, "verification_passed")

    def test_derive_verification_passed_from_errors(self) -> None:
        """Test the derivation logic used in rlm_trace.py."""
        # No verification errors
        errors_ok: list[str] = []
        assert not any("verification_failed" in e for e in errors_ok)

        errors_budget: list[str] = ["budget_exceeded"]
        assert not any("verification_failed" in e for e in errors_budget)

        # Has verification errors
        errors_fail: list[str] = ["verification_failed:no_output"]
        assert any("verification_failed" in e for e in errors_fail)

        errors_multi: list[str] = [
            "budget_exceeded",
            "verification_failed:min_word_count",
        ]
        assert any("verification_failed" in e for e in errors_multi)
