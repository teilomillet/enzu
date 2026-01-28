from __future__ import annotations

import pytest

import enzu.api as api_module
from enzu.models import (
    BudgetUsage,
    ExecutionReport,
    RLMExecutionReport,
    VerificationResult,
)


def _chat_report() -> ExecutionReport:
    return ExecutionReport(
        success=True,
        task_id="t",
        provider="mock",
        model="mock-model",
        output_text="ok",
        verification=VerificationResult(passed=True, reasons=[]),
        budget_usage=BudgetUsage(
            elapsed_seconds=0.0,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            cost_usd=None,
            limits_exceeded=[],
        ),
    )


def _rlm_report() -> RLMExecutionReport:
    return RLMExecutionReport(
        success=True,
        task_id="t",
        provider="mock",
        model="mock-model",
        answer="ok",
        steps=[],
        budget_usage=BudgetUsage(
            elapsed_seconds=0.0,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            cost_usd=None,
            limits_exceeded=[],
        ),
        errors=[],
    )


def test_run_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        api_module.run("task", model="m", mode="bad")


def test_auto_mode_defaults_to_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_run_internal(*_args, **kwargs):
        calls.append(kwargs)
        return _chat_report()

    monkeypatch.setattr(api_module, "_run_internal", fake_run_internal)

    result = api_module.run("task", model="m", provider="p")

    assert result == "ok"
    assert calls[0]["mode"] == "chat"


def test_auto_mode_selects_rlm_with_data(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_run_internal(*_args, **kwargs):
        calls.append(kwargs)
        return _rlm_report()

    monkeypatch.setattr(api_module, "_run_internal", fake_run_internal)

    result = api_module.run("task", model="m", provider="p", data="x")

    assert result == "ok"
    assert calls[0]["mode"] == "rlm"


def test_auto_mode_selects_rlm_for_large_context(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_run_internal(*_args, **kwargs):
        calls.append(kwargs)
        return _rlm_report()

    monkeypatch.setattr(api_module, "_run_internal", fake_run_internal)

    big = "x" * 256_001
    result = api_module.run("task", model="m", provider="p", data=big)

    assert result == "ok"
    assert calls[0]["mode"] == "rlm"


def test_explicit_rlm_mode_uses_empty_data(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_run_internal(*_args, **kwargs):
        calls.append(kwargs)
        return _rlm_report()

    monkeypatch.setattr(api_module, "_run_internal", fake_run_internal)

    result = api_module.run("task", model="m", provider="p", mode="rlm")

    assert result == "ok"
    assert calls[0]["mode"] == "rlm"
    assert calls[0]["data"] == ""
