from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from enzu import run
from enzu.models import BudgetUsage, RLMExecutionReport, TaskSpec
from enzu.runtime import ProviderSpec, RuntimeOptions


@dataclass
class RecordedCall:
    provider: ProviderSpec
    data: str
    options: RuntimeOptions


class RecordingRuntime:
    def __init__(self) -> None:
        self.last_call: Optional[RecordedCall] = None

    def run(
        self,
        *,
        spec: TaskSpec,
        provider: ProviderSpec,
        data: str,
        options: RuntimeOptions,
    ) -> RLMExecutionReport:
        self.last_call = RecordedCall(provider=provider, data=data, options=options)
        return RLMExecutionReport(
            success=True,
            task_id=spec.task_id,
            provider="runtime",
            model=spec.model,
            answer="ok",
            steps=[],
            budget_usage=BudgetUsage(
                elapsed_seconds=0.0,
                input_tokens=None,
                output_tokens=0,
                total_tokens=0,
                cost_usd=None,
                limits_exceeded=[],
            ),
            errors=[],
        )


def test_runtime_used_for_rlm() -> None:
    runtime = RecordingRuntime()
    result = run(
        "Do a thing",
        model="mock-model",
        provider="mock",
        data="context",
        runtime=runtime,
    )

    assert result == "ok"
    assert runtime.last_call is not None
    assert runtime.last_call.provider.name == "mock"
    assert runtime.last_call.data == "context"
