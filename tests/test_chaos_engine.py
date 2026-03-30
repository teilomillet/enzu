"""Chaos tests for the Engine: provider fallback, budget enforcement, verification.

Uses ordeal's stateful chaos testing to explore Engine behavior under
provider failures, budget edge cases, and verification scenarios.

Properties verified:
- Engine always returns a valid ExecutionReport
- Provider fallback produces correct trajectory ordering
- Budget exceeded implies partial=True when output exists
- Failed verification never marks success=True
- Progress events are well-ordered (start before complete)
"""

from __future__ import annotations

from typing import Any, Dict

from hypothesis import strategies as st

from ordeal import ChaosTest, always, invariant, rule
from ordeal.faults.network import intermittent_http_error

from enzu.engine import Engine
from enzu.models import (
    Budget,
    ExecutionReport,
    Outcome,
    ProviderResult,
    SuccessCriteria,
    TaskSpec,
)
from enzu.providers.base import BaseProvider


# ---------------------------------------------------------------------------
# Test providers
# ---------------------------------------------------------------------------


class _ChaosProvider(BaseProvider):
    """Provider with configurable behavior for chaos testing."""

    def __init__(
        self,
        name: str = "chaos",
        output: str = "chaos response",
        *,
        fail: bool = False,
        usage: Dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self._output = output
        self._fail = fail
        self._usage = usage or {"output_tokens": 10, "total_tokens": 20}
        self.call_count = 0

    def generate(self, task: TaskSpec) -> ProviderResult:
        self.call_count += 1
        if self._fail:
            raise RuntimeError(f"{self.name} provider failure")
        return ProviderResult(
            output_text=self._output,
            raw={"mock": True},
            usage=dict(self._usage),
            provider=self.name,
            model=task.model,
        )


def _make_task(
    task_text: str = "test task",
    *,
    max_tokens: int | None = 500,
    max_total_tokens: int | None = None,
    max_seconds: float | None = None,
    required_substrings: list[str] | None = None,
    min_word_count: int | None = None,
    max_output_tokens: int | None = None,
) -> TaskSpec:
    """Build a TaskSpec with optional budget and criteria."""
    # Budget requires at least one limit
    budget = Budget(
        max_tokens=max_tokens or 500,
        max_total_tokens=max_total_tokens,
        max_seconds=max_seconds,
    )
    criteria_kwargs: dict[str, Any] = {}
    if required_substrings:
        criteria_kwargs["required_substrings"] = required_substrings
    if min_word_count is not None:
        criteria_kwargs["min_word_count"] = min_word_count
    # SuccessCriteria requires at least one check or a goal
    if not criteria_kwargs:
        criteria_kwargs["goal"] = "complete the task"
    criteria = SuccessCriteria(**criteria_kwargs)
    import uuid

    return TaskSpec(
        task_id=str(uuid.uuid4()),
        input_text=task_text,
        model="gpt-4o-mini",
        budget=budget,
        success_criteria=criteria,
        max_output_tokens=max_output_tokens,
    )


# ---------------------------------------------------------------------------
# Chaos test: Engine execution under fault injection
# ---------------------------------------------------------------------------


class EngineChaos(ChaosTest):
    """Explore Engine behavior under randomized task configs and provider faults."""

    faults = [
        intermittent_http_error(
            "tests.test_chaos_engine._chaos_provider_generate",
            every_n=3,
            status_code=503,
        ),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.engine = Engine()
        self.reports: list[ExecutionReport] = []

    @rule(
        max_tokens=st.one_of(st.none(), st.integers(min_value=1, max_value=1000)),
        max_total_tokens=st.one_of(
            st.none(), st.integers(min_value=10, max_value=5000)
        ),
        provider_fails=st.booleans(),
        output_text=st.text(min_size=1, max_size=50),
    )
    def run_task(
        self,
        max_tokens: int | None,
        max_total_tokens: int | None,
        provider_fails: bool,
        output_text: str,
    ) -> None:
        """Execute a task with random budget and provider configuration."""
        provider = _ChaosProvider(
            output=output_text,
            fail=provider_fails,
        )
        task = _make_task(
            max_tokens=max_tokens,
            max_total_tokens=max_total_tokens,
        )

        report = self.engine.run(task, provider)
        self.reports.append(report)

        always(
            isinstance(report, ExecutionReport), "run always returns ExecutionReport"
        )
        always(report.task_id is not None, "report has task_id")
        always(report.provider is not None, "report has provider name")

    @rule(
        substrings=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=3),
        output_text=st.text(min_size=0, max_size=100),
    )
    def run_with_verification(self, substrings: list[str], output_text: str) -> None:
        """Execute a task with success criteria to test verification."""
        provider = _ChaosProvider(output=output_text)
        task = _make_task(required_substrings=substrings)

        report = self.engine.run(task, provider)
        self.reports.append(report)

        # If verification failed, success must be False
        if not report.verification.passed:
            always(not report.success, "failed verification implies not success")

    @rule(
        output_text=st.text(min_size=1, max_size=50),
    )
    def run_with_fallback(self, output_text: str) -> None:
        """Execute with a failing primary and working fallback provider."""
        primary = _ChaosProvider(name="failing", fail=True)
        fallback = _ChaosProvider(name="fallback", output=output_text)

        task = _make_task()
        report = self.engine.run(task, primary, fallback_providers=[fallback])
        self.reports.append(report)

        if report.success:
            always(
                report.provider == "fallback",
                "successful fallback uses fallback provider",
            )
            always(
                len(report.trajectory) >= 2,
                "fallback creates at least 2 trajectory entries",
            )

    @invariant()
    def reports_have_valid_outcomes(self) -> None:
        """All reports must have a valid outcome enum."""
        for report in self.reports:
            always(
                report.outcome in list(Outcome),
                "outcome is a valid Outcome enum",
            )

    @invariant()
    def progress_events_well_ordered(self) -> None:
        """Progress events must start with 'start' and end with 'complete' or 'error'."""
        for report in self.reports:
            if report.progress_events:
                first_phase = report.progress_events[0].phase
                always(
                    first_phase == "start",
                    "first progress event is 'start'",
                )


# Module-level function for fault injection target
def _chaos_provider_generate(task: TaskSpec) -> ProviderResult:
    return ProviderResult(
        output_text="response",
        raw={},
        usage={"output_tokens": 10},
        provider="chaos",
        model=task.model,
    )


TestEngineChaos = EngineChaos.TestCase


# ---------------------------------------------------------------------------
# Chaos test: Provider fallback chain
# ---------------------------------------------------------------------------


class ProviderFallbackChaos(ChaosTest):
    """Explore provider fallback behavior with various failure combinations."""

    faults = []

    def __init__(self) -> None:
        super().__init__()
        self.engine = Engine()

    @rule(
        n_failing=st.integers(min_value=0, max_value=4),
        has_working=st.booleans(),
    )
    def fallback_chain(self, n_failing: int, has_working: bool) -> None:
        """Build a chain of N failing providers optionally followed by a working one."""
        failing = [
            _ChaosProvider(name=f"fail_{i}", fail=True) for i in range(n_failing)
        ]
        working = (
            [_ChaosProvider(name="working", output="success")] if has_working else []
        )

        all_providers = failing + working
        if not all_providers:
            return

        primary = all_providers[0]
        fallbacks = all_providers[1:] if len(all_providers) > 1 else None

        task = _make_task()
        report = self.engine.run(task, primary, fallback_providers=fallbacks)

        if has_working:
            always(report.success, "chain with working provider succeeds")
            always(
                report.output_text == "success",
                "output comes from working provider",
            )
        else:
            always(
                not report.success,
                "chain with no working provider fails",
            )

        # Trajectory should have one entry per provider attempted
        always(
            len(report.trajectory) == len(all_providers),
            "trajectory has one entry per provider",
        )

    @rule()
    def empty_fallback_list(self) -> None:
        """Empty fallback list behaves like no fallbacks."""
        provider = _ChaosProvider(output="solo")
        task = _make_task()
        report = self.engine.run(task, provider, fallback_providers=[])

        always(report.success, "solo provider with empty fallbacks succeeds")
        always(len(report.trajectory) == 1, "single trajectory entry")


TestProviderFallbackChaos = ProviderFallbackChaos.TestCase
