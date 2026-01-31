"""
Stress testing harness: run tasks under injected failure conditions.

Integrates with RunMetricsCollector and retry_tracking_context for
full observability of p95 behavior under stress.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from enzu.stress.scenario import CallContext, Scenario
from enzu.metrics.collector import RunMetricsCollector
from enzu.metrics.run_event import RunEvent


@dataclass
class ScenarioReport:
    """
    Results from running a scenario.

    Attributes:
        scenario_name: Name of the scenario.
        runs: Number of runs completed.
        snapshot: Full metrics snapshot from RunMetricsCollector.
    """

    scenario_name: str
    runs: int
    snapshot: Dict[str, Any]


class FailureInjector:
    """
    Injects faults based on scenario rules.

    Call inject() before each provider call to get fault behavior.
    """

    def __init__(
        self,
        scenario: Scenario,
        run_index: int = 0,
        seed: int = 0,
    ) -> None:
        self._scenario = scenario
        self._run_index = run_index
        self._rng = random.Random(seed)
        self._call_index = 0

    def inject(self) -> None:
        """
        Check for fault and apply it (delay + exception).

        Call this before each provider call.
        """
        self._call_index += 1
        ctx = CallContext(
            call_index=self._call_index,
            run_index=self._run_index,
            rng=self._rng,
        )
        fault = self._scenario.decide(ctx)
        if fault is None:
            return

        if fault.delay_seconds > 0:
            time.sleep(fault.delay_seconds)

        if fault.exception is not None:
            raise fault.exception

    @property
    def call_count(self) -> int:
        """Number of calls made so far."""
        return self._call_index


def run_scenarios(
    *,
    task: Callable[[FailureInjector], RunEvent],
    scenarios: Sequence[Scenario],
    runs_per_scenario: int = 50,
    seed: int = 42,
) -> List[ScenarioReport]:
    """
    Run a task under multiple failure scenarios and collect metrics.

    The task function receives a FailureInjector. Call injector.inject()
    before each provider call to trigger fault behavior.

    Args:
        task: Function that executes one run and returns a RunEvent.
              Should call injector.inject() before provider calls.
        scenarios: List of Scenario to test.
        runs_per_scenario: How many runs per scenario (default 50).
        seed: Random seed for reproducibility.

    Returns:
        List of ScenarioReport with metrics for each scenario.

    Example:
        def my_task(injector):
            with retry_tracking_context() as tracker:
                started = utc_now()
                try:
                    injector.inject()  # Before provider call
                    result = provider.complete(...)
                except Exception:
                    ...
                finished = utc_now()
                return RunEvent.from_report_with_tracker(...)

        reports = run_scenarios(
            task=my_task,
            scenarios=[baseline, high_429s],
        )
    """
    reports = []

    for s_idx, scenario in enumerate(scenarios):
        collector = RunMetricsCollector()

        for run_idx in range(runs_per_scenario):
            run_seed = seed + s_idx * 10000 + run_idx
            injector = FailureInjector(
                scenario=scenario,
                run_index=run_idx,
                seed=run_seed,
            )

            event = task(injector)
            collector.observe(event)

        reports.append(
            ScenarioReport(
                scenario_name=scenario.name,
                runs=runs_per_scenario,
                snapshot=collector.snapshot(),
            )
        )

    return reports


def format_report(reports: Sequence[ScenarioReport]) -> str:
    """
    Format scenario reports as human-readable text.

    Args:
        reports: List of ScenarioReport from run_scenarios.

    Returns:
        Formatted string suitable for printing.
    """
    lines = []

    for report in reports:
        s = report.snapshot
        p = s["percentiles"]

        lines.append("=" * 60)
        lines.append(f"SCENARIO: {report.scenario_name}")
        lines.append("=" * 60)
        lines.append(f"Runs: {report.runs}")
        lines.append("")

        lines.append("--- Outcomes ---")
        for outcome, count in sorted(s["outcome_distribution"].items()):
            pct = count / report.runs * 100
            lines.append(f"  {outcome}: {count} ({pct:.1f}%)")

        lines.append("")
        lines.append("--- Retry Metrics ---")
        retry_p = p["retries"]
        lines.append(
            f"  Retries/run: p50={_fmt(retry_p['p50'])} p95={_fmt(retry_p['p95'])}"
        )

        if s["retry_reason_distribution"]:
            lines.append("  By reason:")
            for reason, count in sorted(s["retry_reason_distribution"].items()):
                lines.append(f"    {reason}: {count}")

        lines.append(
            f"  Budget exceeded during retry: {s['budget_exceeded_during_retry']} "
            f"({s['budget_exceeded_during_retry_rate']:.1%})"
        )

        lines.append("")
        lines.append("--- Latency ---")
        elapsed_p = p["elapsed_seconds"]
        lines.append(
            f"  p50={_fmt(elapsed_p['p50'], 3)}s "
            f"p95={_fmt(elapsed_p['p95'], 3)}s "
            f"p99={_fmt(elapsed_p['p99'], 3)}s"
        )

        if s["cost_known_runs"] > 0:
            lines.append("")
            lines.append("--- Cost ---")
            cost_p = p["cost_usd"]
            lines.append(
                f"  p50=${_fmt(cost_p['p50'], 6)} "
                f"p95=${_fmt(cost_p['p95'], 6)} "
                f"p99=${_fmt(cost_p['p99'], 6)}"
            )

        lines.append("")

    return "\n".join(lines)


def _fmt(val: Optional[float], decimals: int = 1) -> str:
    """Format a value, handling None."""
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"
