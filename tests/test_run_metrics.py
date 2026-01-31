"""Tests for run metrics: p50/p95 and terminal state distributions."""

from datetime import datetime, timezone, timedelta

from enzu.metrics import (
    RunEvent,
    RunMetricsCollector,
    get_run_metrics,
    reset_run_metrics,
)
from enzu.models import Outcome, BudgetUsage, ExecutionReport, VerificationResult


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def make_report(
    success: bool = True,
    outcome: Outcome = Outcome.SUCCESS,
    partial: bool = False,
    elapsed: float = 1.0,
    input_tokens: int = 100,
    output_tokens: int = 50,
    cost_usd: float | None = 0.001,
) -> ExecutionReport:
    return ExecutionReport(
        success=success,
        outcome=outcome,
        partial=partial,
        task_id="test-task",
        provider="test-provider",
        model="test-model",
        output_text="test output",
        verification=VerificationResult(passed=success),
        budget_usage=BudgetUsage(
            elapsed_seconds=elapsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost_usd,
        ),
    )


class TestRunEvent:
    def test_from_execution_report(self):
        started = utc_now()
        finished = started + timedelta(seconds=1.5)
        report = make_report(elapsed=1.5)

        event = RunEvent.from_execution_report(
            run_id="test-run-1",
            report=report,
            started_at=started,
            finished_at=finished,
        )

        assert event.run_id == "test-run-1"
        assert event.task_id == "test-task"
        assert event.provider == "test-provider"
        assert event.model == "test-model"
        assert event.outcome == Outcome.SUCCESS
        assert event.success is True
        assert event.partial is False
        assert event.elapsed_seconds == 1.5
        assert event.input_tokens == 100
        assert event.output_tokens == 50
        assert event.total_tokens == 150
        assert event.cost_usd == 0.001

    def test_from_execution_report_with_retries(self):
        report = make_report()
        event = RunEvent.from_execution_report(
            run_id="test",
            report=report,
            started_at=utc_now(),
            finished_at=utc_now(),
            retries=3,
        )
        assert event.retries == 3

    def test_to_log_dict(self):
        report = make_report()
        event = RunEvent.from_execution_report(
            run_id="log-test",
            report=report,
            started_at=utc_now(),
            finished_at=utc_now(),
        )

        log_dict = event.to_log_dict()
        assert log_dict["type"] == "enzu.run_event.v1"
        assert log_dict["run_id"] == "log-test"
        assert log_dict["outcome"] == "success"


class TestRunMetricsCollector:
    def test_observe_single_event(self):
        collector = RunMetricsCollector()
        report = make_report()
        event = RunEvent.from_execution_report(
            run_id="test",
            report=report,
            started_at=utc_now(),
            finished_at=utc_now(),
        )

        collector.observe(event)

        stats = collector.snapshot()
        assert stats["total_runs"] == 1
        assert stats["cost_known_runs"] == 1
        assert stats["cost_unknown_runs"] == 0

    def test_observe_multiple_events(self):
        collector = RunMetricsCollector()

        for i in range(10):
            report = make_report(
                elapsed=float(i + 1),
                cost_usd=0.001 * (i + 1),
            )
            event = RunEvent.from_execution_report(
                run_id=f"test-{i}",
                report=report,
                started_at=utc_now(),
                finished_at=utc_now(),
            )
            collector.observe(event)

        stats = collector.snapshot()
        assert stats["total_runs"] == 10

    def test_percentiles(self):
        collector = RunMetricsCollector()

        for i in range(100):
            report = make_report(elapsed=float(i + 1))
            event = RunEvent.from_execution_report(
                run_id=f"test-{i}",
                report=report,
                started_at=utc_now(),
                finished_at=utc_now(),
            )
            collector.observe(event)

        percentiles = collector.percentiles()
        assert percentiles["elapsed_seconds"]["p50"] is not None
        assert percentiles["elapsed_seconds"]["p95"] is not None
        assert (
            percentiles["elapsed_seconds"]["p50"]
            < percentiles["elapsed_seconds"]["p95"]
        )

    def test_outcome_distribution(self):
        collector = RunMetricsCollector()

        for outcome, count in [
            (Outcome.SUCCESS, 5),
            (Outcome.BUDGET_EXCEEDED, 3),
            (Outcome.TIMEOUT, 2),
        ]:
            for _ in range(count):
                report = make_report(
                    success=outcome == Outcome.SUCCESS,
                    outcome=outcome,
                )
                event = RunEvent.from_execution_report(
                    run_id="test",
                    report=report,
                    started_at=utc_now(),
                    finished_at=utc_now(),
                )
                collector.observe(event)

        dist = collector.outcome_distribution()
        assert dist["success|partial=0"] == 5
        assert dist["budget_exceeded|partial=0"] == 3
        assert dist["timeout|partial=0"] == 2

    def test_partial_tracking(self):
        collector = RunMetricsCollector()

        report = make_report(
            success=False,
            outcome=Outcome.BUDGET_EXCEEDED,
            partial=True,
        )
        event = RunEvent.from_execution_report(
            run_id="partial-test",
            report=report,
            started_at=utc_now(),
            finished_at=utc_now(),
        )
        collector.observe(event)

        dist = collector.outcome_distribution()
        assert dist["budget_exceeded|partial=1"] == 1

    def test_cost_unknown_tracking(self):
        collector = RunMetricsCollector()

        report = make_report(cost_usd=None)
        event = RunEvent.from_execution_report(
            run_id="no-cost",
            report=report,
            started_at=utc_now(),
            finished_at=utc_now(),
        )
        collector.observe(event)

        stats = collector.snapshot()
        assert stats["cost_unknown_runs"] == 1
        assert stats["cost_known_runs"] == 0

    def test_prometheus_format(self):
        collector = RunMetricsCollector()

        report = make_report()
        event = RunEvent.from_execution_report(
            run_id="prom-test",
            report=report,
            started_at=utc_now(),
            finished_at=utc_now(),
        )
        collector.observe(event)

        prom = collector.prometheus_format()
        assert "enzu_runs_total" in prom
        assert 'outcome="success"' in prom
        assert "enzu_run_cost_usd" in prom
        assert "enzu_run_elapsed_seconds" in prom

    def test_empty_collector(self):
        collector = RunMetricsCollector()
        stats = collector.snapshot()

        assert stats["total_runs"] == 0
        assert stats["percentiles"]["elapsed_seconds"]["p50"] is None


class TestGlobalCollector:
    def test_get_run_metrics_singleton(self):
        reset_run_metrics()
        c1 = get_run_metrics()
        c2 = get_run_metrics()
        assert c1 is c2

    def test_reset_run_metrics(self):
        reset_run_metrics()
        c1 = get_run_metrics()

        report = make_report()
        event = RunEvent.from_execution_report(
            run_id="test",
            report=report,
            started_at=utc_now(),
            finished_at=utc_now(),
        )
        c1.observe(event)
        assert c1.snapshot()["total_runs"] == 1

        reset_run_metrics()
        c2 = get_run_metrics()
        assert c2.snapshot()["total_runs"] == 0
