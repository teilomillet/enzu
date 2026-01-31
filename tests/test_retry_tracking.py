"""Tests for retry tracking and budget attribution."""

from datetime import datetime, timezone

from enzu.retries import (
    RetryReason,
    RetryTracker,
    get_retry_tracker,
    retry_tracking_context,
)
from enzu.metrics import RunEvent, RunMetricsCollector
from enzu.models import Outcome, BudgetUsage, ExecutionReport, VerificationResult


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TestRetryTracker:
    def test_record_retry(self):
        tracker = RetryTracker()
        tracker.record_retry(RetryReason.RATE_LIMIT, backoff_seconds=1.5)

        assert tracker.total_retries == 1
        assert tracker.retries_by_reason[RetryReason.RATE_LIMIT] == 1
        assert tracker.backoff_seconds_total == 1.5
        assert tracker.last_retry_reason == RetryReason.RATE_LIMIT

    def test_multiple_retries(self):
        tracker = RetryTracker()
        tracker.record_retry(RetryReason.RATE_LIMIT, backoff_seconds=1.0)
        tracker.record_retry(RetryReason.RATE_LIMIT, backoff_seconds=2.0)
        tracker.record_retry(RetryReason.TIMEOUT, backoff_seconds=1.5)

        assert tracker.total_retries == 3
        assert tracker.retries_by_reason[RetryReason.RATE_LIMIT] == 2
        assert tracker.retries_by_reason[RetryReason.TIMEOUT] == 1
        assert tracker.backoff_seconds_total == 4.5

    def test_mark_budget_exceeded_with_retries(self):
        tracker = RetryTracker()
        tracker.record_retry(RetryReason.RATE_LIMIT)
        tracker.mark_budget_exceeded()

        assert tracker.budget_exceeded_during_retry is True

    def test_mark_budget_exceeded_without_retries(self):
        tracker = RetryTracker()
        tracker.mark_budget_exceeded()

        assert tracker.budget_exceeded_during_retry is False

    def test_to_dict(self):
        tracker = RetryTracker()
        tracker.record_retry(RetryReason.RATE_LIMIT)
        tracker.record_retry(RetryReason.SERVER_ERROR)

        d = tracker.to_dict()
        assert d == {"rate_limit": 1, "server_error": 1}


class TestRetryTrackingContext:
    def test_context_creates_tracker(self):
        with retry_tracking_context() as tracker:
            assert tracker is not None
            assert isinstance(tracker, RetryTracker)

    def test_get_tracker_inside_context(self):
        with retry_tracking_context() as tracker:
            assert get_retry_tracker() is tracker

    def test_get_tracker_outside_context(self):
        assert get_retry_tracker() is None

    def test_nested_contexts(self):
        with retry_tracking_context() as outer:
            outer.record_retry(RetryReason.RATE_LIMIT)

            with retry_tracking_context() as inner:
                inner.record_retry(RetryReason.TIMEOUT)
                assert get_retry_tracker() is inner
                assert inner.total_retries == 1

            assert get_retry_tracker() is outer
            assert outer.total_retries == 1


class TestRunEventWithRetries:
    def make_report(
        self,
        outcome: Outcome = Outcome.SUCCESS,
        limits_exceeded: list | None = None,
    ) -> ExecutionReport:
        return ExecutionReport(
            success=outcome == Outcome.SUCCESS,
            outcome=outcome,
            partial=False,
            task_id="test",
            provider="test",
            model="test",
            output_text="test",
            verification=VerificationResult(passed=True),
            budget_usage=BudgetUsage(
                elapsed_seconds=1.0,
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                cost_usd=0.001,
                limits_exceeded=limits_exceeded or [],
            ),
        )

    def test_from_report_with_tracker(self):
        tracker = RetryTracker()
        tracker.record_retry(RetryReason.RATE_LIMIT, backoff_seconds=1.0)
        tracker.record_retry(RetryReason.TIMEOUT, backoff_seconds=2.0)

        report = self.make_report()
        event = RunEvent.from_report_with_tracker(
            run_id="test",
            report=report,
            started_at=utc_now(),
            finished_at=utc_now(),
            tracker=tracker,
        )

        assert event.retries == 2
        assert event.retries_by_reason == {"rate_limit": 1, "timeout": 1}
        assert event.retry_backoff_seconds == 3.0

    def test_budget_exceeded_during_retry_explicit(self):
        tracker = RetryTracker()
        tracker.record_retry(RetryReason.RATE_LIMIT)
        tracker.mark_budget_exceeded()

        report = self.make_report(
            outcome=Outcome.BUDGET_EXCEEDED,
            limits_exceeded=["tokens"],
        )
        event = RunEvent.from_report_with_tracker(
            run_id="test",
            report=report,
            started_at=utc_now(),
            finished_at=utc_now(),
            tracker=tracker,
        )

        assert event.budget_exceeded_during_retry is True

    def test_budget_exceeded_during_retry_inferred(self):
        report = self.make_report(
            outcome=Outcome.BUDGET_EXCEEDED,
            limits_exceeded=["tokens"],
        )
        event = RunEvent.from_execution_report(
            run_id="test",
            report=report,
            started_at=utc_now(),
            finished_at=utc_now(),
            retries=3,
        )

        assert event.budget_exceeded_during_retry is True

    def test_no_budget_attribution_without_retries(self):
        report = self.make_report(
            outcome=Outcome.BUDGET_EXCEEDED,
            limits_exceeded=["tokens"],
        )
        event = RunEvent.from_execution_report(
            run_id="test",
            report=report,
            started_at=utc_now(),
            finished_at=utc_now(),
            retries=0,
        )

        assert event.budget_exceeded_during_retry is False


class TestCollectorRetryMetrics:
    def make_event(
        self,
        retries: int = 0,
        retries_by_reason: dict | None = None,
        budget_exceeded_during_retry: bool = False,
    ) -> RunEvent:
        return RunEvent(
            run_id="test",
            outcome=Outcome.SUCCESS,
            success=True,
            started_at=utc_now(),
            finished_at=utc_now(),
            elapsed_seconds=1.0,
            retries=retries,
            retries_by_reason=retries_by_reason or {},
            budget_exceeded_during_retry=budget_exceeded_during_retry,
        )

    def test_retry_reason_distribution(self):
        collector = RunMetricsCollector()

        collector.observe(self.make_event(retries=2, retries_by_reason={"rate_limit": 2}))
        collector.observe(self.make_event(retries=1, retries_by_reason={"timeout": 1}))
        collector.observe(self.make_event(retries=3, retries_by_reason={"rate_limit": 2, "server_error": 1}))

        dist = collector.retry_reason_distribution()
        assert dist["rate_limit"] == 4
        assert dist["timeout"] == 1
        assert dist["server_error"] == 1

    def test_budget_exceeded_during_retry_counter(self):
        collector = RunMetricsCollector()

        collector.observe(self.make_event(budget_exceeded_during_retry=False))
        collector.observe(self.make_event(budget_exceeded_during_retry=True))
        collector.observe(self.make_event(budget_exceeded_during_retry=True))

        stats = collector.snapshot()
        assert stats["budget_exceeded_during_retry"] == 2
        assert stats["budget_exceeded_during_retry_rate"] == 2 / 3

    def test_prometheus_includes_retry_metrics(self):
        collector = RunMetricsCollector()
        collector.observe(self.make_event(
            retries=2,
            retries_by_reason={"rate_limit": 2},
            budget_exceeded_during_retry=True,
        ))

        prom = collector.prometheus_format()
        assert 'enzu_retries_total{reason="rate_limit"}' in prom
        assert "enzu_budget_exceeded_during_retry_total" in prom
