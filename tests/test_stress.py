"""Tests for stress testing harness: fault injection and scenario testing."""

import random
from datetime import datetime, timezone

import pytest

from enzu.stress import (
    Scenario,
    Fault,
    CallContext,
    run_scenarios,
    rules,
    faults,
    format_report,
)
from enzu.stress.harness import FailureInjector
from enzu.metrics import RunEvent
from enzu.models import Outcome, BudgetUsage, ExecutionReport, VerificationResult


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def make_report(
    success: bool = True,
    outcome: Outcome = Outcome.SUCCESS,
    elapsed: float = 1.0,
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> ExecutionReport:
    return ExecutionReport(
        success=success,
        outcome=outcome,
        partial=False,
        task_id="stress-test",
        provider="test-provider",
        model="test-model",
        output_text="test output",
        verification=VerificationResult(passed=success),
        budget_usage=BudgetUsage(
            elapsed_seconds=elapsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=0.001,
        ),
    )


class TestFault:
    def test_delay_only(self):
        fault = Fault(delay_seconds=1.5)
        assert fault.delay_seconds == 1.5
        assert fault.exception is None

    def test_exception_only(self):
        exc = ValueError("test error")
        fault = Fault(exception=exc)
        assert fault.delay_seconds == 0.0
        assert fault.exception is exc

    def test_delay_and_exception(self):
        exc = ValueError("test")
        fault = Fault(delay_seconds=2.0, exception=exc)
        assert fault.delay_seconds == 2.0
        assert fault.exception is exc


class TestCallContext:
    def test_creation(self):
        rng = random.Random(42)
        ctx = CallContext(call_index=5, run_index=2, rng=rng)
        assert ctx.call_index == 5
        assert ctx.run_index == 2
        assert ctx.rng is rng


class TestScenario:
    def test_empty_scenario(self):
        scenario = Scenario("baseline", [])
        ctx = CallContext(call_index=1, run_index=0, rng=random.Random(42))
        fault = scenario.decide(ctx)
        assert fault is None

    def test_scenario_with_rules(self):
        # Rule that always triggers
        rule = rules.nth_call({1}, faults.rate_limit_429)
        scenario = Scenario("test", [rule])

        # Call 1 should trigger
        ctx1 = CallContext(call_index=1, run_index=0, rng=random.Random(42))
        fault1 = scenario.decide(ctx1)
        assert fault1 is not None
        assert fault1.exception is not None

        # Call 2 should not trigger
        ctx2 = CallContext(call_index=2, run_index=0, rng=random.Random(42))
        fault2 = scenario.decide(ctx2)
        assert fault2 is None

    def test_scenario_first_match_wins(self):
        # Two rules: first always triggers, second would also trigger
        rule1 = rules.latency(1.0, 1.0)  # Always delay
        rule2 = rules.error_rate(1.0, faults.rate_limit_429)  # Always error

        scenario = Scenario("test", [rule1, rule2])
        ctx = CallContext(call_index=1, run_index=0, rng=random.Random(42))
        fault = scenario.decide(ctx)

        # Should get latency fault (first rule), not error
        assert fault is not None
        assert fault.delay_seconds == 1.0
        assert fault.exception is None

    def test_scenario_addition(self):
        s1 = Scenario("s1", [rules.latency(0.5, 1.0)])
        s2 = Scenario("s2", [rules.error_rate(0.3, faults.timeout)])

        combined = s1 + s2
        assert combined.name == "s1+s2"
        assert len(combined.rules) == 2


class TestLatencyRule:
    def test_always_trigger(self):
        rule = rules.latency(p=1.0, delay_seconds=2.5)
        ctx = CallContext(call_index=1, run_index=0, rng=random.Random(42))
        fault = rule.maybe_fault(ctx)
        assert fault is not None
        assert fault.delay_seconds == 2.5
        assert fault.exception is None

    def test_never_trigger(self):
        rule = rules.latency(p=0.0, delay_seconds=2.5)
        ctx = CallContext(call_index=1, run_index=0, rng=random.Random(42))
        fault = rule.maybe_fault(ctx)
        assert fault is None

    def test_probabilistic(self):
        rule = rules.latency(p=0.5, delay_seconds=1.0)
        rng = random.Random(42)

        # Run multiple times to check probability
        triggered = 0
        for i in range(100):
            ctx = CallContext(call_index=i, run_index=0, rng=rng)
            fault = rule.maybe_fault(ctx)
            if fault is not None:
                triggered += 1

        # Should be roughly 50% (allow some variance)
        assert 30 < triggered < 70


class TestErrorRateRule:
    def test_always_trigger(self):
        rule = rules.error_rate(p=1.0, exc_factory=faults.rate_limit_429)
        ctx = CallContext(call_index=1, run_index=0, rng=random.Random(42))
        fault = rule.maybe_fault(ctx)
        assert fault is not None
        assert fault.exception is not None

    def test_never_trigger(self):
        rule = rules.error_rate(p=0.0, exc_factory=faults.rate_limit_429)
        ctx = CallContext(call_index=1, run_index=0, rng=random.Random(42))
        fault = rule.maybe_fault(ctx)
        assert fault is None

    def test_creates_new_exception_each_time(self):
        rule = rules.error_rate(p=1.0, exc_factory=faults.rate_limit_429)
        ctx = CallContext(call_index=1, run_index=0, rng=random.Random(42))

        fault1 = rule.maybe_fault(ctx)
        fault2 = rule.maybe_fault(ctx)

        # Should get different exception instances
        assert fault1 is not None
        assert fault2 is not None
        assert fault1.exception is not fault2.exception


class TestBurstRule:
    def test_burst_window(self):
        rule = rules.burst(start=5, length=3, exc_factory=faults.server_error_500)
        rng = random.Random(42)

        # Before burst
        ctx = CallContext(call_index=4, run_index=0, rng=rng)
        assert rule.maybe_fault(ctx) is None

        # During burst (5, 6, 7)
        for call_idx in [5, 6, 7]:
            ctx = CallContext(call_index=call_idx, run_index=0, rng=rng)
            fault = rule.maybe_fault(ctx)
            assert fault is not None
            assert fault.exception is not None

        # After burst
        ctx = CallContext(call_index=8, run_index=0, rng=rng)
        assert rule.maybe_fault(ctx) is None


class TestNthCallRule:
    def test_specific_calls(self):
        rule = rules.nth_call({1, 5, 10}, faults.timeout)
        rng = random.Random(42)

        # Test calls 1-11
        for call_idx in range(1, 12):
            ctx = CallContext(call_index=call_idx, run_index=0, rng=rng)
            fault = rule.maybe_fault(ctx)

            if call_idx in {1, 5, 10}:
                assert fault is not None
                assert fault.exception is not None
            else:
                assert fault is None


class TestFaults:
    def test_rate_limit_429(self):
        exc = faults.rate_limit_429()
        assert exc.__class__.__name__ == "RateLimitError"

    def test_timeout(self):
        exc = faults.timeout()
        assert exc.__class__.__name__ == "APITimeoutError"

    def test_connection_error(self):
        exc = faults.connection_error()
        assert exc.__class__.__name__ == "APIConnectionError"

    def test_server_error_500(self):
        exc = faults.server_error_500()
        assert exc.__class__.__name__ == "APIStatusError"

    def test_server_error_503(self):
        exc = faults.server_error_503()
        assert exc.__class__.__name__ == "APIStatusError"


class TestFailureInjector:
    def test_no_faults(self):
        scenario = Scenario("baseline", [])
        injector = FailureInjector(scenario, run_index=0, seed=42)

        # Should not raise
        injector.inject()
        injector.inject()

        assert injector.call_count == 2

    def test_inject_increments_call_count(self):
        scenario = Scenario("baseline", [])
        injector = FailureInjector(scenario, run_index=0, seed=42)

        assert injector.call_count == 0
        injector.inject()
        assert injector.call_count == 1
        injector.inject()
        assert injector.call_count == 2

    def test_inject_raises_exception(self):
        scenario = Scenario("errors", [rules.error_rate(1.0, faults.rate_limit_429)])
        injector = FailureInjector(scenario, run_index=0, seed=42)

        with pytest.raises(Exception):
            injector.inject()

    def test_inject_delay(self):
        scenario = Scenario("slow", [rules.latency(1.0, 0.01)])  # 10ms delay
        injector = FailureInjector(scenario, run_index=0, seed=42)

        import time
        start = time.time()
        injector.inject()
        elapsed = time.time() - start

        # Should have delayed at least 10ms (allow some tolerance)
        assert elapsed >= 0.009


class TestRunScenarios:
    def test_baseline_scenario(self):
        def task(injector):
            injector.inject()  # No faults, should not raise
            report = make_report()
            started = utc_now()
            return RunEvent.from_execution_report(
                run_id=f"run_{injector.call_count}",
                report=report,
                started_at=started,
                finished_at=started,
            )

        scenarios = [Scenario("baseline", [])]
        reports = run_scenarios(task=task, scenarios=scenarios, runs_per_scenario=10)

        assert len(reports) == 1
        report = reports[0]
        assert report.scenario_name == "baseline"
        assert report.runs == 10
        assert report.snapshot["total_runs"] == 10

    def test_multiple_scenarios(self):
        def task(injector):
            injector.inject()
            report = make_report()
            started = utc_now()
            return RunEvent.from_execution_report(
                run_id=f"run_{injector.call_count}",
                report=report,
                started_at=started,
                finished_at=started,
            )

        scenarios = [
            Scenario("baseline", []),
            Scenario("slow", [rules.latency(0.5, 0.01)]),
        ]

        reports = run_scenarios(task=task, scenarios=scenarios, runs_per_scenario=5)

        assert len(reports) == 2
        assert reports[0].scenario_name == "baseline"
        assert reports[1].scenario_name == "slow"
        assert reports[0].runs == 5
        assert reports[1].runs == 5

    def test_error_scenario_captured(self):
        """Test that errors from fault injection are handled properly."""
        def task(injector):
            try:
                injector.inject()  # May raise
                report = make_report(success=True)
            except Exception:
                # Simulate failure outcome
                report = make_report(success=False, outcome=Outcome.PROVIDER_ERROR)

            started = utc_now()
            return RunEvent.from_execution_report(
                run_id=f"run_{injector.call_count}",
                report=report,
                started_at=started,
                finished_at=started,
            )

        # Every call fails
        scenarios = [Scenario("errors", [rules.error_rate(1.0, faults.rate_limit_429)])]
        reports = run_scenarios(task=task, scenarios=scenarios, runs_per_scenario=10)

        assert len(reports) == 1
        snapshot = reports[0].snapshot

        # All runs should have failed
        assert snapshot["outcome_distribution"]["provider_error|partial=0"] == 10


class TestFormatReport:
    def test_format_single_scenario(self):
        def task(injector):
            injector.inject()
            report = make_report()
            started = utc_now()
            return RunEvent.from_execution_report(
                run_id=f"run_{injector.call_count}",
                report=report,
                started_at=started,
                finished_at=started,
            )

        scenarios = [Scenario("baseline", [])]
        reports = run_scenarios(task=task, scenarios=scenarios, runs_per_scenario=5)

        output = format_report(reports)

        # Check for expected sections
        assert "SCENARIO: baseline" in output
        assert "Runs: 5" in output
        assert "--- Outcomes ---" in output
        assert "--- Retry Metrics ---" in output
        assert "--- Latency ---" in output

    def test_format_multiple_scenarios(self):
        def task(injector):
            injector.inject()
            report = make_report()
            started = utc_now()
            return RunEvent.from_execution_report(
                run_id=f"run_{injector.call_count}",
                report=report,
                started_at=started,
                finished_at=started,
            )

        scenarios = [
            Scenario("baseline", []),
            Scenario("slow", [rules.latency(0.5, 0.01)]),
        ]
        reports = run_scenarios(task=task, scenarios=scenarios, runs_per_scenario=3)

        output = format_report(reports)

        assert "SCENARIO: baseline" in output
        assert "SCENARIO: slow" in output
