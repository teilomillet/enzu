"""
Stress Testing Demo: Test p95 behavior before production.

This example demonstrates how to use enzu's stress testing harness to:
1. Inject failure conditions (429s, timeouts, latency, error bursts)
2. Measure p95 cost/run and terminal state distributions
3. Identify how your system behaves under degraded conditions

Run:
    export OPENROUTER_API_KEY=...
    python examples/stress_testing_demo.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from enzu import Enzu  # noqa: E402
from enzu.stress import Scenario, run_scenarios, rules, faults, format_report  # noqa: E402
from enzu.retries import retry_tracking_context  # noqa: E402
from enzu.metrics import RunEvent  # noqa: E402


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def create_task_function(client: Enzu, prompt: str, token_limit: int):
    """
    Create a task function that uses the FailureInjector.

    The task function receives a FailureInjector and calls inject()
    before each provider call to trigger fault behavior.
    """
    def task(injector):
        with retry_tracking_context() as tracker:
            started = utc_now()

            try:
                # Inject fault before provider call
                injector.inject()

                # Run the actual task
                report = client.run(prompt, tokens=token_limit, return_report=True)

            except Exception:
                # Handle any exceptions from injection or actual errors
                # In this demo, we'll let them propagate for simplicity
                raise

            finished = utc_now()

            # Create RunEvent with retry tracking
            event = RunEvent.from_report_with_tracker(
                run_id=f"stress_{injector.call_count}",
                report=report,
                started_at=started,
                finished_at=finished,
                tracker=tracker,
            )

            return event

    return task


def main():
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY or OPENAI_API_KEY")

    provider = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "openai"
    model = "openai/gpt-4o-mini" if provider == "openrouter" else "gpt-4o-mini"

    client = Enzu(provider=provider, model=model)

    # Simple task for testing
    prompt = "Say hello in one sentence."
    token_limit = 100

    # Define scenarios
    scenarios = [
        Scenario("baseline", []),

        Scenario("high_429s", [
            rules.error_rate(0.3, faults.rate_limit_429),
        ]),

        Scenario("slow_provider", [
            rules.latency(0.5, 2.0),
        ]),

        Scenario("error_burst", [
            rules.burst(start=2, length=5, exc_factory=faults.server_error_503),
        ]),

        Scenario("mixed_failures", [
            rules.error_rate(0.2, faults.rate_limit_429),
            rules.latency(0.3, 1.5),
        ]),
    ]

    print("=" * 60)
    print("STRESS TESTING DEMO")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Task: {prompt}")
    print(f"Token limit: {token_limit}")
    print(f"Scenarios: {len(scenarios)}")
    print("Runs per scenario: 50")
    print()

    # Run scenarios
    task_fn = create_task_function(client, prompt, token_limit)

    print("Running scenarios (this may take a few minutes)...")
    print()

    reports = run_scenarios(
        task=task_fn,
        scenarios=scenarios,
        runs_per_scenario=50,
        seed=42,
    )

    # Print formatted report
    print(format_report(reports))

    # Print insights
    print("=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)

    for report in reports:
        s = report.snapshot
        scenario_name = report.scenario_name

        success_rate = s["outcome_distribution"].get("success", 0) / report.runs
        p95_retries = s["percentiles"]["retries"]["p95"]
        budget_exceeded_rate = s["budget_exceeded_during_retry_rate"]

        print(f"\n{scenario_name}:")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  p95 retries/run: {p95_retries:.1f}")
        print(f"  Budget exceeded during retry: {budget_exceeded_rate:.1%}")

        # Highlight issues
        if success_rate < 0.95:
            print("  ⚠️  Low success rate - consider increasing retry limits")
        if p95_retries > 5:
            print("  ⚠️  High p95 retries - potential cost multiplier")
        if budget_exceeded_rate > 0.1:
            print("  ⚠️  Budget exhaustion during retries - retry storms detected")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Compare p95 metrics across scenarios")
    print("2. Identify which failure modes cause budget exhaustion")
    print("3. Test with your actual workload and token limits")
    print("4. Set SLOs based on p95/p99, not averages")
    print("\nSee docs/STRESS_TESTING.md for more details.")


if __name__ == "__main__":
    main()
