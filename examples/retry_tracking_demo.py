"""
Retry Tracking Demo: See retry behavior as a budget signal.

This example demonstrates how retries contribute to budget exhaustion
and how to track them for production observability.

Run:
    export OPENROUTER_API_KEY=...
    python examples/retry_tracking_demo.py
"""

import os
import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from enzu import Enzu  # noqa: E402
from enzu.metrics import RunEvent, RunMetricsCollector  # noqa: E402
from enzu.retries import retry_tracking_context  # noqa: E402


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def main():
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY or OPENAI_API_KEY")

    collector = RunMetricsCollector()

    provider = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "openai"
    model = "openai/gpt-4o-mini" if provider == "openrouter" else "gpt-4o-mini"

    client = Enzu(provider=provider, model=model)

    print("=== RETRY TRACKING DEMO ===\n")

    print("1. Simulating runs with retry tracking context...")
    print("   (Real retries happen transparently inside provider calls)\n")

    for i in range(5):
        run_id = str(uuid.uuid4())[:8]

        with retry_tracking_context() as tracker:
            started = utc_now()

            report = client.run(
                f"Say '{i}' and nothing else.",
                tokens=50,
                return_report=True,
            )

            finished = utc_now()

            event = RunEvent.from_report_with_tracker(
                run_id=run_id,
                report=report,
                started_at=started,
                finished_at=finished,
                tracker=tracker,
            )
            collector.observe(event)

            status = "✓" if report.success else "✗"
            retries = tracker.total_retries
            print(f"   {status} Run {i+1}: {report.outcome.value}, retries={retries}")

    print("\n2. Simulating runs with synthetic retry data...")
    print("   (For demo purposes, showing what metrics look like with retries)\n")

    synthetic_scenarios = [
        {"retries": 0, "reason": None, "budget_hit": False},
        {"retries": 2, "reason": "rate_limit", "budget_hit": False},
        {"retries": 5, "reason": "rate_limit", "budget_hit": True},
        {"retries": 1, "reason": "timeout", "budget_hit": False},
        {"retries": 3, "reason": "server_error", "budget_hit": True},
    ]

    for i, scenario in enumerate(synthetic_scenarios):
        retries_by_reason = {}
        if scenario["reason"]:
            retries_by_reason[scenario["reason"]] = scenario["retries"]

        event = RunEvent(
            run_id=f"synthetic-{i}",
            outcome="budget_exceeded" if scenario["budget_hit"] else "success",
            success=not scenario["budget_hit"],
            started_at=utc_now(),
            finished_at=utc_now(),
            elapsed_seconds=1.0 + scenario["retries"] * 2.0,
            retries=scenario["retries"],
            retries_by_reason=retries_by_reason,
            budget_exceeded_during_retry=scenario["budget_hit"] and scenario["retries"] > 0,
        )
        collector.observe(event)

        status = "✓" if event.success else "✗"
        attr = f"retries={scenario['retries']}"
        if scenario["budget_hit"]:
            attr += ", BUDGET HIT"
        print(f"   {status} Synthetic {i+1}: {attr}")

    print("\n" + "=" * 60)
    print("METRICS SNAPSHOT")
    print("=" * 60)

    stats = collector.snapshot()

    print(f"\nTotal runs: {stats['total_runs']}")

    print("\n--- Retry Percentiles ---")
    p = stats["percentiles"]["retries"]
    print(f"  p50: {p['p50']:.1f} retries/run")
    print(f"  p95: {p['p95']:.1f} retries/run")
    print(f"  p99: {p['p99']:.1f} retries/run")

    print("\n--- Retry Reason Distribution ---")
    for reason, count in stats["retry_reason_distribution"].items():
        print(f"  {reason}: {count}")

    print("\n--- Budget Attribution ---")
    print(f"  Runs where budget exceeded during retry: {stats['budget_exceeded_during_retry']}")
    print(f"  Rate: {stats['budget_exceeded_during_retry_rate']:.1%}")

    print("\n--- Outcome Distribution ---")
    for outcome, count in stats["outcome_distribution"].items():
        print(f"  {outcome}: {count}")

    print("\n" + "=" * 60)
    print("PROMETHEUS FORMAT (retry metrics)")
    print("=" * 60)
    prom = collector.prometheus_format()
    for line in prom.split("\n"):
        if "retry" in line.lower() or "budget_exceeded_during" in line.lower():
            print(line)

    print("\n=== KEY INSIGHT ===")
    print("When p95 retries/run is high AND budget_exceeded_during_retry_rate")
    print("is significant, you have a correlated retry storm problem.")
    print("Mitigate with: jitter, queueing, concurrency limits, circuit breakers.")


if __name__ == "__main__":
    main()
