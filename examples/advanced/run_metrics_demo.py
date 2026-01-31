"""
Run Metrics Demo: p50/p95 cost/run + terminal state distributions.

This example demonstrates how to use enzu's first-class run metrics to:
1. Track per-run cost, time, and token usage
2. Compute p50/p95 percentiles for tail behavior analysis
3. Analyze terminal state distributions (success vs budget_exceeded, etc.)

Run:
    export OPENROUTER_API_KEY=...
    python examples/run_metrics_demo.py
"""

import os
import sys
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from enzu import Enzu, Outcome  # noqa: E402
from enzu.metrics import RunEvent, get_run_metrics, reset_run_metrics  # noqa: E402


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def main():
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY or OPENAI_API_KEY")

    reset_run_metrics()
    collector = get_run_metrics()

    provider = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "openai"
    model = "openai/gpt-4o-mini" if provider == "openrouter" else "gpt-4o-mini"

    client = Enzu(provider=provider, model=model)

    tasks = [
        ("Say hello in one sentence.", 100),
        ("What is 2+2? Answer with just the number.", 50),
        ("Write a haiku about Python.", 100),
        ("Explain recursion in one sentence.", 100),
        ("Count from 1 to 5.", 50),
        ("Write 500 words about AI.", 20),
        ("Name three colors.", 50),
        ("What day comes after Monday?", 50),
    ]

    print("Running tasks and collecting metrics...\n")

    for task, token_limit in tasks:
        run_id = str(uuid.uuid4())[:8]
        started_at = utc_now()

        report = client.run(task, tokens=token_limit, return_report=True)

        finished_at = utc_now()

        event = RunEvent.from_execution_report(
            run_id=run_id,
            report=report,
            started_at=started_at,
            finished_at=finished_at,
        )

        collector.observe(event)

        status = "✓" if report.success else "✗"
        outcome = report.outcome.value
        tokens = report.budget_usage.total_tokens or 0
        print(f"  {status} [{outcome:16}] {tokens:5} tokens | {task[:40]}...")

    print("\n" + "=" * 60)
    print("METRICS SNAPSHOT")
    print("=" * 60)

    stats = collector.snapshot()

    print(f"\nTotal runs: {stats['total_runs']}")
    print(f"Cost coverage: {stats['cost_coverage']:.1%} of runs have cost data")

    print("\n--- Percentiles (tail behavior) ---")
    percentiles = stats["percentiles"]

    for metric in ["elapsed_seconds", "cost_usd", "total_tokens"]:
        p = percentiles[metric]
        p50 = f"{p['p50']:.4f}" if p["p50"] is not None else "N/A"
        p95 = f"{p['p95']:.4f}" if p["p95"] is not None else "N/A"
        p99 = f"{p['p99']:.4f}" if p["p99"] is not None else "N/A"
        print(f"  {metric:16} p50={p50:>10}  p95={p95:>10}  p99={p99:>10}")

    print("\n--- Terminal State Distribution ---")
    for outcome_key, count in sorted(stats["outcome_distribution"].items()):
        print(f"  {outcome_key}: {count}")

    print("\n--- Averages ---")
    avgs = stats["averages"]
    for metric, value in avgs.items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")

    print("\n" + "=" * 60)
    print("PROMETHEUS FORMAT (excerpt)")
    print("=" * 60)
    prom = collector.prometheus_format()
    for line in prom.split("\n")[:20]:
        print(line)
    print("...")

    print("\n" + "=" * 60)
    print("JSON LOG FORMAT (single event)")
    print("=" * 60)
    print(json.dumps(event.to_log_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
