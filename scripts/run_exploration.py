#!/usr/bin/env python3
"""Run ordeal coverage-guided exploration on enzu chaos tests.

Usage:
    python scripts/run_exploration.py                # quick (30s each)
    python scripts/run_exploration.py --deep          # deep (120s each)
    python scripts/run_exploration.py --class BudgetControllerChaos
"""

from __future__ import annotations

import argparse
import sys
import time

from ordeal.explore import Explorer


# All chaos test classes to explore
TEST_CLASSES = {
    "BudgetControllerChaos": ("tests.test_chaos_budget", "BudgetControllerChaos"),
    "ConcurrentBudgetChaos": ("tests.test_chaos_budget", "ConcurrentBudgetChaos"),
    "TinyBudgetChaos": ("tests.test_chaos_budget", "TinyBudgetChaos"),
    "EngineChaos": ("tests.test_chaos_engine", "EngineChaos"),
    "ProviderFallbackChaos": ("tests.test_chaos_engine", "ProviderFallbackChaos"),
    "SessionStateChaos": ("tests.test_chaos_session", "SessionStateChaos"),
    "ExchangeSerializationChaos": (
        "tests.test_chaos_session",
        "ExchangeSerializationChaos",
    ),
    "HistoryFormattingChaos": ("tests.test_chaos_session", "HistoryFormattingChaos"),
    "FloatingPointDriftProbe": (
        "tests.test_chaos_deep_probe",
        "FloatingPointDriftProbe",
    ),
    "SessionCostDriftProbe": ("tests.test_chaos_deep_probe", "SessionCostDriftProbe"),
    "AuditLogCompletenessProbe": (
        "tests.test_chaos_deep_probe",
        "AuditLogCompletenessProbe",
    ),
}


def load_class(module_path: str, class_name: str):
    import importlib
    import sys
    from pathlib import Path

    # Ensure project root is on sys.path for test imports
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)

    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def run_exploration(
    test_class,
    name: str,
    max_time: float,
    steps_per_run: int,
    seed: int = 42,
) -> dict:
    """Run Explorer on a single test class and return results."""
    explorer = Explorer(
        test_class,
        target_modules=["enzu.budget", "enzu.engine", "enzu.session", "enzu.models"],
        seed=seed,
        max_checkpoints=128,
        checkpoint_prob=0.5,
        checkpoint_strategy="energy",
        fault_toggle_prob=0.3,
        record_traces=True,
    )

    def on_progress(snap):
        print(
            f"  [{name}] runs={snap.total_runs}, edges={snap.unique_edges}, "
            f"failures={snap.failures}, checkpoints={snap.checkpoints}",
            end="\r",
        )

    result = explorer.run(
        max_time=max_time,
        steps_per_run=steps_per_run,
        shrink=True,
        progress=on_progress,
    )
    print()  # clear progress line
    return {
        "name": name,
        "runs": result.total_runs,
        "steps": result.total_steps,
        "unique_edges": result.unique_edges,
        "failures": len(result.failures),
        "failure_details": [
            f"{f.error_type}: {f.error_message}" for f in result.failures
        ],
        "checkpoints_saved": result.checkpoints_saved,
        "duration": result.duration_seconds,
    }


def main():
    parser = argparse.ArgumentParser(description="Run ordeal exploration on enzu")
    parser.add_argument(
        "--deep", action="store_true", help="Deep exploration (120s each)"
    )
    parser.add_argument("--class", dest="cls", help="Run a specific test class")
    parser.add_argument("--time", type=float, help="Custom time per class (seconds)")
    parser.add_argument("--steps", type=int, default=50, help="Steps per run")
    args = parser.parse_args()

    if args.time:
        max_time = args.time
    elif args.deep:
        max_time = 120.0
    else:
        max_time = 30.0

    classes_to_run = TEST_CLASSES
    if args.cls:
        if args.cls not in TEST_CLASSES:
            print(f"Unknown class: {args.cls}")
            print(f"Available: {', '.join(TEST_CLASSES)}")
            sys.exit(1)
        classes_to_run = {args.cls: TEST_CLASSES[args.cls]}

    print(f"Ordeal Explorer — {max_time}s per class, {args.steps} steps/run")
    print(f"Classes: {len(classes_to_run)}")
    print("=" * 70)

    all_results = []
    total_failures = 0
    total_edges = 0

    for name, (module, cls_name) in classes_to_run.items():
        print(f"\n>>> Exploring {name}...")
        test_cls = load_class(module, cls_name)
        start = time.time()
        result = run_exploration(test_cls, name, max_time, args.steps)
        elapsed = time.time() - start
        all_results.append(result)
        total_failures += result["failures"]
        total_edges += result["unique_edges"]

        status = "FAILURES FOUND" if result["failures"] else "clean"
        print(
            f"  {status}: {result['runs']} runs, {result['unique_edges']} edges, "
            f"{result['checkpoints_saved']} checkpoints, {elapsed:.1f}s"
        )
        if result["failure_details"]:
            for detail in result["failure_details"]:
                print(f"    BUG: {detail}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Classes explored: {len(all_results)}")
    print(f"Total unique edges: {total_edges}")
    print(f"Total failures:     {total_failures}")
    print()

    for r in all_results:
        marker = "BUG" if r["failures"] else "OK "
        print(
            f"  [{marker}] {r['name']:40s} "
            f"edges={r['unique_edges']:5d}  runs={r['runs']:5d}  "
            f"failures={r['failures']}"
        )

    if total_failures:
        print(f"\n{total_failures} FAILURE(S) FOUND — see details above")
        sys.exit(1)
    else:
        print("\nAll clean.")


if __name__ == "__main__":
    main()
