#!/usr/bin/env python3
"""
Report Service Demo: Goal + Corpus + Hard Budgets -> Bounded Output

Run: uv run examples/production/report_service/demo.py
"""

import json
import time
from pathlib import Path
from enzu import Enzu, Outcome

DOCS_DIR = Path(__file__).parent / "docs"
OUTPUT_DIR = Path(__file__).parent
BUDGET_TOKENS, BUDGET_SECONDS, BUDGET_COST = 1200, 120, 0.50


def load_corpus() -> str:
    parts = []
    for p in sorted(DOCS_DIR.glob("*.txt")):
        parts.append(f"=== {p.name} ===\n{p.read_text()}\n")
    return "\n".join(parts)


def main() -> None:
    print("=" * 60)
    print("REPORT SERVICE DEMO")
    print("=" * 60)

    corpus = load_corpus()
    doc_count = len(list(DOCS_DIR.glob("*.txt")))
    print(f"\nCorpus: {doc_count} docs, {len(corpus)} chars")
    print(f"Budget: {BUDGET_TOKENS} tokens, {BUDGET_SECONDS}s, ${BUDGET_COST}")

    task = """You are a research analyst. Analyze ALL provided documents and write a detailed report.

## EXECUTIVE SUMMARY
Write 2-3 sentences summarizing the overall theme.

## KEY FINDINGS
List 3-5 findings. Each MUST cite a source document by exact filename, e.g. (source: ai_safety_overview.txt)

## CONNECTIONS MAP
List 3+ relationships between documents:
- [ai_safety_overview.txt] <-> [reward_modeling_paper.txt]: explain connection
- [evaluation_frameworks.txt] <-> [governance_framework.txt]: explain connection
(continue with more connections)

## RECOMMENDATIONS
List 2-3 actionable recommendations based on the findings.

Write the full report now. Include all sections."""

    client = Enzu()
    start = time.time()
    report = client.run(
        task,
        data=corpus,
        tokens=BUDGET_TOKENS,
        seconds=BUDGET_SECONDS,
        cost=BUDGET_COST,
        return_report=True,
    )
    elapsed = time.time() - start

    answer = getattr(report, "answer", None) or getattr(report, "output_text", None)
    usage = report.budget_usage

    print(f"\nOutcome: {report.outcome.value}, Partial: {report.partial}")
    print(
        f"Elapsed: {elapsed:.1f}s, Tokens: {usage.output_tokens}/{usage.total_tokens}"
    )
    if usage.cost_usd:
        print(f"Cost: ${usage.cost_usd:.4f}")
    if usage.limits_exceeded:
        print(f"Limits exceeded: {usage.limits_exceeded}")

    if answer:
        (OUTPUT_DIR / "report.md").write_text(f"# AI Safety Report\n\n{answer}\n")
        print("\nWrote: report.md")

    trace = {
        "outcome": report.outcome.value,
        "partial": report.partial,
        "budget": {
            "tokens": BUDGET_TOKENS,
            "seconds": BUDGET_SECONDS,
            "cost": BUDGET_COST,
        },
        "usage": {
            "output": usage.output_tokens,
            "total": usage.total_tokens,
            "elapsed": usage.elapsed_seconds,
            "cost": usage.cost_usd,
            "limits_exceeded": usage.limits_exceeded,
        },
        "docs": doc_count,
    }
    (OUTPUT_DIR / "trace.json").write_text(json.dumps(trace, indent=2))
    print("Wrote: trace.json")

    print("\n" + "=" * 60)
    print(
        "SUCCESS"
        if report.outcome == Outcome.SUCCESS
        else f"OUTCOME: {report.outcome.value}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
