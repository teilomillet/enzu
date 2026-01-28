import argparse
import os
import sys
from typing import List

from dotenv import load_dotenv

from enzu import Budget, Engine, OpenAICompatProvider, SuccessCriteria, TaskSpec


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run a chat task example.")
    parser.add_argument("--task-id", default="example-task")
    parser.add_argument("--input-text", required=True)
    parser.add_argument("--model", default=os.getenv("OPENROUTER_MODEL"))
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY"))
    parser.add_argument("--referer", default=os.getenv("OPENROUTER_REFERER"))
    parser.add_argument("--app-name", default=os.getenv("OPENROUTER_APP_NAME"))
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--max-total-tokens", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--require-substring", action="append", default=[])
    parser.add_argument("--require-regex", action="append", default=[])
    parser.add_argument("--min-word-count", type=int)
    parser.add_argument("--temperature", type=float)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model:
        raise ValueError("Model is required via --model or OPENROUTER_MODEL.")
    if not args.api_key:
        raise ValueError("API key is required via --api-key or OPENROUTER_API_KEY.")
    if not any(
        [
            args.max_output_tokens,
            args.max_total_tokens,
            args.max_seconds,
            args.max_cost_usd,
        ]
    ):
        raise ValueError("At least one budget limit is required.")
    if not any([args.require_substring, args.require_regex, args.min_word_count]):
        raise ValueError("At least one success criterion is required.")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    budget = Budget(
        max_output_tokens=args.max_output_tokens,
        max_total_tokens=args.max_total_tokens,
        max_seconds=args.max_seconds,
        max_cost_usd=args.max_cost_usd,
    )
    criteria = SuccessCriteria(
        required_substrings=_dedupe(args.require_substring),
        required_regex=_dedupe(args.require_regex),
        min_word_count=args.min_word_count,
    )
    task = TaskSpec(
        task_id=args.task_id,
        input_text=args.input_text,
        model=args.model,
        budget=budget,
        success_criteria=criteria,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
    )

    headers = {}
    if args.referer:
        headers["HTTP-Referer"] = args.referer
    if args.app_name:
        headers["X-Title"] = args.app_name
    provider = OpenAICompatProvider(
        name="openrouter",
        api_key=args.api_key,
        base_url="https://openrouter.ai/api/v1",
        headers=headers if headers else None,
    )
    engine = Engine()

    # Streaming progress exposes live generation and verification states.
    def on_progress(event) -> None:
        payload = f"{event.phase}: {event.message}"
        print(payload)

    report = engine.run(task, provider, on_progress=on_progress)
    print("\n=== Final Report ===")
    print(f"success: {report.success}")
    print(f"limits_exceeded: {report.budget_usage.limits_exceeded}")
    print(f"verification_passed: {report.verification.passed}")
    if report.output_text:
        print("\nOutput:\n")
        print(report.output_text)
    return 0


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        if value in seen:
            continue
        result.append(value)
        seen.add(value)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
