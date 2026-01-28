import argparse
import os
import sys

from dotenv import load_dotenv

from enzu import Budget, OpenAICompatProvider, SuccessCriteria, TaskSpec
from enzu.rlm import RLMEngine


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run an RLM task example.")
    parser.add_argument("--task-id", default="rlm-example")
    parser.add_argument("--input-text", required=True)
    parser.add_argument("--context-file", required=True)
    parser.add_argument("--model", default=os.getenv("OPENROUTER_MODEL"))
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY"))
    parser.add_argument("--referer", default=os.getenv("OPENROUTER_REFERER"))
    parser.add_argument("--app-name", default=os.getenv("OPENROUTER_APP_NAME"))
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--max-total-tokens", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=8,
        help="Maximum RLM steps before stopping.",
    )
    parser.add_argument("--require-substring", action="append", default=[])
    parser.add_argument("--require-regex", action="append", default=[])
    parser.add_argument("--min-word-count", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model:
        raise ValueError("Model is required via --model or OPENROUTER_MODEL.")
    if not args.api_key:
        raise ValueError("API key is required via --api-key or OPENROUTER_API_KEY.")
    if not os.path.exists(args.context_file):
        raise ValueError("Context file not found.")
    if not any(
        [
            args.max_output_tokens,
            args.max_total_tokens,
            args.max_seconds,
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

    with open(args.context_file, "r", encoding="utf-8") as handle:
        context = handle.read()

    budget = Budget(
        max_output_tokens=args.max_output_tokens,
        max_total_tokens=args.max_total_tokens,
        max_seconds=args.max_seconds,
    )
    criteria = SuccessCriteria(
        required_substrings=args.require_substring,
        required_regex=args.require_regex,
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
    engine = RLMEngine(max_steps=args.max_iterations)

    # Progress events track each RLM step without dumping full context.
    def on_progress(message: str) -> None:
        print(message)

    report = engine.run(task, provider, data=context, on_progress=on_progress)
    if args.verbose and report.steps:
        step0 = report.steps[0]
        for line in step0.prompt.splitlines():
            if line.startswith("Context length:"):
                print(line)
                break
        print("\n=== Step 0 Model Output ===")
        print(step0.model_output)
        if len(report.steps) > 1:
            print("\n=== Step 1 Model Output ===")
            print(report.steps[1].model_output)
        if report.errors:
            print("\n=== Errors ===")
            for error in report.errors:
                print(error)
    print("\n=== Final Report ===")
    print(f"success: {report.success}")
    print(f"limits_exceeded: {report.budget_usage.limits_exceeded}")
    if report.answer:
        print("\nAnswer:\n")
        print(report.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
