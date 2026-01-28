import argparse
from pathlib import Path
from typing import Optional

from enzu import Session, SessionBudgetExceeded


def _read_text(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8").strip()
    return text or None


def main() -> int:
    parser = argparse.ArgumentParser(description="File-based multi-turn researcher using Session.")
    parser.add_argument("--session", default="research_session.json", help="Session JSON path.")
    parser.add_argument("--prompt", required=True, help="Path to the research prompt text file.")
    parser.add_argument("--data", help="Optional path to supporting context text.")
    parser.add_argument("--output", default="research_output.txt", help="Path to write the answer.")
    parser.add_argument("--provider", default="openrouter", help="Provider name for new sessions.")
    parser.add_argument("--model", help="Model name for new sessions.")
    parser.add_argument("--max-cost-usd", type=float, help="Session cost cap.")
    parser.add_argument("--max-tokens", type=int, help="Session token cap.")
    parser.add_argument("--tokens", type=int, help="Max output tokens for this turn.")
    parser.add_argument("--seconds", type=float, help="Max seconds for this turn.")
    parser.add_argument("--cost", type=float, help="Max cost for this turn.")
    parser.add_argument("--max-steps", type=int, help="Max RLM steps for this turn.")
    args = parser.parse_args()

    session_path = Path(args.session)
    prompt_text = _read_text(args.prompt)
    if not prompt_text:
        raise ValueError("Prompt file is empty.")

    if session_path.exists():
        session = Session.load(str(session_path))
        if args.model and session.model != args.model:
            raise ValueError("Session model differs from --model; start a new session.")
        if args.provider and session.provider != args.provider:
            raise ValueError("Session provider differs from --provider; start a new session.")
    else:
        if not args.model:
            raise ValueError("--model is required for a new session.")
        session = Session(
            model=args.model,
            provider=args.provider,
            max_cost_usd=args.max_cost_usd,
            max_tokens=args.max_tokens,
        )

    data_text = _read_text(args.data)

    try:
        # Session prepends history to data; auto mode resolves to RLM once history exists.
        answer = session.run(
            prompt_text,
            data=data_text,
            tokens=args.tokens,
            seconds=args.seconds,
            cost=args.cost,
            max_steps=args.max_steps,
        )
    except SessionBudgetExceeded as exc:
        raise SystemExit(str(exc)) from exc

    Path(args.output).write_text(str(answer), encoding="utf-8")
    session.save(str(session_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
