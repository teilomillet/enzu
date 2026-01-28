import os
from pathlib import Path

from enzu import Session, SessionBudgetExceeded

SESSION_PATH = Path("research_session.json")
PROMPT_PATH = Path("research_prompt.txt")
OUTPUT_PATH = Path("research_output.txt")

MODEL = os.getenv("OPENROUTER_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
PROVIDER = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "openai"


def main() -> None:
    if not PROMPT_PATH.exists():
        raise SystemExit("Create research_prompt.txt with your question.")
    prompt_text = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise SystemExit("research_prompt.txt is empty.")

    if SESSION_PATH.exists():
        session = Session.load(str(SESSION_PATH))
    else:
        session = Session(
            model=MODEL,
            provider=PROVIDER,
            max_cost_usd=0.05,
            max_tokens=1200,
        )

    try:
        answer = session.run(
            prompt_text,
            tokens=400,
            max_steps=4,
        )
    except SessionBudgetExceeded as exc:
        raise SystemExit(str(exc)) from exc

    OUTPUT_PATH.write_text(str(answer), encoding="utf-8")
    session.save(str(SESSION_PATH))


if __name__ == "__main__":
    main()
