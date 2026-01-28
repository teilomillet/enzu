import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from enzu import run


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"history": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("State file must be a JSON object.")
    history = payload.get("history", [])
    if not isinstance(history, list):
        raise ValueError("State file must include a list field named 'history'.")
    return {"history": history}


def _format_history(history: List[Dict[str, Any]], max_chars: int) -> str:
    if not history:
        return ""
    parts: List[str] = []
    total = 0
    for turn in reversed(history):
        user = str(turn.get("user", "")).strip()
        assistant = str(turn.get("assistant", "")).strip()
        if not user and not assistant:
            continue
        block = f"User: {user}\nAssistant: {assistant}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    if not parts:
        return ""
    parts.reverse()
    return "== Previous Conversation ==\n" + "\n".join(parts) + "\n== End Previous ==\n"


def _build_prompt(system: str, history_text: str, user_text: str) -> str:
    system = system.strip()
    user_text = user_text.strip()
    sections: List[str] = []
    if system:
        sections.append(system)
    if history_text:
        sections.append(history_text)
    sections.append(f"User: {user_text}\nAssistant:")
    return "\n\n".join(sections).strip()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="File-based chatbot using enzu run().")
    parser.add_argument("--state", default="chat_state.json", help="JSON file storing chat history.")
    parser.add_argument("--input", required=True, help="Path to user input text file.")
    parser.add_argument("--output", default="chat_output.txt", help="Path to write assistant response.")
    parser.add_argument("--report", help="Optional path to write JSON report.")
    parser.add_argument("--provider", default="openrouter", help="Provider name.")
    parser.add_argument("--model", required=True, help="Model name.")
    parser.add_argument("--system", default="You are a helpful assistant.", help="System instruction prefix.")
    parser.add_argument("--max-history-chars", type=int, default=8000, help="Max history chars to include.")
    parser.add_argument("--tokens", type=int, help="Max output tokens.")
    args = parser.parse_args()

    state_path = Path(args.state)
    input_path = Path(args.input)
    output_path = Path(args.output)

    user_text = input_path.read_text(encoding="utf-8").strip()
    if not user_text:
        raise ValueError("Input file is empty.")

    state = _load_state(state_path)
    # Persist history in a file so each process can rebuild the prompt.
    history_text = _format_history(state["history"], args.max_history_chars)
    prompt = _build_prompt(args.system, history_text, user_text)

    # Force chat mode; history is embedded in the prompt.
    report = run(
        prompt,
        provider=args.provider,
        model=args.model,
        mode="chat",
        tokens=args.tokens,
        return_report=True,
    )
    answer = getattr(report, "output_text", None) or ""

    output_path.write_text(answer, encoding="utf-8")

    state["history"].append({"user": user_text, "assistant": answer})
    _write_json(state_path, state)

    if args.report and hasattr(report, "model_dump"):
        model_dump = getattr(report, "model_dump")
        _write_json(Path(args.report), model_dump(mode="json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
