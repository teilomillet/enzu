#!/usr/bin/env python3
"""
Multi-turn chat with history persistence.

This example demonstrates stateful conversation with:
- History truncation to fit context windows (8k char limit)
- State persistence between runs via JSON
- Simple file-based input/output interface

Run:
    # Create your input
    echo "Hello! What can you help me with?" > chat_input.txt

    # Set your API key
    export OPENAI_API_KEY=sk-...

    # Run the chat
    python examples/production/file_chatbot.py

    # Check the response
    cat chat_output.txt

    # Continue the conversation
    echo "Tell me more about that" > chat_input.txt
    python examples/production/file_chatbot.py
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from enzu import run

STATE_PATH = Path("chat_state.json")
INPUT_PATH = Path("chat_input.txt")
OUTPUT_PATH = Path("chat_output.txt")

MODEL = os.getenv("OPENROUTER_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
PROVIDER = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "openai"


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"history": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload.get("history", []) if isinstance(payload, dict) else []
    return {"history": history}


def format_history(history: List[Dict[str, Any]], max_chars: int = 8000) -> str:
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


def main() -> None:
    if not INPUT_PATH.exists():
        raise SystemExit("Create chat_input.txt with your prompt.")

    user_text = INPUT_PATH.read_text(encoding="utf-8").strip()
    if not user_text:
        raise SystemExit("chat_input.txt is empty.")

    state = load_state(STATE_PATH)
    history_text = format_history(state["history"])
    prompt = f"{history_text}\nUser: {user_text}\nAssistant:".strip()

    report = run(
        prompt,
        provider=PROVIDER,
        model=MODEL,
        mode="chat",
        tokens=300,
        return_report=True,
    )
    answer = getattr(report, "output_text", None) or ""

    OUTPUT_PATH.write_text(answer, encoding="utf-8")
    state["history"].append({"user": user_text, "assistant": answer})
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
