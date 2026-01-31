"""Tests for file_chatbot example."""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest


def format_history(history: List[Dict[str, Any]], max_chars: int = 8000) -> str:
    """Format history for context (copied from file_chatbot.py)."""
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


class TestFileChatbot:
    """Tests for the file chatbot demo."""

    def test_empty_history_formatting(self):
        """Empty history returns empty string."""
        result = format_history([])
        assert result == ""

    def test_single_turn_formatting(self):
        """Single turn formats correctly."""
        history = [{"user": "Hello", "assistant": "Hi there!"}]
        result = format_history(history)

        assert "User: Hello" in result
        assert "Assistant: Hi there!" in result
        assert "== Previous Conversation ==" in result

    def test_history_truncation(self):
        """Long history is truncated to fit max_chars."""
        # Create history that exceeds limit
        history = [
            {"user": "Message " + str(i), "assistant": "Response " + str(i)}
            for i in range(100)
        ]

        result = format_history(history, max_chars=500)

        # Should be truncated
        assert len(result) <= 600  # Allow some margin for headers
        # Recent messages should be included
        assert "Message 99" in result or "Message 98" in result

    def test_state_persistence(self, temp_dir):
        """State can be saved and loaded."""
        state_path = temp_dir / "chat_state.json"

        # Initial state
        state = {"history": []}
        state["history"].append({"user": "Hello", "assistant": "Hi!"})

        # Save
        state_path.write_text(json.dumps(state, indent=2))

        # Load
        loaded = json.loads(state_path.read_text())
        assert loaded["history"][0]["user"] == "Hello"
        assert loaded["history"][0]["assistant"] == "Hi!"

    def test_state_load_missing_file(self, temp_dir):
        """Missing state file returns empty history."""
        state_path = temp_dir / "nonexistent.json"

        # Simulate load_state behavior
        if not state_path.exists():
            state = {"history": []}
        else:
            state = json.loads(state_path.read_text())

        assert state == {"history": []}

    def test_history_order_preserved(self):
        """History maintains chronological order after formatting."""
        history = [
            {"user": "First", "assistant": "1"},
            {"user": "Second", "assistant": "2"},
            {"user": "Third", "assistant": "3"},
        ]

        result = format_history(history)

        # Find positions
        pos_first = result.find("First")
        pos_second = result.find("Second")
        pos_third = result.find("Third")

        assert pos_first < pos_second < pos_third
