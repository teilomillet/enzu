"""Tests for Session conversation persistence."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from enzu import Session, SessionBudgetExceeded
from enzu.session import Exchange, _format_history
from tests.providers.mock_provider import MockProvider


def test_format_history_empty() -> None:
    """Empty exchanges returns empty string."""
    assert _format_history([]) == ""


def test_format_history_single() -> None:
    """Single exchange formats correctly."""
    exchanges = [Exchange(user="Hello", assistant="Hi there")]
    result = _format_history(exchanges)
    assert "== Previous Conversation ==" in result
    assert "User: Hello" in result
    assert "Assistant: Hi there" in result


def test_format_history_with_data() -> None:
    """Exchange with data snippet shows data indicator."""
    exchanges = [Exchange(user="Find bug", assistant="Found it", data_snippet="log data")]
    result = _format_history(exchanges)
    assert "[Data provided:" in result


def test_format_history_truncates() -> None:
    """History truncates when over max_chars."""
    exchanges = [
        Exchange(user="A" * 100, assistant="B" * 100),
        Exchange(user="C" * 100, assistant="D" * 100),
        Exchange(user="E" * 100, assistant="F" * 100),
    ]
    # With small max_chars, should drop oldest
    result = _format_history(exchanges, max_chars=300)
    # Most recent should be present
    assert "E" * 50 in result or "F" * 50 in result


def test_session_run_appends_exchange() -> None:
    """Each run() appends to conversation history."""
    mock_provider = MockProvider(main_outputs=[
        '```python\nFINAL("First answer")\n```',
        '```python\nFINAL("Second answer")\n```',
    ])

    # Patch at runtime level where RLM mode resolves providers
    with patch("enzu.runtime.local.resolve_provider", return_value=mock_provider):
        session = Session(model="mock")

        session.run("First task", cost=1.00)
        assert len(session) == 1
        assert session.exchanges[0].user == "First task"
        assert session.exchanges[0].assistant == "First answer"

        session.run("Second task", cost=1.00)
        assert len(session) == 2
        assert session.exchanges[1].user == "Second task"


def test_session_run_passes_history_as_data() -> None:
    """Subsequent runs receive history in data."""
    outputs = [
        '```python\nFINAL("Found the bug")\n```',
        '```python\nprint(context[1][:100])\nFINAL("Fixed it")\n```',
    ]
    mock_provider = MockProvider(main_outputs=outputs)

    # Patch at runtime level where RLM mode resolves providers
    with patch("enzu.runtime.local.resolve_provider", return_value=mock_provider):
        session = Session(model="mock")

        session.run("Find the bug", data="error logs", cost=1.00)
        session.run("Fix it", cost=1.00)

        # Second call should have received history
        # Check that provider was called twice
        assert len(mock_provider.calls) == 2


def test_session_save_load() -> None:
    """Session can be saved and loaded."""
    session = Session(model="gpt-4", provider="openai")
    session.exchanges.append(Exchange(user="Test", assistant="Response"))
    session.total_cost_usd = 0.05

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        session.save(path)

        loaded = Session.load(path, api_key="test-key")
        assert loaded.model == "gpt-4"
        assert loaded.provider == "openai"
        assert len(loaded.exchanges) == 1
        assert loaded.exchanges[0].user == "Test"
        assert loaded.total_cost_usd == 0.05
    finally:
        Path(path).unlink()


def test_session_clear() -> None:
    """clear() removes all exchanges."""
    session = Session(model="mock")
    session.exchanges.append(Exchange(user="Test", assistant="Response"))
    assert len(session) == 1

    session.clear()
    assert len(session) == 0


def test_session_to_dict() -> None:
    """Session serializes correctly."""
    session = Session(model="gpt-4")
    session.exchanges.append(Exchange(user="Hello", assistant="Hi"))

    d = session.to_dict()
    assert d["model"] == "gpt-4"
    assert len(d["exchanges"]) == 1
    assert d["exchanges"][0]["user"] == "Hello"


def test_session_history_property() -> None:
    """history property returns list of dicts."""
    session = Session(model="mock")
    session.exchanges.append(Exchange(user="A", assistant="B"))
    session.exchanges.append(Exchange(user="C", assistant="D"))

    history = session.history
    assert len(history) == 2
    assert history[0]["user"] == "A"
    assert history[1]["user"] == "C"


def test_session_repr() -> None:
    """__repr__ shows useful info."""
    session = Session(model="gpt-4")
    session.total_cost_usd = 0.123
    session.exchanges.append(Exchange(user="X", assistant="Y"))

    r = repr(session)
    assert "gpt-4" in r
    assert "exchanges=1" in r
    assert "0.123" in r


# Budget cap tests


def test_session_budget_cap_raises_when_exceeded() -> None:
    """Session raises SessionBudgetExceeded when cost cap is reached."""
    session = Session(model="mock", max_cost_usd=1.00)
    session.total_cost_usd = 1.00  # Simulate cost

    with pytest.raises(SessionBudgetExceeded) as exc_info:
        session.run("Do something", cost=1.00)

    assert exc_info.value.cost_used == 1.00
    assert exc_info.value.cost_cap == 1.00
    assert "raise_cost_cap" in str(exc_info.value)


def test_session_budget_cap_allows_under_limit() -> None:
    """Session allows runs when under budget cap."""
    mock_provider = MockProvider(main_outputs=[
        '```python\nFINAL("Done")\n```'
    ])

    # Patch at runtime level where RLM mode resolves providers
    with patch("enzu.runtime.local.resolve_provider", return_value=mock_provider):
        session = Session(model="mock", max_cost_usd=10.00)
        result = session.run("Task", cost=5.00)

    assert result == "Done"


def test_session_raise_cap() -> None:
    """raise_cost_cap increases the budget limit."""
    session = Session(model="mock", max_cost_usd=5.00)
    session.total_cost_usd = 5.00

    session.raise_cost_cap(10.00)

    assert session.max_cost_usd == 10.00
    assert session.remaining_budget == 5.00


def test_session_raise_cap_must_be_higher() -> None:
    """raise_cost_cap rejects lower values."""
    session = Session(model="mock", max_cost_usd=10.00)

    with pytest.raises(ValueError) as exc_info:
        session.raise_cost_cap(5.00)

    assert "higher" in str(exc_info.value)


def test_session_remaining_budget() -> None:
    """remaining_budget property works correctly."""
    session = Session(model="mock", max_cost_usd=10.00)
    assert session.remaining_budget == 10.00

    session.total_cost_usd = 3.50
    assert session.remaining_budget == 6.50


def test_session_remaining_budget_unlimited() -> None:
    """remaining_budget is None when no cap set."""
    session = Session(model="mock")
    assert session.remaining_budget is None


def test_session_budget_cap_persists() -> None:
    """Budget cap is saved and loaded."""
    session = Session(model="gpt-4", max_cost_usd=25.00)
    session.total_cost_usd = 10.00

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        session.save(path)
        loaded = Session.load(path)

        assert loaded.max_cost_usd == 25.00
        assert loaded.total_cost_usd == 10.00
        assert loaded.remaining_budget == 15.00
    finally:
        Path(path).unlink()


# Token budget cap tests


def test_session_token_cap_raises_when_exceeded() -> None:
    """Session raises SessionBudgetExceeded when token cap is reached."""
    session = Session(model="mock", max_tokens=1000)
    session.total_tokens = 1000  # Simulate token usage

    with pytest.raises(SessionBudgetExceeded) as exc_info:
        session.run("Do something", cost=1.00)

    assert exc_info.value.tokens_used == 1000
    assert exc_info.value.tokens_cap == 1000
    assert "raise_token_cap" in str(exc_info.value)


def test_session_token_cap_allows_under_limit() -> None:
    """Session allows runs when under token cap."""
    mock_provider = MockProvider(main_outputs=[
        '```python\nFINAL("Done")\n```'
    ])

    # Patch at runtime level where RLM mode resolves providers
    with patch("enzu.runtime.local.resolve_provider", return_value=mock_provider):
        session = Session(model="mock", max_tokens=10000)
        result = session.run("Task", cost=5.00)

    assert result == "Done"


def test_session_raise_token_cap() -> None:
    """raise_token_cap increases the token limit."""
    session = Session(model="mock", max_tokens=5000)
    session.total_tokens = 5000

    session.raise_token_cap(10000)

    assert session.max_tokens == 10000
    assert session.remaining_tokens == 5000


def test_session_raise_token_cap_must_be_higher() -> None:
    """raise_token_cap rejects lower values."""
    session = Session(model="mock", max_tokens=10000)

    with pytest.raises(ValueError) as exc_info:
        session.raise_token_cap(5000)

    assert "higher" in str(exc_info.value)


def test_session_remaining_tokens() -> None:
    """remaining_tokens property works correctly."""
    session = Session(model="mock", max_tokens=10000)
    assert session.remaining_tokens == 10000

    session.total_tokens = 3500
    assert session.remaining_tokens == 6500


def test_session_remaining_tokens_unlimited() -> None:
    """remaining_tokens is None when no cap set."""
    session = Session(model="mock")
    assert session.remaining_tokens is None


def test_session_token_cap_persists() -> None:
    """Token cap is saved and loaded."""
    session = Session(model="gpt-4", max_tokens=50000)
    session.total_tokens = 20000

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        session.save(path)
        loaded = Session.load(path)

        assert loaded.max_tokens == 50000
        assert loaded.total_tokens == 20000
        assert loaded.remaining_tokens == 30000
    finally:
        Path(path).unlink()


def test_session_both_caps() -> None:
    """Session can have both cost and token caps."""
    session = Session(model="mock", max_cost_usd=10.00, max_tokens=5000)

    assert session.max_cost_usd == 10.00
    assert session.max_tokens == 5000
    assert session.remaining_budget == 10.00
    assert session.remaining_tokens == 5000
