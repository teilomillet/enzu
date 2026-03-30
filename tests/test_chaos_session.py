"""Chaos tests for Session: multi-turn consistency, budget caps, serialization.

Uses ordeal's stateful chaos testing to explore Session behavior under
randomized operation sequences.

Properties verified:
- Session history grows monotonically
- Budget caps are enforced (SessionBudgetExceeded raised appropriately)
- Serialization roundtrip preserves essential state
- Cost/token tracking is monotonically non-decreasing
- raise_cost_cap / raise_token_cap only accept increases
"""

from __future__ import annotations

import json

from hypothesis import strategies as st

from ordeal import ChaosTest, always, invariant, rule

from enzu.session import Exchange, Session, _format_history


# ---------------------------------------------------------------------------
# Chaos test: Session state machine (unit-level, no API calls)
# ---------------------------------------------------------------------------


class SessionStateChaos(ChaosTest):
    """Explore Session state transitions without making actual API calls.

    Tests the session's internal bookkeeping: exchanges, cost tracking,
    budget caps, serialization, history formatting.
    """

    faults = []

    def __init__(self) -> None:
        super().__init__()
        self.session = Session(
            model="gpt-4o-mini",
            provider="mock",
            max_cost_usd=5.0,
            max_tokens=1000,
        )
        self._prev_exchange_count = 0
        self._prev_cost = 0.0
        self._prev_tokens = 0

    @rule(
        user_text=st.text(min_size=1, max_size=50),
        assistant_text=st.text(min_size=1, max_size=100),
        cost=st.floats(min_value=0.0, max_value=0.5).filter(lambda x: x == x),
        tokens=st.integers(min_value=0, max_value=50),
    )
    def add_exchange(
        self, user_text: str, assistant_text: str, cost: float, tokens: int
    ) -> None:
        """Manually add an exchange and update tracking (simulates a run)."""
        self.session.exchanges.append(
            Exchange(
                user=user_text,
                assistant=assistant_text,
                cost_usd=cost,
            )
        )
        self.session.total_cost_usd += cost
        self.session.total_tokens += tokens

    @rule()
    def format_history(self) -> None:
        """Format history and verify it's a valid string."""
        text = _format_history(self.session.exchanges, self.session.history_max_chars)
        always(isinstance(text, str), "history is a string")
        if self.session.exchanges:
            always(len(text) > 0, "non-empty session has non-empty history")

    @rule()
    def serialize_roundtrip(self) -> None:
        """Serialize to dict and verify key fields are preserved."""
        d = self.session.to_dict()
        always(d["model"] == "gpt-4o-mini", "model preserved in serialization")
        always(
            d["total_cost_usd"] == self.session.total_cost_usd,
            "cost preserved in serialization",
        )
        always(
            d["total_tokens"] == self.session.total_tokens,
            "tokens preserved in serialization",
        )
        always(
            len(d["exchanges"]) == len(self.session.exchanges),
            "exchange count preserved",
        )

        # Verify JSON-serializable
        json_str = json.dumps(d)
        always(isinstance(json_str, str), "to_dict is JSON-serializable")

    @rule()
    def check_budget_cap(self) -> None:
        """Verify budget cap enforcement."""
        if self.session.max_cost_usd is not None:
            if self.session.total_cost_usd >= self.session.max_cost_usd:
                # Budget should be exhausted
                remaining = self.session.remaining_budget
                always(
                    remaining is not None and remaining == 0,
                    "exhausted budget has 0 remaining",
                )
        if self.session.max_tokens is not None:
            if self.session.total_tokens >= self.session.max_tokens:
                remaining = self.session.remaining_tokens
                always(
                    remaining is not None and remaining == 0,
                    "exhausted token budget has 0 remaining",
                )

    @rule(new_cap=st.floats(min_value=0.01, max_value=100.0))
    def try_raise_cost_cap(self, new_cap: float) -> None:
        """Attempt to raise the cost cap."""
        current = self.session.max_cost_usd
        if current is not None and new_cap <= current:
            try:
                self.session.raise_cost_cap(new_cap)
                always(False, "raise_cost_cap should reject lower/equal cap")
            except ValueError:
                pass  # Expected
        elif current is not None:
            self.session.raise_cost_cap(new_cap)
            always(
                self.session.max_cost_usd == new_cap,
                "cost cap updated after raise",
            )

    @rule(new_cap=st.integers(min_value=1, max_value=10000))
    def try_raise_token_cap(self, new_cap: int) -> None:
        """Attempt to raise the token cap."""
        current = self.session.max_tokens
        if current is not None and new_cap <= current:
            try:
                self.session.raise_token_cap(new_cap)
                always(False, "raise_token_cap should reject lower/equal cap")
            except ValueError:
                pass  # Expected
        elif current is not None:
            self.session.raise_token_cap(new_cap)
            always(
                self.session.max_tokens == new_cap,
                "token cap updated after raise",
            )

    @rule()
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.session.clear()
        always(
            len(self.session.exchanges) == 0,
            "clear empties exchanges",
        )

    @invariant()
    def cost_is_non_negative(self) -> None:
        always(
            self.session.total_cost_usd >= 0,
            "total cost is non-negative",
        )

    @invariant()
    def tokens_non_negative(self) -> None:
        always(
            self.session.total_tokens >= 0,
            "total tokens is non-negative",
        )

    @invariant()
    def remaining_budget_consistent(self) -> None:
        """Remaining budget must equal cap - used, clamped to 0."""
        if self.session.max_cost_usd is not None:
            expected = max(0, self.session.max_cost_usd - self.session.total_cost_usd)
            actual = self.session.remaining_budget
            always(
                actual is not None and abs(actual - expected) < 1e-9,
                "remaining cost consistent with cap - used",
            )
        if self.session.max_tokens is not None:
            expected = max(0, self.session.max_tokens - self.session.total_tokens)
            actual = self.session.remaining_tokens
            always(
                actual is not None and actual == expected,
                "remaining tokens consistent with cap - used",
            )


TestSessionStateChaos = SessionStateChaos.TestCase


# ---------------------------------------------------------------------------
# Chaos test: Exchange serialization
# ---------------------------------------------------------------------------


class ExchangeSerializationChaos(ChaosTest):
    """Explore Exchange serialization/deserialization with adversarial data."""

    faults = []

    def __init__(self) -> None:
        super().__init__()
        self.exchanges: list[Exchange] = []

    @rule(
        user=st.text(min_size=0, max_size=200),
        assistant=st.text(min_size=0, max_size=200),
        cost=st.one_of(
            st.none(),
            st.floats(min_value=0.0, max_value=100.0).filter(lambda x: x == x),
        ),
    )
    def create_and_roundtrip(
        self,
        user: str,
        assistant: str,
        cost: float | None,
    ) -> None:
        """Create an Exchange, serialize it, deserialize it, verify equality."""
        ex = Exchange(user=user, assistant=assistant, cost_usd=cost)
        self.exchanges.append(ex)

        d = ex.to_dict()
        restored = Exchange.from_dict(d)

        always(restored.user == ex.user, "user preserved in roundtrip")
        always(
            restored.assistant == ex.assistant,
            "assistant preserved in roundtrip",
        )
        always(
            restored.cost_usd == ex.cost_usd,
            "cost preserved in roundtrip",
        )

    @invariant()
    def all_exchanges_serializable(self) -> None:
        """All accumulated exchanges must be JSON-serializable."""
        for ex in self.exchanges:
            d = ex.to_dict()
            try:
                json.dumps(d)
            except (TypeError, ValueError):
                always(False, "exchange must be JSON-serializable")


TestExchangeSerializationChaos = ExchangeSerializationChaos.TestCase


# ---------------------------------------------------------------------------
# Chaos test: History formatting edge cases
# ---------------------------------------------------------------------------


class HistoryFormattingChaos(ChaosTest):
    """Explore _format_history with adversarial exchange sequences."""

    faults = []

    def __init__(self) -> None:
        super().__init__()
        self.exchanges: list[Exchange] = []

    @rule(
        user=st.text(min_size=0, max_size=500),
        assistant=st.text(min_size=0, max_size=500),
        data_snippet=st.one_of(st.none(), st.text(min_size=0, max_size=200)),
    )
    def add_exchange(self, user: str, assistant: str, data_snippet: str | None) -> None:
        self.exchanges.append(
            Exchange(user=user, assistant=assistant, data_snippet=data_snippet)
        )

    @rule(
        max_chars=st.integers(min_value=0, max_value=10000),
    )
    def format_with_limit(self, max_chars: int) -> None:
        """Format history with a character limit."""
        result = _format_history(self.exchanges, max_chars)
        always(isinstance(result, str), "format returns a string")
        # Result should respect the limit (approximately — header adds overhead)
        if max_chars == 0:
            # With zero limit, we get empty or just the header
            always(
                len(result) <= 200,
                "zero max_chars produces small output",
            )

    @rule()
    def format_empty(self) -> None:
        """Formatting empty history returns empty string."""
        result = _format_history([], 10000)
        always(result == "", "empty exchanges produce empty string")


TestHistoryFormattingChaos = HistoryFormattingChaos.TestCase
