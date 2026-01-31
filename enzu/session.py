"""
Session: conversation persistence for goal-oriented execution.

Maintains conversation history across multiple run() calls.
Each exchange (user prompt + assistant response) becomes context for subsequent calls.

Usage:
    session = Session(model="gpt-4")
    session.run("Find the bug", data=logs, cost=5.00)
    session.run("Fix it")  # Model has context from previous exchange
    session.save("session.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from enzu.models import ExecutionReport, RLMExecutionReport

if TYPE_CHECKING:
    from enzu.providers.base import BaseProvider


class SessionBudgetExceeded(Exception):
    """Raised when session budget cap (cost or tokens) is exceeded."""

    def __init__(
        self,
        *,
        cost_used: Optional[float] = None,
        cost_cap: Optional[float] = None,
        tokens_used: Optional[int] = None,
        tokens_cap: Optional[int] = None,
    ):
        self.cost_used = cost_used
        self.cost_cap = cost_cap
        self.tokens_used = tokens_used
        self.tokens_cap = tokens_cap

        if cost_cap is not None and cost_used is not None:
            msg = (
                f"Session budget exceeded: ${cost_used:.2f} used of ${cost_cap:.2f} cap. "
                f"Call session.raise_cost_cap(new_amount) to continue."
            )
        elif tokens_cap is not None and tokens_used is not None:
            msg = (
                f"Session token limit exceeded: {tokens_used:,} used of {tokens_cap:,} cap. "
                f"Call session.raise_token_cap(new_amount) to continue."
            )
        else:
            msg = "Session budget exceeded."
        super().__init__(msg)


@dataclass
class Exchange:
    """Single conversation turn: user prompt + assistant response."""

    user: str
    assistant: str
    data_snippet: Optional[str] = None  # First 500 chars of data, for context
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    cost_usd: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user": self.user,
            "assistant": self.assistant,
            "data_snippet": self.data_snippet,
            "timestamp": self.timestamp,
            "cost_usd": self.cost_usd,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Exchange":
        return cls(
            user=d["user"],
            assistant=d["assistant"],
            data_snippet=d.get("data_snippet"),
            timestamp=d.get("timestamp", datetime.now().isoformat()),
            cost_usd=d.get("cost_usd"),
        )


def _format_history(exchanges: List[Exchange], max_chars: int = 10000) -> str:
    """
    Format conversation history as text for model context.

    Returns empty string if no history.
    Truncates oldest exchanges if over max_chars.
    """
    if not exchanges:
        return ""

    parts = []
    total = 0

    # Build from most recent, then reverse
    for ex in reversed(exchanges):
        part = f"User: {ex.user}\n"
        if ex.data_snippet:
            part += f"[Data provided: {len(ex.data_snippet)} chars]\n"
        part += f"Assistant: {ex.assistant}\n"

        if total + len(part) > max_chars:
            break
        parts.append(part)
        total += len(part)

    if not parts:
        return ""

    parts.reverse()
    return "== Previous Conversation ==\n" + "\n".join(parts) + "\n== End Previous ==\n"


class Session:
    """
    Conversation session with history persistence.

    See enzu.terminology for full documentation on token terminology.

    Each run() call:
        1. Prepends conversation history to data
        2. Executes the task
        3. Appends the exchange to history

    Attributes:
        max_tokens: Cumulative OUTPUT tokens limit across all run() calls.
        total_tokens: Cumulative OUTPUT tokens consumed (primary billing metric).

    Example:
        session = Session(model="gpt-4")
        session.run("Find the bug", data=logs, cost=5.00)
        session.run("Fix it")  # Has context from previous
        session.save("debug_session.json")
    """

    def __init__(
        self,
        model: str,
        *,
        provider: Optional[Union[str, "BaseProvider"]] = None,
        api_key: Optional[str] = None,
        history_max_chars: int = 10000,
        max_cost_usd: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self.model = model
        self.provider: Union[str, "BaseProvider"] = provider or "openrouter"
        self.api_key = api_key
        self.history_max_chars = history_max_chars
        self.max_cost_usd = max_cost_usd
        self.max_tokens = max_tokens
        self.exchanges: List[Exchange] = []
        self.total_cost_usd: float = 0.0
        self.total_tokens: int = 0
        self.created_at: str = datetime.now().isoformat()

    def run(
        self,
        task: str,
        *,
        data: Optional[str] = None,
        tokens: Optional[int] = None,
        seconds: Optional[float] = None,
        cost: Optional[float] = None,
        contains: Optional[List[str]] = None,
        matches: Optional[List[str]] = None,
        min_words: Optional[int] = None,
        goal: Optional[str] = None,
        temperature: Optional[float] = None,
        max_steps: Optional[int] = None,
        on_progress: Optional[Callable[[str], None]] = None,
        return_report: bool = False,
    ) -> Union[str, ExecutionReport, RLMExecutionReport]:
        """
        Run a task with conversation history as context.

        History is prepended to data, so the model sees previous exchanges.
        After completion, this exchange is added to history.

        Raises:
            SessionBudgetExceeded: If session budget cap (cost or tokens) would be exceeded.
        """
        # Check budget caps before execution
        if self.max_cost_usd is not None and self.total_cost_usd >= self.max_cost_usd:
            raise SessionBudgetExceeded(
                cost_used=self.total_cost_usd,
                cost_cap=self.max_cost_usd,
            )
        if self.max_tokens is not None and self.total_tokens >= self.max_tokens:
            raise SessionBudgetExceeded(
                tokens_used=self.total_tokens,
                tokens_cap=self.max_tokens,
            )

        # Import here to avoid circular dependency
        from enzu.api import run as enzu_run

        # Format history as text
        history_text = _format_history(self.exchanges, self.history_max_chars)

        # Combine history with new data
        # Ensure combined_data is never None (data parameter accepts Optional[str] but we want str)
        if history_text:
            if data:
                combined_data = history_text + "\n== Current Data ==\n" + data
            else:
                combined_data = history_text
        else:
            combined_data = data or ""

        # Execute
        report = enzu_run(
            task,
            model=self.model,
            provider=self.provider,
            data=combined_data,
            tokens=tokens,
            seconds=seconds,
            cost=cost,
            contains=contains,
            matches=matches,
            min_words=min_words,
            goal=goal,
            temperature=temperature,
            api_key=self.api_key,
            max_steps=max_steps,
            on_progress=on_progress,
            return_report=True,
        )

        # Extract answer and usage
        # When return_report=True, enzu_run always returns ExecutionReport or RLMExecutionReport, never str
        # Track output_tokens (primary billing metric) not total_tokens
        if isinstance(report, RLMExecutionReport):
            answer = report.answer or ""
            run_cost = report.budget_usage.cost_usd
            run_tokens = report.budget_usage.output_tokens
        elif isinstance(report, ExecutionReport):
            answer = report.output_text or ""
            run_cost = report.budget_usage.cost_usd
            run_tokens = report.budget_usage.output_tokens
        else:
            # Fallback: should not happen when return_report=True
            answer = str(report) if report else ""
            run_cost = None
            run_tokens = None

        # Track usage
        if run_cost:
            self.total_cost_usd += run_cost
        if run_tokens:
            self.total_tokens += run_tokens

        # Append exchange to history
        data_snippet = data[:500] if data else None
        self.exchanges.append(
            Exchange(
                user=task,
                assistant=answer,
                data_snippet=data_snippet,
                cost_usd=run_cost,
            )
        )

        if return_report:
            return report
        return answer

    def clear(self) -> None:
        """Clear conversation history."""
        self.exchanges = []

    def raise_cost_cap(self, new_cap: float) -> None:
        """Raise the cost budget cap. Must be higher than current cap."""
        if self.max_cost_usd is not None and new_cap <= self.max_cost_usd:
            raise ValueError(
                f"New cap ${new_cap:.2f} must be higher than current ${self.max_cost_usd:.2f}"
            )
        self.max_cost_usd = new_cap

    def raise_cap(self, new_cap: float) -> None:
        """Raise the cost budget cap. Must be higher than current cap."""
        self.raise_cost_cap(new_cap)

    def raise_token_cap(self, new_cap: int) -> None:
        """Raise the token budget cap. Must be higher than current cap."""
        if self.max_tokens is not None and new_cap <= self.max_tokens:
            raise ValueError(
                f"New cap {new_cap:,} must be higher than current {self.max_tokens:,}"
            )
        self.max_tokens = new_cap

    @property
    def remaining_budget(self) -> Optional[float]:
        """Remaining cost budget. None if unlimited."""
        if self.max_cost_usd is None:
            return None
        return max(0, self.max_cost_usd - self.total_cost_usd)

    @property
    def remaining_tokens(self) -> Optional[int]:
        """Remaining output token budget. None if unlimited."""
        if self.max_tokens is None:
            return None
        return max(0, self.max_tokens - self.total_tokens)

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Get conversation history as list of dicts."""
        return [ex.to_dict() for ex in self.exchanges]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session for persistence."""
        return {
            "model": self.model,
            "provider": self.provider,
            "created_at": self.created_at,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "max_cost_usd": self.max_cost_usd,
            "max_tokens": self.max_tokens,
            "history_max_chars": self.history_max_chars,
            "exchanges": [ex.to_dict() for ex in self.exchanges],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], api_key: Optional[str] = None) -> "Session":
        """Load session from dict."""
        session = cls(
            model=d["model"],
            provider=d.get("provider", "openrouter"),
            api_key=api_key,
            history_max_chars=d.get("history_max_chars", 10000),
            max_cost_usd=d.get("max_cost_usd"),
            max_tokens=d.get("max_tokens"),
        )
        session.created_at = d.get("created_at", datetime.now().isoformat())
        session.total_cost_usd = d.get("total_cost_usd", 0.0)
        session.total_tokens = d.get("total_tokens", 0)
        session.exchanges = [Exchange.from_dict(ex) for ex in d.get("exchanges", [])]
        return session

    def save(self, path: str) -> None:
        """Save session to JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str, api_key: Optional[str] = None) -> "Session":
        """Load session from JSON file."""
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data, api_key=api_key)

    def __len__(self) -> int:
        """Number of exchanges in history."""
        return len(self.exchanges)

    def __repr__(self) -> str:
        return f"Session(model={self.model!r}, exchanges={len(self.exchanges)}, cost=${self.total_cost_usd:.4f})"
