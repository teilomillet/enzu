"""
Retry tracking with contextvars for per-run visibility.

Provides:
- RetryReason enum for low-cardinality classification
- RetryTracker dataclass for per-run retry accounting
- Context manager for scoping tracking to a run
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class RetryReason(str, Enum):
    """
    Low-cardinality retry reason classification.

    Used for metrics labels and budget attribution.
    """

    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"
    CONNECTION_ERROR = "connection_error"
    UNKNOWN = "unknown"


@dataclass
class RetryTracker:
    """
    Tracks retry counts and reasons for a single run.

    Attributes:
        total_retries: Number of retry attempts (not including initial attempt).
        retries_by_reason: Breakdown by RetryReason.
        backoff_seconds_total: Total time spent in backoff delays.
        budget_exceeded_during_retry: True if budget was exhausted while retrying.
        last_retry_reason: Most recent retry reason (for attribution).
    """

    total_retries: int = 0
    retries_by_reason: Dict[RetryReason, int] = field(default_factory=dict)
    backoff_seconds_total: float = 0.0
    budget_exceeded_during_retry: bool = False
    last_retry_reason: Optional[RetryReason] = None

    def record_retry(
        self,
        reason: RetryReason,
        *,
        backoff_seconds: float = 0.0,
    ) -> None:
        """
        Record a retry attempt.

        Args:
            reason: Why the retry occurred.
            backoff_seconds: How long we waited before retrying.
        """
        self.total_retries += 1
        self.retries_by_reason[reason] = self.retries_by_reason.get(reason, 0) + 1
        self.backoff_seconds_total += backoff_seconds
        self.last_retry_reason = reason

    def mark_budget_exceeded(self) -> None:
        """
        Mark that budget was exceeded during retry attempts.

        Only sets the flag if we had prior retries (for accurate attribution).
        """
        if self.total_retries > 0:
            self.budget_exceeded_during_retry = True

    def to_dict(self) -> Dict[str, int]:
        """
        Convert retries_by_reason to a serializable dict.

        Returns:
            Dict mapping reason string to count.
        """
        return {reason.value: count for reason, count in self.retries_by_reason.items()}


_retry_tracker_var: contextvars.ContextVar[Optional[RetryTracker]] = (
    contextvars.ContextVar("enzu_retry_tracker", default=None)
)


def get_retry_tracker() -> Optional[RetryTracker]:
    """
    Get the current run's retry tracker.

    Returns:
        RetryTracker if inside a retry_tracking_context, else None.
    """
    return _retry_tracker_var.get()


class retry_tracking_context:
    """
    Context manager for per-run retry tracking.

    Creates a new RetryTracker scoped to the run.
    Works with both sync and async code via contextvars.

    Usage:
        with retry_tracking_context() as tracker:
            # ... execute run ...
            print(tracker.total_retries)
    """

    def __init__(self) -> None:
        self._token: Optional[contextvars.Token[Optional[RetryTracker]]] = None
        self._tracker: Optional[RetryTracker] = None

    def __enter__(self) -> RetryTracker:
        self._tracker = RetryTracker()
        self._token = _retry_tracker_var.set(self._tracker)
        return self._tracker

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._token is not None:
            _retry_tracker_var.reset(self._token)

    async def __aenter__(self) -> RetryTracker:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)
