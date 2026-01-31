"""
Retry tracking for production observability.

Tracks retry counts, reasons, and budget attribution per run.
Uses contextvars for thread-safe, async-compatible tracking.

Usage:
    from enzu.retries import retry_tracking_context, get_retry_tracker

    with retry_tracking_context() as tracker:
        # ... run execution with retries ...
        print(f"Total retries: {tracker.total_retries}")
        print(f"By reason: {tracker.retries_by_reason}")
"""

from enzu.retries.tracking import (
    RetryReason,
    RetryTracker,
    get_retry_tracker,
    retry_tracking_context,
)

__all__ = [
    "RetryReason",
    "RetryTracker",
    "get_retry_tracker",
    "retry_tracking_context",
]
