"""
Retry decorator with exponential backoff and tracking.

Provides automatic retry for transient LLM provider errors:
- 429 Rate Limit
- 5xx Server Errors
- Connection/Timeout Errors

Integrates with retry tracking for per-run observability.
"""

import time
import random
from functools import wraps
from typing import Tuple, Type

from openai import APIConnectionError, RateLimitError, APITimeoutError, APIStatusError

from enzu.retries.tracking import RetryReason, get_retry_tracker

RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)


def _classify_exception(exc: Exception) -> RetryReason:
    """
    Classify an exception into a RetryReason.

    Args:
        exc: The exception that triggered the retry.

    Returns:
        Low-cardinality RetryReason for metrics/attribution.
    """
    if isinstance(exc, RateLimitError):
        return RetryReason.RATE_LIMIT
    if isinstance(exc, APITimeoutError):
        return RetryReason.TIMEOUT
    if isinstance(exc, APIConnectionError):
        return RetryReason.CONNECTION_ERROR
    if isinstance(exc, APIStatusError):
        if exc.status_code == 429:
            return RetryReason.RATE_LIMIT
        if exc.status_code >= 500:
            return RetryReason.SERVER_ERROR
    return RetryReason.UNKNOWN


def is_retryable_status(exc: APIStatusError) -> bool:
    """Check if status code is retryable (429, 5xx)."""
    return exc.status_code == 429 or exc.status_code >= 500


def with_retry(max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """
    Decorator for exponential backoff retry with tracking.

    Automatically retries on:
    - RateLimitError (429)
    - APITimeoutError
    - APIConnectionError
    - APIStatusError with 5xx status

    Tracks retries via contextvars if a retry_tracking_context is active.

    Args:
        max_attempts: Maximum number of attempts (including initial).
        base_delay: Base delay in seconds for exponential backoff.
        max_delay: Maximum delay cap in seconds.

    Returns:
        Decorated function with retry behavior.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except RETRYABLE_EXCEPTIONS as e:
                    last_exception = e
                    reason = _classify_exception(e)
                except APIStatusError as e:
                    if not is_retryable_status(e):
                        raise
                    last_exception = e
                    reason = _classify_exception(e)

                if attempt < max_attempts - 1:
                    delay = min(
                        base_delay * (2**attempt) + random.uniform(0, 1), max_delay
                    )

                    tracker = get_retry_tracker()
                    if tracker is not None:
                        tracker.record_retry(reason, backoff_seconds=delay)

                    time.sleep(delay)

            assert last_exception is not None
            raise last_exception

        return wrapper

    return decorator
