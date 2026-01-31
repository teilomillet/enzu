import time
import random
from functools import wraps
from openai import APIConnectionError, RateLimitError, APITimeoutError, APIStatusError

RETRYABLE_EXCEPTIONS = (APIConnectionError, RateLimitError, APITimeoutError)


def is_retryable_status(exc: APIStatusError) -> bool:
    """Check if status code is retryable (429, 5xx)."""
    return exc.status_code == 429 or exc.status_code >= 500


def with_retry(max_attempts=3, base_delay=1.0, max_delay=60.0):
    """Decorator for exponential backoff retry."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except RETRYABLE_EXCEPTIONS as e:
                    last_exception = e
                except APIStatusError as e:
                    if not is_retryable_status(e):
                        raise  # Don't retry 4xx errors (except 429)
                    last_exception = e

                if attempt < max_attempts - 1:
                    delay = min(
                        base_delay * (2**attempt) + random.uniform(0, 1), max_delay
                    )
                    time.sleep(delay)
            assert last_exception is not None
            raise last_exception

        return wrapper

    return decorator
