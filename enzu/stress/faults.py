"""
Factory functions for creating OpenAI-compatible exceptions.

These exceptions are recognized by @with_retry and will trigger
retry behavior just like real provider errors.
"""

from __future__ import annotations

import httpx
from openai import RateLimitError, APITimeoutError, APIConnectionError, APIStatusError


def _mock_request() -> httpx.Request:
    """Create a minimal mock request for exception constructors."""
    return httpx.Request("POST", "https://mock.local/v1/chat/completions")


def rate_limit_429(message: str = "Rate limit exceeded (mock)") -> RateLimitError:
    """
    Create a 429 RateLimitError.

    This will trigger @with_retry and be classified as RetryReason.RATE_LIMIT.
    """
    req = _mock_request()
    resp = httpx.Response(429, request=req)
    return RateLimitError(message, response=resp, body=None)


def timeout(message: str = "Request timed out (mock)") -> APITimeoutError:
    """
    Create a timeout error.

    This will trigger @with_retry and be classified as RetryReason.TIMEOUT.
    """
    req = _mock_request()
    return APITimeoutError(request=req)


def connection_error(message: str = "Connection failed (mock)") -> APIConnectionError:
    """
    Create a connection error.

    This will trigger @with_retry and be classified as RetryReason.CONNECTION_ERROR.
    """
    req = _mock_request()
    return APIConnectionError(message=message, request=req)


def server_error_500(message: str = "Internal server error (mock)") -> APIStatusError:
    """
    Create a 500 server error.

    This will trigger @with_retry and be classified as RetryReason.SERVER_ERROR.
    """
    req = _mock_request()
    resp = httpx.Response(500, request=req)
    return APIStatusError(message, response=resp, body=None)


def server_error_503(message: str = "Service unavailable (mock)") -> APIStatusError:
    """
    Create a 503 service unavailable error.

    This will trigger @with_retry and be classified as RetryReason.SERVER_ERROR.
    """
    req = _mock_request()
    resp = httpx.Response(503, request=req)
    return APIStatusError(message, response=resp, body=None)
