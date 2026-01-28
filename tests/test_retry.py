from __future__ import annotations

import httpx
import pytest

from openai import APIConnectionError, APITimeoutError, APIStatusError, RateLimitError

import enzu.retry as retry_module


def _request() -> httpx.Request:
    return httpx.Request("GET", "https://example.com")


def _response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code=status_code, request=_request())


def test_with_retry_retries_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    # Covers retryable exception path with exponential backoff.
    sleeps: list[float] = []

    monkeypatch.setattr(retry_module.random, "uniform", lambda _a, _b: 0.0)
    monkeypatch.setattr(retry_module.time, "sleep", lambda d: sleeps.append(d))

    calls = {"count": 0}

    @retry_module.with_retry(max_attempts=3, base_delay=1.0, max_delay=10.0)
    def _work():
        calls["count"] += 1
        if calls["count"] < 3:
            raise APIConnectionError(message="boom", request=_request())
        return "ok"

    assert _work() == "ok"
    assert calls["count"] == 3
    assert sleeps == [1.0, 2.0]


def test_with_retry_does_not_retry_non_retryable_status() -> None:
    @retry_module.with_retry(max_attempts=3)
    def _work():
        raise APIStatusError("bad", response=_response(400), body=None)

    with pytest.raises(APIStatusError):
        _work()


def test_with_retry_retries_on_retryable_status() -> None:
    calls = {"count": 0}

    @retry_module.with_retry(max_attempts=2)
    def _work():
        calls["count"] += 1
        if calls["count"] == 1:
            raise APIStatusError("server error", response=_response(500), body=None)
        return "ok"

    assert _work() == "ok"
    assert calls["count"] == 2


def test_with_retry_retries_on_rate_limit() -> None:
    calls = {"count": 0}

    @retry_module.with_retry(max_attempts=2)
    def _work():
        calls["count"] += 1
        if calls["count"] == 1:
            raise RateLimitError("limit", response=_response(429), body=None)
        return "ok"

    assert _work() == "ok"


def test_with_retry_retries_on_timeout() -> None:
    calls = {"count": 0}

    @retry_module.with_retry(max_attempts=2)
    def _work():
        calls["count"] += 1
        if calls["count"] == 1:
            raise APITimeoutError(request=_request())
        return "ok"

    assert _work() == "ok"
