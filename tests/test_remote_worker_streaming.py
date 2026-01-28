"""Tests for remote worker SSE streaming."""

from __future__ import annotations

import json
import time
from typing import List
from unittest.mock import patch, MagicMock

import pytest

from enzu.models import BudgetUsage, RLMExecutionReport, TaskSpec, Budget, SuccessCriteria
from enzu.runtime import ProviderSpec, RuntimeOptions
from enzu.runtime.distributed import RemoteWorker


def make_spec(task_id: str = "test") -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        input_text="test",
        model="mock",
        budget=Budget(max_tokens=100),
        success_criteria=SuccessCriteria(goal="test"),
    )


def make_report(task_id: str = "test") -> RLMExecutionReport:
    return RLMExecutionReport(
        success=True,
        task_id=task_id,
        provider="mock",
        model="mock",
        answer="ok",
        steps=[],
        budget_usage=BudgetUsage(
            elapsed_seconds=0.1,
            input_tokens=10,
            output_tokens=10,
            total_tokens=20,
            cost_usd=0.001,
            limits_exceeded=[],
        ),
        errors=[],
    )


class MockSSEResponse:
    """Mock httpx streaming response for SSE."""

    def __init__(self, events: List[dict], delay: float = 0.01):
        self.events = events
        self.delay = delay
        self.status_code = 200

    def raise_for_status(self):
        pass

    def iter_lines(self):
        for event in self.events:
            time.sleep(self.delay)
            yield f"event: {event['event']}"
            yield f"data: {json.dumps(event['data'])}"
            yield ""  # Empty line marks end of event

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MockStreamingClient:
    """Mock httpx.Client that returns SSE responses."""

    def __init__(self, events: List[dict]):
        self.events = events
        self.requests = []

    def stream(self, method: str, url: str, **kwargs):
        self.requests.append({"method": method, "url": url, **kwargs})
        return MockSSEResponse(self.events)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_run_streaming_receives_progress_events():
    """RemoteWorker.run_streaming() receives and processes SSE events."""
    report = make_report("stream-test")
    events = [
        {"event": "progress", "data": {"message": "Starting execution"}},
        {"event": "step", "data": {"step": 1, "status": "Generating code"}},
        {"event": "step", "data": {"step": 2, "status": "Executing"}},
        {"event": "subcall", "data": {"depth": 1, "phase": "llm_query started"}},
        {"event": "subcall", "data": {"depth": 1, "phase": "llm_query completed"}},
        {"event": "complete", "data": report.model_dump()},
    ]

    progress_messages = []

    def on_progress(msg: str):
        progress_messages.append(msg)

    worker = RemoteWorker(endpoint="http://test:8080")

    with patch("httpx.Client", return_value=MockStreamingClient(events)):
        options = RuntimeOptions(on_progress=on_progress)
        result = worker.run_streaming(
            spec=make_spec("stream-test"),
            provider=ProviderSpec(name="mock"),
            data="test data",
            options=options,
        )

    assert result.success
    assert result.task_id == "stream-test"
    assert len(progress_messages) >= 4  # At least progress, steps, subcalls


def test_run_streaming_handles_error_event():
    """RemoteWorker.run_streaming() handles error events."""
    events = [
        {"event": "progress", "data": {"message": "Starting"}},
        {"event": "error", "data": {"error": "Something went wrong"}},
    ]

    worker = RemoteWorker(endpoint="http://test:8080")

    with patch("httpx.Client", return_value=MockStreamingClient(events)):
        with pytest.raises(RuntimeError) as exc_info:
            worker.run_streaming(
                spec=make_spec(),
                provider=ProviderSpec(name="mock"),
                data="test",
                options=RuntimeOptions(),
            )

    assert "Something went wrong" in str(exc_info.value)


def test_run_streaming_no_complete_event_raises():
    """RemoteWorker.run_streaming() raises if no complete event."""
    events = [
        {"event": "progress", "data": {"message": "Starting"}},
        # Missing complete event
    ]

    worker = RemoteWorker(endpoint="http://test:8080")

    with patch("httpx.Client", return_value=MockStreamingClient(events)):
        with pytest.raises(RuntimeError) as exc_info:
            worker.run_streaming(
                spec=make_spec(),
                provider=ProviderSpec(name="mock"),
                data="test",
                options=RuntimeOptions(),
            )

    assert "did not return a result" in str(exc_info.value)


def test_run_auto_selects_streaming_when_on_progress_set():
    """RemoteWorker.run() uses streaming when on_progress is provided."""
    report = make_report()
    events = [{"event": "complete", "data": report.model_dump()}]

    worker = RemoteWorker(endpoint="http://test:8080")
    progress_called = []

    with patch("httpx.Client", return_value=MockStreamingClient(events)):
        result = worker.run(
            spec=make_spec(),
            provider=ProviderSpec(name="mock"),
            data="test",
            options=RuntimeOptions(on_progress=lambda m: progress_called.append(m)),
        )

    assert result.success


def test_run_uses_blocking_when_no_progress():
    """RemoteWorker.run() uses blocking mode when no on_progress."""
    report = make_report()

    class MockBlockingClient:
        def post(self, url, **kwargs):
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = report.model_dump()
            response.raise_for_status = MagicMock()
            return response

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    worker = RemoteWorker(endpoint="http://test:8080")

    with patch("httpx.Client", return_value=MockBlockingClient()):
        result = worker.run(
            spec=make_spec(),
            provider=ProviderSpec(name="mock"),
            data="test",
            options=RuntimeOptions(),  # No on_progress
        )

    assert result.success


def test_worker_stats_updated_on_streaming():
    """Worker stats are updated correctly after streaming execution."""
    report = make_report()
    events = [{"event": "complete", "data": report.model_dump()}]

    worker = RemoteWorker(endpoint="http://test:8080")
    assert worker.stats.completed == 0
    assert worker.stats.active == 0

    with patch("httpx.Client", return_value=MockStreamingClient(events)):
        worker.run_streaming(
            spec=make_spec(),
            provider=ProviderSpec(name="mock"),
            data="test",
            options=RuntimeOptions(),
        )

    assert worker.stats.completed == 1
    assert worker.stats.active == 0


def test_worker_stats_updated_on_streaming_error():
    """Worker stats track failures from streaming."""
    events = [{"event": "error", "data": {"error": "boom"}}]

    worker = RemoteWorker(endpoint="http://test:8080")

    with patch("httpx.Client", return_value=MockStreamingClient(events)):
        try:
            worker.run_streaming(
                spec=make_spec(),
                provider=ProviderSpec(name="mock"),
                data="test",
                options=RuntimeOptions(),
            )
        except RuntimeError:
            pass

    assert worker.stats.failed == 1
    assert worker.stats.active == 0


def test_streaming_sends_auth_header():
    """Streaming requests include Authorization header."""
    report = make_report()
    events = [{"event": "complete", "data": report.model_dump()}]

    mock_client = MockStreamingClient(events)
    worker = RemoteWorker(endpoint="http://test:8080", _secret="test-secret")

    with patch("httpx.Client", return_value=mock_client):
        worker.run_streaming(
            spec=make_spec(),
            provider=ProviderSpec(name="mock"),
            data="test",
            options=RuntimeOptions(),
        )

    assert len(mock_client.requests) == 1
    headers = mock_client.requests[0].get("headers", {})
    assert headers.get("Authorization") == "Bearer test-secret"
    assert headers.get("Accept") == "text/event-stream"


def test_streaming_endpoint_url():
    """Streaming uses /run/stream endpoint."""
    report = make_report()
    events = [{"event": "complete", "data": report.model_dump()}]

    mock_client = MockStreamingClient(events)
    worker = RemoteWorker(endpoint="http://test:8080")

    with patch("httpx.Client", return_value=mock_client):
        worker.run_streaming(
            spec=make_spec(),
            provider=ProviderSpec(name="mock"),
            data="test",
            options=RuntimeOptions(),
        )

    assert mock_client.requests[0]["url"] == "http://test:8080/run/stream"


# Server-side tests (require starlette)
try:
    from starlette.testclient import TestClient
    from enzu.runtime.worker import app, get_worker_state
    import enzu.runtime.worker as worker_module

    STARLETTE_AVAILABLE = app is not None
except ImportError:
    STARLETTE_AVAILABLE = False


@pytest.mark.skipif(not STARLETTE_AVAILABLE, reason="Starlette not installed")
class TestWorkerStreamingEndpoint:
    """Tests for the /run/stream endpoint."""

    @pytest.fixture(autouse=True)
    def reset_worker_state(self):
        """Reset worker state before each test."""
        worker_module._worker_state = None
        yield
        worker_module._worker_state = None

    def test_stream_endpoint_exists(self):
        """The /run/stream endpoint is registered."""
        assert app is not None
        client = TestClient(app)

        # Should return 401 without auth, not 404
        response = client.post("/run/stream", json={})
        assert response.status_code in (401, 400)  # Auth required or bad request

    def test_stream_requires_auth(self):
        """Streaming endpoint requires authentication when secret is set."""
        import os
        old_secret = os.environ.get("ENZU_WORKER_SECRET")
        os.environ["ENZU_WORKER_SECRET"] = "test-secret"

        try:
            worker_module._worker_state = None  # Reset to pick up new secret
            assert app is not None
            client = TestClient(app)

            response = client.post(
                "/run/stream",
                json={"spec": {}, "provider": {}, "data": "", "options": {}},
            )
            assert response.status_code == 401

            # With auth should proceed (may fail on payload validation)
            response = client.post(
                "/run/stream",
                json={"spec": {}, "provider": {}, "data": "", "options": {}},
                headers={"Authorization": "Bearer test-secret"},
            )
            # Should get past auth (400 = bad payload, not 401)
            assert response.status_code in (400, 500)
        finally:
            if old_secret:
                os.environ["ENZU_WORKER_SECRET"] = old_secret
            else:
                os.environ.pop("ENZU_WORKER_SECRET", None)

    def test_stream_returns_event_stream_content_type(self):
        """Streaming endpoint returns correct content type."""
        # Create a mock runtime that returns immediately
        from enzu.runtime.local import LocalRuntime

        class MockRuntime(LocalRuntime):
            def run(self, **kwargs):
                return make_report()

        worker_module._worker_state = None
        state = get_worker_state()
        state.runtime = MockRuntime()

        assert app is not None
        client = TestClient(app)

        spec = make_spec()
        response = client.post(
            "/run/stream",
            json={
                "spec": spec.model_dump(),
                "provider": {"name": "mock"},
                "data": "test",
                "options": {"max_steps": 1},
            },
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
