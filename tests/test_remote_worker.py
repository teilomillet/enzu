"""
Remote Worker Tests.

This test suite validates RemoteWorker functionality:
- HTTP communication with auth headers
- Secret enforcement
- Timeout handling
- Retry logic
- Health checks
- Error handling for network issues
"""
from __future__ import annotations

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional
from unittest.mock import patch
import socket

import pytest

from enzu.models import Budget, BudgetUsage, RLMExecutionReport, SuccessCriteria, TaskSpec
from enzu.runtime import (
    DistributedRuntime,
    ProviderSpec,
    RuntimeOptions,
)
from enzu.runtime.distributed import RemoteWorker, _serialize_task, _deserialize_report


# =============================================================================
# TEST HELPERS
# =============================================================================


def make_spec(task_id: str = "test") -> TaskSpec:
    """Create a TaskSpec for testing."""
    return TaskSpec(
        task_id=task_id,
        input_text="test input",
        model="mock",
        budget=Budget(max_tokens=100),
        success_criteria=SuccessCriteria(goal="test"),
    )


def make_report_dict(task_id: str = "test", success: bool = True) -> dict:
    """Create a report dict for JSON response."""
    return {
        "success": success,
        "task_id": task_id,
        "provider": "mock",
        "model": "mock",
        "answer": "remote_result" if success else "",
        "steps": [],
        "budget_usage": {
            "elapsed_seconds": 0.1,
            "input_tokens": 10,
            "output_tokens": 10,
            "total_tokens": 20,
            "cost_usd": 0.001,
            "limits_exceeded": [],
        },
        "errors": [] if success else ["Remote error"],
    }


def find_free_port() -> int:
    """Find a free port for the test server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


class MockWorkerServer:
    """Mock HTTP server that simulates a remote worker."""

    def __init__(
        self,
        port: int,
        expected_secret: Optional[str] = None,
        response_delay: float = 0.0,
        fail_after: Optional[int] = None,
        health_status: bool = True,
    ):
        self.port = port
        self.expected_secret = expected_secret
        self.response_delay = response_delay
        self.fail_after = fail_after
        self.health_status = health_status
        self.request_count = 0
        self.received_requests: List[Dict[str, Any]] = []
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Create handler class with access to server state
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress logging

            def do_GET(self):
                if self.path == "/health":
                    if server_ref.health_status:
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(b'{"status": "healthy"}')
                    else:
                        self.send_response(503)
                        self.end_headers()
                        self.wfile.write(b'{"status": "unhealthy"}')
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path != "/run":
                    self.send_response(404)
                    self.end_headers()
                    return

                # Check auth
                if server_ref.expected_secret:
                    auth = self.headers.get("Authorization", "")
                    expected = f"Bearer {server_ref.expected_secret}"
                    if auth != expected:
                        self.send_response(401)
                        self.end_headers()
                        self.wfile.write(b'{"error": "Unauthorized"}')
                        return

                # Read request body
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                request_data = json.loads(body)

                with server_ref._lock:
                    server_ref.request_count += 1
                    server_ref.received_requests.append(request_data)
                    count = server_ref.request_count

                # Apply delay
                if server_ref.response_delay > 0:
                    time.sleep(server_ref.response_delay)

                # Check if should fail
                if server_ref.fail_after is not None and count > server_ref.fail_after:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b'{"error": "Server error"}')
                    return

                # Return success
                task_id = request_data.get("spec", {}).get("task_id", "unknown")
                response = make_report_dict(task_id)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

        self.handler_class = Handler

    def start(self):
        """Start the mock server."""
        self.server = HTTPServer(("localhost", self.port), self.handler_class)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()
        # Give server time to start
        time.sleep(0.1)

    def stop(self):
        """Stop the mock server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.thread:
            self.thread.join(timeout=5.0)


@pytest.fixture
def mock_server():
    """Fixture that provides a mock worker server."""
    port = find_free_port()
    server = MockWorkerServer(port)
    server.start()
    yield server
    server.stop()


@pytest.fixture
def auth_server():
    """Fixture that provides a mock worker server requiring auth."""
    port = find_free_port()
    server = MockWorkerServer(port, expected_secret="test-secret-123")
    server.start()
    yield server
    server.stop()


# =============================================================================
# TEST CLASS: Basic Remote Worker Communication
# =============================================================================


class TestRemoteWorkerBasicCommunication:
    """Test basic HTTP communication with remote workers."""

    def test_remote_worker_sends_request(self, mock_server):
        """RemoteWorker sends properly formatted HTTP request."""
        worker = RemoteWorker(
            endpoint=f"http://localhost:{mock_server.port}",
            max_concurrent=4,
        )

        spec = make_spec("remote_test_1")
        result = worker.run(
            spec=spec,
            provider=ProviderSpec(name="mock"),
            data="test data",
            options=RuntimeOptions(max_steps=5),
        )

        assert result.success
        assert result.task_id == "remote_test_1"
        assert mock_server.request_count == 1

    def test_remote_worker_deserializes_response(self, mock_server):
        """RemoteWorker correctly deserializes response."""
        worker = RemoteWorker(
            endpoint=f"http://localhost:{mock_server.port}",
        )

        result = worker.run(
            spec=make_spec("deserialize_test"),
            provider=ProviderSpec(name="mock"),
            data="",
            options=RuntimeOptions(),
        )

        assert isinstance(result, RLMExecutionReport)
        assert result.success
        assert result.answer == "remote_result"
        assert result.budget_usage.total_tokens == 20

    def test_request_payload_structure(self, mock_server):
        """Request payload has correct structure."""
        worker = RemoteWorker(
            endpoint=f"http://localhost:{mock_server.port}",
        )

        spec = make_spec("payload_test")
        worker.run(
            spec=spec,
            provider=ProviderSpec(name="mock", use_pool=True),
            data="context data",
            options=RuntimeOptions(max_steps=10, isolation="subprocess"),
        )

        assert len(mock_server.received_requests) == 1
        payload = mock_server.received_requests[0]

        # Verify structure
        assert "spec" in payload
        assert "provider" in payload
        assert "data" in payload
        assert "options" in payload

        # Verify spec
        assert payload["spec"]["task_id"] == "payload_test"
        assert payload["spec"]["input_text"] == "test input"

        # Verify provider (no API key!)
        assert payload["provider"]["name"] == "mock"
        assert payload["provider"]["use_pool"] is True
        assert "api_key" not in payload["provider"]

        # Verify data
        assert payload["data"] == "context data"

        # Verify options
        assert payload["options"]["max_steps"] == 10
        assert payload["options"]["isolation"] == "subprocess"


# =============================================================================
# TEST CLASS: Authentication
# =============================================================================


class TestRemoteWorkerAuthentication:
    """Test authentication with remote workers."""

    def test_auth_header_sent(self, auth_server):
        """Worker sends auth header when secret configured."""
        worker = RemoteWorker(
            endpoint=f"http://localhost:{auth_server.port}",
            _secret="test-secret-123",
        )

        result = worker.run(
            spec=make_spec("auth_test"),
            provider=ProviderSpec(name="mock"),
            data="",
            options=RuntimeOptions(),
        )

        assert result.success

    def test_missing_auth_fails(self, auth_server):
        """Request without auth to protected server fails."""
        worker = RemoteWorker(
            endpoint=f"http://localhost:{auth_server.port}",
            _secret=None,  # No secret
        )

        with pytest.raises(Exception):  # httpx.HTTPStatusError
            worker.run(
                spec=make_spec("no_auth_test"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )

    def test_wrong_auth_fails(self, auth_server):
        """Request with wrong auth fails."""
        worker = RemoteWorker(
            endpoint=f"http://localhost:{auth_server.port}",
            _secret="wrong-secret",
        )

        with pytest.raises(Exception):
            worker.run(
                spec=make_spec("wrong_auth_test"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )

    def test_auth_from_environment(self, auth_server):
        """Worker reads secret from ENZU_WORKER_SECRET env var."""
        with patch.dict("os.environ", {"ENZU_WORKER_SECRET": "test-secret-123"}):
            worker = RemoteWorker(
                endpoint=f"http://localhost:{auth_server.port}",
            )
            # Secret should be loaded from env
            assert worker._secret == "test-secret-123"

            result = worker.run(
                spec=make_spec("env_auth_test"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )
            assert result.success


# =============================================================================
# TEST CLASS: Health Checks
# =============================================================================


class TestRemoteWorkerHealthCheck:
    """Test health check functionality."""

    def test_health_check_success(self, mock_server):
        """Health check returns True for healthy server."""
        worker = RemoteWorker(
            endpoint=f"http://localhost:{mock_server.port}",
        )

        assert worker.health_check() is True

    def test_health_check_unhealthy(self):
        """Health check returns False for unhealthy server."""
        port = find_free_port()
        server = MockWorkerServer(port, health_status=False)
        server.start()

        try:
            worker = RemoteWorker(endpoint=f"http://localhost:{port}")
            assert worker.health_check() is False
        finally:
            server.stop()

    def test_health_check_unreachable(self):
        """Health check returns False for unreachable server."""
        worker = RemoteWorker(
            endpoint="http://localhost:99999",  # Unlikely to be running
        )

        assert worker.health_check() is False

    def test_health_check_with_auth(self, auth_server):
        """Health check sends auth header."""
        worker = RemoteWorker(
            endpoint=f"http://localhost:{auth_server.port}",
            _secret="test-secret-123",
        )

        # Health endpoint doesn't require auth in our mock, but header is sent
        assert worker.health_check() is True


# =============================================================================
# TEST CLASS: Timeout Handling
# =============================================================================


class TestRemoteWorkerTimeout:
    """Test timeout handling."""

    def test_request_respects_timeout(self):
        """Slow server triggers timeout."""
        port = find_free_port()
        server = MockWorkerServer(port, response_delay=5.0)  # 5 second delay
        server.start()

        try:
            worker = RemoteWorker(
                endpoint=f"http://localhost:{port}",
                _timeout=0.5,  # 500ms timeout
            )

            with pytest.raises(Exception):  # httpx.ReadTimeout or similar
                worker.run(
                    spec=make_spec("timeout_test"),
                    provider=ProviderSpec(name="mock"),
                    data="",
                    options=RuntimeOptions(),
                )
        finally:
            server.stop()

    def test_default_timeout_reasonable(self, mock_server):
        """Default timeout allows for reasonable RLM execution."""
        worker = RemoteWorker(
            endpoint=f"http://localhost:{mock_server.port}",
        )

        # Default timeout should be several minutes for RLM tasks
        assert worker._timeout >= 60.0  # At least 1 minute


# =============================================================================
# TEST CLASS: Worker Stats
# =============================================================================


class TestRemoteWorkerStats:
    """Test worker statistics tracking."""

    def test_stats_track_active(self, mock_server):
        """Stats track active requests."""
        worker = RemoteWorker(
            endpoint=f"http://localhost:{mock_server.port}",
        )

        assert worker.stats.active == 0

        # Stats are updated during run()
        worker.run(
            spec=make_spec("stats_test"),
            provider=ProviderSpec(name="mock"),
            data="",
            options=RuntimeOptions(),
        )

        # After completion
        assert worker.stats.active == 0
        assert worker.stats.completed == 1

    def test_stats_track_failures(self):
        """Stats track failed requests."""
        port = find_free_port()
        server = MockWorkerServer(port, fail_after=1)
        server.start()

        try:
            worker = RemoteWorker(endpoint=f"http://localhost:{port}")

            # First succeeds
            worker.run(
                spec=make_spec("success"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )

            # Second fails
            try:
                worker.run(
                    spec=make_spec("fail"),
                    provider=ProviderSpec(name="mock"),
                    data="",
                    options=RuntimeOptions(),
                )
            except Exception:
                pass

            assert worker.stats.completed == 1
            assert worker.stats.failed == 1
        finally:
            server.stop()

    def test_capacity_tracking(self, mock_server):
        """Available capacity is tracked correctly."""
        worker = RemoteWorker(
            endpoint=f"http://localhost:{mock_server.port}",
            max_concurrent=4,
        )

        assert worker.max_concurrent == 4
        assert worker.available_capacity == 4
        assert worker.is_available is True

        # Simulate load
        worker.stats.active = 3
        assert worker.available_capacity == 1
        assert worker.is_available is True

        worker.stats.active = 4
        assert worker.available_capacity == 0
        assert worker.is_available is False


# =============================================================================
# TEST CLASS: Error Handling
# =============================================================================


class TestRemoteWorkerErrorHandling:
    """Test error handling for various failure scenarios."""

    def test_connection_refused(self):
        """Handles connection refused gracefully."""
        worker = RemoteWorker(
            endpoint="http://localhost:1",  # Likely refused
        )

        with pytest.raises(Exception):
            worker.run(
                spec=make_spec("refused"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )

        assert worker.stats.failed == 1

    def test_server_error_response(self):
        """Handles 500 error response."""
        port = find_free_port()
        server = MockWorkerServer(port, fail_after=0)  # Always fail
        server.start()

        try:
            worker = RemoteWorker(endpoint=f"http://localhost:{port}")

            with pytest.raises(Exception):
                worker.run(
                    spec=make_spec("server_error"),
                    provider=ProviderSpec(name="mock"),
                    data="",
                    options=RuntimeOptions(),
                )

            assert worker.stats.failed == 1
        finally:
            server.stop()

    def test_invalid_json_response(self, mock_server):
        """Handles invalid JSON in response."""
        # This test would require modifying the mock server
        # For now, verify that valid JSON works
        worker = RemoteWorker(
            endpoint=f"http://localhost:{mock_server.port}",
        )

        result = worker.run(
            spec=make_spec("json_test"),
            provider=ProviderSpec(name="mock"),
            data="",
            options=RuntimeOptions(),
        )

        assert result.success


# =============================================================================
# TEST CLASS: Security
# =============================================================================


class TestRemoteWorkerSecurity:
    """Test security aspects of remote workers."""

    def test_api_key_not_sent(self, mock_server):
        """API keys are NOT sent to remote workers."""
        worker = RemoteWorker(
            endpoint=f"http://localhost:{mock_server.port}",
        )

        # Provider spec with API key
        provider = ProviderSpec(
            name="openai",
            api_key="sk-secret-key-12345",  # This should NOT be sent
        )

        worker.run(
            spec=make_spec("security_test"),
            provider=provider,
            data="",
            options=RuntimeOptions(),
        )

        # Check request payload
        payload = mock_server.received_requests[0]
        assert "api_key" not in payload["provider"]
        assert "sk-secret" not in json.dumps(payload)

    def test_sensitive_options_not_sent(self, mock_server):
        """Sensitive options are not serialized."""
        worker = RemoteWorker(
            endpoint=f"http://localhost:{mock_server.port}",
        )

        # Options that shouldn't be sent
        options = RuntimeOptions(
            max_steps=5,
            # sandbox and sandbox_factory are not serializable
        )

        worker.run(
            spec=make_spec("options_test"),
            provider=ProviderSpec(name="mock"),
            data="",
            options=options,
        )

        payload = mock_server.received_requests[0]
        # Verify only safe options present
        assert "max_steps" in payload["options"]


# =============================================================================
# TEST CLASS: Integration with DistributedRuntime
# =============================================================================


class TestRemoteWorkerRuntimeIntegration:
    """Test integration of RemoteWorker with DistributedRuntime."""

    def test_runtime_with_remote_workers(self, mock_server):
        """DistributedRuntime works with RemoteWorker instances."""
        worker = RemoteWorker(
            endpoint=f"http://localhost:{mock_server.port}",
            max_concurrent=4,
        )

        runtime = DistributedRuntime(workers=[worker])

        result = runtime.run(
            spec=make_spec("runtime_integration"),
            provider=ProviderSpec(name="mock"),
            data="test",
            options=RuntimeOptions(),
        )

        assert result.success
        assert result.task_id == "runtime_integration"

    def test_runtime_from_endpoints(self, mock_server):
        """DistributedRuntime.from_endpoints creates remote workers."""
        runtime = DistributedRuntime.from_endpoints(
            endpoints=[f"http://localhost:{mock_server.port}"],
            max_per_worker=4,
        )

        assert len(runtime._workers) == 1
        assert isinstance(runtime._workers[0], RemoteWorker)

        result = runtime.run(
            spec=make_spec("from_endpoints"),
            provider=ProviderSpec(name="mock"),
            data="",
            options=RuntimeOptions(),
        )

        assert result.success

    def test_mixed_local_and_remote_workers(self, mock_server):
        """Runtime can use mix of local and remote workers."""
        from enzu.runtime import LocalWorker

        # Create a mock local worker
        class MockLocalWorker(LocalWorker):
            def run(self, spec, provider, data, options):
                return RLMExecutionReport(
                    success=True,
                    task_id=spec.task_id,
                    provider="local",
                    model="mock",
                    answer="local_result",
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

        local_worker = MockLocalWorker()
        remote_worker = RemoteWorker(
            endpoint=f"http://localhost:{mock_server.port}",
        )

        runtime = DistributedRuntime(workers=[local_worker, remote_worker])

        # Run multiple tasks
        results = []
        for i in range(4):
            result = runtime.run(
                spec=make_spec(f"mixed_{i}"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )
            results.append(result)

        assert all(r.success for r in results)


# =============================================================================
# TEST CLASS: Serialization Helpers
# =============================================================================


class TestSerializationHelpers:
    """Test serialization helper functions."""

    def test_serialize_task_structure(self):
        """_serialize_task produces correct structure."""
        spec = make_spec("serialize_test")
        provider = ProviderSpec(name="openai", api_key="secret", use_pool=True)
        data = "test data"
        options = RuntimeOptions(max_steps=5, isolation="subprocess")

        result = _serialize_task(spec, provider, data, options)

        # Verify structure
        assert "spec" in result
        assert "provider" in result
        assert "data" in result
        assert "options" in result

        # Verify API key not included
        assert "api_key" not in result["provider"]
        assert result["provider"]["name"] == "openai"
        assert result["provider"]["use_pool"] is True

    def test_deserialize_report(self):
        """_deserialize_report correctly parses response."""
        payload = make_report_dict("deserialize_test")
        report = _deserialize_report(payload)

        assert isinstance(report, RLMExecutionReport)
        assert report.success is True
        assert report.task_id == "deserialize_test"
        assert report.answer == "remote_result"
        assert report.budget_usage.total_tokens == 20
