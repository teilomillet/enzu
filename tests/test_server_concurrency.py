"""
Concurrent HTTP Server Tests

Validates that the FastAPI server can handle 10+ concurrent users with:
- Session isolation - each user's conversation stays separate
- Request/response correlation - right answers go to right users
- No cross-contamination between sessions
- Correct behavior under concurrent load

Uses mock LLM responses to test isolation without actual API calls.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from enzu.server import create_app
from enzu.server.services.session_manager import reset_session_store
from enzu.server.config import reset_settings
from enzu.models import ProviderResult, TaskSpec
from enzu.providers.base import BaseProvider


# =============================================================================
# MOCK PROVIDER FOR ISOLATION TESTING
# =============================================================================


class IsolationTrackingProvider(BaseProvider):
    """
    Mock provider that returns responses containing the input's unique marker.

    This allows us to verify that:
    1. Each session gets its own responses
    2. No cross-contamination between concurrent requests
    3. Session history is maintained correctly
    """

    name = "isolation_tracker"

    def __init__(self) -> None:
        self._calls: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._active_count = 0
        self._peak_concurrent = 0

    def stream(self, task: TaskSpec, on_progress=None) -> ProviderResult:
        """Return response containing the task's unique marker."""
        # Track concurrency
        with self._lock:
            self._active_count += 1
            if self._active_count > self._peak_concurrent:
                self._peak_concurrent = self._active_count

        try:
            # Simulate some latency
            time.sleep(0.05)

            # Extract marker from input - look for MARKER-XXX pattern
            input_text = task.input_text or ""
            marker = None
            for word in input_text.split():
                if word.startswith("MARKER-"):
                    marker = word
                    break

            if not marker:
                marker = f"NO-MARKER-{uuid.uuid4().hex[:8]}"

            # Log the call
            with self._lock:
                self._calls.append(
                    {
                        "task_id": task.task_id,
                        "input_preview": input_text[:100],
                        "marker": marker,
                        "timestamp": time.time(),
                    }
                )

            # Return response that echoes the marker
            response_text = f"Response for {marker}: This is the answer containing {marker} for verification."

            return ProviderResult(
                output_text=response_text,
                raw={"mock": True, "marker": marker},
                usage={"output_tokens": 50, "total_tokens": 100, "prompt_tokens": 50},
                provider=self.name,
                model=task.model,
            )

        finally:
            with self._lock:
                self._active_count -= 1

    def generate(self, task: TaskSpec) -> ProviderResult:
        return self.stream(task)

    @property
    def calls(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._calls)

    @property
    def peak_concurrent(self) -> int:
        with self._lock:
            return self._peak_concurrent

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()
            self._peak_concurrent = 0
            self._active_count = 0


# Global mock provider instance
_mock_provider = IsolationTrackingProvider()


def get_mock_provider(*args, **kwargs) -> IsolationTrackingProvider:
    """Return the global mock provider."""
    return _mock_provider


# =============================================================================
# TEST DATA STRUCTURES
# =============================================================================


@dataclass
class UserSimulation:
    """Simulates a user's session with multiple requests."""

    user_id: int
    session_id: Optional[str] = None
    marker: str = ""
    requests_sent: int = 0
    responses: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    isolation_valid: bool = True

    def __post_init__(self):
        self.marker = f"MARKER-USER{self.user_id:03d}-{uuid.uuid4().hex[:8]}"


@dataclass
class ConcurrencyTestResult:
    """Results from concurrent user test."""

    total_users: int
    successful_users: int
    failed_users: int
    isolation_violations: int
    cross_contamination_found: List[str]
    peak_concurrent: int
    total_requests: int
    total_duration_ms: float

    def to_report(self) -> str:
        return f"""
{"=" * 66}
        HTTP SERVER CONCURRENCY TEST REPORT
{"=" * 66}
 Total users: {self.total_users}
 Successful: {self.successful_users}
 Failed: {self.failed_users}
 Success rate: {self.successful_users / self.total_users:.1%}
{"=" * 66}
 ISOLATION CHECK
{"-" * 66}
 Isolation violations: {self.isolation_violations}
 Cross-contamination cases: {len(self.cross_contamination_found)}
{"=" * 66}
 PERFORMANCE
{"-" * 66}
 Peak concurrent requests: {self.peak_concurrent}
 Total requests processed: {self.total_requests}
 Total duration: {self.total_duration_ms:.1f}ms
{"=" * 66}
"""


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def mock_provider():
    """Fresh mock provider for each test."""
    _mock_provider.reset()
    return _mock_provider


@pytest.fixture
def app():
    """Create fresh app for each test."""
    reset_session_store()
    reset_settings()
    return create_app()


@pytest.fixture
def client(app):
    """Sync test client."""
    return TestClient(app)


@pytest.fixture
async def async_client(app):
    """Async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def run_user_session_sync(
    user: UserSimulation,
    base_url: str,
    num_turns: int = 3,
) -> UserSimulation:
    """
    Simulate a user session with multiple turns (synchronous version).

    Each turn:
    1. Creates session if needed
    2. Sends request with user's unique marker
    3. Verifies response contains the marker
    4. Checks no other user's markers are present
    """
    import httpx

    try:
        with httpx.Client(base_url=base_url, timeout=30.0) as client:
            # Create session
            resp = client.post(
                "/v1/sessions",
                json={"model": "mock-model"},
            )
            if resp.status_code != 200:
                user.errors.append(f"Session creation failed: {resp.status_code}")
                return user

            user.session_id = resp.json()["session_id"]

            # Run multiple turns
            for turn in range(num_turns):
                task = f"Turn {turn + 1}: Analyze data with {user.marker}"

                resp = client.post(
                    f"/v1/sessions/{user.session_id}/run",
                    json={"task": task},
                    headers={"X-Request-ID": f"req-{user.user_id}-{turn}"},
                )

                if resp.status_code != 200:
                    user.errors.append(
                        f"Turn {turn + 1} failed: {resp.status_code} - {resp.text}"
                    )
                    continue

                answer = resp.json().get("answer", "")
                user.responses.append(answer)
                user.requests_sent += 1

                # Verify own marker is in response
                if user.marker not in answer:
                    user.isolation_valid = False
                    user.errors.append(f"Turn {turn + 1}: Own marker not in response")

    except Exception as e:
        user.errors.append(f"Exception: {str(e)}")

    return user


async def run_user_session_async(
    user: UserSimulation,
    client: AsyncClient,
    num_turns: int = 3,
) -> UserSimulation:
    """
    Simulate a user session with multiple turns (async version).
    """
    try:
        # Create session
        resp = await client.post(
            "/v1/sessions",
            json={"model": "mock-model"},
        )
        if resp.status_code != 200:
            user.errors.append(f"Session creation failed: {resp.status_code}")
            return user

        user.session_id = resp.json()["session_id"]

        # Run multiple turns
        for turn in range(num_turns):
            task = f"Turn {turn + 1}: Analyze data with {user.marker}"

            resp = await client.post(
                f"/v1/sessions/{user.session_id}/run",
                json={"task": task},
                headers={"X-Request-ID": f"req-{user.user_id}-{turn}"},
            )

            if resp.status_code != 200:
                user.errors.append(
                    f"Turn {turn + 1} failed: {resp.status_code} - {resp.text}"
                )
                continue

            answer = resp.json().get("answer", "")
            user.responses.append(answer)
            user.requests_sent += 1

            # Verify own marker is in response
            if user.marker not in answer:
                user.isolation_valid = False
                user.errors.append(f"Turn {turn + 1}: Own marker not in response")

    except Exception as e:
        user.errors.append(f"Exception: {str(e)}")

    return user


def check_cross_contamination(users: List[UserSimulation]) -> List[str]:
    """
    Check if any user's response contains another user's marker.

    Returns list of contamination descriptions.
    """
    contaminations = []

    for user in users:
        for response in user.responses:
            for other_user in users:
                if other_user.user_id != user.user_id:
                    if other_user.marker in response:
                        contaminations.append(
                            f"User {user.user_id}'s response contains User {other_user.user_id}'s marker"
                        )

    return contaminations


# =============================================================================
# TESTS
# =============================================================================


class TestServerConcurrency:
    """Test suite for concurrent HTTP server operations."""

    def test_health_endpoint(self, client):
        """Health endpoint works without auth."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    @patch("enzu.server.services.enzu_service.Enzu")
    def test_single_session_flow(self, mock_enzu_class, client, mock_provider):
        """Single user session flow works correctly."""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.model = "mock-model"
        mock_instance.provider = "mock"
        mock_enzu_class.return_value = mock_instance

        # Mock the run method to return a proper report
        mock_report = MagicMock()
        mock_report.output_text = "Test response with MARKER-TEST"
        mock_report.answer = "Test response with MARKER-TEST"
        mock_report.budget_usage = MagicMock()
        mock_report.budget_usage.total_tokens = 100
        mock_report.budget_usage.prompt_tokens = 50
        mock_report.budget_usage.completion_tokens = 50
        mock_report.budget_usage.cost_usd = 0.001

        # Make isinstance check work
        from enzu.models import ExecutionReport

        mock_report.__class__ = ExecutionReport

        mock_instance.run.return_value = mock_report

        # Create session
        resp = client.post("/v1/sessions", json={"model": "mock-model"})
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]
        assert session_id.startswith("sess-")

        # Get session state
        resp = client.get(f"/v1/sessions/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["exchange_count"] == 0

    def test_session_not_found(self, client):
        """404 for non-existent session."""
        resp = client.get("/v1/sessions/sess-nonexistent")
        assert resp.status_code == 404

    @patch("enzu.server.services.enzu_service.Enzu")
    def test_delete_session(self, mock_enzu_class, client):
        """Can delete a session."""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.model = "mock-model"
        mock_instance.provider = "mock"
        mock_enzu_class.return_value = mock_instance

        # Create session
        resp = client.post("/v1/sessions", json={"model": "mock-model"})
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]

        # Delete it
        resp = client.delete(f"/v1/sessions/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

        # Verify gone
        resp = client.get(f"/v1/sessions/{session_id}")
        assert resp.status_code == 404

    @pytest.mark.skip(
        reason="Multiprocessing tests can't use mocking - use test_10_concurrent_users_async instead"
    )
    def test_10_concurrent_users_with_sessions(self, app, mock_provider):
        """
        Test 10 concurrent users, each with their own session.

        NOTE: This test requires multiprocessing which doesn't work with mocking.
        Use test_10_concurrent_users_async for actual concurrency testing.
        """
        pass

    @pytest.mark.anyio
    @patch("enzu.api.run")
    async def test_10_concurrent_users_async(self, mock_api_run, app, mock_provider):
        """
        Async version of concurrent users test.

        Uses httpx async client for better concurrency testing.
        """
        num_users = 10
        num_turns = 3

        # Setup mock that echoes markers - patches enzu.api.run which Session.run() calls
        def create_mock_report(*args, **kwargs):
            task = args[0] if args else kwargs.get("task", "")
            marker = "NO-MARKER"
            for word in str(task).split():
                if word.startswith("MARKER-"):
                    marker = word
                    break

            # Check if return_report=True
            return_report = kwargs.get("return_report", False)

            from enzu.models import ExecutionReport, BudgetUsage, VerificationResult

            report = ExecutionReport(
                success=True,
                task_id="test-task",
                provider="mock",
                model="mock-model",
                output_text=f"Response containing {marker}",
                verification=VerificationResult(passed=True),
                budget_usage=BudgetUsage(
                    elapsed_seconds=0.1,
                    input_tokens=50,
                    output_tokens=50,
                    total_tokens=100,
                    cost_usd=0.001,
                ),
                errors=[],
            )

            if return_report:
                return report
            return report.output_text

        mock_api_run.side_effect = create_mock_report

        # Create users
        users = [UserSimulation(user_id=i) for i in range(num_users)]

        start_time = time.time()

        # Run all users concurrently using async
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test", timeout=30.0
        ) as client:
            tasks = [run_user_session_async(user, client, num_turns) for user in users]
            users = await asyncio.gather(*tasks)

        total_duration_ms = (time.time() - start_time) * 1000

        # Analyze results
        successful = [u for u in users if u.session_id and not u.errors]
        failed = [u for u in users if not u.session_id or u.errors]
        contaminations = check_cross_contamination(users)
        isolation_violations = sum(1 for u in users if not u.isolation_valid)

        result = ConcurrencyTestResult(
            total_users=num_users,
            successful_users=len(successful),
            failed_users=len(failed),
            isolation_violations=isolation_violations,
            cross_contamination_found=contaminations,
            peak_concurrent=mock_api_run.call_count,
            total_requests=sum(u.requests_sent for u in users),
            total_duration_ms=total_duration_ms,
        )

        print(result.to_report())

        # Print any errors
        for user in failed:
            print(f"User {user.user_id} errors: {user.errors}")

        # Assertions
        assert len(successful) >= num_users * 0.9, (
            f"Expected at least 90% success, got {len(successful)}/{num_users}"
        )

        assert isolation_violations == 0, (
            f"Found {isolation_violations} isolation violations"
        )

        assert len(contaminations) == 0, f"Found cross-contamination: {contaminations}"

    @pytest.mark.skip(
        reason="Multiprocessing tests can't use mocking - session locking tested via async test"
    )
    @patch("enzu.server.services.enzu_service.Enzu")
    def test_session_locking_prevents_race(self, mock_enzu_class, app):
        """
        Test that session locking prevents race conditions.

        When two requests hit the same session simultaneously,
        one should wait or fail gracefully.
        """
        # Setup slow mock
        call_count = [0]
        call_lock = threading.Lock()

        def slow_run(*args, **kwargs):
            with call_lock:
                call_count[0] += 1
            time.sleep(0.5)  # Slow response

            mock_report = MagicMock()
            mock_report.output_text = "Slow response"
            mock_report.answer = "Slow response"
            mock_report.budget_usage = MagicMock()
            mock_report.budget_usage.total_tokens = 100
            mock_report.budget_usage.prompt_tokens = 50
            mock_report.budget_usage.completion_tokens = 50
            mock_report.budget_usage.cost_usd = 0.001

            from enzu.models import ExecutionReport

            mock_report.__class__ = ExecutionReport
            return mock_report

        mock_instance = MagicMock()
        mock_instance.model = "mock-model"
        mock_instance.provider = "mock"
        mock_instance.run.side_effect = slow_run
        mock_enzu_class.return_value = mock_instance

        # Use uvicorn for real server
        import uvicorn
        from multiprocessing import Process
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]

        def run_server():
            uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")

        server_process = Process(target=run_server)
        server_process.start()
        time.sleep(1.0)

        try:
            import httpx

            base_url = f"http://127.0.0.1:{port}"

            with httpx.Client(base_url=base_url, timeout=10.0) as client:
                # Create session
                resp = client.post("/v1/sessions", json={})
                session_id = resp.json()["session_id"]

                # Send two concurrent requests to same session
                results = []
                errors = []

                def send_request(req_id: int):
                    try:
                        resp = client.post(
                            f"/v1/sessions/{session_id}/run",
                            json={"task": f"Request {req_id}"},
                            headers={"X-Request-ID": f"req-{req_id}"},
                        )
                        results.append((req_id, resp.status_code))
                    except Exception as e:
                        errors.append((req_id, str(e)))

                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [executor.submit(send_request, i) for i in range(2)]
                    [f.result() for f in futures]

                # One should succeed (200), one should be locked out (409) or both succeed sequentially
                status_codes = [r[1] for r in results]

                # Either both succeed (sequential execution due to lock)
                # or one gets 409 (conflict due to lock)
                valid_outcomes = (
                    status_codes == [200, 200]
                    or 200 in status_codes
                    and 409 in status_codes
                )

                print(f"Results: {results}")
                print(f"Errors: {errors}")

                assert valid_outcomes or len(errors) > 0, (
                    f"Unexpected results: {results}, errors: {errors}"
                )

        finally:
            server_process.terminate()
            server_process.join(timeout=5)

    def test_request_id_propagation(self, client):
        """Request ID is propagated in response headers."""
        # Without X-Request-ID - should generate one
        resp = client.get("/health")
        assert "X-Request-ID" in resp.headers

        # With X-Request-ID - should echo it
        resp = client.get("/health", headers={"X-Request-ID": "my-custom-id"})
        assert resp.headers.get("X-Request-ID") == "my-custom-id"


class TestStatelessRun:
    """Tests for stateless /v1/run endpoint."""

    @patch("enzu.server.services.enzu_service.Enzu")
    def test_run_endpoint(self, mock_enzu_class, client):
        """Basic run endpoint works."""
        mock_report = MagicMock()
        mock_report.output_text = "Test answer"
        mock_report.answer = "Test answer"
        mock_report.budget_usage = MagicMock()
        mock_report.budget_usage.total_tokens = 100
        mock_report.budget_usage.prompt_tokens = 50
        mock_report.budget_usage.completion_tokens = 50
        mock_report.budget_usage.cost_usd = 0.001

        from enzu.models import ExecutionReport

        mock_report.__class__ = ExecutionReport

        mock_instance = MagicMock()
        mock_instance.model = "gpt-4o"
        mock_instance.run.return_value = mock_report
        mock_enzu_class.return_value = mock_instance

        resp = client.post(
            "/v1/run",
            json={"task": "What is 2+2?"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "request_id" in data

    @pytest.mark.skip(
        reason="Multiprocessing tests can't use mocking - use async tests instead"
    )
    @patch("enzu.server.services.enzu_service.Enzu")
    def test_concurrent_run_requests(self, mock_enzu_class, app):
        """Multiple concurrent run requests are handled correctly."""
        call_markers = []
        call_lock = threading.Lock()

        def marker_run(task, *args, **kwargs):
            # Extract marker from task
            marker = None
            for word in task.split():
                if word.startswith("MARKER-"):
                    marker = word
                    break

            with call_lock:
                call_markers.append(marker)

            time.sleep(0.05)  # Small delay

            mock_report = MagicMock()
            mock_report.output_text = f"Answer for {marker}"
            mock_report.answer = f"Answer for {marker}"
            mock_report.budget_usage = MagicMock()
            mock_report.budget_usage.total_tokens = 100
            mock_report.budget_usage.prompt_tokens = 50
            mock_report.budget_usage.completion_tokens = 50
            mock_report.budget_usage.cost_usd = 0.001

            from enzu.models import ExecutionReport

            mock_report.__class__ = ExecutionReport
            return mock_report

        mock_instance = MagicMock()
        mock_instance.model = "gpt-4o"
        mock_instance.run.side_effect = marker_run
        mock_enzu_class.return_value = mock_instance

        # Start server
        import uvicorn
        from multiprocessing import Process
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]

        def run_server():
            uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")

        server_process = Process(target=run_server)
        server_process.start()
        time.sleep(1.0)

        try:
            import httpx

            base_url = f"http://127.0.0.1:{port}"

            num_requests = 10
            markers = [f"MARKER-RUN-{i:03d}" for i in range(num_requests)]
            results = []

            def send_run(marker: str):
                with httpx.Client(base_url=base_url, timeout=30.0) as client:
                    resp = client.post(
                        "/v1/run",
                        json={"task": f"Process {marker}"},
                    )
                    return (
                        marker,
                        resp.status_code,
                        resp.json() if resp.status_code == 200 else None,
                    )

            with ThreadPoolExecutor(max_workers=num_requests) as executor:
                futures = [executor.submit(send_run, m) for m in markers]
                results = [f.result() for f in futures]

            # All should succeed
            successes = [r for r in results if r[1] == 200]
            assert len(successes) == num_requests

            # Each response should contain its own marker
            for marker, status, data in results:
                if data:
                    assert marker in data["answer"], (
                        f"Response for {marker} doesn't contain marker: {data['answer']}"
                    )

        finally:
            server_process.terminate()
            server_process.join(timeout=5)


# =============================================================================
# SYNC TEST FOR DIRECT EXECUTION
# =============================================================================


@pytest.mark.skip(
    reason="Multiprocessing tests can't use mocking - use test_10_concurrent_users_async instead"
)
def test_10_concurrent_users_sync():
    """
    Synchronous test that can run without pytest-anyio.

    This creates a real HTTP server and tests concurrent access.
    """
    from enzu.server import create_app
    from enzu.server.services.session_manager import reset_session_store
    from enzu.server.config import reset_settings
    from unittest.mock import patch, MagicMock

    reset_session_store()
    reset_settings()

    app = create_app()

    # Setup mock
    def create_mock_run(*args, **kwargs):
        task = args[0] if args else kwargs.get("task", "")
        marker = "NO-MARKER"
        for word in task.split():
            if word.startswith("MARKER-"):
                marker = word
                break

        mock_report = MagicMock()
        mock_report.output_text = f"Response containing {marker}"
        mock_report.answer = f"Response containing {marker}"
        mock_report.budget_usage = MagicMock()
        mock_report.budget_usage.total_tokens = 100
        mock_report.budget_usage.prompt_tokens = 50
        mock_report.budget_usage.completion_tokens = 50
        mock_report.budget_usage.cost_usd = 0.001

        from enzu.models import ExecutionReport

        mock_report.__class__ = ExecutionReport
        return mock_report

    with patch("enzu.server.services.enzu_service.Enzu") as mock_enzu_class:
        mock_instance = MagicMock()
        mock_instance.model = "mock-model"
        mock_instance.provider = "mock"
        mock_instance.run.side_effect = create_mock_run
        mock_enzu_class.return_value = mock_instance

        # Start server
        import uvicorn
        from multiprocessing import Process
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]

        def run_server():
            uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")

        server_process = Process(target=run_server)
        server_process.start()
        time.sleep(1.5)

        try:
            num_users = 10
            num_turns = 3
            users = [UserSimulation(user_id=i) for i in range(num_users)]

            base_url = f"http://127.0.0.1:{port}"
            start_time = time.time()

            with ThreadPoolExecutor(max_workers=num_users) as executor:
                futures = [
                    executor.submit(run_user_session_sync, user, base_url, num_turns)
                    for user in users
                ]
                users = [f.result() for f in futures]

            total_duration_ms = (time.time() - start_time) * 1000

            # Analyze
            successful = [u for u in users if u.session_id and not u.errors]
            failed = [u for u in users if not u.session_id or u.errors]
            contaminations = check_cross_contamination(users)

            print(f"""
{"=" * 66}
        SYNC CONCURRENT TEST RESULTS
{"=" * 66}
 Total users: {num_users}
 Successful: {len(successful)}
 Failed: {len(failed)}
 Cross-contamination: {len(contaminations)}
 Total duration: {total_duration_ms:.1f}ms
{"=" * 66}
""")

            for user in failed:
                print(f"User {user.user_id} errors: {user.errors}")

            assert len(successful) >= num_users * 0.9
            assert len(contaminations) == 0

            print("SUCCESS: All 10 concurrent users passed isolation test")

        finally:
            server_process.terminate()
            server_process.join(timeout=5)


if __name__ == "__main__":
    test_10_concurrent_users_sync()
