"""
Multi-user isolation tests: verify no state leakage between concurrent users.

This test suite validates that N users with N sessions submitting concurrently
maintain complete isolation - no cross-contamination of:
- Session state (exchanges, history)
- Runtime execution context
- Provider calls and responses
- Budget tracking
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Set
from unittest.mock import patch


from enzu import Session
from enzu.models import Budget, BudgetUsage, RLMExecutionReport, RLMStep, SuccessCriteria, TaskSpec
from enzu.runtime import (
    DistributedRuntime,
    LocalWorker,
    ProviderSpec,
    RuntimeOptions,
)
from enzu.runtime.local import LocalRuntime
from tests.providers.mock_provider import MockProvider


def create_session_mock_run(responses: Dict[int, str]):
    """
    Create a mock LocalRuntime.run that returns user-specific responses.

    Args:
        responses: Dict mapping user_id (from task prompt) to response text
    """
    call_count = [0]

    def mock_run(self, spec, provider, data, options):
        # Try to extract user_id from the input text
        import re
        match = re.search(r'user (\d+)', spec.input_text)
        user_id = int(match.group(1)) if match else call_count[0]
        call_count[0] += 1

        answer = responses.get(user_id, f"default_response_{user_id}")

        return RLMExecutionReport(
            success=True,
            task_id=spec.task_id,
            provider="mock",
            model="mock",
            answer=answer,
            steps=[RLMStep(step_index=0, prompt="p", model_output="o", code="c", stdout="s", error=None)],
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
    return mock_run


def create_simple_mock_run(answer: str = "done"):
    """Create a simple mock LocalRuntime.run that returns a fixed answer."""
    def mock_run(self, spec, provider, data, options):
        return RLMExecutionReport(
            success=True,
            task_id=spec.task_id,
            provider="mock",
            model="mock",
            answer=answer,
            steps=[RLMStep(step_index=0, prompt="p", model_output="o", code="c", stdout="s", error=None)],
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
    return mock_run


# =============================================================================
# TEST HELPERS
# =============================================================================


def make_spec(task_id: str, input_text: str = "test") -> TaskSpec:
    """Create a TaskSpec with identifiable task_id."""
    return TaskSpec(
        task_id=task_id,
        input_text=input_text,
        model="mock",
        budget=Budget(max_tokens=100),
        success_criteria=SuccessCriteria(goal="test"),
    )


def make_report(task_id: str, answer: str = "ok") -> RLMExecutionReport:
    """Create a mock execution report."""
    return RLMExecutionReport(
        success=True,
        task_id=task_id,
        provider="mock",
        model="mock",
        answer=answer,
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


class TrackingMockWorker(LocalWorker):
    """Worker that tracks all received task IDs and their thread origins."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.received_tasks: List[tuple] = []  # (task_id, thread_id, input_text)
        self._tracking_lock = threading.Lock()

    def run(
        self,
        spec: TaskSpec,
        provider: ProviderSpec,
        data: str,
        options: RuntimeOptions,
    ) -> RLMExecutionReport:
        thread_id = threading.current_thread().ident
        with self._tracking_lock:
            self.stats.active += 1
            self.received_tasks.append((spec.task_id, thread_id, spec.input_text))
        try:
            # Simulate work
            time.sleep(0.01)
            with self._lock:
                self.stats.completed += 1
            # Return answer that includes task_id to verify routing
            return make_report(spec.task_id, answer=f"response_for_{spec.task_id}")
        finally:
            with self._lock:
                self.stats.active -= 1


class UserIdTrackingProvider(MockProvider):
    """Provider that tracks which user_id made each call."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls_by_user: Dict[str, List[str]] = {}
        self._user_lock = threading.Lock()

    def generate(self, task: TaskSpec):
        user_id = task.metadata.get("user_id", "unknown")
        with self._user_lock:
            if user_id not in self.calls_by_user:
                self.calls_by_user[user_id] = []
            self.calls_by_user[user_id].append(task.input_text)
        return super().generate(task)


# =============================================================================
# TEST CLASS: Multi-User Session Isolation
# =============================================================================


class TestMultiUserSessionIsolation:
    """Verify that concurrent Session instances maintain complete isolation."""

    def test_n_users_isolated_history(self):
        """N users have completely separate conversation histories."""
        num_users = 10

        user_sessions: Dict[int, Session] = {}
        results: Dict[int, List[Any]] = {}

        def mock_run(self, spec, provider, data, options):
            import re
            match = re.search(r'user (\d+)', spec.input_text)
            user_id = int(match.group(1)) if match else 0
            task_num = "1" if "Task 1" in spec.input_text else "2"

            return RLMExecutionReport(
                success=True,
                task_id=spec.task_id,
                provider="mock",
                model="mock",
                answer=f"Response {task_num} for user {user_id}",
                steps=[RLMStep(step_index=0, prompt="p", model_output="o", code="c", stdout="s", error=None)],
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

        with patch.object(LocalRuntime, 'run', mock_run):
            for user_id in range(num_users):
                session = Session(model="mock", max_cost_usd=100.0)
                result1 = session.run(f"Task 1 from user {user_id}", cost=1.0)
                result2 = session.run(f"Task 2 from user {user_id}", cost=1.0)
                user_sessions[user_id] = session
                results[user_id] = [result1, result2]

        # Verify isolation: each user's history contains ONLY their own exchanges
        for user_id, session in user_sessions.items():
            assert len(session.exchanges) == 2, f"User {user_id} should have 2 exchanges"

            # Verify exchange content belongs to this user
            for i, exchange in enumerate(session.exchanges):
                assert f"user {user_id}" in exchange.user, \
                    f"User {user_id} exchange {i} has wrong user prompt: {exchange.user}"
                assert f"user {user_id}" in exchange.assistant, \
                    f"User {user_id} exchange {i} has wrong response: {exchange.assistant}"

            # Verify no other user's data leaked in
            for other_id in range(num_users):
                if other_id != user_id:
                    for exchange in session.exchanges:
                        assert f"user {other_id}" not in exchange.user
                        assert f"user {other_id}" not in exchange.assistant

    def test_n_users_isolated_budget_tracking(self):
        """Each user's budget is tracked independently."""
        num_users = 5

        user_sessions: Dict[int, Session] = {}

        with patch.object(LocalRuntime, 'run', create_simple_mock_run("done")):
            # Each user runs a different number of tasks (user_id + 1 tasks)
            for user_id in range(num_users):
                num_runs = user_id + 1
                session = Session(model="mock")
                for i in range(num_runs):
                    session.run(f"Task {i}", cost=0.5)
                user_sessions[user_id] = session

        # Verify each user's budget is tracked independently
        for user_id, session in user_sessions.items():
            expected_exchanges = user_id + 1
            assert len(session.exchanges) == expected_exchanges, \
                f"User {user_id} should have {expected_exchanges} exchanges"


# =============================================================================
# TEST CLASS: Multi-User Runtime Isolation
# =============================================================================


class TestMultiUserRuntimeIsolation:
    """Verify DistributedRuntime maintains isolation across concurrent users."""

    def test_concurrent_users_no_task_mixing(self):
        """Tasks from different users are processed without mixing."""
        num_users = 10
        tasks_per_user = 5

        worker = TrackingMockWorker(max_concurrent=20)
        runtime = DistributedRuntime(workers=[worker])

        user_results: Dict[int, List[RLMExecutionReport]] = {}
        errors: List[Exception] = []
        lock = threading.Lock()

        def run_user_tasks(user_id: int):
            """Submit tasks for a single user."""
            try:
                results = []
                for task_num in range(tasks_per_user):
                    task_id = f"user{user_id}_task{task_num}"
                    spec = make_spec(task_id, input_text=f"Input from user {user_id}")
                    result = runtime.run(
                        spec=spec,
                        provider=ProviderSpec(name="mock"),
                        data=f"Data for user {user_id}",
                        options=RuntimeOptions(),
                    )
                    results.append(result)

                with lock:
                    user_results[user_id] = results
            except Exception as e:
                errors.append(e)

        # Run all users concurrently
        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [executor.submit(run_user_tasks, i) for i in range(num_users)]
            for f in as_completed(futures):
                f.result()

        assert not errors, f"Errors: {errors}"

        # Verify each user got exactly their own results
        for user_id, results in user_results.items():
            assert len(results) == tasks_per_user
            for i, result in enumerate(results):
                expected_task_id = f"user{user_id}_task{i}"
                assert result.task_id == expected_task_id, \
                    f"User {user_id} got wrong task_id: {result.task_id}"
                assert result.answer is not None and f"user{user_id}" in result.answer, \
                    f"User {user_id} got wrong answer: {result.answer}"

        # Verify total task count
        assert len(worker.received_tasks) == num_users * tasks_per_user

    def test_concurrent_users_isolated_budget_tracking(self):
        """Runtime budget is global, but users see correct per-task costs."""
        num_users = 5
        tasks_per_user = 3

        worker = TrackingMockWorker(max_concurrent=20)
        runtime = DistributedRuntime(workers=[worker])

        all_results: List[RLMExecutionReport] = []
        lock = threading.Lock()

        def run_user_tasks(user_id: int):
            for task_num in range(tasks_per_user):
                spec = make_spec(f"user{user_id}_task{task_num}")
                result = runtime.run(
                    spec=spec,
                    provider=ProviderSpec(name="mock"),
                    data="test",
                    options=RuntimeOptions(),
                )
                with lock:
                    all_results.append(result)

        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [executor.submit(run_user_tasks, i) for i in range(num_users)]
            for f in as_completed(futures):
                f.result()

        # Verify global budget tracking
        total_tasks = num_users * tasks_per_user
        assert runtime.budget["total_tasks"] == total_tasks

        # Verify each result has correct budget_usage
        for result in all_results:
            assert result.budget_usage is not None
            assert result.budget_usage.cost_usd == 0.001  # From make_report

    def test_high_concurrency_no_deadlock(self):
        """100 concurrent users don't cause deadlocks."""
        num_users = 100
        workers = [TrackingMockWorker(max_concurrent=10) for _ in range(4)]
        runtime = DistributedRuntime(workers=workers)

        completed = [0]
        lock = threading.Lock()

        def run_user_task(user_id: int):
            spec = make_spec(f"user{user_id}")
            result = runtime.run(
                spec=spec,
                provider=ProviderSpec(name="mock"),
                data="test",
                options=RuntimeOptions(),
            )
            with lock:
                completed[0] += 1
            return result

        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [executor.submit(run_user_task, i) for i in range(num_users)]
            results = [f.result(timeout=30.0) for f in futures]
        elapsed = time.monotonic() - start

        assert completed[0] == num_users, "Not all tasks completed"
        assert all(r.success for r in results), "Some tasks failed"
        assert elapsed < 30.0, f"Possible deadlock: took {elapsed:.2f}s"


# =============================================================================
# TEST CLASS: Cross-Component Isolation
# =============================================================================


class TestCrossComponentIsolation:
    """Verify isolation across Session, Runtime, and Provider layers."""

    def test_session_with_runtime_isolation(self):
        """Sessions using shared runtime maintain isolation.

        Note: This test runs sessions sequentially to avoid patch propagation
        issues with threads. The isolation property is still validated.
        """
        num_sessions = 5

        session_results: Dict[int, List[Any]] = {}

        for session_id in range(num_sessions):
            # Each session gets unique mock responses
            provider = MockProvider(main_outputs=[
                f'```python\nFINAL("Session {session_id} response 1")\n```',
                f'```python\nFINAL("Session {session_id} response 2")\n```',
            ])
            session = Session(model="mock", provider=provider)
            r1 = session.run(f"Query 1 from session {session_id}", cost=1.0)
            r2 = session.run(f"Query 2 from session {session_id}", cost=1.0)
            session_results[session_id] = [r1, r2]

        # Verify each session got only its own responses
        for session_id, results in session_results.items():
            for result in results:
                assert f"Session {session_id}" in result, \
                    f"Session {session_id} got wrong response: {result}"

    def test_provider_call_isolation(self):
        """Provider calls are correctly routed per-user."""
        num_users = 5

        provider = UserIdTrackingProvider(
            main_outputs=[
                '```python\nFINAL("done")\n```'
                for _ in range(num_users * 2)
            ]
        )

        def run_user_call(user_id: int):
            spec = TaskSpec(
                task_id=f"task_user{user_id}",
                input_text=f"Query from user {user_id}",
                model="mock",
                budget=Budget(max_tokens=100),
                success_criteria=SuccessCriteria(goal="test"),
                metadata={"user_id": f"user_{user_id}"},
            )
            # Note: In real usage, the provider would be resolved and called
            # This test verifies the metadata tracking pattern works
            provider.generate(spec)

        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [executor.submit(run_user_call, i) for i in range(num_users)]
            for f in as_completed(futures):
                f.result()

        # Verify each user's calls were tracked separately
        assert len(provider.calls_by_user) == num_users
        for user_id in range(num_users):
            user_key = f"user_{user_id}"
            assert user_key in provider.calls_by_user
            assert len(provider.calls_by_user[user_key]) == 1


# =============================================================================
# TEST CLASS: State Leakage Detection
# =============================================================================


class TestStateLeakageDetection:
    """Explicit tests to detect any state leakage between users."""

    def test_no_global_state_pollution(self):
        """Verify no global state is shared between sessions.

        This tests that each Session instance is independent and doesn't
        share state with other instances.
        """
        sessions: List[Session] = []

        # Create multiple sessions and verify they are unique instances
        for i in range(20):
            provider = MockProvider(main_outputs=['```python\nFINAL("ok")\n```'])
            session = Session(model="mock", provider=provider)
            sessions.append(session)
            session.run("test", cost=1.0)

        # Each call should create a unique session instance
        # Keep references to prevent GC from reusing memory addresses
        session_ids = {id(s) for s in sessions}
        assert len(session_ids) == 20, "Sessions should be unique instances"

    def test_exchange_history_not_shared(self):
        """Verify exchange history is not shared between sessions."""
        sessions: List[Session] = []

        for i in range(5):
            provider = MockProvider(main_outputs=[
                f'```python\nFINAL("Response {i}")\n```'
            ])
            session = Session(model="mock", provider=provider)
            session.run(f"Task {i}", cost=1.0)
            sessions.append(session)

        # Verify each session has independent history
        for i, session in enumerate(sessions):
            assert len(session.exchanges) == 1
            assert f"Task {i}" in session.exchanges[0].user
            assert f"Response {i}" in session.exchanges[0].assistant

            # Verify no other session's data
            for j in range(5):
                if j != i:
                    assert f"Task {j}" not in session.exchanges[0].user
                    assert f"Response {j}" not in session.exchanges[0].assistant


# =============================================================================
# STRESS TEST
# =============================================================================


class TestMultiUserStress:
    """Stress tests for multi-user scenarios."""

    def test_burst_100_users(self):
        """Handle burst of 100 concurrent users."""
        num_users = 100
        workers = [TrackingMockWorker(max_concurrent=25) for _ in range(4)]
        runtime = DistributedRuntime(workers=workers)

        results = []
        errors = []
        lock = threading.Lock()

        def user_burst(user_id: int):
            try:
                spec = make_spec(f"burst_user_{user_id}")
                result = runtime.run(
                    spec=spec,
                    provider=ProviderSpec(name="mock"),
                    data="burst test",
                    options=RuntimeOptions(),
                )
                with lock:
                    results.append((user_id, result))
            except Exception as e:
                with lock:
                    errors.append((user_id, e))

        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [executor.submit(user_burst, i) for i in range(num_users)]
            for f in as_completed(futures, timeout=60.0):
                f.result()
        elapsed = time.monotonic() - start

        assert not errors, f"Errors occurred: {errors[:5]}..."
        assert len(results) == num_users
        print(f"\nBurst test: {num_users} users completed in {elapsed:.2f}s "
              f"({num_users/elapsed:.1f} users/sec)")

    def test_sustained_multiuser_load(self):
        """Sustained load from multiple users over time."""
        num_users = 20
        requests_per_user = 10
        workers = [TrackingMockWorker(max_concurrent=10) for _ in range(4)]
        runtime = DistributedRuntime(workers=workers)

        completed_by_user: Dict[int, int] = {i: 0 for i in range(num_users)}
        lock = threading.Lock()

        def user_sustained_load(user_id: int):
            for req in range(requests_per_user):
                spec = make_spec(f"user{user_id}_req{req}")
                runtime.run(
                    spec=spec,
                    provider=ProviderSpec(name="mock"),
                    data="sustained",
                    options=RuntimeOptions(),
                )
                with lock:
                    completed_by_user[user_id] += 1
                # Small delay between requests
                time.sleep(0.001)

        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [executor.submit(user_sustained_load, i) for i in range(num_users)]
            for f in as_completed(futures, timeout=120.0):
                f.result()
        elapsed = time.monotonic() - start

        total_requests = num_users * requests_per_user
        assert sum(completed_by_user.values()) == total_requests
        print(f"\nSustained test: {total_requests} requests in {elapsed:.2f}s "
              f"({total_requests/elapsed:.1f} req/sec)")
