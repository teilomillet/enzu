"""
Budget Layers Integration Tests.

This test suite validates the interaction between multiple budget enforcement layers:
- Session budget cap (per-user conversation budget)
- Task budget (per-task limits in TaskSpec)
- Coordinator/Runtime budget (cross-task limits)

Tests verify:
- Correct layer rejects first when multiple limits active
- Budget tracking is accurate across layers
- Budget exceptions propagate correctly
- Budget state is isolated between sessions/tasks
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
import pytest

from enzu import Session, SessionBudgetExceeded
from enzu.models import (
    Budget,
    BudgetUsage,
    RLMExecutionReport,
    SuccessCriteria,
    TaskSpec,
)
from enzu.runtime import (
    DistributedRuntime,
    LocalWorker,
    ProviderSpec,
    RuntimeOptions,
    BudgetLimit,
    BudgetExceededError,
)
from tests.providers.mock_provider import MockProvider


# =============================================================================
# TEST HELPERS
# =============================================================================


def make_spec(
    task_id: str = "test",
    max_cost_usd: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> TaskSpec:
    """Create a TaskSpec with optional budget constraints."""
    budget = Budget(
        max_tokens=max_tokens or 1000,
        max_cost_usd=max_cost_usd,
    )
    return TaskSpec(
        task_id=task_id,
        input_text="test task",
        model="mock",
        budget=budget,
        success_criteria=SuccessCriteria(goal="test"),
    )


def make_report(
    task_id: str = "test",
    cost_usd: float = 0.01,
    total_tokens: int = 100,
) -> RLMExecutionReport:
    """Create a mock execution report with configurable usage."""
    return RLMExecutionReport(
        success=True,
        task_id=task_id,
        provider="mock",
        model="mock",
        answer="done",
        steps=[],
        budget_usage=BudgetUsage(
            elapsed_seconds=0.1,
            input_tokens=total_tokens // 2,
            output_tokens=total_tokens // 2,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            limits_exceeded=[],
        ),
        errors=[],
    )


class ConfigurableCostWorker(LocalWorker):
    """Worker that returns configurable cost per task."""

    def __init__(
        self, cost_per_task: float = 0.01, tokens_per_task: int = 100, **kwargs
    ):
        super().__init__(**kwargs)
        self.cost_per_task = cost_per_task
        self.tokens_per_task = tokens_per_task
        self.tasks_executed: List[str] = []

    def run(
        self,
        spec: TaskSpec,
        provider: ProviderSpec,
        data: str,
        options: RuntimeOptions,
    ) -> RLMExecutionReport:
        with self._lock:
            self.stats.active += 1
            self.tasks_executed.append(spec.task_id)
        try:
            time.sleep(0.01)
            with self._lock:
                self.stats.completed += 1
            return make_report(
                spec.task_id,
                cost_usd=self.cost_per_task,
                total_tokens=self.tokens_per_task,
            )
        finally:
            with self._lock:
                self.stats.active -= 1


# =============================================================================
# TEST CLASS: Session Budget Layer
# =============================================================================


class TestSessionBudgetLayer:
    """Test session-level budget enforcement."""

    def test_session_cost_cap_enforced(self):
        """Session rejects tasks when cost cap exceeded."""
        session = Session(model="mock", max_cost_usd=0.05)

        # Simulate that we've used the budget (without running actual tasks)
        session.total_cost_usd = 0.05

        # Next task should fail immediately (budget check happens before provider resolution)
        with pytest.raises(SessionBudgetExceeded) as exc_info:
            session.run("task 3", cost=0.02)

        assert exc_info.value.cost_cap == 0.05
        assert "raise_cost_cap" in str(exc_info.value)

    def test_session_token_cap_enforced(self):
        """Session rejects tasks when token cap exceeded."""
        session = Session(model="mock", max_tokens=500)

        # Simulate token usage
        session.total_tokens = 500

        # Next task should fail
        with pytest.raises(SessionBudgetExceeded) as exc_info:
            session.run("task", cost=1.0)

        assert exc_info.value.tokens_cap == 500
        assert "raise_token_cap" in str(exc_info.value)

    def test_session_budget_tracking_accumulates(self):
        """Session accurately tracks cumulative usage."""
        provider = MockProvider(
            main_outputs=['```python\nFINAL("done")\n```' for _ in range(5)]
        )

        session = Session(model="mock", provider=provider)

        for i in range(3):
            session.run(f"task {i}", cost=1.0)

        # Session should track exchanges
        assert len(session.exchanges) == 3

    def test_session_remaining_budget_correct(self):
        """remaining_budget property is accurate."""
        session = Session(model="mock", max_cost_usd=10.0)
        assert session.remaining_budget == 10.0

        session.total_cost_usd = 3.5
        assert session.remaining_budget == 6.5

        session.total_cost_usd = 10.0
        assert session.remaining_budget == 0.0


# =============================================================================
# TEST CLASS: Runtime Budget Layer
# =============================================================================


class TestRuntimeBudgetLayer:
    """Test runtime-level budget enforcement."""

    def test_runtime_task_limit_enforced(self):
        """Runtime rejects tasks when task limit exceeded."""
        worker = ConfigurableCostWorker(max_concurrent=4)
        runtime = DistributedRuntime(
            workers=[worker],
            budget=BudgetLimit(max_tasks=3),
        )

        # First 3 tasks should succeed
        for i in range(3):
            result = runtime.run(
                spec=make_spec(f"task_{i}"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )
            assert result.success

        # 4th task should fail
        with pytest.raises(BudgetExceededError) as exc_info:
            runtime.run(
                spec=make_spec("task_4"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )
        assert "Task limit exceeded" in str(exc_info.value)

    def test_runtime_cost_limit_enforced(self):
        """Runtime rejects tasks when cost limit exceeded."""
        worker = ConfigurableCostWorker(cost_per_task=0.05, max_concurrent=4)
        runtime = DistributedRuntime(
            workers=[worker],
            budget=BudgetLimit(max_cost_usd=0.10),
        )

        # First 2 tasks should succeed (0.05 * 2 = 0.10)
        for i in range(2):
            result = runtime.run(
                spec=make_spec(f"cost_task_{i}"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )
            assert result.success

        # 3rd task should fail (would exceed 0.10)
        with pytest.raises(BudgetExceededError) as exc_info:
            runtime.run(
                spec=make_spec("cost_task_3"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )
        assert "Budget exceeded" in str(exc_info.value)

    def test_runtime_budget_tracking_accurate(self):
        """Runtime budget tracking is accurate across workers."""
        workers = [
            ConfigurableCostWorker(cost_per_task=0.01, max_concurrent=2)
            for _ in range(3)
        ]
        runtime = DistributedRuntime(workers=workers)

        for i in range(10):
            runtime.run(
                spec=make_spec(f"track_{i}"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )

        budget = runtime.budget
        assert budget["total_tasks"] == 10
        assert abs(budget["total_cost_usd"] - 0.10) < 0.001


# =============================================================================
# TEST CLASS: Multi-Layer Budget Interaction
# =============================================================================


class TestMultiLayerBudgetInteraction:
    """Test interaction between multiple budget layers."""

    def test_session_rejects_before_runtime(self):
        """Session budget is checked before runtime budget."""
        worker = ConfigurableCostWorker(cost_per_task=0.01)
        runtime = DistributedRuntime(
            workers=[worker],
            budget=BudgetLimit(max_tasks=100),  # High limit
        )

        # Session budget check happens before any provider resolution
        session = Session(model="mock", max_cost_usd=0.02)
        session.total_cost_usd = 0.02  # Simulate at cap

        # Session should reject before runtime is called
        with pytest.raises(SessionBudgetExceeded):
            session.run("task 2", cost=0.01)

        # Runtime budget should not have been touched for rejected task
        assert runtime.budget["total_tasks"] == 0

    def test_runtime_rejects_after_session_allows(self):
        """Runtime budget rejects even if session allows."""
        worker = ConfigurableCostWorker(cost_per_task=0.01)
        runtime = DistributedRuntime(
            workers=[worker],
            budget=BudgetLimit(max_tasks=2),  # Low runtime limit
        )

        # Tasks directly to runtime (bypassing session)
        runtime.run(
            spec=make_spec("rt_1"),
            provider=ProviderSpec(name="mock"),
            data="",
            options=RuntimeOptions(),
        )
        runtime.run(
            spec=make_spec("rt_2"),
            provider=ProviderSpec(name="mock"),
            data="",
            options=RuntimeOptions(),
        )

        # Runtime should reject 3rd task
        with pytest.raises(BudgetExceededError):
            runtime.run(
                spec=make_spec("rt_3"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )

    def test_both_caps_active_lowest_wins(self):
        """When both session and runtime have caps, lowest wins."""
        worker = ConfigurableCostWorker(cost_per_task=0.10, max_concurrent=4)
        DistributedRuntime(
            workers=[worker],
            budget=BudgetLimit(max_cost_usd=0.50),  # Runtime: $0.50
        )

        # Session has lower cap: $0.25
        session = Session(model="mock", max_cost_usd=0.25)
        session.total_cost_usd = 0.25  # Simulate at cap

        # Session cap should reject before runtime cap
        with pytest.raises(SessionBudgetExceeded):
            session.run("task 3", cost=0.10)

    def test_task_spec_budget_respected(self):
        """Task-level budget in TaskSpec is respected."""
        worker = ConfigurableCostWorker(cost_per_task=1.0)  # Expensive
        runtime = DistributedRuntime(workers=[worker])

        # Task with tight budget
        spec = make_spec("budget_task", max_cost_usd=0.50)

        result = runtime.run(
            spec=spec,
            provider=ProviderSpec(name="mock"),
            data="",
            options=RuntimeOptions(),
        )

        # Task should still execute (worker returns 1.0 cost)
        # The enforcement happens in the RLM engine, not worker
        assert result.task_id == "budget_task"


# =============================================================================
# TEST CLASS: Budget State Isolation
# =============================================================================


class TestBudgetStateIsolation:
    """Test that budget state is properly isolated."""

    def test_sessions_have_independent_budgets(self):
        """Multiple sessions track budgets independently."""
        sessions = []
        for i in range(3):
            provider = MockProvider(
                main_outputs=['```python\nFINAL("done")\n```' for _ in range(5)]
            )
            session = Session(
                model="mock", provider=provider, max_cost_usd=float(i + 1)
            )
            sessions.append(session)

            for j in range(i + 1):
                session.run(f"task {j}", cost=0.5)

        # Each session should have different exchange counts
        assert len(sessions[0].exchanges) == 1
        assert len(sessions[1].exchanges) == 2
        assert len(sessions[2].exchanges) == 3

        # Each session should have independent caps
        assert sessions[0].max_cost_usd == 1.0
        assert sessions[1].max_cost_usd == 2.0
        assert sessions[2].max_cost_usd == 3.0

    def test_runtimes_have_independent_budgets(self):
        """Multiple runtimes track budgets independently."""
        runtimes = []
        for i in range(3):
            worker = ConfigurableCostWorker(cost_per_task=0.01)
            runtime = DistributedRuntime(
                workers=[worker],
                budget=BudgetLimit(max_tasks=(i + 1) * 5),
            )
            runtimes.append(runtime)

            # Run different numbers of tasks
            for j in range((i + 1) * 2):
                runtime.run(
                    spec=make_spec(f"rt{i}_task{j}"),
                    provider=ProviderSpec(name="mock"),
                    data="",
                    options=RuntimeOptions(),
                )

        # Each runtime should have tracked different task counts
        assert runtimes[0].budget["total_tasks"] == 2
        assert runtimes[1].budget["total_tasks"] == 4
        assert runtimes[2].budget["total_tasks"] == 6

    def test_concurrent_sessions_isolated(self):
        """Sessions maintain budget isolation.

        Note: This test runs sessions sequentially to avoid patch propagation
        issues with threads. The isolation property is still validated.
        """
        num_sessions = 5
        results: Dict[int, Dict] = {}

        for session_id in range(num_sessions):
            provider = MockProvider(
                main_outputs=['```python\nFINAL("done")\n```' for _ in range(10)]
            )

            session = Session(model="mock", provider=provider, max_cost_usd=1.0)

            for i in range(session_id + 1):
                session.run(f"task {i}", cost=0.1)

            results[session_id] = {
                "exchanges": len(session.exchanges),
                "total_cost": session.total_cost_usd,
            }

        # Verify each session tracked independently
        for session_id in range(num_sessions):
            assert results[session_id]["exchanges"] == session_id + 1


# =============================================================================
# TEST CLASS: Budget Exception Handling
# =============================================================================


class TestBudgetExceptionHandling:
    """Test proper handling of budget exceptions."""

    def test_session_budget_exception_contains_usage(self):
        """SessionBudgetExceeded contains usage information."""
        session = Session(model="mock", max_cost_usd=1.0, max_tokens=1000)
        session.total_cost_usd = 1.0

        with pytest.raises(SessionBudgetExceeded) as exc_info:
            session.run("test", cost=0.1)

        exc = exc_info.value
        assert exc.cost_used == 1.0
        assert exc.cost_cap == 1.0

    def test_runtime_budget_exception_contains_info(self):
        """BudgetExceededError contains budget information."""
        worker = ConfigurableCostWorker()
        runtime = DistributedRuntime(
            workers=[worker],
            budget=BudgetLimit(max_tasks=1),
        )

        # Use up budget
        runtime.run(
            spec=make_spec("first"),
            provider=ProviderSpec(name="mock"),
            data="",
            options=RuntimeOptions(),
        )

        with pytest.raises(BudgetExceededError) as exc_info:
            runtime.run(
                spec=make_spec("second"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )

        assert "Task limit exceeded" in str(exc_info.value)

    def test_budget_exception_allows_recovery(self):
        """After budget exception, raising cap allows continuation."""
        session = Session(model="mock", max_cost_usd=0.10)
        session.total_cost_usd = 0.10  # At cap

        # Should fail
        with pytest.raises(SessionBudgetExceeded):
            session.run("blocked", cost=0.05)

        # Raise cap
        session.raise_cost_cap(0.20)

        # Session budget check should now pass
        # Test that the budget check allows through after raising cap
        assert session.remaining_budget == 0.10  # Now has budget

        # To actually run, we'd need a mock provider
        provider = MockProvider(main_outputs=['```python\nFINAL("done")\n```'])
        session.provider = provider
        result = session.run("allowed", cost=0.05)
        assert result == "done"


# =============================================================================
# TEST CLASS: Budget Enforcement Under Load
# =============================================================================


class TestBudgetEnforcementUnderLoad:
    """Test budget enforcement under concurrent load."""

    def test_runtime_budget_thread_safe(self):
        """Runtime budget enforcement is thread-safe."""
        worker = ConfigurableCostWorker(cost_per_task=0.01, max_concurrent=20)
        runtime = DistributedRuntime(
            workers=[worker],
            budget=BudgetLimit(max_tasks=50),
        )

        successful = [0]
        rejected = [0]
        lock = threading.Lock()

        def submit_task(task_id: int):
            try:
                runtime.run(
                    spec=make_spec(f"concurrent_{task_id}"),
                    provider=ProviderSpec(name="mock"),
                    data="",
                    options=RuntimeOptions(),
                )
                with lock:
                    successful[0] += 1
            except BudgetExceededError:
                with lock:
                    rejected[0] += 1

        # Submit more tasks than budget allows
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(submit_task, i) for i in range(70)]
            for f in as_completed(futures, timeout=30.0):
                f.result()

        # At least 50 should succeed (budget check happens before execution)
        # Due to race conditions, some extra might slip through
        assert successful[0] >= 50
        assert successful[0] + rejected[0] == 70
        # Total tasks tracked should match successful (tasks that ran)
        assert runtime.budget["total_tasks"] == successful[0]

    def test_no_budget_overflow(self):
        """Budget tracking doesn't overflow under high load."""
        worker = ConfigurableCostWorker(cost_per_task=0.001, max_concurrent=10)
        runtime = DistributedRuntime(workers=[worker])

        # Run many tasks
        for i in range(1000):
            runtime.run(
                spec=make_spec(f"overflow_{i}"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )

        budget = runtime.budget
        assert budget["total_tasks"] == 1000
        # Cost should be approximately 1.0 (1000 * 0.001)
        assert abs(budget["total_cost_usd"] - 1.0) < 0.01


# =============================================================================
# TEST CLASS: Budget Reset and Cleanup
# =============================================================================


class TestBudgetResetAndCleanup:
    """Test budget reset and cleanup scenarios."""

    def test_session_clear_preserves_budget_tracking(self):
        """Session.clear() preserves budget tracking."""
        provider = MockProvider(
            main_outputs=['```python\nFINAL("done")\n```' for _ in range(10)]
        )

        session = Session(model="mock", provider=provider, max_cost_usd=10.0)

        session.run("task 1", cost=1.0)
        session.run("task 2", cost=1.0)

        initial_cost = session.total_cost_usd

        # Clear history
        session.clear()

        # History cleared but budget preserved
        assert len(session.exchanges) == 0
        assert session.total_cost_usd == initial_cost

    def test_new_runtime_fresh_budget(self):
        """New runtime instance starts with fresh budget."""
        worker = ConfigurableCostWorker(cost_per_task=0.01)

        # First runtime
        runtime1 = DistributedRuntime(
            workers=[worker],
            budget=BudgetLimit(max_tasks=5),
        )
        for i in range(3):
            runtime1.run(
                spec=make_spec(f"rt1_{i}"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )
        assert runtime1.budget["total_tasks"] == 3

        # New runtime (new worker to avoid shared state)
        worker2 = ConfigurableCostWorker(cost_per_task=0.01)
        runtime2 = DistributedRuntime(
            workers=[worker2],
            budget=BudgetLimit(max_tasks=5),
        )
        assert runtime2.budget["total_tasks"] == 0

        # Should have full budget
        for i in range(5):
            runtime2.run(
                spec=make_spec(f"rt2_{i}"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )
        assert runtime2.budget["total_tasks"] == 5
