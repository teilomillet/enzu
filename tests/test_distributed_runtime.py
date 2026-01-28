"""Tests for the distributed runtime."""

from __future__ import annotations

import time
from concurrent.futures import Future
from typing import List

from enzu.models import BudgetUsage, RLMExecutionReport, TaskSpec
from enzu.runtime import (
    DistributedRuntime,
    LocalWorker,
    ProviderSpec,
    RuntimeOptions,
    LeastLoadedScheduler,
    RoundRobinScheduler,
    AdaptiveScheduler,
    BudgetLimit,
    BudgetExceededError,
)


def make_spec(task_id: str = "test") -> TaskSpec:
    from enzu.models import Budget, SuccessCriteria

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


class MockLocalWorker(LocalWorker):
    """Worker that returns immediately without calling LLM."""

    def __init__(self, delay: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay
        self.calls: List[str] = []

    def run(
        self,
        spec: TaskSpec,
        provider: ProviderSpec,
        data: str,
        options: RuntimeOptions,
    ) -> RLMExecutionReport:
        with self._lock:
            self.stats.active += 1
            self.calls.append(spec.task_id)
        try:
            if self.delay > 0:
                time.sleep(self.delay)
            with self._lock:
                self.stats.completed += 1
            return make_report(spec.task_id)
        finally:
            with self._lock:
                self.stats.active -= 1


def test_distributed_runtime_sync_execution():
    """Sync run() executes on a worker."""
    worker = MockLocalWorker()
    runtime = DistributedRuntime(workers=[worker])

    result = runtime.run(
        spec=make_spec("task1"),
        provider=ProviderSpec(name="mock"),
        data="test data",
        options=RuntimeOptions(),
    )

    assert result.success
    assert result.task_id == "task1"
    assert "task1" in worker.calls


def test_distributed_runtime_async_submit():
    """submit() returns Future."""
    worker = MockLocalWorker(delay=0.05)
    runtime = DistributedRuntime(workers=[worker])

    future = runtime.submit(
        spec=make_spec("async1"),
        provider=ProviderSpec(name="mock"),
        data="test",
        options=RuntimeOptions(),
    )

    assert isinstance(future, Future)
    result = future.result(timeout=5.0)
    assert result.success
    assert result.task_id == "async1"


def test_distributed_runtime_parallel_execution():
    """Multiple tasks execute in parallel."""
    workers = [MockLocalWorker(delay=0.05, max_concurrent=2) for _ in range(2)]
    runtime = DistributedRuntime(workers=workers)

    start = time.monotonic()
    futures = [
        runtime.submit(
            spec=make_spec(f"parallel{i}"),
            provider=ProviderSpec(name="mock"),
            data="test",
            options=RuntimeOptions(),
        )
        for i in range(4)
    ]
    results = [f.result(timeout=5.0) for f in futures]
    elapsed = time.monotonic() - start

    assert all(r.success for r in results)
    # 4 tasks, 4 total capacity, should run in ~1 batch
    assert elapsed < 0.2  # Would be 0.2+ if sequential


def test_least_loaded_scheduler():
    """LeastLoadedScheduler picks worker with most capacity."""
    w1 = MockLocalWorker(max_concurrent=4)
    w2 = MockLocalWorker(max_concurrent=4)
    w1.stats.active = 3  # Almost full
    w2.stats.active = 1  # Mostly empty

    scheduler = LeastLoadedScheduler()
    selected = scheduler.select([w1, w2])

    assert selected is w2  # More capacity


def test_round_robin_scheduler():
    """RoundRobinScheduler cycles through workers."""
    w1 = MockLocalWorker()
    w2 = MockLocalWorker()
    w3 = MockLocalWorker()

    scheduler = RoundRobinScheduler()
    selections = [scheduler.select([w1, w2, w3]) for _ in range(6)]

    assert selections == [w1, w2, w3, w1, w2, w3]


def test_adaptive_scheduler_prefers_fast_workers():
    """AdaptiveScheduler favors workers with lower avg duration."""
    w1 = MockLocalWorker(max_concurrent=4)
    w2 = MockLocalWorker(max_concurrent=4)
    w1.stats.completed = 10
    w1.stats.total_seconds = 10.0  # 1.0s avg
    w2.stats.completed = 10
    w2.stats.total_seconds = 1.0  # 0.1s avg

    scheduler = AdaptiveScheduler()
    selected = scheduler.select([w1, w2])

    assert selected is w2  # Faster


def test_runtime_stats():
    """stats() aggregates across workers."""
    workers = [MockLocalWorker(max_concurrent=4) for _ in range(3)]
    workers[0].stats.active = 2
    workers[0].stats.completed = 5
    workers[1].stats.active = 1
    workers[1].stats.completed = 3
    workers[2].stats.failed = 1

    runtime = DistributedRuntime(workers=workers)
    stats = runtime.stats

    assert stats["workers"] == 3
    assert stats["capacity"] == 12
    assert stats["active"] == 3
    assert stats["completed"] == 8
    assert stats["failed"] == 1
    assert stats["utilization"] == 0.25


def test_runtime_map():
    """map() executes multiple specs in parallel."""
    workers = [MockLocalWorker(max_concurrent=4) for _ in range(2)]
    runtime = DistributedRuntime(workers=workers)

    specs = [make_spec(f"map{i}") for i in range(4)]
    completed = []

    def on_complete(i: int, result: RLMExecutionReport):
        completed.append((i, result.task_id))

    results = runtime.map(
        specs,
        provider=ProviderSpec(name="mock"),
        data="test",
        options=RuntimeOptions(),
        on_complete=on_complete,
    )

    assert len(results) == 4
    assert all(r.success for r in results)
    assert len(completed) == 4


def test_runtime_context_manager():
    """Runtime can be used as context manager."""
    workers = [MockLocalWorker() for _ in range(2)]
    with DistributedRuntime(workers=workers) as runtime:
        result = runtime.run(
            spec=make_spec(),
            provider=ProviderSpec(name="mock"),
            data="test",
            options=RuntimeOptions(),
        )
        assert result.success
    # Shutdown should have been called


def test_runtime_auto_creates_workers():
    """Default creates workers based on CPU count."""
    import os

    runtime = DistributedRuntime()
    expected = os.cpu_count() or 4
    assert len(runtime._workers) == expected


def test_runtime_shutdown_rejects_new_tasks():
    """After shutdown, new tasks are rejected."""
    runtime = DistributedRuntime(num_workers=1)
    runtime.shutdown(wait=True)

    try:
        runtime.run(
            spec=make_spec(),
            provider=ProviderSpec(name="mock"),
            data="test",
            options=RuntimeOptions(),
        )
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "shut down" in str(e)


def test_budget_tracking():
    """Budget is tracked across tasks."""
    workers = [MockLocalWorker() for _ in range(2)]
    runtime = DistributedRuntime(workers=workers)

    # Run a few tasks
    for i in range(3):
        runtime.run(
            spec=make_spec(f"budget{i}"),
            provider=ProviderSpec(name="mock"),
            data="test",
            options=RuntimeOptions(),
        )

    assert runtime.budget["total_tasks"] == 3
    assert "total_cost_usd" in runtime.budget


def test_budget_limit_by_tasks():
    """Budget limit by task count is enforced."""
    workers = [MockLocalWorker() for _ in range(2)]
    budget = BudgetLimit(max_tasks=2)
    runtime = DistributedRuntime(workers=workers, budget=budget)

    # First two tasks should succeed
    runtime.run(
        spec=make_spec("ok1"),
        provider=ProviderSpec(name="mock"),
        data="test",
        options=RuntimeOptions(),
    )
    runtime.run(
        spec=make_spec("ok2"),
        provider=ProviderSpec(name="mock"),
        data="test",
        options=RuntimeOptions(),
    )

    # Third task should fail
    try:
        runtime.run(
            spec=make_spec("fail"),
            provider=ProviderSpec(name="mock"),
            data="test",
            options=RuntimeOptions(),
        )
        assert False, "Should have raised BudgetExceededError"
    except BudgetExceededError as e:
        assert "Task limit exceeded" in str(e)
