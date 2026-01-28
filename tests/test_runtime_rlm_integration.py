"""
Runtime ↔ RLM Engine Integration Tests.

This test suite validates that DistributedRuntime correctly dispatches
RLM tasks and maintains isolation per task.

Tests cover:
- Full task dispatch through DistributedRuntime
- Multi-step RLM-like execution through workers
- Subcall handling simulation
- Budget enforcement across runtime layers
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

import pytest

from enzu.models import (
    Budget,
    BudgetUsage,
    RLMExecutionReport,
    RLMStep,
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


# =============================================================================
# TEST HELPERS
# =============================================================================


def make_spec(
    task_id: str = "test",
    input_text: str = "Solve this problem",
    max_cost_usd: Optional[float] = 1.0,
) -> TaskSpec:
    """Create a TaskSpec for testing."""
    return TaskSpec(
        task_id=task_id,
        input_text=input_text,
        model="mock",
        budget=Budget(max_tokens=1000, max_cost_usd=max_cost_usd),
        success_criteria=SuccessCriteria(goal="Complete the task"),
        max_output_tokens=1000,
    )


def make_report(
    task_id: str = "test",
    answer: str = "result",
    success: bool = True,
    steps: int = 1,
    cost_usd: float = 0.001,
) -> RLMExecutionReport:
    """Create a mock RLM execution report."""
    return RLMExecutionReport(
        success=success,
        task_id=task_id,
        provider="mock",
        model="mock",
        answer=answer,
        steps=[
            RLMStep(
                step_index=i,
                prompt=f"Step {i + 1} prompt",
                model_output="```python\nFINAL(result)\n```",
                code="FINAL(result)",
                stdout="result",
                error=None,
            )
            for i in range(steps)
        ],
        budget_usage=BudgetUsage(
            elapsed_seconds=0.1 * steps,
            input_tokens=10 * steps,
            output_tokens=10 * steps,
            total_tokens=20 * steps,
            cost_usd=cost_usd,
            limits_exceeded=[],
        ),
        errors=[] if success else ["Simulated error"],
    )


class RLMSimulatingWorker(LocalWorker):
    """
    Worker that simulates RLM execution by returning configurable reports.

    This allows testing the runtime dispatch path without actually running
    the RLM engine, giving precise control over execution behavior.
    """

    def __init__(
        self,
        responses: Optional[Dict[str, RLMExecutionReport]] = None,
        default_answer: str = "default_result",
        default_steps: int = 1,
        delay: float = 0.01,
        fail_tasks: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.responses = responses or {}
        self.default_answer = default_answer
        self.default_steps = default_steps
        self.delay = delay
        self.fail_tasks = fail_tasks or []
        self.executed_tasks: List[str] = []
        self._exec_lock = threading.Lock()

    def run(
        self,
        spec: TaskSpec,
        provider: ProviderSpec,
        data: str,
        options: RuntimeOptions,
    ) -> RLMExecutionReport:
        start = time.monotonic()
        with self._lock:
            self.stats.active += 1

        try:
            with self._exec_lock:
                self.executed_tasks.append(spec.task_id)

            if self.delay > 0:
                time.sleep(self.delay)

            # Check for task-specific failure
            if spec.task_id in self.fail_tasks:
                with self._lock:
                    self.stats.failed += 1
                return make_report(
                    task_id=spec.task_id,
                    answer="",
                    success=False,
                    steps=1,
                )

            # Check for task-specific response
            if spec.task_id in self.responses:
                report = self.responses[spec.task_id]
            else:
                # Generate default response with task_id in answer
                report = make_report(
                    task_id=spec.task_id,
                    answer=f"{self.default_answer}_for_{spec.task_id}",
                    steps=self.default_steps,
                )

            with self._lock:
                self.stats.completed += 1
                self.stats.total_seconds += time.monotonic() - start

            return report

        finally:
            with self._lock:
                self.stats.active -= 1


# =============================================================================
# TEST CLASS: Basic Runtime-RLM Integration
# =============================================================================


class TestRuntimeRLMBasicIntegration:
    """Test basic integration between DistributedRuntime and RLM execution."""

    def test_single_task_dispatch(self):
        """Single task dispatched through runtime executes correctly."""
        worker = RLMSimulatingWorker(default_answer="hello_world")
        runtime = DistributedRuntime(workers=[worker])

        result = runtime.run(
            spec=make_spec("task1", "Say hello"),
            provider=ProviderSpec(name="mock"),
            data="",
            options=RuntimeOptions(max_steps=5),
        )

        assert result.success
        assert result.task_id == "task1"
        assert result.answer is not None and "hello_world" in result.answer
        assert "task1" in worker.executed_tasks

    def test_multiple_sequential_tasks(self):
        """Multiple tasks execute sequentially through same worker."""
        worker = RLMSimulatingWorker()
        runtime = DistributedRuntime(workers=[worker])

        results = []
        for i in range(3):
            result = runtime.run(
                spec=make_spec(f"seq_task_{i}"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )
            results.append(result)

        assert all(r.success for r in results)
        assert [r.task_id for r in results] == ["seq_task_0", "seq_task_1", "seq_task_2"]
        assert worker.executed_tasks == ["seq_task_0", "seq_task_1", "seq_task_2"]

    def test_parallel_task_dispatch(self):
        """Multiple tasks dispatch in parallel to multiple workers."""
        workers = [
            RLMSimulatingWorker(default_answer=f"worker_{i}", max_concurrent=4)
            for i in range(2)
        ]
        runtime = DistributedRuntime(workers=workers)

        # Submit tasks asynchronously
        futures = [
            runtime.submit(
                spec=make_spec(f"parallel_{i}"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )
            for i in range(4)
        ]

        results = [f.result(timeout=10.0) for f in futures]

        assert all(r.success for r in results)
        # Verify distribution across workers
        total_executed = sum(len(w.executed_tasks) for w in workers)
        assert total_executed == 4


# =============================================================================
# TEST CLASS: Multi-Step RLM Execution
# =============================================================================


class TestRuntimeRLMMultiStep:
    """Test multi-step RLM-like execution through runtime."""

    def test_multistep_execution(self):
        """Multi-step task completes through runtime."""
        worker = RLMSimulatingWorker(
            responses={
                "multistep_task": make_report(
                    task_id="multistep_task",
                    answer="completed_after_3_steps",
                    steps=3,
                )
            }
        )
        runtime = DistributedRuntime(workers=[worker])

        result = runtime.run(
            spec=make_spec("multistep_task"),
            provider=ProviderSpec(name="mock"),
            data="",
            options=RuntimeOptions(max_steps=10),
        )

        assert result.success
        assert result.answer is not None and "completed_after_3_steps" in result.answer
        assert len(result.steps) == 3

    def test_varying_step_counts(self):
        """Different tasks have different step counts."""
        worker = RLMSimulatingWorker(
            responses={
                f"task_{i}": make_report(task_id=f"task_{i}", steps=i + 1)
                for i in range(5)
            }
        )
        runtime = DistributedRuntime(workers=[worker])

        for i in range(5):
            result = runtime.run(
                spec=make_spec(f"task_{i}"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )
            assert result.success
            assert len(result.steps) == i + 1


# =============================================================================
# TEST CLASS: Subcall Handling
# =============================================================================


class TestRuntimeRLMSubcalls:
    """Test RLM subcall-like handling through runtime dispatch."""

    def test_subcall_resolution(self):
        """Tasks with subcall-like behavior resolve correctly."""
        worker = RLMSimulatingWorker(
            responses={
                "subcall_task": make_report(
                    task_id="subcall_task",
                    answer="main_with_found_value_123",
                    steps=2,  # One for main, one for subcall
                )
            }
        )
        runtime = DistributedRuntime(workers=[worker])

        result = runtime.run(
            spec=make_spec("subcall_task"),
            provider=ProviderSpec(name="mock"),
            data="",
            options=RuntimeOptions(),
        )

        assert result.success
        assert result.answer is not None and "found_value_123" in result.answer

    def test_multiple_subcalls(self):
        """Tasks with multiple subcalls are handled."""
        worker = RLMSimulatingWorker(
            responses={
                "multi_subcall": make_report(
                    task_id="multi_subcall",
                    answer="alpha_beta_combined",
                    steps=3,
                )
            }
        )
        runtime = DistributedRuntime(workers=[worker])

        result = runtime.run(
            spec=make_spec("multi_subcall"),
            provider=ProviderSpec(name="mock"),
            data="",
            options=RuntimeOptions(),
        )

        assert result.success
        assert result.answer is not None and "alpha" in result.answer
        assert result.answer is not None and "beta" in result.answer


# =============================================================================
# TEST CLASS: Task Isolation
# =============================================================================


class TestRuntimeTaskIsolation:
    """Test that tasks maintain isolation in the runtime."""

    def test_tasks_have_isolated_results(self):
        """Each task gets its own isolated result."""
        worker = RLMSimulatingWorker(max_concurrent=4)
        runtime = DistributedRuntime(workers=[worker])

        # Submit multiple tasks concurrently
        futures = []
        for i in range(10):
            futures.append(
                runtime.submit(
                    spec=make_spec(f"isolated_{i}"),
                    provider=ProviderSpec(name="mock"),
                    data=f"data_{i}",
                    options=RuntimeOptions(),
                )
            )

        results = [f.result(timeout=10.0) for f in futures]

        # Each result should have its own task_id in the answer
        for i, result in enumerate(results):
            assert result.success
            assert result.task_id == f"isolated_{i}"
            assert f"isolated_{i}" in result.answer

    def test_parallel_tasks_isolated(self):
        """Parallel tasks don't interfere with each other."""
        workers = [
            RLMSimulatingWorker(max_concurrent=5, delay=0.02)
            for _ in range(2)
        ]
        runtime = DistributedRuntime(workers=workers)

        # Submit many tasks in parallel
        futures = [
            runtime.submit(
                spec=make_spec(f"parallel_{i}"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )
            for i in range(20)
        ]

        results = [f.result(timeout=30.0) for f in futures]

        # All should succeed with correct task IDs
        assert all(r.success for r in results)
        task_ids = {r.task_id for r in results}
        expected_ids = {f"parallel_{i}" for i in range(20)}
        assert task_ids == expected_ids


# =============================================================================
# TEST CLASS: Budget Enforcement
# =============================================================================


class TestRuntimeRLMBudget:
    """Test budget enforcement across runtime and RLM layers."""

    def test_runtime_budget_limits_tasks(self):
        """Runtime-level budget limit stops new tasks."""
        worker = RLMSimulatingWorker()
        runtime = DistributedRuntime(
            workers=[worker],
            budget=BudgetLimit(max_tasks=3),
        )

        # First 3 should succeed
        for i in range(3):
            result = runtime.run(
                spec=make_spec(f"budget_task_{i}"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )
            assert result.success

        # 4th should fail with budget exceeded
        with pytest.raises(BudgetExceededError) as exc_info:
            runtime.run(
                spec=make_spec("budget_task_3"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )
        assert "Task limit exceeded" in str(exc_info.value)

    def test_task_budget_tracked_by_runtime(self):
        """Runtime tracks cumulative cost from executions."""
        worker = RLMSimulatingWorker()
        runtime = DistributedRuntime(workers=[worker])

        for i in range(5):
            runtime.run(
                spec=make_spec(f"tracked_{i}"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )

        budget = runtime.budget
        assert budget["total_tasks"] == 5
        assert "total_cost_usd" in budget

    def test_cost_limit_enforcement(self):
        """Runtime cost limit is enforced."""
        # Worker returns 0.05 cost per task
        worker = RLMSimulatingWorker(
            responses={
                f"cost_task_{i}": make_report(
                    task_id=f"cost_task_{i}",
                    cost_usd=0.05,
                )
                for i in range(10)
            }
        )
        runtime = DistributedRuntime(
            workers=[worker],
            budget=BudgetLimit(max_cost_usd=0.10),
        )

        # First 2 should succeed (0.05 * 2 = 0.10)
        for i in range(2):
            result = runtime.run(
                spec=make_spec(f"cost_task_{i}"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )
            assert result.success

        # 3rd should fail
        with pytest.raises(BudgetExceededError):
            runtime.run(
                spec=make_spec("cost_task_2"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )


# =============================================================================
# TEST CLASS: Error Handling
# =============================================================================


class TestRuntimeRLMErrorHandling:
    """Test error handling in runtime-RLM integration."""

    def test_failed_task_captured_in_report(self):
        """Failed tasks return failure reports."""
        worker = RLMSimulatingWorker(fail_tasks=["error_task"])
        runtime = DistributedRuntime(workers=[worker])

        result = runtime.run(
            spec=make_spec("error_task"),
            provider=ProviderSpec(name="mock"),
            data="",
            options=RuntimeOptions(),
        )

        assert not result.success
        assert result.task_id == "error_task"
        assert len(result.errors) > 0

    def test_worker_failure_doesnt_crash_runtime(self):
        """Worker failure is isolated and doesn't crash the runtime."""
        worker = RLMSimulatingWorker(fail_tasks=["fail_task"])
        runtime = DistributedRuntime(workers=[worker])

        # First task succeeds
        result1 = runtime.run(
            spec=make_spec("ok_task"),
            provider=ProviderSpec(name="mock"),
            data="",
            options=RuntimeOptions(),
        )
        assert result1.success

        # Second task fails
        result2 = runtime.run(
            spec=make_spec("fail_task"),
            provider=ProviderSpec(name="mock"),
            data="",
            options=RuntimeOptions(),
        )
        assert not result2.success

        # Third task should still work
        result3 = runtime.run(
            spec=make_spec("another_ok"),
            provider=ProviderSpec(name="mock"),
            data="",
            options=RuntimeOptions(),
        )
        assert result3.success

        # Runtime should still be operational
        assert not runtime._shutdown


# =============================================================================
# TEST CLASS: Concurrent RLM Execution
# =============================================================================


class TestConcurrentRLMExecution:
    """Test concurrent RLM-like execution through runtime."""

    def test_map_executes_in_parallel(self):
        """Runtime.map() executes multiple tasks in parallel."""
        workers = [
            RLMSimulatingWorker(max_concurrent=4, delay=0.01)
            for _ in range(2)
        ]
        runtime = DistributedRuntime(workers=workers)

        specs = [make_spec(f"map_task_{i}") for i in range(8)]
        completed = []

        def on_complete(idx: int, result: RLMExecutionReport):
            completed.append((idx, result.task_id))

        results = runtime.map(
            specs,
            provider=ProviderSpec(name="mock"),
            data="",
            options=RuntimeOptions(),
            on_complete=on_complete,
        )

        assert len(results) == 8
        assert all(r.success for r in results)
        assert len(completed) == 8

    def test_high_concurrency_execution(self):
        """High concurrency execution maintains correctness."""
        num_tasks = 50
        num_workers = 4

        workers = [
            RLMSimulatingWorker(max_concurrent=20, delay=0.005)
            for _ in range(num_workers)
        ]
        runtime = DistributedRuntime(workers=workers)

        start = time.monotonic()
        futures = [
            runtime.submit(
                spec=make_spec(f"concurrent_{i}"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )
            for i in range(num_tasks)
        ]
        results = [f.result(timeout=30.0) for f in futures]
        elapsed = time.monotonic() - start

        successful = sum(1 for r in results if r.success)
        print(f"\nConcurrent execution: {successful}/{num_tasks} succeeded in {elapsed:.2f}s")

        assert successful == num_tasks
        assert all(r.task_id == f"concurrent_{i}" for i, r in enumerate(results))


# =============================================================================
# TEST CLASS: Integration with RLM Features
# =============================================================================


class TestRLMFeatureIntegration:
    """Test integration with specific RLM-like features through runtime."""

    def test_task_with_data_context(self):
        """Tasks receive and can use data context."""
        worker = RLMSimulatingWorker(
            responses={
                "data_task": make_report(
                    task_id="data_task",
                    answer="processed_important_data",
                )
            }
        )
        runtime = DistributedRuntime(workers=[worker])

        result = runtime.run(
            spec=make_spec("data_task", "Process this data"),
            provider=ProviderSpec(name="mock"),
            data="Important data payload here",
            options=RuntimeOptions(),
        )

        assert result.success
        assert result.answer is not None and "processed" in result.answer

    def test_verification_criteria(self):
        """Tasks with verification criteria work through runtime."""
        worker = RLMSimulatingWorker(
            responses={
                "verify_task": make_report(
                    task_id="verify_task",
                    answer="output_with_expected_keyword_here",
                )
            }
        )
        runtime = DistributedRuntime(workers=[worker])

        spec = TaskSpec(
            task_id="verify_task",
            input_text="Generate output with keyword",
            model="mock",
            budget=Budget(max_tokens=1000),
            success_criteria=SuccessCriteria(
                required_substrings=["expected_keyword"],
            ),
        )

        result = runtime.run(
            spec=spec,
            provider=ProviderSpec(name="mock"),
            data="",
            options=RuntimeOptions(verify_on_final=True),
        )

        assert result.success
        assert result.answer is not None and "expected_keyword" in result.answer

    def test_worker_stats_accurate(self):
        """Worker stats accurately reflect RLM-like executions."""
        worker = RLMSimulatingWorker(delay=0.01)
        runtime = DistributedRuntime(workers=[worker])

        for i in range(5):
            runtime.run(
                spec=make_spec(f"stats_{i}"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )

        assert worker.stats.completed == 5
        assert worker.stats.failed == 0
        assert worker.stats.total_seconds > 0

        # Check runtime aggregate stats
        stats = runtime.stats
        assert stats["completed"] == 5
        assert stats["workers"] == 1
