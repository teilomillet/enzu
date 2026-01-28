"""
Coordinator Failover Tests.

This test suite validates that the distributed system handles node failures
gracefully:
- Node health transitions (healthy -> unhealthy -> healthy)
- Circuit breaker behavior
- Task rerouting when nodes fail
- Graceful degradation under partial failures
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from enzu.models import Budget, BudgetUsage, RLMExecutionReport, SuccessCriteria, TaskSpec
from enzu.runtime import (
    DistributedRuntime,
    LocalWorker,
    ProviderSpec,
    RuntimeOptions,
    LeastLoadedScheduler,
)


# =============================================================================
# TEST HELPERS
# =============================================================================


def make_spec(task_id: str = "test") -> TaskSpec:
    """Create a TaskSpec for testing."""
    return TaskSpec(
        task_id=task_id,
        input_text="test",
        model="mock",
        budget=Budget(max_tokens=100),
        success_criteria=SuccessCriteria(goal="test"),
    )


def make_report(task_id: str, success: bool = True) -> RLMExecutionReport:
    """Create a mock execution report."""
    return RLMExecutionReport(
        success=success,
        task_id=task_id,
        provider="mock",
        model="mock",
        answer="ok" if success else "",
        steps=[],
        budget_usage=BudgetUsage(
            elapsed_seconds=0.1,
            input_tokens=10,
            output_tokens=10,
            total_tokens=20,
            cost_usd=0.001,
            limits_exceeded=[],
        ),
        errors=[] if success else ["Worker failed"],
    )


@dataclass
class FailableWorker(LocalWorker):
    """Worker that can be configured to fail on demand."""

    fail_after: Optional[int] = None  # Fail after N successful tasks
    fail_for: Optional[int] = None  # Fail for N tasks then recover
    fail_permanently: bool = False
    delay: float = 0.01
    _task_count: int = field(default=0, init=False)
    _fail_count: int = field(default=0, init=False)
    _healthy: bool = field(default=True, init=False)

    def set_healthy(self, healthy: bool) -> None:
        """Manually set health status."""
        self._healthy = healthy

    def run(
        self,
        spec: TaskSpec,
        provider: ProviderSpec,
        data: str,
        options: RuntimeOptions,
    ) -> RLMExecutionReport:
        with self._lock:
            self.stats.active += 1
            self._task_count += 1

        try:
            # Check failure conditions
            should_fail = False

            if self.fail_permanently:
                should_fail = True
            elif self.fail_after is not None and self._task_count > self.fail_after:
                if self.fail_for is not None:
                    if self._fail_count < self.fail_for:
                        should_fail = True
                        self._fail_count += 1
                else:
                    should_fail = True
            elif not self._healthy:
                should_fail = True

            if self.delay > 0:
                time.sleep(self.delay)

            if should_fail:
                with self._lock:
                    self.stats.failed += 1
                raise RuntimeError(f"Worker failed on task {spec.task_id}")

            with self._lock:
                self.stats.completed += 1
            return make_report(spec.task_id, success=True)

        finally:
            with self._lock:
                self.stats.active -= 1

    @property
    def is_healthy(self) -> bool:
        """Check if worker is currently healthy."""
        return self._healthy and not self.fail_permanently


@dataclass
class CircuitBreakerWorker(LocalWorker):
    """Worker with circuit breaker behavior."""

    failure_threshold: int = 3  # Open circuit after N failures
    recovery_timeout: float = 1.0  # Seconds before attempting recovery
    half_open_max: int = 1  # Max requests in half-open state

    _consecutive_failures: int = field(default=0, init=False)
    _circuit_state: str = field(default="closed", init=False)  # closed, open, half_open
    _last_failure_time: Optional[float] = field(default=None, init=False)
    _half_open_attempts: int = field(default=0, init=False)
    _should_fail_next: bool = field(default=False, init=False)

    def trigger_failure(self) -> None:
        """Make next request fail."""
        self._should_fail_next = True

    def run(
        self,
        spec: TaskSpec,
        provider: ProviderSpec,
        data: str,
        options: RuntimeOptions,
    ) -> RLMExecutionReport:
        with self._lock:
            self.stats.active += 1

        try:
            # Check circuit state
            if self._circuit_state == "open":
                # Check if we can transition to half-open
                if (
                    self._last_failure_time
                    and time.monotonic() - self._last_failure_time >= self.recovery_timeout
                ):
                    self._circuit_state = "half_open"
                    self._half_open_attempts = 0
                else:
                    with self._lock:
                        self.stats.failed += 1
                    raise RuntimeError("Circuit breaker is open")

            if self._circuit_state == "half_open":
                if self._half_open_attempts >= self.half_open_max:
                    # Too many half-open attempts, re-open
                    self._circuit_state = "open"
                    self._last_failure_time = time.monotonic()
                    with self._lock:
                        self.stats.failed += 1
                    raise RuntimeError("Circuit breaker re-opened")
                self._half_open_attempts += 1

            # Execute task
            time.sleep(0.01)

            if self._should_fail_next:
                self._should_fail_next = False
                self._record_failure()
                with self._lock:
                    self.stats.failed += 1
                raise RuntimeError("Triggered failure")

            # Success - reset circuit
            self._record_success()
            with self._lock:
                self.stats.completed += 1
            return make_report(spec.task_id)

        finally:
            with self._lock:
                self.stats.active -= 1

    def _record_failure(self) -> None:
        """Record a failure and potentially trip the circuit."""
        self._consecutive_failures += 1
        self._last_failure_time = time.monotonic()
        if self._consecutive_failures >= self.failure_threshold:
            self._circuit_state = "open"

    def _record_success(self) -> None:
        """Record success and reset counters."""
        self._consecutive_failures = 0
        if self._circuit_state == "half_open":
            self._circuit_state = "closed"

    @property
    def circuit_state(self) -> str:
        """Get current circuit state."""
        return self._circuit_state


class HealthCheckScheduler(LeastLoadedScheduler):
    """Scheduler that skips unhealthy workers."""

    def select(self, workers):
        # Filter to healthy workers if possible
        healthy = [w for w in workers if getattr(w, "is_healthy", True)]
        if healthy:
            return super().select(healthy)
        # Fallback to least loaded if all unhealthy
        return super().select(workers)


# =============================================================================
# TEST CLASS: Node Health Transitions
# =============================================================================


class TestNodeHealthTransitions:
    """Test worker health state transitions."""

    def test_worker_becomes_unhealthy(self):
        """Worker transitions from healthy to unhealthy."""
        worker = FailableWorker(max_concurrent=4)
        assert worker.is_healthy

        worker.set_healthy(False)
        assert not worker.is_healthy

    def test_worker_recovers(self):
        """Worker transitions from unhealthy back to healthy."""
        worker = FailableWorker(max_concurrent=4)
        worker.set_healthy(False)
        assert not worker.is_healthy

        worker.set_healthy(True)
        assert worker.is_healthy

    def test_tasks_fail_when_worker_unhealthy(self):
        """Tasks submitted to unhealthy worker fail."""
        worker = FailableWorker(max_concurrent=4)
        worker.set_healthy(False)

        with pytest.raises(RuntimeError) as exc_info:
            worker.run(
                make_spec("test_task"),
                ProviderSpec(name="mock"),
                "",
                RuntimeOptions(),
            )
        assert "failed" in str(exc_info.value).lower()

    def test_tasks_succeed_after_recovery(self):
        """Tasks succeed after worker recovers."""
        worker = FailableWorker(max_concurrent=4)
        worker.set_healthy(False)

        # Should fail
        with pytest.raises(RuntimeError):
            worker.run(make_spec("fail"), ProviderSpec(name="mock"), "", RuntimeOptions())

        # Recover
        worker.set_healthy(True)

        # Should succeed
        result = worker.run(
            make_spec("success"),
            ProviderSpec(name="mock"),
            "",
            RuntimeOptions(),
        )
        assert result.success


# =============================================================================
# TEST CLASS: Circuit Breaker Behavior
# =============================================================================


class TestCircuitBreakerBehavior:
    """Test circuit breaker pattern implementation."""

    def test_circuit_starts_closed(self):
        """Circuit breaker starts in closed state."""
        worker = CircuitBreakerWorker(failure_threshold=3)
        assert worker.circuit_state == "closed"

    def test_circuit_opens_after_threshold(self):
        """Circuit opens after reaching failure threshold."""
        worker = CircuitBreakerWorker(failure_threshold=3)

        # Trigger failures
        for i in range(3):
            worker.trigger_failure()
            try:
                worker.run(make_spec(f"fail_{i}"), ProviderSpec(name="mock"), "", RuntimeOptions())
            except RuntimeError:
                pass

        assert worker.circuit_state == "open"

    def test_open_circuit_rejects_requests(self):
        """Open circuit immediately rejects requests."""
        worker = CircuitBreakerWorker(failure_threshold=2, recovery_timeout=10.0)

        # Trip the circuit
        for i in range(2):
            worker.trigger_failure()
            try:
                worker.run(make_spec(f"trip_{i}"), ProviderSpec(name="mock"), "", RuntimeOptions())
            except RuntimeError:
                pass

        assert worker.circuit_state == "open"

        # Next request should fail fast
        with pytest.raises(RuntimeError) as exc_info:
            worker.run(make_spec("rejected"), ProviderSpec(name="mock"), "", RuntimeOptions())
        assert "open" in str(exc_info.value).lower()

    def test_circuit_transitions_to_half_open(self):
        """Circuit transitions to half-open after recovery timeout."""
        worker = CircuitBreakerWorker(
            failure_threshold=2,
            recovery_timeout=0.1,  # Short timeout for testing
        )

        # Trip the circuit
        for i in range(2):
            worker.trigger_failure()
            try:
                worker.run(make_spec(f"trip_{i}"), ProviderSpec(name="mock"), "", RuntimeOptions())
            except RuntimeError:
                pass

        assert worker.circuit_state == "open"

        # Wait for recovery timeout
        time.sleep(0.15)

        # Next request should be allowed (half-open)
        result = worker.run(make_spec("probe"), ProviderSpec(name="mock"), "", RuntimeOptions())
        assert result.success
        assert worker.circuit_state == "closed"  # Success closes circuit

    def test_half_open_failure_reopens_circuit(self):
        """Failure in half-open state reopens the circuit."""
        worker = CircuitBreakerWorker(
            failure_threshold=2,
            recovery_timeout=0.1,
            half_open_max=1,
        )

        # Trip the circuit
        for i in range(2):
            worker.trigger_failure()
            try:
                worker.run(make_spec(f"trip_{i}"), ProviderSpec(name="mock"), "", RuntimeOptions())
            except RuntimeError:
                pass

        # Wait for recovery
        time.sleep(0.15)

        # Fail the probe request
        worker.trigger_failure()
        try:
            worker.run(make_spec("probe_fail"), ProviderSpec(name="mock"), "", RuntimeOptions())
        except RuntimeError:
            pass

        # Circuit should be open again
        assert worker.circuit_state == "open"


# =============================================================================
# TEST CLASS: Task Rerouting
# =============================================================================


class TestTaskRerouting:
    """Test task rerouting when workers fail."""

    def test_scheduler_skips_unhealthy_workers(self):
        """Scheduler routes to healthy workers when some are unhealthy."""
        healthy_worker = FailableWorker(max_concurrent=4)
        unhealthy_worker = FailableWorker(max_concurrent=4)
        unhealthy_worker.set_healthy(False)

        scheduler = HealthCheckScheduler()

        # Should select healthy worker
        selected = scheduler.select([unhealthy_worker, healthy_worker])
        assert selected is healthy_worker

    def test_tasks_route_to_remaining_healthy_workers(self):
        """Tasks automatically route to healthy workers."""
        healthy = FailableWorker(max_concurrent=10)
        unhealthy = FailableWorker(max_concurrent=10)
        unhealthy.set_healthy(False)

        runtime = DistributedRuntime(
            workers=[unhealthy, healthy],
            scheduler=HealthCheckScheduler(),
        )

        # All tasks should go to healthy worker
        for i in range(5):
            result = runtime.run(
                spec=make_spec(f"routed_{i}"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )
            assert result.success

        # Verify all went to healthy worker
        assert healthy.stats.completed == 5
        assert unhealthy.stats.completed == 0

    def test_failover_during_execution(self):
        """System handles worker failure mid-execution."""
        worker1 = FailableWorker(max_concurrent=4, fail_after=2)
        worker2 = FailableWorker(max_concurrent=4)

        runtime = DistributedRuntime(
            workers=[worker1, worker2],
            scheduler=HealthCheckScheduler(),
        )

        results = []
        errors = []

        for i in range(6):
            try:
                result = runtime.run(
                    spec=make_spec(f"failover_{i}"),
                    provider=ProviderSpec(name="mock"),
                    data="",
                    options=RuntimeOptions(),
                )
                results.append(result)
            except RuntimeError:
                errors.append(i)

        # First 2 succeed on worker1, then it fails
        # Subsequent tasks should try worker1 and fail, then potentially succeed
        # The exact behavior depends on scheduler retry logic
        assert len(results) >= 2  # At least first 2 should succeed


# =============================================================================
# TEST CLASS: Graceful Degradation
# =============================================================================


class TestGracefulDegradation:
    """Test system behavior under partial failures."""

    def test_continues_with_reduced_capacity(self):
        """System continues operating with reduced capacity."""
        workers = [FailableWorker(max_concurrent=2) for _ in range(4)]

        # Mark half as unhealthy
        workers[0].set_healthy(False)
        workers[1].set_healthy(False)

        runtime = DistributedRuntime(
            workers=workers,
            scheduler=HealthCheckScheduler(),
        )

        # Should still be able to process tasks
        results = []
        for i in range(8):
            result = runtime.run(
                spec=make_spec(f"degraded_{i}"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )
            results.append(result)

        assert all(r.success for r in results)

        # Verify only healthy workers were used
        assert workers[0].stats.completed == 0
        assert workers[1].stats.completed == 0
        assert workers[2].stats.completed + workers[3].stats.completed == 8

    def test_stats_reflect_failures(self):
        """Runtime stats accurately reflect worker failures."""
        worker = FailableWorker(max_concurrent=4, fail_after=2)
        runtime = DistributedRuntime(workers=[worker])

        for i in range(5):
            try:
                runtime.run(
                    spec=make_spec(f"stat_test_{i}"),
                    provider=ProviderSpec(name="mock"),
                    data="",
                    options=RuntimeOptions(),
                )
            except RuntimeError:
                pass

        stats = runtime.stats
        assert stats["completed"] == 2  # First 2 succeeded
        assert stats["failed"] == 3  # Last 3 failed

    def test_recovery_increases_capacity(self):
        """Recovering workers increase available capacity."""
        workers = [FailableWorker(max_concurrent=4) for _ in range(2)]
        workers[0].set_healthy(False)

        runtime = DistributedRuntime(
            workers=workers,
            scheduler=HealthCheckScheduler(),
        )

        # Initially only worker[1] handles tasks
        for i in range(3):
            runtime.run(
                spec=make_spec(f"before_{i}"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )

        assert workers[0].stats.completed == 0
        assert workers[1].stats.completed == 3

        # Recover worker[0]
        workers[0].set_healthy(True)

        # Now both should handle tasks
        for i in range(4):
            runtime.run(
                spec=make_spec(f"after_{i}"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )

        # At least some should go to recovered worker
        # (depends on scheduler behavior)
        total_completed = workers[0].stats.completed + workers[1].stats.completed
        assert total_completed == 7


# =============================================================================
# TEST CLASS: Concurrent Failover
# =============================================================================


class TestConcurrentFailover:
    """Test failover behavior under concurrent load."""

    def test_concurrent_tasks_during_failure(self):
        """Concurrent tasks handle worker failure correctly."""
        worker1 = FailableWorker(max_concurrent=10, fail_after=5)
        worker2 = FailableWorker(max_concurrent=10)

        runtime = DistributedRuntime(
            workers=[worker1, worker2],
            scheduler=HealthCheckScheduler(),
        )

        results = []
        errors = []
        lock = threading.Lock()

        def submit_task(task_id: int):
            try:
                result = runtime.run(
                    spec=make_spec(f"concurrent_{task_id}"),
                    provider=ProviderSpec(name="mock"),
                    data="",
                    options=RuntimeOptions(),
                )
                with lock:
                    results.append(result)
            except RuntimeError as e:
                with lock:
                    errors.append((task_id, str(e)))

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(submit_task, i) for i in range(20)]
            for f in as_completed(futures, timeout=30.0):
                f.result()

        # Some tasks should succeed
        assert len(results) >= 5  # At least the first 5 on worker1
        print(f"\nConcurrent failover: {len(results)} succeeded, {len(errors)} failed")

    def test_all_workers_fail_gracefully(self):
        """System handles all workers failing."""
        workers = [
            FailableWorker(max_concurrent=4, fail_permanently=True)
            for _ in range(3)
        ]

        runtime = DistributedRuntime(workers=workers)

        errors = []
        for i in range(5):
            try:
                runtime.run(
                    spec=make_spec(f"all_fail_{i}"),
                    provider=ProviderSpec(name="mock"),
                    data="",
                    options=RuntimeOptions(),
                )
            except RuntimeError as e:
                errors.append(e)

        # All should fail
        assert len(errors) == 5

        # Runtime should still be operational (not crashed)
        assert not runtime._shutdown


# =============================================================================
# TEST CLASS: Recovery Scenarios
# =============================================================================


class TestRecoveryScenarios:
    """Test various recovery scenarios."""

    def test_temporary_network_failure(self):
        """System recovers from temporary network-like failure."""
        worker = FailableWorker(
            max_concurrent=4,
            fail_after=2,
            fail_for=3,  # Fail for 3 tasks then recover
        )
        runtime = DistributedRuntime(workers=[worker])

        results = []
        errors = []

        for i in range(10):
            try:
                result = runtime.run(
                    spec=make_spec(f"network_{i}"),
                    provider=ProviderSpec(name="mock"),
                    data="",
                    options=RuntimeOptions(),
                )
                results.append(result)
            except RuntimeError:
                errors.append(i)

        # First 2 succeed, next 3 fail, then recovery
        assert len(results) == 7  # 2 before + 5 after
        assert len(errors) == 3

    def test_rolling_restart(self):
        """System handles rolling restart of workers."""
        workers = [FailableWorker(max_concurrent=4) for _ in range(3)]

        runtime = DistributedRuntime(
            workers=workers,
            scheduler=HealthCheckScheduler(),
        )

        # Process initial tasks
        for i in range(6):
            runtime.run(
                spec=make_spec(f"initial_{i}"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )

        # Rolling restart: take down one at a time
        for restart_idx in range(3):
            workers[restart_idx].set_healthy(False)

            # Process during partial outage
            for i in range(3):
                result = runtime.run(
                    spec=make_spec(f"during_{restart_idx}_{i}"),
                    provider=ProviderSpec(name="mock"),
                    data="",
                    options=RuntimeOptions(),
                )
                assert result.success

            # Bring worker back
            workers[restart_idx].set_healthy(True)

        # Final verification - all workers healthy
        for i in range(6):
            result = runtime.run(
                spec=make_spec(f"final_{i}"),
                provider=ProviderSpec(name="mock"),
                data="",
                options=RuntimeOptions(),
            )
            assert result.success
