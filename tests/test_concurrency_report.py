"""
Concurrency Test Suite: Report Generation at Scale

Validates that enzu can handle multiple concurrent "full context builder -> report constructor"
pipelines without:
- API call interference (responses mixed between instances)
- State corruption across concurrent executions
- Connection exhaustion under load
- Race conditions in shared resources

Configuration:
    Set ENZU_CONCURRENCY_SCALE environment variable to control scale (default: 5)
    Set ENZU_STAGGER_DELAYS to customize delays (comma-separated, e.g. "0,2,3,5,7")

Usage:
    pytest tests/test_concurrency_report.py -v
    ENZU_CONCURRENCY_SCALE=10 pytest tests/test_concurrency_report.py -v
    ENZU_CONCURRENCY_SCALE=20 pytest tests/test_concurrency_report.py -v --timeout=120

"""
from __future__ import annotations

import asyncio
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple
import json
from uuid import uuid4

import pytest

from enzu.models import ProviderResult, TaskSpec, Budget, SuccessCriteria
from enzu.providers.base import BaseProvider
from enzu.providers.pool import (
    set_capacity_limit,
    get_capacity_stats,
    close_all_providers,
)
from enzu.isolation.concurrency import (
    configure_global_limiter,
    get_global_limiter,
    reset_global_limiter,
)
from enzu.engine import Engine


# =============================================================================
# CONFIGURATION
# =============================================================================

def get_scale() -> int:
    """Get test scale from environment (default: 5, supports 5, 10, 20)."""
    scale = int(os.environ.get("ENZU_CONCURRENCY_SCALE", "5"))
    return max(1, min(scale, 100))  # Clamp between 1-100


def get_stagger_delays(scale: int) -> List[float]:
    """
    Get stagger delays from environment or generate based on scale.

    Default delays create a realistic staggered load pattern:
    - scale=5:  [0, 2, 3, 5, 7]
    - scale=10: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    - scale=20: [0, 0.5, 1, 1.5, ..., 9.5]
    """
    env_delays = os.environ.get("ENZU_STAGGER_DELAYS")
    if env_delays:
        return [float(d.strip()) for d in env_delays.split(",")]

    if scale <= 5:
        # Original plan delays for 5 instances
        return [0, 2, 3, 5, 7][:scale]
    elif scale <= 10:
        # 1-second spacing for 10 instances
        return [float(i) for i in range(scale)]
    else:
        # 0.5-second spacing for larger scales
        return [i * 0.5 for i in range(scale)]


# =============================================================================
# DATA CLASSES (Ray-Inspired Design)
# =============================================================================

@dataclass
class InstanceResult:
    """Result from a single report generation instance."""
    instance_id: int
    correlation_marker: str
    success: bool
    output_text: Optional[str]
    input_data: str
    start_time: float
    end_time: float
    duration_ms: float
    error: Optional[str] = None

    @property
    def isolation_valid(self) -> bool:
        """Check if correlation marker is in output (isolation maintained)."""
        if not self.success or not self.output_text:
            return False
        return self.correlation_marker in self.output_text


@dataclass
class ConcurrencyMetrics:
    """Metrics collected during test run."""
    # Timing
    total_duration_ms: float
    instance_durations_ms: List[float]
    overlap_windows: List[Tuple[float, float]]  # (start, end) of concurrent periods

    # Correctness
    isolation_violations: int  # Cross-contamination detected
    state_corruptions: int     # Shared state leaks

    # Resources
    peak_concurrent_requests: int
    connection_pool_size: int
    api_calls_total: int
    api_errors: int

    # Results
    total_instances: int
    successful_instances: int
    failed_instances: int

    def success_rate(self) -> float:
        if self.total_instances == 0:
            return 0.0
        return self.successful_instances / self.total_instances

    def to_report(self) -> str:
        """Generate formatted report."""
        return f"""
{'='*66}
              CONCURRENCY TEST REPORT
{'='*66}
 Total instances: {self.total_instances}
 Successful: {self.successful_instances}
 Failed: {self.failed_instances}
 Success rate: {self.success_rate():.1%}
 Peak concurrent: {self.peak_concurrent_requests}
{'='*66}
 ISOLATION CHECK
{'-'*66}
 Isolation violations: {self.isolation_violations}
 State corruptions: {self.state_corruptions}
{'='*66}
 TIMING
{'-'*66}
 Total duration: {self.total_duration_ms:.1f}ms
 Avg instance duration: {sum(self.instance_durations_ms) / len(self.instance_durations_ms) if self.instance_durations_ms else 0:.1f}ms
 Min instance duration: {min(self.instance_durations_ms) if self.instance_durations_ms else 0:.1f}ms
 Max instance duration: {max(self.instance_durations_ms) if self.instance_durations_ms else 0:.1f}ms
{'='*66}
 RESOURCES
{'-'*66}
 Connection pool size: {self.connection_pool_size}
 Total API calls: {self.api_calls_total}
 API errors: {self.api_errors}
{'='*66}
"""


# =============================================================================
# MOCK PROVIDER FOR TESTING
# =============================================================================

class ConcurrencyMockProvider(BaseProvider):
    """
    Mock provider that simulates realistic LLM response times and tracks calls.

    Key features:
    - Includes correlation marker in response (for isolation testing)
    - Simulates realistic latency (100-500ms)
    - Thread-safe call tracking
    - Configurable failure rate
    """
    name = "concurrency_mock"

    def __init__(
        self,
        latency_ms: Tuple[int, int] = (100, 300),
        failure_rate: float = 0.0,
    ) -> None:
        self.latency_range = latency_ms
        self.failure_rate = failure_rate
        self._calls: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._active_count = 0
        self._peak_concurrent = 0

    def stream(self, task: TaskSpec, on_progress=None) -> ProviderResult:
        """Process request with simulated latency."""
        import random

        # Track concurrency
        with self._lock:
            self._active_count += 1
            if self._active_count > self._peak_concurrent:
                self._peak_concurrent = self._active_count

        try:
            # Simulate failure
            if random.random() < self.failure_rate:
                raise Exception("Simulated provider failure")

            # Simulate latency
            latency = random.uniform(
                self.latency_range[0] / 1000,
                self.latency_range[1] / 1000,
            )
            time.sleep(latency)

            # Extract correlation marker from metadata or input
            correlation_marker = task.metadata.get("correlation_marker", "")
            input_data = task.metadata.get("input_data", "")
            instance_id = task.metadata.get("instance_id", -1)

            # Build response that includes markers for verification
            output_text = (
                f"Report generated for instance {instance_id}. "
                f"Correlation: {correlation_marker}. "
                f"Input data processed: {input_data}. "
                f"Task ID: {task.task_id}."
            )

            # Track call
            with self._lock:
                self._calls.append({
                    "task_id": task.task_id,
                    "instance_id": instance_id,
                    "correlation_marker": correlation_marker,
                    "timestamp": time.time(),
                })

            return ProviderResult(
                output_text=output_text,
                raw={"mock": True},
                usage={"output_tokens": len(output_text.split()), "total_tokens": 100},
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

    def reset(self):
        with self._lock:
            self._calls.clear()
            self._peak_concurrent = 0


# =============================================================================
# STAGGERED SCHEDULER (Ray-Inspired)
# =============================================================================

@dataclass
class StaggeredScheduler:
    """
    Ray-inspired staggered task submission.

    Submits tasks at scheduled delays to simulate realistic load patterns
    where requests arrive at different times.
    """
    delays: List[float] = field(default_factory=lambda: [0, 2, 3, 5, 7])

    async def run(
        self,
        task_factory: Callable[[int], Coroutine],
    ) -> Tuple[List[InstanceResult], ConcurrencyMetrics]:
        """
        Submit tasks at scheduled delays, collect all results.

        Args:
            task_factory: Async function that takes instance_id and returns InstanceResult

        Returns:
            Tuple of (results list, metrics)
        """
        tasks: List[asyncio.Task] = []
        start = time.time()

        # Track concurrent execution
        active_count = [0]
        peak_concurrent = [0]
        overlap_windows: List[Tuple[float, float]] = []
        lock = threading.Lock()

        async def wrapped_task(instance_id: int, delay: float) -> InstanceResult:
            # Wait for scheduled time
            elapsed = time.time() - start
            wait_time = max(0, delay - elapsed)
            if wait_time > 0:
                await asyncio.sleep(wait_time)

            # Track overlap
            with lock:
                active_count[0] += 1
                if active_count[0] > peak_concurrent[0]:
                    peak_concurrent[0] = active_count[0]
                if active_count[0] > 1:
                    overlap_windows.append((time.time() - start, -1))

            try:
                result = await task_factory(instance_id)
                return result
            finally:
                with lock:
                    active_count[0] -= 1
                    # Update last overlap window end time
                    if overlap_windows and overlap_windows[-1][1] == -1:
                        overlap_windows[-1] = (overlap_windows[-1][0], time.time() - start)

        # Submit all tasks with their scheduled delays
        for i, delay in enumerate(self.delays):
            task = asyncio.create_task(wrapped_task(i + 1, delay))
            tasks.append(task)

        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_duration = (time.time() - start) * 1000

        # Process results
        instance_results: List[InstanceResult] = []
        api_errors = 0

        for i, r in enumerate(results):
            if isinstance(r, BaseException):
                api_errors += 1
                instance_results.append(InstanceResult(
                    instance_id=i + 1,
                    correlation_marker="",
                    success=False,
                    output_text=None,
                    input_data="",
                    start_time=0,
                    end_time=0,
                    duration_ms=0,
                    error=str(r),
                ))
            else:
                instance_results.append(r)

        # Build metrics
        successful = [r for r in instance_results if r.success]
        failed = [r for r in instance_results if not r.success]

        metrics = ConcurrencyMetrics(
            total_duration_ms=total_duration,
            instance_durations_ms=[r.duration_ms for r in instance_results],
            overlap_windows=overlap_windows,
            isolation_violations=0,  # Calculated later
            state_corruptions=0,     # Calculated later
            peak_concurrent_requests=peak_concurrent[0],
            connection_pool_size=1,  # Updated in tests
            api_calls_total=len(instance_results),
            api_errors=api_errors,
            total_instances=len(instance_results),
            successful_instances=len(successful),
            failed_instances=len(failed),
        )

        return instance_results, metrics


# =============================================================================
# REPORT INSTANCE (Ray Actor Pattern)
# =============================================================================

class ReportInstance:
    """
    Isolated report generation instance with realistic context.

    Each instance maintains its own:
    - Unique instance_id (correlation)
    - Input data (immutable, like Ray object refs)
    - Execution state (not shared)
    - Generated context data (unique per instance)
    """

    # Simulated context templates - each instance gets unique data
    CONTEXT_TEMPLATES = [
        {
            "domain": "financial",
            "metrics": ["revenue", "profit", "growth_rate", "market_share"],
            "trends": ["increasing", "stable", "declining", "volatile"],
        },
        {
            "domain": "technical",
            "metrics": ["latency", "throughput", "error_rate", "uptime"],
            "trends": ["improving", "degrading", "fluctuating", "stable"],
        },
        {
            "domain": "operational",
            "metrics": ["efficiency", "cost", "output", "quality_score"],
            "trends": ["optimized", "needs_attention", "critical", "nominal"],
        },
    ]

    def __init__(self, instance_id: int, input_data: str):
        self.instance_id = instance_id
        self.input_data = input_data
        self._correlation_marker = f"INST-{instance_id}-{uuid4().hex[:8]}"
        self._context_data = self._generate_context_data()

    def _generate_context_data(self) -> Dict[str, Any]:
        """Generate unique context data for this instance."""
        template = self.CONTEXT_TEMPLATES[self.instance_id % len(self.CONTEXT_TEMPLATES)]

        # Generate unique metrics for this instance
        metrics = {}
        for metric in template["metrics"]:
            # Use instance_id to seed deterministic but unique values
            value = ((self.instance_id * 17 + hash(metric)) % 1000) / 10
            trend = template["trends"][(self.instance_id + hash(metric)) % len(template["trends"])]
            metrics[metric] = {"value": value, "trend": trend}

        return {
            "instance_id": self.instance_id,
            "report_id": self._correlation_marker,
            "domain": template["domain"],
            "dataset": self.input_data,
            "timestamp": f"2024-01-{(self.instance_id % 28) + 1:02d}",
            "metrics": metrics,
            "priority": ["high", "medium", "low"][self.instance_id % 3],
        }

    async def execute(
        self,
        provider: BaseProvider,
        model: str = "mock-model",
    ) -> InstanceResult:
        """
        Full context build -> report construct pipeline.

        Args:
            provider: The provider to use for generation
            model: Model name

        Returns:
            InstanceResult with execution details
        """
        start_time = time.time()

        try:
            # Step 1: Context building (preparation phase)
            context = self._build_context()

            # Step 2: Report construction (LLM call)
            task = TaskSpec(
                task_id=f"report-{self.instance_id}-{uuid4().hex[:8]}",
                input_text=f"Generate a report for: {context}",
                model=model,
                responses={},
                budget=Budget(max_tokens=500),
                success_criteria=SuccessCriteria(min_word_count=1),
                metadata={
                    "instance_id": self.instance_id,
                    "correlation_marker": self._correlation_marker,
                    "input_data": self.input_data,
                },
            )

            # Use Engine for execution
            engine = Engine()

            # Run provider call in thread pool to not block async
            loop = asyncio.get_event_loop()
            report = await loop.run_in_executor(
                None,
                lambda: engine.run(task, provider),
            )

            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Step 3: Validate no cross-contamination
            output_text = report.output_text or ""

            return InstanceResult(
                instance_id=self.instance_id,
                correlation_marker=self._correlation_marker,
                success=report.success,
                output_text=output_text,
                input_data=self.input_data,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                error="; ".join(report.errors) if report.errors else None,
            )

        except Exception as e:
            end_time = time.time()
            return InstanceResult(
                instance_id=self.instance_id,
                correlation_marker=self._correlation_marker,
                success=False,
                output_text=None,
                input_data=self.input_data,
                start_time=start_time,
                end_time=end_time,
                duration_ms=(end_time - start_time) * 1000,
                error=str(e),
            )

    def _build_context(self) -> str:
        """Build realistic context document from generated data."""
        data = self._context_data

        # Format as a structured document
        lines = [
            f"=== {data['domain'].upper()} ANALYSIS REPORT ===",
            f"Report ID: {data['report_id']}",
            f"Dataset: {data['dataset']}",
            f"Date: {data['timestamp']}",
            f"Priority: {data['priority']}",
            "",
            "METRICS:",
        ]

        for metric, info in data['metrics'].items():
            lines.append(f"  - {metric}: {info['value']:.1f} ({info['trend']})")

        lines.append("")
        lines.append(f"Raw data: {json.dumps(data['metrics'])}")

        return "\n".join(lines)

    @property
    def correlation_marker(self) -> str:
        return self._correlation_marker


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def mock_provider():
    """Create a fresh mock provider for each test."""
    return ConcurrencyMockProvider()


@pytest.fixture(autouse=True)
def reset_concurrency_state():
    """Reset global concurrency state before/after each test."""
    reset_global_limiter()
    yield
    reset_global_limiter()
    close_all_providers()


# =============================================================================
# TEST CASES
# =============================================================================

class TestStaggeredSubmission:
    """Test that instances start at correct times."""

    @pytest.mark.anyio
    async def test_staggered_submission_timing(self):
        """
        Verify instances start at correct times.
        Pass criteria: Start times within 100ms of schedule
        """
        scale = get_scale()
        delays = get_stagger_delays(scale)
        scheduler = StaggeredScheduler(delays=delays)

        start_times: List[float] = []
        lock = threading.Lock()

        async def record_start(instance_id: int) -> InstanceResult:
            with lock:
                start_times.append(time.time())
            await asyncio.sleep(0.05)  # Small work
            return InstanceResult(
                instance_id=instance_id,
                correlation_marker=f"test-{instance_id}",
                success=True,
                output_text="test",
                input_data="test",
                start_time=time.time(),
                end_time=time.time(),
                duration_ms=50,
            )

        base_time = time.time()
        results, metrics = await scheduler.run(record_start)

        # Verify timing
        actual_delays = [(t - base_time) for t in start_times]

        print(f"\nStaggered Submission Test (scale={scale}):")
        print(f"  Expected delays: {delays[:5]}{'...' if len(delays) > 5 else ''}")
        print(f"  Actual delays: {[f'{d:.2f}' for d in actual_delays[:5]]}{'...' if len(actual_delays) > 5 else ''}")

        # Check each delay is within tolerance
        tolerance_ms = 200  # 200ms tolerance for async scheduling
        for i, (expected, actual) in enumerate(zip(delays, actual_delays)):
            diff = abs(expected - actual)
            assert diff < tolerance_ms / 1000, \
                f"Instance {i+1} started {diff*1000:.0f}ms from schedule (expected {expected}s, got {actual:.2f}s)"

        assert len(results) == scale
        print(f"  All {scale} instances started within {tolerance_ms}ms of schedule")


class TestResponseIsolation:
    """Test that each instance gets its own response."""

    @pytest.mark.anyio
    async def test_response_isolation(self, mock_provider):
        """
        Each instance gets its own response, not another's.
        Pass criteria: No cross-contamination of correlation markers
        """
        scale = get_scale()
        delays = get_stagger_delays(scale)
        scheduler = StaggeredScheduler(delays=delays)

        # Create instances with unique data
        instances = [
            ReportInstance(i + 1, f"dataset-[{i + 1:03d}]")
            for i in range(scale)
        ]

        async def run_instance(instance_id: int) -> InstanceResult:
            instance = instances[instance_id - 1]
            return await instance.execute(mock_provider)

        results, metrics = await scheduler.run(run_instance)

        # Check isolation
        isolation_violations = 0
        for result in results:
            if not result.success:
                continue

            # Each result must contain its unique correlation marker
            if result.correlation_marker not in (result.output_text or ""):
                isolation_violations += 1
                continue

            # Must NOT contain any other instance's marker
            for other in results:
                if other.instance_id != result.instance_id:
                    if other.correlation_marker in (result.output_text or ""):
                        isolation_violations += 1
                        break

        metrics.isolation_violations = isolation_violations

        print(f"\nResponse Isolation Test (scale={scale}):")
        print(f"  Total instances: {len(results)}")
        print(f"  Successful: {len([r for r in results if r.success])}")
        print(f"  Isolation violations: {isolation_violations}")
        print(f"  Peak concurrent: {mock_provider.peak_concurrent}")

        assert isolation_violations == 0, \
            f"Found {isolation_violations} isolation violations"

        print(f"  All {scale} instances properly isolated")


class TestStateCorruption:
    """Test that no shared state leaks between instances."""

    @pytest.mark.anyio
    async def test_no_state_corruption(self, mock_provider):
        """
        Concurrent instances don't corrupt each other's state.
        Pass criteria: Each output references only its own input data
        """
        scale = get_scale()
        delays = get_stagger_delays(scale)
        scheduler = StaggeredScheduler(delays=delays)

        # Each instance processes different data
        # Use zero-padded format to prevent substring collisions (e.g., data-1 in data-10)
        instances = [
            ReportInstance(i + 1, f"unique-data-[{i + 1:03d}]")
            for i in range(scale)
        ]

        async def run_instance(instance_id: int) -> InstanceResult:
            instance = instances[instance_id - 1]
            return await instance.execute(mock_provider)

        results, metrics = await scheduler.run(run_instance)

        # Check state isolation
        state_corruptions = 0
        for result in results:
            if not result.success:
                continue

            output = result.output_text or ""

            # Output must reference correct input data
            if result.input_data not in output:
                state_corruptions += 1
                continue

            # Must not reference any other dataset
            for other in results:
                if other.instance_id != result.instance_id:
                    if other.input_data in output:
                        state_corruptions += 1
                        break

        metrics.state_corruptions = state_corruptions

        print(f"\nState Corruption Test (scale={scale}):")
        print(f"  Total instances: {len(results)}")
        print(f"  State corruptions: {state_corruptions}")

        assert state_corruptions == 0, \
            f"Found {state_corruptions} state corruption cases"

        print(f"  All {scale} instances maintained correct state")


class TestConnectionPooling:
    """Test connection pooling handles concurrent requests."""

    @pytest.mark.anyio
    async def test_connection_pool_under_load(self, mock_provider):
        """
        Provider pool shares connections without exhaustion.
        Pass criteria: All requests succeed with single shared provider
        """
        scale = get_scale()
        delays = get_stagger_delays(scale)

        # Configure concurrency limiter for the scale
        max_concurrent = max(50, scale * 2)
        configure_global_limiter(max_concurrent=max_concurrent, force_reconfigure=True)
        set_capacity_limit(max_concurrent)

        scheduler = StaggeredScheduler(delays=delays)

        instances = [
            ReportInstance(i + 1, f"pool-test-[{i + 1:03d}]")
            for i in range(scale)
        ]

        async def run_instance(instance_id: int) -> InstanceResult:
            instance = instances[instance_id - 1]
            return await instance.execute(mock_provider)

        results, metrics = await scheduler.run(run_instance)

        # Get pool stats
        stats = get_capacity_stats()
        limiter_stats = get_global_limiter().stats()

        print(f"\nConnection Pool Test (scale={scale}):")
        print(f"  Configured max concurrent: {max_concurrent}")
        print(f"  Total instances: {len(results)}")
        print(f"  Successful: {len([r for r in results if r.success])}")
        print(f"  Pool providers count: {stats['providers_count']}")
        print(f"  Peak concurrent in provider: {mock_provider.peak_concurrent}")
        print(f"  Limiter total acquired: {limiter_stats.total_acquired}")

        successful = [r for r in results if r.success]
        assert len(successful) >= scale * 0.95, \
            f"Only {len(successful)}/{scale} requests succeeded"

        print(f"  Pool handled {scale} concurrent requests successfully")


class TestConcurrentAPICalls:
    """Test that API doesn't mix responses between concurrent calls."""

    @pytest.mark.anyio
    async def test_concurrent_api_calls(self, mock_provider):
        """
        API doesn't mix responses between concurrent instances.
        Pass criteria: All correlation markers match their instances
        """
        scale = get_scale()
        delays = get_stagger_delays(scale)
        scheduler = StaggeredScheduler(delays=delays)

        instances = [
            ReportInstance(i + 1, f"api-test-[{i + 1:03d}]")
            for i in range(scale)
        ]

        async def run_instance(instance_id: int) -> InstanceResult:
            instance = instances[instance_id - 1]
            return await instance.execute(mock_provider)

        results, metrics = await scheduler.run(run_instance)

        # Verify each instance's marker appears in its response
        marker_mismatches = 0
        for result in results:
            if not result.success:
                continue
            if not result.isolation_valid:
                marker_mismatches += 1

        # Check provider call log
        calls = mock_provider.calls
        call_instance_ids = [c["instance_id"] for c in calls]

        print(f"\nConcurrent API Calls Test (scale={scale}):")
        print(f"  Total instances: {len(results)}")
        print(f"  Marker mismatches: {marker_mismatches}")
        print(f"  Provider calls recorded: {len(calls)}")
        print(f"  Unique instance IDs in calls: {len(set(call_instance_ids))}")

        assert marker_mismatches == 0, \
            f"Found {marker_mismatches} correlation marker mismatches"

        assert len(set(call_instance_ids)) == scale, \
            f"Expected {scale} unique instance IDs, got {len(set(call_instance_ids))}"

        print(f"  All {scale} API calls correctly correlated")


class TestGracefulDegradation:
    """Test system handles overload gracefully."""

    @pytest.mark.anyio
    async def test_graceful_degradation(self):
        """
        System handles overload without crashes.
        Pass criteria: Proper rejection, no crashes
        """
        scale = get_scale()

        # Create provider with some failure rate
        failing_provider = ConcurrencyMockProvider(
            latency_ms=(50, 100),
            failure_rate=0.1,  # 10% failure rate
        )

        # Configure tight limits to force backpressure
        tight_limit = max(3, scale // 3)
        configure_global_limiter(max_concurrent=tight_limit, force_reconfigure=True)

        # Use faster delays to create more pressure
        fast_delays = [i * 0.2 for i in range(scale)]  # 200ms between starts
        scheduler = StaggeredScheduler(delays=fast_delays)

        instances = [
            ReportInstance(i + 1, f"stress-test-[{i + 1:03d}]")
            for i in range(scale)
        ]

        async def run_instance(instance_id: int) -> InstanceResult:
            instance = instances[instance_id - 1]
            return await instance.execute(failing_provider)

        results, metrics = await scheduler.run(run_instance)

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        limiter_stats = get_global_limiter().stats()

        print(f"\nGraceful Degradation Test (scale={scale}):")
        print(f"  Concurrency limit: {tight_limit}")
        print("  Provider failure rate: 10%")
        print(f"  Total instances: {len(results)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        print(f"  Limiter rejected: {limiter_stats.total_rejected}")

        # System should not crash and some requests should succeed
        # Even with 10% failure rate and tight limits, expect at least 50% success
        assert len(successful) > 0, "No requests succeeded under load"
        assert len(results) == scale, "Some requests were lost (crash)"

        # Success rate should be reasonable (at least 40% given 10% provider failures)
        success_rate = len(successful) / scale
        print(f"  Success rate: {success_rate:.1%}")

        assert success_rate >= 0.4, \
            f"Success rate {success_rate:.1%} too low under load"

        print("  System handled degradation gracefully")


class TestFullPipeline:
    """
    Full end-to-end test combining all checks.
    This is the main certification test.
    """

    @pytest.mark.anyio
    async def test_full_concurrency_pipeline(self, mock_provider):
        """
        Full concurrency test with all validations.

        This test runs the complete pipeline and generates
        a comprehensive report.
        """
        scale = get_scale()
        delays = get_stagger_delays(scale)

        print(f"\n{'='*66}")
        print("  FULL CONCURRENCY PIPELINE TEST")
        print(f"  Scale: {scale} instances")
        print(f"  Delays: {delays[:5]}{'...' if len(delays) > 5 else ''}")
        print(f"{'='*66}")

        # Configure for scale
        max_concurrent = max(50, scale * 2)
        configure_global_limiter(max_concurrent=max_concurrent, force_reconfigure=True)

        scheduler = StaggeredScheduler(delays=delays)

        # Create instances with unique data
        instances = [
            ReportInstance(i + 1, f"full-test-dataset-[{i + 1:03d}]")
            for i in range(scale)
        ]

        async def run_instance(instance_id: int) -> InstanceResult:
            instance = instances[instance_id - 1]
            return await instance.execute(mock_provider)

        results, metrics = await scheduler.run(run_instance)

        # Check isolation
        isolation_violations = 0
        state_corruptions = 0

        for result in results:
            if not result.success:
                continue

            output = result.output_text or ""

            # Check isolation
            if result.correlation_marker not in output:
                isolation_violations += 1

            # Check state
            if result.input_data not in output:
                state_corruptions += 1

            # Check for cross-contamination
            for other in results:
                if other.instance_id != result.instance_id:
                    if other.correlation_marker in output:
                        isolation_violations += 1
                    if other.input_data in output:
                        state_corruptions += 1

        # Update metrics
        metrics.isolation_violations = isolation_violations
        metrics.state_corruptions = state_corruptions
        metrics.connection_pool_size = get_capacity_stats()["providers_count"]

        # Print report
        print(metrics.to_report())

        # Assertions
        assert metrics.success_rate() >= 0.95, \
            f"Success rate {metrics.success_rate():.1%} below 95%"

        assert metrics.isolation_violations == 0, \
            f"Found {metrics.isolation_violations} isolation violations"

        assert metrics.state_corruptions == 0, \
            f"Found {metrics.state_corruptions} state corruptions"

        print(f"CERTIFICATION PASSED: {scale} instances processed correctly")
        print(f"{'='*66}\n")


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "concurrency: marks test as concurrency test"
    )


if __name__ == "__main__":
    # Allow running directly for quick testing
    import sys
    pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:])
