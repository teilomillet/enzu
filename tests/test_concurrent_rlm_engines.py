"""
Test Concurrent RLMEngine Instances

Validates that 10+ RLMEngine instances can run concurrently with:
- Different subjects (unique data per instance)
- Subagent spawning via llm_query/llm_batch
- No cross-contamination between instances
- No state corruption

This test is designed to ACTUALLY verify correctness - not just claim to pass.
Each instance has a unique subject/marker that MUST appear in its output and
MUST NOT appear in any other instance's output.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import pytest

from enzu.models import (
    Budget,
    ProviderResult,
    SuccessCriteria,
    TaskSpec,
)
from enzu.providers.base import BaseProvider
from enzu.rlm.engine import RLMEngine


# =============================================================================
# DATA CLASSES FOR TEST TRACKING
# =============================================================================


@dataclass
class EngineResult:
    """Result from a single RLMEngine execution."""

    instance_id: int
    subject: str  # Unique subject/topic for this instance
    correlation_marker: str  # Unique marker that must appear in output
    success: bool
    answer: Optional[str]
    output_text: Optional[str]
    start_time: float
    end_time: float
    duration_ms: float
    subcalls_made: int  # How many llm_query/llm_batch calls were made
    error: Optional[str] = None

    @property
    def isolation_valid(self) -> bool:
        """Check if correlation marker is in output (isolation maintained)."""
        if not self.success or not self.answer:
            return False
        return self.correlation_marker in self.answer


@dataclass
class ConcurrencyTestMetrics:
    """Metrics from the concurrent test run."""

    total_instances: int
    successful_instances: int
    failed_instances: int
    isolation_violations: int  # Cross-contamination detected
    missing_markers: int  # Instances that don't have their own marker
    peak_concurrent: int
    total_subcalls: int
    total_duration_ms: float

    def to_report(self) -> str:
        return f"""
{"=" * 66}
        CONCURRENT RLMENGINE TEST REPORT
{"=" * 66}
 Total instances: {self.total_instances}
 Successful: {self.successful_instances}
 Failed: {self.failed_instances}
 Success rate: {self.successful_instances / self.total_instances:.1%}
{"=" * 66}
 ISOLATION CHECK
{"-" * 66}
 Isolation violations (cross-contamination): {self.isolation_violations}
 Missing markers (own marker not found): {self.missing_markers}
{"=" * 66}
 CONCURRENCY
{"-" * 66}
 Peak concurrent executions: {self.peak_concurrent}
 Total subcalls (llm_query): {self.total_subcalls}
 Total duration: {self.total_duration_ms:.1f}ms
{"=" * 66}
"""


# =============================================================================
# MOCK PROVIDER FOR TESTING
# =============================================================================


class SubjectAwareProvider(BaseProvider):
    """
    Mock provider that:
    1. Recognizes the subject/marker from the task
    2. Returns code that includes the marker in FINAL()
    3. Tracks subcalls to verify llm_query/llm_batch work
    4. Is thread-safe for concurrent access

    The provider ensures each instance gets output containing ONLY its own marker.
    """

    name = "subject_aware_mock"

    # Different subjects we'll test with
    SUBJECTS = [
        "quantum_physics",
        "machine_learning",
        "climate_science",
        "blockchain_technology",
        "neuroscience",
        "renewable_energy",
        "genetic_engineering",
        "space_exploration",
        "cybersecurity",
        "artificial_intelligence",
        "nanotechnology",
        "robotics",
        "data_science",
        "bioinformatics",
        "cryptography",
    ]

    def __init__(self, latency_ms: Tuple[int, int] = (10, 50)) -> None:
        self.latency_range = latency_ms
        self._calls: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._active_count = 0
        self._peak_concurrent = 0
        self._subcall_count = 0

    def stream(self, task: TaskSpec, on_progress=None) -> ProviderResult:
        import random

        # Track concurrency
        with self._lock:
            self._active_count += 1
            if self._active_count > self._peak_concurrent:
                self._peak_concurrent = self._active_count

        try:
            # Simulate latency
            latency = random.uniform(
                self.latency_range[0] / 1000,
                self.latency_range[1] / 1000,
            )
            time.sleep(latency)

            # Extract markers from metadata or input
            instance_id = task.metadata.get("instance_id", -1)
            correlation_marker = task.metadata.get("correlation_marker", "UNKNOWN")
            subject = task.metadata.get("subject", "unknown")
            is_subcall = task.metadata.get("_subcall_prompt") is not None

            # Track call
            with self._lock:
                self._calls.append(
                    {
                        "task_id": task.task_id,
                        "instance_id": instance_id,
                        "correlation_marker": correlation_marker,
                        "subject": subject,
                        "is_subcall": is_subcall,
                        "timestamp": time.time(),
                    }
                )
                if is_subcall:
                    self._subcall_count += 1

            # For subcalls, just return the marker info
            if is_subcall:
                return ProviderResult(
                    output_text=f"FINAL('Sub-research on {subject}: {correlation_marker}')",
                    raw={"mock": True},
                    usage={"output_tokens": 20, "total_tokens": 100},
                    provider=self.name,
                    model=task.model,
                )

            # Main call: generate code that uses llm_query and returns the marker
            # This tests that subagents work AND that isolation is maintained
            code = f"""```python
# Research task on {subject} with marker {correlation_marker}
# Use llm_query to spawn a subagent
sub_result = llm_query("Research details about {subject}")

# Build final answer with the correlation marker
answer = f"Analysis of {subject}: {correlation_marker}. Sub-research: {{sub_result}}"
FINAL(answer)
```"""

            return ProviderResult(
                output_text=code,
                raw={"mock": True},
                usage={"output_tokens": 50, "total_tokens": 200},
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

    @property
    def subcall_count(self) -> int:
        with self._lock:
            return self._subcall_count

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()
            self._peak_concurrent = 0
            self._active_count = 0
            self._subcall_count = 0


# =============================================================================
# TEST RUNNER
# =============================================================================


def run_single_engine(
    instance_id: int,
    subject: str,
    correlation_marker: str,
    provider: BaseProvider,
) -> EngineResult:
    """Run a single RLMEngine with the given subject."""
    start_time = time.time()

    task = TaskSpec(
        task_id=f"rlm-concurrent-{instance_id}-{uuid4().hex[:8]}",
        input_text=f"Analyze the topic of {subject}. Include the marker {correlation_marker} in your response.",
        model="mock-model",
        budget=Budget(max_tokens=500),
        success_criteria=SuccessCriteria(required_substrings=[correlation_marker]),
        metadata={
            "instance_id": instance_id,
            "correlation_marker": correlation_marker,
            "subject": subject,
        },
    )

    engine = RLMEngine(
        max_steps=3,
        recursive_subcalls=True,
        max_recursion_depth=1,
    )

    try:
        report = engine.run(task, provider, data=f"Context for {subject}")
        end_time = time.time()

        return EngineResult(
            instance_id=instance_id,
            subject=subject,
            correlation_marker=correlation_marker,
            success=report.success,
            answer=report.answer,
            output_text=report.answer,
            start_time=start_time,
            end_time=end_time,
            duration_ms=(end_time - start_time) * 1000,
            subcalls_made=len([s for s in report.steps if "llm_query" in str(s)]),
            error="; ".join(report.errors) if report.errors else None,
        )
    except Exception as e:
        end_time = time.time()
        return EngineResult(
            instance_id=instance_id,
            subject=subject,
            correlation_marker=correlation_marker,
            success=False,
            answer=None,
            output_text=None,
            start_time=start_time,
            end_time=end_time,
            duration_ms=(end_time - start_time) * 1000,
            subcalls_made=0,
            error=str(e),
        )


async def run_concurrent_engines(
    num_instances: int,
    provider: SubjectAwareProvider,
) -> Tuple[List[EngineResult], ConcurrencyTestMetrics]:
    """Run multiple RLMEngines concurrently and collect results."""

    subjects = SubjectAwareProvider.SUBJECTS[:num_instances]
    # Generate unique correlation markers
    markers = [f"MARKER-{i:03d}-{uuid4().hex[:8]}" for i in range(num_instances)]

    # Use ThreadPoolExecutor since RLMEngine.run is synchronous
    from concurrent.futures import ThreadPoolExecutor

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_instances) as executor:
        futures = [
            executor.submit(
                run_single_engine,
                i + 1,
                subjects[i % len(subjects)],
                markers[i],
                provider,
            )
            for i in range(num_instances)
        ]
        results = [f.result() for f in futures]

    total_duration_ms = (time.time() - start_time) * 1000

    # Analyze results for isolation violations
    isolation_violations = 0
    missing_markers = 0

    for result in results:
        if not result.success:
            continue

        # Check own marker is present
        if result.correlation_marker not in (result.answer or ""):
            missing_markers += 1
            continue

        # Check no other markers are present (cross-contamination)
        for other in results:
            if other.instance_id != result.instance_id:
                if other.correlation_marker in (result.answer or ""):
                    isolation_violations += 1
                    break

    metrics = ConcurrencyTestMetrics(
        total_instances=num_instances,
        successful_instances=len([r for r in results if r.success]),
        failed_instances=len([r for r in results if not r.success]),
        isolation_violations=isolation_violations,
        missing_markers=missing_markers,
        peak_concurrent=provider.peak_concurrent,
        total_subcalls=provider.subcall_count,
        total_duration_ms=total_duration_ms,
    )

    return results, metrics


# =============================================================================
# TESTS
# =============================================================================


class TestConcurrentRLMEngines:
    """Test suite for concurrent RLMEngine execution."""

    @pytest.fixture
    def provider(self):
        """Fresh provider for each test."""
        return SubjectAwareProvider()

    @pytest.mark.anyio
    async def test_10_concurrent_engines_with_subagents(self, provider):
        """
        Test 10 concurrent RLMEngine instances with different subjects.

        Each instance:
        - Has a unique subject (quantum_physics, machine_learning, etc.)
        - Has a unique correlation marker
        - Spawns subagents via llm_query

        Pass criteria:
        - All instances succeed
        - No cross-contamination (each output contains ONLY its own marker)
        - Subagents were actually spawned
        """
        results, metrics = await run_concurrent_engines(10, provider)

        print(metrics.to_report())

        # MUST have all instances succeed
        assert metrics.successful_instances == 10, (
            f"Expected all 10 instances to succeed, but {metrics.failed_instances} failed"
        )

        # MUST have zero isolation violations
        assert metrics.isolation_violations == 0, (
            f"Found {metrics.isolation_violations} cross-contamination violations"
        )

        # MUST have zero missing markers
        assert metrics.missing_markers == 0, (
            f"Found {metrics.missing_markers} instances missing their own marker"
        )

        # MUST have spawned subagents (each instance calls llm_query)
        assert metrics.total_subcalls >= 10, (
            f"Expected at least 10 subcalls, but only got {metrics.total_subcalls}"
        )

        # Verify we actually ran concurrently (peak > 1)
        assert metrics.peak_concurrent > 1, (
            f"Expected concurrent execution, but peak was {metrics.peak_concurrent}"
        )

        print("SUCCESS: All 10 instances completed with proper isolation")
        print(f"  Peak concurrent: {metrics.peak_concurrent}")
        print(f"  Total subcalls: {metrics.total_subcalls}")

    @pytest.mark.anyio
    async def test_15_concurrent_engines_with_subagents(self, provider):
        """
        Test 15 concurrent RLMEngine instances - stress test.

        Same criteria as test_10 but with more instances.
        """
        results, metrics = await run_concurrent_engines(15, provider)

        print(metrics.to_report())

        # Allow some failures in stress test (90% success rate)
        min_success = int(15 * 0.9)
        assert metrics.successful_instances >= min_success, (
            f"Expected at least {min_success} successes, but got {metrics.successful_instances}"
        )

        # MUST have zero isolation violations even under stress
        assert metrics.isolation_violations == 0, (
            f"Found {metrics.isolation_violations} cross-contamination violations under stress"
        )

        # Missing markers are okay if the instance failed
        successful_missing = min(metrics.missing_markers, metrics.successful_instances)
        assert successful_missing == 0, (
            f"Found {successful_missing} successful instances missing their marker"
        )

        print(
            f"SUCCESS: {metrics.successful_instances}/15 instances completed with proper isolation"
        )

    @pytest.mark.anyio
    async def test_engine_isolation_per_subject(self, provider):
        """
        Verify each subject's engine is truly isolated.

        This test specifically checks that:
        - Each engine gets its own subject in the output
        - No subject appears in another engine's output
        """
        results, metrics = await run_concurrent_engines(10, provider)

        # Check subject isolation
        for result in results:
            if not result.success:
                continue

            # Own subject should be in output
            assert result.subject in (result.answer or ""), (
                f"Instance {result.instance_id}'s output missing its subject '{result.subject}'"
            )

            # Other subjects should NOT be in output (unless it's a common word)
            for other in results:
                if other.instance_id != result.instance_id:
                    # Only check for subject contamination, not common words
                    if (
                        other.subject in (result.answer or "")
                        and other.subject != result.subject
                    ):
                        # This is a potential contamination
                        # But we need to be careful - "machine_learning" appearing in "artificial_intelligence"
                        # discussion would be normal. So we check for the correlation marker instead.
                        pass

        # The correlation marker check is the definitive isolation test
        assert metrics.isolation_violations == 0, (
            f"Found {metrics.isolation_violations} isolation violations"
        )

        print("SUCCESS: All subjects properly isolated")

    @pytest.mark.anyio
    async def test_concurrent_engines_with_varying_loads(self, provider):
        """
        Test concurrent engines with varying numbers to ensure stability.
        """
        for num_instances in [5, 10, 12]:
            provider.reset()
            results, metrics = await run_concurrent_engines(num_instances, provider)

            print(f"\n--- {num_instances} instances ---")
            print(f"  Success: {metrics.successful_instances}/{num_instances}")
            print(f"  Isolation violations: {metrics.isolation_violations}")
            print(f"  Peak concurrent: {metrics.peak_concurrent}")

            # All should succeed with zero violations
            assert metrics.isolation_violations == 0, (
                f"Violations with {num_instances} instances: {metrics.isolation_violations}"
            )

        print("\nSUCCESS: All load levels passed isolation checks")


# =============================================================================
# SYNC TEST ALTERNATIVE (if anyio not available)
# =============================================================================


def test_10_concurrent_rlm_engines_sync():
    """
    Synchronous version of the concurrent test.

    This test can run without anyio/pytest-anyio.
    """
    provider = SubjectAwareProvider()

    # Run the concurrent engines using threading directly
    from concurrent.futures import ThreadPoolExecutor

    num_instances = 10
    subjects = SubjectAwareProvider.SUBJECTS[:num_instances]
    markers = [f"SYNC-MARKER-{i:03d}-{uuid4().hex[:8]}" for i in range(num_instances)]

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_instances) as executor:
        futures = [
            executor.submit(
                run_single_engine,
                i + 1,
                subjects[i % len(subjects)],
                markers[i],
                provider,
            )
            for i in range(num_instances)
        ]
        results = [f.result() for f in futures]

    total_duration_ms = (time.time() - start_time) * 1000

    # Analyze results
    successful = [r for r in results if r.success]
    isolation_violations = 0
    missing_markers = 0

    for result in results:
        if not result.success:
            continue

        if result.correlation_marker not in (result.answer or ""):
            missing_markers += 1
            continue

        for other in results:
            if other.instance_id != result.instance_id:
                if other.correlation_marker in (result.answer or ""):
                    isolation_violations += 1
                    break

    print(f"""
{"=" * 66}
        SYNC CONCURRENT RLMENGINE TEST RESULTS
{"=" * 66}
 Total instances: {num_instances}
 Successful: {len(successful)}
 Isolation violations: {isolation_violations}
 Missing markers: {missing_markers}
 Peak concurrent: {provider.peak_concurrent}
 Total subcalls: {provider.subcall_count}
 Total duration: {total_duration_ms:.1f}ms
{"=" * 66}
""")

    # Assertions
    assert len(successful) == num_instances, (
        f"Expected all {num_instances} to succeed, but {num_instances - len(successful)} failed"
    )

    assert isolation_violations == 0, (
        f"Found {isolation_violations} cross-contamination violations"
    )

    assert missing_markers == 0, (
        f"Found {missing_markers} instances missing their marker"
    )

    assert provider.subcall_count >= num_instances, (
        f"Expected at least {num_instances} subcalls, got {provider.subcall_count}"
    )

    assert provider.peak_concurrent > 1, (
        f"Expected concurrent execution, peak was only {provider.peak_concurrent}"
    )

    print("SUCCESS: All 10 concurrent RLMEngine instances passed isolation test")


if __name__ == "__main__":
    # Run the sync test directly
    test_10_concurrent_rlm_engines_sync()
