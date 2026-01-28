#!/usr/bin/env python3
"""
Concurrency Benchmark: Report Generation at Scale

Standalone benchmark script for validating concurrent report generation.
Supports real API integration and configurable scale.

Usage:
    # Run with mock provider (default)
    python scripts/concurrency_benchmark.py

    # Run with 10 instances
    python scripts/concurrency_benchmark.py --scale 10

    # Run with 20 instances and real API
    python scripts/concurrency_benchmark.py --scale 20 --real-api

    # Custom delays
    python scripts/concurrency_benchmark.py --delays 0,1,2,3,4

    # Full options
    python scripts/concurrency_benchmark.py --scale 10 --real-api --model gpt-4o-mini --provider openrouter

"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple
from uuid import uuid4

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from enzu.models import ProviderResult, TaskSpec, Budget, SuccessCriteria
from enzu.providers.base import BaseProvider
from enzu.providers.pool import (
    set_capacity_limit,
    get_capacity_stats,
    close_all_providers,
)
from enzu.isolation.concurrency import (
    configure_global_limiter,
    reset_global_limiter,
)
from enzu.engine import Engine
from enzu.api import resolve_provider


# =============================================================================
# DATA CLASSES
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
        if not self.success or not self.output_text:
            return False
        return self.correlation_marker in self.output_text


@dataclass
class ConcurrencyMetrics:
    """Metrics collected during benchmark run."""
    total_duration_ms: float
    instance_durations_ms: List[float]
    overlap_windows: List[Tuple[float, float]]
    isolation_violations: int
    state_corruptions: int
    peak_concurrent_requests: int
    connection_pool_size: int
    api_calls_total: int
    api_errors: int
    total_instances: int
    successful_instances: int
    failed_instances: int

    def success_rate(self) -> float:
        if self.total_instances == 0:
            return 0.0
        return self.successful_instances / self.total_instances


# =============================================================================
# MOCK PROVIDER
# =============================================================================

class BenchmarkMockProvider(BaseProvider):
    """Mock provider for benchmark testing without real API."""
    name = "benchmark_mock"

    def __init__(self, latency_ms: Tuple[int, int] = (100, 300)):
        self.latency_range = latency_ms
        self._calls: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._active_count = 0
        self._peak_concurrent = 0

    def stream(self, task: TaskSpec, on_progress=None) -> ProviderResult:
        import random

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

            correlation_marker = task.metadata.get("correlation_marker", "")
            input_data = task.metadata.get("input_data", "")
            instance_id = task.metadata.get("instance_id", -1)

            output_text = (
                f"Report generated for instance {instance_id}. "
                f"Correlation: {correlation_marker}. "
                f"Input data processed: {input_data}. "
                f"Task ID: {task.task_id}."
            )

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
    def peak_concurrent(self) -> int:
        with self._lock:
            return self._peak_concurrent


# =============================================================================
# STAGGERED SCHEDULER
# =============================================================================

@dataclass
class StaggeredScheduler:
    """Ray-inspired staggered task submission."""
    delays: List[float] = field(default_factory=lambda: [0, 2, 3, 5, 7])

    async def run(
        self,
        task_factory: Callable[[int], Coroutine],
    ) -> Tuple[List[InstanceResult], ConcurrencyMetrics]:
        tasks: List[asyncio.Task] = []
        start = time.time()

        active_count = [0]
        peak_concurrent = [0]
        overlap_windows: List[Tuple[float, float]] = []
        lock = threading.Lock()

        async def wrapped_task(instance_id: int, delay: float) -> InstanceResult:
            elapsed = time.time() - start
            wait_time = max(0, delay - elapsed)
            if wait_time > 0:
                await asyncio.sleep(wait_time)

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
                    if overlap_windows and overlap_windows[-1][1] == -1:
                        overlap_windows[-1] = (overlap_windows[-1][0], time.time() - start)

        for i, delay in enumerate(self.delays):
            task = asyncio.create_task(wrapped_task(i + 1, delay))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_duration = (time.time() - start) * 1000

        instance_results: List[InstanceResult] = []
        api_errors = 0

        for i, r in enumerate(results):
            if isinstance(r, Exception):
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
                assert isinstance(r, InstanceResult)
                instance_results.append(r)

        successful = [r for r in instance_results if r.success]
        failed = [r for r in instance_results if not r.success]

        metrics = ConcurrencyMetrics(
            total_duration_ms=total_duration,
            instance_durations_ms=[r.duration_ms for r in instance_results],
            overlap_windows=overlap_windows,
            isolation_violations=0,
            state_corruptions=0,
            peak_concurrent_requests=peak_concurrent[0],
            connection_pool_size=1,
            api_calls_total=len(instance_results),
            api_errors=api_errors,
            total_instances=len(instance_results),
            successful_instances=len(successful),
            failed_instances=len(failed),
        )

        return instance_results, metrics


# =============================================================================
# REPORT INSTANCE
# =============================================================================

class ReportInstance:
    """
    Isolated report generation instance with realistic context.

    Each instance gets unique context data that simulates a real document
    the model must analyze. This tests actual LLM processing, not just
    marker echoing.
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
        start_time = time.time()

        try:
            # Build realistic context for analysis
            context = self._build_context()

            # Create a task that requires actual analysis, not just marker echoing
            task = TaskSpec(
                task_id=f"report-{self.instance_id}-{uuid4().hex[:8]}",
                input_text=f"""Analyze the following data and generate a brief executive summary report.

DATA TO ANALYZE:
{context}

REQUIREMENTS:
1. Start with: "Report ID: {self._correlation_marker}"
2. Include the dataset name: {self.input_data}
3. Summarize the key metrics and their trends
4. Provide 2-3 actionable insights based on the data
5. End with a recommendation

Keep the report concise (150-300 words).""",
                model=model,
                responses={},
                budget=Budget(max_tokens=600),
                success_criteria=SuccessCriteria(min_word_count=50),
                metadata={
                    "instance_id": self.instance_id,
                    "correlation_marker": self._correlation_marker,
                    "input_data": self.input_data,
                },
            )

            engine = Engine()
            loop = asyncio.get_event_loop()
            report = await loop.run_in_executor(
                None,
                lambda: engine.run(task, provider),
            )

            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            return InstanceResult(
                instance_id=self.instance_id,
                correlation_marker=self._correlation_marker,
                success=report.success,
                output_text=report.output_text or "",
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
        import json
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
# BENCHMARK RUNNER
# =============================================================================

def get_stagger_delays(scale: int, custom_delays: Optional[str] = None) -> List[float]:
    """Get stagger delays based on scale or custom input."""
    if custom_delays:
        return [float(d.strip()) for d in custom_delays.split(",")]

    if scale <= 5:
        return [0, 2, 3, 5, 7][:scale]
    elif scale <= 10:
        return [float(i) for i in range(scale)]
    else:
        return [i * 0.5 for i in range(scale)]


def print_report(metrics: ConcurrencyMetrics, results: List[InstanceResult], args):
    """Print detailed benchmark report."""
    print()
    print("=" * 70)
    print("              CONCURRENCY BENCHMARK REPORT")
    print("=" * 70)
    print(f" Schedule: {get_stagger_delays(args.scale, args.delays)[:5]}{'...' if args.scale > 5 else ''}")
    print(f" Total instances: {metrics.total_instances}")
    print(f" Peak concurrent: {metrics.peak_concurrent_requests}")
    print(f" Provider: {'Real API' if args.real_api else 'Mock'}")
    if args.real_api:
        print(f" Model: {args.model}")
        print(f" Provider: {args.provider}")
    print("=" * 70)
    print(" RESULTS")
    print("-" * 70)

    for result in results:
        status = "PASS" if result.success else "FAIL"
        isolation = "OK" if result.isolation_valid else "VIOLATION"
        error = f" - {result.error[:40]}..." if result.error and len(result.error) > 40 else (f" - {result.error}" if result.error else "")

        if result.success:
            print(f" Instance {result.instance_id}: {status} ({result.duration_ms:.1f}ms) - Isolation {isolation}")
        else:
            print(f" Instance {result.instance_id}: {status} ({result.duration_ms:.1f}ms){error}")

    # Show verbose output if requested
    if args.verbose and results:
        print("=" * 70)
        print(" SAMPLE OUTPUTS")
        print("-" * 70)
        for result in results[:3]:  # Show first 3
            if result.output_text:
                output_preview = result.output_text[:500] + "..." if len(result.output_text) > 500 else result.output_text
                print(f"\n Instance {result.instance_id} output:")
                print(f"   Input data: {result.input_data}")
                print(f"   Marker: {result.correlation_marker}")
                print(f"   Output ({len(result.output_text)} chars):")
                for line in output_preview.split('\n')[:10]:
                    print(f"     {line}")
                if len(result.output_text) > 500 or len(output_preview.split('\n')) > 10:
                    print("     ...")

    print("=" * 70)
    print(" SUMMARY")
    print("-" * 70)

    if metrics.success_rate() >= 0.95 and metrics.isolation_violations == 0:
        status = "ALL CHECKS PASSED"
    else:
        status = "CHECKS FAILED"

    print(f" Status: {status}")
    print(f" Success rate: {metrics.success_rate():.1%}")
    print(f" Isolation violations: {metrics.isolation_violations}")
    print(f" State corruptions: {metrics.state_corruptions}")
    print(f" Total duration: {metrics.total_duration_ms:.1f}ms")
    print(f" Avg instance duration: {sum(metrics.instance_durations_ms) / len(metrics.instance_durations_ms) if metrics.instance_durations_ms else 0:.1f}ms")
    print(f" API errors: {metrics.api_errors}")
    print("=" * 70)
    print()


async def run_benchmark(args):
    """Run the benchmark with specified configuration."""
    scale = args.scale
    delays = get_stagger_delays(scale, args.delays)

    print("\nStarting concurrency benchmark...")
    print(f"  Scale: {scale} instances")
    print(f"  Mode: {'Real API' if args.real_api else 'Mock provider'}")
    if args.real_api:
        print(f"  Provider: {args.provider}")
        print(f"  Model: {args.model}")

    # Configure concurrency limits
    max_concurrent = max(50, scale * 2)
    reset_global_limiter()
    configure_global_limiter(max_concurrent=max_concurrent, force_reconfigure=True)
    set_capacity_limit(max_concurrent)

    # Create provider
    if args.real_api:
        provider = resolve_provider(
            args.provider,
            use_pool=True,
        )
    else:
        provider = BenchmarkMockProvider(latency_ms=(100, 300))

    scheduler = StaggeredScheduler(delays=delays)

    # Create instances
    # Use zero-padded format to prevent substring collisions (e.g., data-1 in data-10)
    instances = [
        ReportInstance(i + 1, f"benchmark-dataset-[{i + 1:03d}]")
        for i in range(scale)
    ]

    async def run_instance(instance_id: int) -> InstanceResult:
        instance = instances[instance_id - 1]
        return await instance.execute(provider, model=args.model)

    # Run benchmark
    print(f"\nRunning {scale} instances with staggered delays...")
    results, metrics = await scheduler.run(run_instance)

    # Check isolation and state
    isolation_violations = 0
    state_corruptions = 0

    for result in results:
        if not result.success:
            continue

        output = result.output_text or ""

        # For mock provider, check markers in output
        # For real API, we can't guarantee the model echoes markers
        if not args.real_api:
            if result.correlation_marker not in output:
                isolation_violations += 1

            if result.input_data not in output:
                state_corruptions += 1

            for other in results:
                if other.instance_id != result.instance_id:
                    if other.correlation_marker in output:
                        isolation_violations += 1
                    if other.input_data in output:
                        state_corruptions += 1

    metrics.isolation_violations = isolation_violations
    metrics.state_corruptions = state_corruptions
    metrics.connection_pool_size = get_capacity_stats()["providers_count"]

    # Print report
    print_report(metrics, results, args)

    # Save artifacts if requested
    if args.save_artifacts:
        from pathlib import Path
        import json as json_module

        artifact_dir = Path(args.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving artifacts to {artifact_dir}/...")

        # Save each instance result as a separate artifact
        for result in results:
            artifact = {
                "instance_id": result.instance_id,
                "correlation_marker": result.correlation_marker,
                "input_data": result.input_data,
                "success": result.success,
                "duration_ms": result.duration_ms,
                "error": result.error,
                "output_text": result.output_text,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": args.model if args.real_api else "mock",
                "provider": args.provider if args.real_api else "mock",
            }

            artifact_path = artifact_dir / f"instance-{result.instance_id:03d}.json"
            artifact_path.write_text(json_module.dumps(artifact, indent=2, ensure_ascii=False))

        # Save summary
        summary = {
            "benchmark_config": {
                "scale": args.scale,
                "real_api": args.real_api,
                "model": args.model if args.real_api else "mock",
                "provider": args.provider if args.real_api else "mock",
            },
            "metrics": {
                "total_instances": metrics.total_instances,
                "successful_instances": metrics.successful_instances,
                "failed_instances": metrics.failed_instances,
                "success_rate": metrics.success_rate(),
                "isolation_violations": metrics.isolation_violations,
                "state_corruptions": metrics.state_corruptions,
                "peak_concurrent": metrics.peak_concurrent_requests,
                "total_duration_ms": metrics.total_duration_ms,
                "avg_duration_ms": sum(metrics.instance_durations_ms) / len(metrics.instance_durations_ms) if metrics.instance_durations_ms else 0,
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        summary_path = artifact_dir / "summary.json"
        summary_path.write_text(json_module.dumps(summary, indent=2))

        print(f"  Saved {len(results)} instance artifacts + summary.json")

    # Cleanup
    close_all_providers()

    # Return exit code
    if metrics.success_rate() >= 0.95 and metrics.isolation_violations == 0:
        return 0
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Concurrency Benchmark for Report Generation at Scale",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with mock provider (default, 5 instances)
    python scripts/concurrency_benchmark.py

    # Run with 10 instances
    python scripts/concurrency_benchmark.py --scale 10

    # Run with 20 instances
    python scripts/concurrency_benchmark.py --scale 20

    # Run with real API
    python scripts/concurrency_benchmark.py --real-api --model gpt-4o-mini

    # Custom delays (seconds)
    python scripts/concurrency_benchmark.py --delays 0,1,2,3,4

        """
    )

    parser.add_argument(
        "--scale",
        type=int,
        default=5,
        help="Number of concurrent instances (default: 5, supports 1-100)"
    )

    parser.add_argument(
        "--delays",
        type=str,
        default=None,
        help="Custom stagger delays in seconds, comma-separated (e.g., '0,2,3,5,7')"
    )

    parser.add_argument(
        "--real-api",
        action="store_true",
        help="Use real API instead of mock provider"
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="openrouter",
        help="Provider name when using real API (default: openrouter)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-mini",
        help="Model name when using real API (default: openai/gpt-4o-mini)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show sample outputs from each instance"
    )

    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Save each report as a JSON artifact file"
    )

    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="artifacts/concurrency-test",
        help="Directory to save artifacts (default: artifacts/concurrency-test)"
    )

    args = parser.parse_args()

    # Validate scale
    args.scale = max(1, min(args.scale, 100))

    # Run benchmark
    exit_code = asyncio.run(run_benchmark(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
