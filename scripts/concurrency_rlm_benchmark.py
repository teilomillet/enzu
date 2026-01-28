#!/usr/bin/env python3
"""
RLM Concurrency Benchmark: Full Pipeline at Scale

This benchmark tests the FULL enzu pipeline with:
- RLMEngine (not simple Engine)
- Python sandbox execution
- Sub-agent spawning (llm_query, llm_batch)
- Context building and accumulation
- Artifact persistence

Usage:
    # Run with mock (no API costs)
    python scripts/concurrency_rlm_benchmark.py --scale 5

    # Run with real API
    python scripts/concurrency_rlm_benchmark.py --scale 5 --real-api

    # With research tools (requires EXA_API_KEY)
    python scripts/concurrency_rlm_benchmark.py --scale 3 --real-api --with-research

"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple
from uuid import uuid4

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from enzu.models import TaskSpec, Budget, SuccessCriteria, RLMExecutionReport, ProviderResult
from enzu.providers.base import BaseProvider
from enzu.providers.pool import (
    set_capacity_limit,
    close_all_providers,
)
from enzu.isolation.concurrency import (
    configure_global_limiter,
    reset_global_limiter,
)
from enzu.rlm import RLMEngine
from enzu.api import resolve_provider


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RLMInstanceResult:
    """Result from a single RLM pipeline instance."""
    instance_id: int
    correlation_marker: str
    success: bool
    answer: Optional[str]
    input_data: str
    start_time: float
    end_time: float
    duration_ms: float
    steps_executed: int
    sub_calls_made: int
    context_sources: int
    artifact_path: Optional[str]
    error: Optional[str] = None
    raw_report: Optional[Dict[str, Any]] = None


@dataclass
class RLMConcurrencyMetrics:
    """Metrics collected during RLM benchmark run."""
    total_duration_ms: float
    instance_durations_ms: List[float]
    total_instances: int
    successful_instances: int
    failed_instances: int
    total_steps: int
    total_sub_calls: int
    total_context_sources: int
    peak_concurrent: int
    api_errors: int

    def success_rate(self) -> float:
        if self.total_instances == 0:
            return 0.0
        return self.successful_instances / self.total_instances


# =============================================================================
# MOCK PROVIDER FOR RLM TESTING
# =============================================================================

class RLMMockProvider(BaseProvider):
    """
    Mock provider that simulates RLM-style responses with code generation.

    Returns Python code that the RLM sandbox will execute, including
    calls to llm_query() for sub-agent spawning.
    """
    name = "rlm_mock"

    def __init__(self, latency_ms: Tuple[int, int] = (100, 300)):
        self.latency_range = latency_ms
        self._calls: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._call_count = 0

    def stream(self, task: TaskSpec, on_progress=None) -> ProviderResult:
        from enzu.models import ProviderResult
        import random

        with self._lock:
            self._call_count += 1
            call_num = self._call_count

        # Simulate latency
        latency = random.uniform(
            self.latency_range[0] / 1000,
            self.latency_range[1] / 1000,
        )
        time.sleep(latency)

        # Extract context from task
        input_text = task.input_text or ""
        metadata = task.metadata or {}

        # Check if this is a sub-call (llm_query)
        is_subcall = "SUBCALL:" in input_text or metadata.get("is_subcall")

        if is_subcall:
            # Sub-call response - return analysis text
            output = "Analysis complete. The data shows positive trends with key metrics improving. Recommendation: continue monitoring."
        else:
            # Main RLM call - return Python code
            correlation_marker = metadata.get("correlation_marker", "UNKNOWN")
            instance_id = metadata.get("instance_id", 0)
            dataset = metadata.get("dataset", "unknown")

            if call_num == 1:
                # First step: gather data via sub-calls
                output = f'''```python
# Step 1: Analyze data using sub-agents
print("Starting analysis for instance {instance_id}")

# Use sub-agent to analyze metrics
metrics_analysis = llm_query("SUBCALL: Analyze the metrics data and trends")
print(f"Metrics analysis: {{metrics_analysis[:100]}}")

# Use sub-agent to generate insights
insights = llm_query("SUBCALL: Generate 3 actionable insights based on the data")
print(f"Insights: {{insights[:100]}}")

# Store intermediate results
intermediate = {{
    "instance_id": {instance_id},
    "correlation_marker": "{correlation_marker}",
    "dataset": "{dataset}",
    "metrics_analysis": metrics_analysis,
    "insights": insights,
}}
print(f"Intermediate results stored")
```'''
            else:
                # Second step: generate final report
                output = f'''```python
# Step 2: Generate final report
report = f"""
**Executive Summary Report**

**Report ID:** {correlation_marker}
**Dataset:** {dataset}
**Instance:** {instance_id}

**Analysis:**
The data analysis reveals significant patterns. Key metrics show improvement
with a positive trajectory. Sub-agent analysis confirmed the trends.

**Key Findings:**
1. Performance metrics trending upward
2. Efficiency indicators stable
3. Growth potential identified

**Recommendation:**
Continue current strategy with minor optimizations.

*Generated via RLM pipeline with sub-agent delegation*
"""

FINAL(report)
```'''

        with self._lock:
            self._calls.append({
                "call_num": call_num,
                "is_subcall": is_subcall,
                "timestamp": time.time(),
            })

        return ProviderResult(
            output_text=output,
            raw={"mock": True},
            usage={"output_tokens": len(output.split()), "total_tokens": 200},
            provider=self.name,
            model=task.model,
        )

    def generate(self, task: TaskSpec) -> ProviderResult:
        return self.stream(task)

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count

    @property
    def subcall_count(self) -> int:
        with self._lock:
            return sum(1 for c in self._calls if c.get("is_subcall"))


# =============================================================================
# RLM PIPELINE INSTANCE
# =============================================================================

class RLMPipelineInstance:
    """
    Full RLM pipeline instance with context building and sub-agent spawning.

    This simulates a real "context builder -> report constructor" pipeline.
    """

    CONTEXT_TEMPLATES = [
        {"domain": "financial", "metrics": ["revenue", "profit", "growth_rate"]},
        {"domain": "technical", "metrics": ["latency", "throughput", "error_rate"]},
        {"domain": "operational", "metrics": ["efficiency", "cost", "quality"]},
    ]

    def __init__(self, instance_id: int, input_data: str, artifact_dir: str):
        self.instance_id = instance_id
        self.input_data = input_data
        self.artifact_dir = artifact_dir
        self._correlation_marker = f"RLM-{instance_id}-{uuid4().hex[:8]}"
        self._context_data = self._generate_context_data()

    def _generate_context_data(self) -> Dict[str, Any]:
        """Generate unique context data for this instance."""
        template = self.CONTEXT_TEMPLATES[self.instance_id % len(self.CONTEXT_TEMPLATES)]

        metrics = {}
        for metric in template["metrics"]:
            value = ((self.instance_id * 17 + hash(metric)) % 1000) / 10
            metrics[metric] = {"value": value, "trend": "increasing" if value > 50 else "stable"}

        return {
            "instance_id": self.instance_id,
            "report_id": self._correlation_marker,
            "domain": template["domain"],
            "dataset": self.input_data,
            "metrics": metrics,
        }

    async def execute(
        self,
        provider: BaseProvider,
        model: str = "mock-model",
        with_research: bool = False,
    ) -> RLMInstanceResult:
        """
        Execute full RLM pipeline.

        This runs:
        1. Context preparation
        2. RLMEngine execution (with Python sandbox)
        3. Sub-agent spawning (llm_query/llm_batch)
        4. Artifact saving
        """
        start_time = time.time()

        # Reset context store for this instance (thread-local would be better)
        # For now, we'll track context in the result
        context_sources = 0

        try:
            # Build context document
            context_doc = self._build_context_document()

            # Build task for RLMEngine
            task = TaskSpec(
                task_id=f"rlm-pipeline-{self.instance_id}-{uuid4().hex[:8]}",
                input_text=f"""Analyze the following data and generate an executive summary report.

You have access to llm_query() to delegate analysis tasks to sub-agents.

DATA TO ANALYZE:
{context_doc}

REQUIREMENTS:
1. Use llm_query() to analyze different aspects of the data
2. Combine the sub-agent results into a coherent report
3. The report MUST include: Report ID: {self._correlation_marker}
4. The report MUST reference: {self.input_data}
5. Call FINAL(report) with your final report

Generate Python code to accomplish this task.""",
                model=model,
                responses={},
                budget=Budget(max_tokens=12000, max_total_tokens=50000),
                success_criteria=SuccessCriteria(
                    required_substrings=[self._correlation_marker],
                    min_word_count=50,
                ),
                metadata={
                    "instance_id": self.instance_id,
                    "correlation_marker": self._correlation_marker,
                    "dataset": self.input_data,
                },
            )

            # Create RLMEngine with sub-agent support
            engine = RLMEngine(
                max_steps=4,
                recursive_subcalls=True,  # Enable llm_query to spawn sub-RLMs
                inject_search_tools=with_research,  # Enable research() if requested
                allowed_imports=["json", "re", "math"],
            )

            # Run in thread pool to not block async
            loop = asyncio.get_event_loop()
            report: RLMExecutionReport = await loop.run_in_executor(
                None,
                lambda: engine.run(task, provider, data=json.dumps(self._context_data)),
            )

            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Count steps and sub-calls from report
            steps_executed = len(report.steps) if report.steps else 0
            sub_calls_made = 0
            for step in (report.steps or []):
                # Count llm_query calls in code
                code = step.code or ""
                if "llm_query" in code:
                    sub_calls_made += code.count("llm_query")
                if "llm_batch" in code:
                    sub_calls_made += code.count("llm_batch")

            # Save artifact
            artifact_path = self._save_artifact(report, duration_ms)

            return RLMInstanceResult(
                instance_id=self.instance_id,
                correlation_marker=self._correlation_marker,
                success=report.success,
                answer=report.answer,
                input_data=self.input_data,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                steps_executed=steps_executed,
                sub_calls_made=sub_calls_made,
                context_sources=context_sources,
                artifact_path=artifact_path,
                error="; ".join(report.errors) if report.errors else None,
                raw_report=report.model_dump(mode="json") if report else None,
            )

        except Exception as e:
            end_time = time.time()
            return RLMInstanceResult(
                instance_id=self.instance_id,
                correlation_marker=self._correlation_marker,
                success=False,
                answer=None,
                input_data=self.input_data,
                start_time=start_time,
                end_time=end_time,
                duration_ms=(end_time - start_time) * 1000,
                steps_executed=0,
                sub_calls_made=0,
                context_sources=0,
                artifact_path=None,
                error=str(e),
            )

    def _build_context_document(self) -> str:
        """Build context document for analysis."""
        data = self._context_data

        lines = [
            f"=== {data['domain'].upper()} ANALYSIS CONTEXT ===",
            f"Report ID: {data['report_id']}",
            f"Dataset: {data['dataset']}",
            f"Domain: {data['domain']}",
            "",
            "METRICS:",
        ]

        for metric, info in data['metrics'].items():
            lines.append(f"  - {metric}: {info['value']:.1f} ({info['trend']})")

        lines.append("")
        lines.append(f"Raw JSON: {json.dumps(data['metrics'])}")

        return "\n".join(lines)

    def _save_artifact(self, report: RLMExecutionReport, duration_ms: float) -> str:
        """Save full artifact with report and metadata."""
        artifact_dir = Path(self.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        artifact = {
            "instance_id": self.instance_id,
            "correlation_marker": self._correlation_marker,
            "input_data": self.input_data,
            "context_data": self._context_data,
            "success": report.success,
            "answer": report.answer,
            "errors": report.errors,
            "duration_ms": duration_ms,
            "steps": [
                {
                    "step_index": s.step_index,
                    "code": s.code[:500] if s.code else None,  # Truncate for readability
                    "stdout": s.stdout[:500] if s.stdout else None,
                    "error": s.error,
                }
                for s in (report.steps or [])
            ],
            "budget_usage": report.budget_usage.model_dump() if report.budget_usage else None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        artifact_path = artifact_dir / f"rlm-instance-{self.instance_id:03d}.json"
        artifact_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))

        return str(artifact_path)

    @property
    def correlation_marker(self) -> str:
        return self._correlation_marker


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
    ) -> Tuple[List[RLMInstanceResult], RLMConcurrencyMetrics]:
        tasks: List[asyncio.Task] = []
        start = time.time()

        active_count = [0]
        peak_concurrent = [0]
        lock = threading.Lock()

        async def wrapped_task(instance_id: int, delay: float) -> RLMInstanceResult:
            elapsed = time.time() - start
            wait_time = max(0, delay - elapsed)
            if wait_time > 0:
                await asyncio.sleep(wait_time)

            with lock:
                active_count[0] += 1
                if active_count[0] > peak_concurrent[0]:
                    peak_concurrent[0] = active_count[0]

            try:
                return await task_factory(instance_id)
            finally:
                with lock:
                    active_count[0] -= 1

        for i, delay in enumerate(self.delays):
            task = asyncio.create_task(wrapped_task(i + 1, delay))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_duration = (time.time() - start) * 1000

        instance_results: List[RLMInstanceResult] = []
        api_errors = 0
        total_steps = 0
        total_sub_calls = 0
        total_context_sources = 0

        for i, r in enumerate(results):
            if isinstance(r, Exception):
                api_errors += 1
                instance_results.append(RLMInstanceResult(
                    instance_id=i + 1,
                    correlation_marker="",
                    success=False,
                    answer=None,
                    input_data="",
                    start_time=0,
                    end_time=0,
                    duration_ms=0,
                    steps_executed=0,
                    sub_calls_made=0,
                    context_sources=0,
                    artifact_path=None,
                    error=str(r),
                ))
            else:
                assert isinstance(r, RLMInstanceResult)
                instance_results.append(r)
                total_steps += r.steps_executed
                total_sub_calls += r.sub_calls_made
                total_context_sources += r.context_sources

        successful = [r for r in instance_results if r.success]
        failed = [r for r in instance_results if not r.success]

        metrics = RLMConcurrencyMetrics(
            total_duration_ms=total_duration,
            instance_durations_ms=[r.duration_ms for r in instance_results],
            total_instances=len(instance_results),
            successful_instances=len(successful),
            failed_instances=len(failed),
            total_steps=total_steps,
            total_sub_calls=total_sub_calls,
            total_context_sources=total_context_sources,
            peak_concurrent=peak_concurrent[0],
            api_errors=api_errors,
        )

        return instance_results, metrics


# =============================================================================
# REPORTING
# =============================================================================

def print_report(metrics: RLMConcurrencyMetrics, results: List[RLMInstanceResult], args):
    """Print detailed RLM benchmark report."""
    print()
    print("=" * 70)
    print("          RLM CONCURRENCY BENCHMARK REPORT")
    print("=" * 70)
    print(f" Scale: {args.scale} instances")
    print(f" Mode: {'Real API' if args.real_api else 'Mock'}")
    print(f" Research: {'Enabled' if args.with_research else 'Disabled'}")
    if args.real_api:
        print(f" Model: {args.model}")
        print(f" Provider: {args.provider}")
    print("=" * 70)
    print(" RLM EXECUTION DETAILS")
    print("-" * 70)
    print(f" Total RLM steps executed: {metrics.total_steps}")
    print(f" Total sub-agent calls: {metrics.total_sub_calls}")
    print(f" Total context sources: {metrics.total_context_sources}")
    print(f" Peak concurrent pipelines: {metrics.peak_concurrent}")
    print("=" * 70)
    print(" INSTANCE RESULTS")
    print("-" * 70)

    for result in results:
        status = "PASS" if result.success else "FAIL"
        steps_info = f"steps={result.steps_executed}, subcalls={result.sub_calls_made}"

        if result.success:
            marker_ok = result.correlation_marker in (result.answer or "")
            isolation = "ISOLATED" if marker_ok else "MARKER_MISSING"
            print(f" Instance {result.instance_id}: {status} ({result.duration_ms:.0f}ms) - {steps_info} - {isolation}")
        else:
            error_short = result.error[:40] + "..." if result.error and len(result.error) > 40 else result.error
            print(f" Instance {result.instance_id}: {status} ({result.duration_ms:.0f}ms) - {error_short}")

    if args.verbose:
        print("=" * 70)
        print(" SAMPLE OUTPUTS")
        print("-" * 70)
        for result in results[:2]:
            if result.answer:
                print(f"\n Instance {result.instance_id} answer ({len(result.answer)} chars):")
                for line in result.answer.split('\n')[:15]:
                    print(f"   {line}")
                if len(result.answer.split('\n')) > 15:
                    print("   ...")

    print("=" * 70)
    print(" SUMMARY")
    print("-" * 70)

    status = "ALL CHECKS PASSED" if metrics.success_rate() >= 0.95 else "CHECKS FAILED"
    print(f" Status: {status}")
    print(f" Success rate: {metrics.success_rate():.1%}")
    print(f" Total duration: {metrics.total_duration_ms:.0f}ms")
    print(f" Avg instance duration: {sum(metrics.instance_durations_ms) / len(metrics.instance_durations_ms) if metrics.instance_durations_ms else 0:.0f}ms")
    print(f" API errors: {metrics.api_errors}")
    print(f" Artifacts saved to: {args.artifact_dir}")
    print("=" * 70)
    print()


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def get_stagger_delays(scale: int) -> List[float]:
    """Get stagger delays based on scale."""
    if scale <= 5:
        return [0, 2, 3, 5, 7][:scale]
    elif scale <= 10:
        return [float(i) for i in range(scale)]
    else:
        return [i * 0.5 for i in range(scale)]


async def run_benchmark(args):
    """Run the RLM benchmark."""
    scale = args.scale
    delays = get_stagger_delays(scale)

    print("\nStarting RLM concurrency benchmark...")
    print(f"  Scale: {scale} instances")
    print(f"  Mode: {'Real API' if args.real_api else 'Mock provider'}")
    print(f"  Research: {'Enabled' if args.with_research else 'Disabled'}")
    if args.real_api:
        print(f"  Provider: {args.provider}")
        print(f"  Model: {args.model}")

    # Configure concurrency
    max_concurrent = max(50, scale * 3)
    reset_global_limiter()
    configure_global_limiter(max_concurrent=max_concurrent, force_reconfigure=True)
    set_capacity_limit(max_concurrent)

    # Create provider
    if args.real_api:
        provider = resolve_provider(args.provider, use_pool=True)
    else:
        provider = RLMMockProvider(latency_ms=(100, 300))

    scheduler = StaggeredScheduler(delays=delays)

    # Create artifact directory
    artifact_dir = args.artifact_dir
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)

    # Create instances
    instances = [
        RLMPipelineInstance(i + 1, f"rlm-dataset-[{i + 1:03d}]", artifact_dir)
        for i in range(scale)
    ]

    async def run_instance(instance_id: int) -> RLMInstanceResult:
        instance = instances[instance_id - 1]
        return await instance.execute(
            provider,
            model=args.model,
            with_research=args.with_research,
        )

    # Run benchmark
    print(f"\nRunning {scale} RLM pipeline instances...")
    results, metrics = await scheduler.run(run_instance)

    # Save summary
    summary = {
        "benchmark_type": "rlm_pipeline",
        "config": {
            "scale": args.scale,
            "real_api": args.real_api,
            "with_research": args.with_research,
            "model": args.model if args.real_api else "mock",
            "provider": args.provider if args.real_api else "mock",
        },
        "metrics": {
            "total_instances": metrics.total_instances,
            "successful_instances": metrics.successful_instances,
            "failed_instances": metrics.failed_instances,
            "success_rate": metrics.success_rate(),
            "total_rlm_steps": metrics.total_steps,
            "total_sub_calls": metrics.total_sub_calls,
            "peak_concurrent": metrics.peak_concurrent,
            "total_duration_ms": metrics.total_duration_ms,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    summary_path = Path(artifact_dir) / "rlm-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Print report
    print_report(metrics, results, args)

    # Cleanup
    close_all_providers()

    return 0 if metrics.success_rate() >= 0.95 else 1


def main():
    parser = argparse.ArgumentParser(
        description="RLM Concurrency Benchmark - Full Pipeline Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--scale", type=int, default=5,
        help="Number of concurrent RLM pipeline instances (default: 5)"
    )

    parser.add_argument(
        "--real-api", action="store_true",
        help="Use real API instead of mock provider"
    )

    parser.add_argument(
        "--provider", type=str, default="openrouter",
        help="Provider name when using real API (default: openrouter)"
    )

    parser.add_argument(
        "--model", type=str, default="openai/gpt-4o-mini",
        help="Model name (default: openai/gpt-4o-mini)"
    )

    parser.add_argument(
        "--with-research", action="store_true",
        help="Enable research tools (requires EXA_API_KEY)"
    )

    parser.add_argument(
        "--artifact-dir", type=str, default="artifacts/rlm-concurrency-test",
        help="Directory to save artifacts"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed output"
    )

    args = parser.parse_args()
    args.scale = max(1, min(args.scale, 20))  # Limit to 20 for RLM (more expensive)

    exit_code = asyncio.run(run_benchmark(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
