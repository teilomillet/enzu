"""
Test RLM with Container Isolation and Recursive Sub-tasks

Validates that RLM works correctly with container isolation:
1. Basic RLM execution with recursive sub-tasks via IPC
2. Multiple concurrent instances with container isolation
3. Short and long running tasks
4. IPC communication for llm_query/llm_batch in containers

Uses OpenRouter API with model "z-ai/glm-4.7" from .env file.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from uuid import uuid4

import pytest
from dotenv import load_dotenv

from enzu import Budget, SuccessCriteria, TaskSpec
from enzu.api import _resolve_provider
from enzu.rlm.engine import RLMEngine
from enzu.isolation.container import is_container_available

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


# Configuration
MODEL = "z-ai/glm-4.7"
PROVIDER = "openrouter"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Skip tests if API key not available
# These are integration tests requiring live API access
pytestmark = [
    pytest.mark.skipif(
        not OPENROUTER_API_KEY,
        reason="OPENROUTER_API_KEY not set in environment",
    ),
    pytest.mark.integration,
]


@dataclass
class ContainerTestResult:
    """Result from a container isolation test."""
    test_name: str
    success: bool
    answer: Optional[str]
    duration_ms: float
    subcalls_made: int
    error: Optional[str] = None
    isolation_mode: Optional[str] = None


def get_provider():
    """Get OpenRouter provider instance."""
    return _resolve_provider(PROVIDER, api_key=OPENROUTER_API_KEY)


# =============================================================================
# Basic IPC Tests
# =============================================================================

def test_rlm_container_isolation_basic_recursive_subcall():
    """
    Test basic RLM execution with container isolation and recursive sub-task.
    
    Verifies:
    - Container isolation works
    - llm_query works via IPC
    - Recursive sub-tasks execute correctly
    """
    if not is_container_available():
        pytest.skip("Docker not available, skipping container test")
    
    provider = get_provider()
    
    task = TaskSpec(
        task_id=f"container-basic-{uuid4().hex[:8]}",
        input_text=(
            "You need to analyze a topic by delegating to a sub-agent. "
            "Use llm_query to ask a sub-agent: 'What are the key features of Python programming language?' "
            "Then return the sub-agent's response as your final answer."
        ),
        model=MODEL,
        budget=Budget(max_tokens=500, max_total_tokens=2000),
        success_criteria=SuccessCriteria(
            min_word_count=10,
            required_substrings=["Python"],
        ),
    )
    
    engine = RLMEngine(
        isolation="container",
        max_steps=5,
        recursive_subcalls=True,
        max_recursion_depth=1,
        subcall_max_steps=3,
    )
    
    start_time = time.time()
    report = engine.run(task, provider, data="Context about programming languages")
    duration_ms = (time.time() - start_time) * 1000
    
    # Count subcalls by checking steps
    subcalls = len([s for s in report.steps if "llm_query" in str(s).lower()])
    
    assert report.success, f"RLM failed: {report.errors}"
    assert report.answer is not None, "No answer returned"
    assert "Python" in report.answer, f"Answer should mention Python: {report.answer}"
    assert subcalls > 0, "Should have made at least one subcall via llm_query"
    
    print("✓ Container isolation test passed")
    print(f"  Duration: {duration_ms:.0f}ms")
    print(f"  Subcalls: {subcalls}")
    print(f"  Answer length: {len(report.answer)} chars")


def test_rlm_subprocess_isolation_basic_recursive_subcall():
    """
    Test basic RLM execution with subprocess isolation and recursive sub-task.
    
    Verifies:
    - Subprocess isolation works
    - llm_query works via IPC
    - Recursive sub-tasks execute correctly
    """
    provider = get_provider()
    
    task = TaskSpec(
        task_id=f"subprocess-basic-{uuid4().hex[:8]}",
        input_text=(
            "You need to analyze a topic by delegating to a sub-agent. "
            "Use llm_query to ask a sub-agent: 'What are the main benefits of using Docker containers?' "
            "Then return the sub-agent's response as your final answer."
        ),
        model=MODEL,
        # Budget must account for subcall token usage including research() calls
        # which can accumulate large context. Subcalls with search tools can use
        # 10k+ tokens per call. Set limits high enough to not constrain the test.
        budget=Budget(max_tokens=20000, max_total_tokens=100000),
        success_criteria=SuccessCriteria(
            min_word_count=10,
            required_substrings=["Docker"],
        ),
    )
    
    engine = RLMEngine(
        isolation="subprocess",
        max_steps=5,
        recursive_subcalls=True,
        max_recursion_depth=1,
        subcall_max_steps=3,
    )
    
    start_time = time.time()
    report = engine.run(task, provider, data="Context about containerization")
    duration_ms = (time.time() - start_time) * 1000
    
    # Count subcalls
    subcalls = len([s for s in report.steps if "llm_query" in str(s).lower()])
    
    assert report.success, f"RLM failed: {report.errors}"
    assert report.answer is not None, "No answer returned"
    assert "Docker" in report.answer, f"Answer should mention Docker: {report.answer}"
    assert subcalls > 0, "Should have made at least one subcall via llm_query"
    
    print("✓ Subprocess isolation test passed")
    print(f"  Duration: {duration_ms:.0f}ms")
    print(f"  Subcalls: {subcalls}")
    print(f"  Answer length: {len(report.answer)} chars")


def test_rlm_container_isolation_llm_batch():
    """
    Test llm_batch with container isolation.
    
    Verifies:
    - llm_batch works via IPC in containers
    - Multiple parallel sub-tasks execute correctly
    """
    if not is_container_available():
        pytest.skip("Docker not available, skipping container test")
    
    provider = get_provider()
    
    task = TaskSpec(
        task_id=f"container-batch-{uuid4().hex[:8]}",
        input_text=(
            "You need to analyze multiple topics in parallel. "
            "Use llm_batch to ask sub-agents about: "
            "1. 'What is machine learning?' "
            "2. 'What is artificial intelligence?' "
            "Then combine both responses into your final answer."
        ),
        model=MODEL,
        budget=Budget(max_tokens=500, max_total_tokens=2000),
        success_criteria=SuccessCriteria(
            min_word_count=20,
        ),
    )
    
    engine = RLMEngine(
        isolation="container",
        max_steps=5,
        recursive_subcalls=True,
        max_recursion_depth=1,
        subcall_max_steps=3,
    )
    
    start_time = time.time()
    report = engine.run(task, provider, data="Context about AI")
    duration_ms = (time.time() - start_time) * 1000
    
    assert report.success, f"RLM failed: {report.errors}"
    assert report.answer is not None, "No answer returned"
    assert len(report.answer) > 50, "Answer should be substantial"
    
    print("✓ Container llm_batch test passed")
    print(f"  Duration: {duration_ms:.0f}ms")
    print(f"  Answer length: {len(report.answer)} chars")


# =============================================================================
# Scaling Tests - Multiple Concurrent Instances
# =============================================================================

def run_single_container_engine(
    instance_id: int,
    topic: str,
    marker: str,
    isolation_mode: str = "container",
) -> ContainerTestResult:
    """Run a single RLMEngine with container/subprocess isolation."""
    if isolation_mode == "container" and not is_container_available():
        return ContainerTestResult(
            test_name=f"instance-{instance_id}",
            success=False,
            answer=None,
            duration_ms=0,
            subcalls_made=0,
            error="Docker not available",
            isolation_mode=isolation_mode,
        )
    
    provider = get_provider()
    
    task = TaskSpec(
        task_id=f"concurrent-{isolation_mode}-{instance_id}-{uuid4().hex[:8]}",
        input_text=(
            f"Analyze the topic: {topic}. "
            f"Use llm_query to ask a sub-agent: 'Research key points about {topic}'. "
            f"Include the marker '{marker}' in your final answer."
        ),
        model=MODEL,
        budget=Budget(max_tokens=2000, max_total_tokens=12000),
        success_criteria=SuccessCriteria(
            required_substrings=[marker],
            min_word_count=10,
        ),
    )
    
    engine = RLMEngine(
        isolation=isolation_mode,
        max_steps=4,
        recursive_subcalls=True,
        max_recursion_depth=1,
        subcall_max_steps=2,
    )
    
    start_time = time.time()
    try:
        report = engine.run(task, provider, data=f"Context about {topic}")
        duration_ms = (time.time() - start_time) * 1000
        
        subcalls = len([s for s in report.steps if "llm_query" in str(s).lower()])
        
        return ContainerTestResult(
            test_name=f"instance-{instance_id}",
            success=report.success,
            answer=report.answer,
            duration_ms=duration_ms,
            subcalls_made=subcalls,
            error="; ".join(report.errors) if report.errors else None,
            isolation_mode=isolation_mode,
        )
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return ContainerTestResult(
            test_name=f"instance-{instance_id}",
            success=False,
            answer=None,
            duration_ms=duration_ms,
            subcalls_made=0,
            error=str(e),
            isolation_mode=isolation_mode,
        )


def test_rlm_container_isolation_5_concurrent():
    """
    Test 5 concurrent RLM instances with container isolation.
    
    Verifies:
    - Multiple containers can run simultaneously
    - IPC works correctly under concurrency
    - No cross-contamination between instances
    """
    if not is_container_available():
        pytest.skip("Docker not available, skipping container test")
    
    topics = [
        "quantum computing",
        "blockchain technology",
        "renewable energy",
        "space exploration",
        "biotechnology",
    ]
    markers = [f"MARKER-{i:02d}-{uuid4().hex[:6]}" for i in range(5)]
    
    from concurrent.futures import ThreadPoolExecutor
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(
                run_single_container_engine,
                i + 1,
                topics[i],
                markers[i],
                "container",
            )
            for i in range(5)
        ]
        results = [f.result() for f in futures]
    
    total_duration_ms = (time.time() - start_time) * 1000
    
    # Analyze results
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    # Check isolation: each marker should only appear in its own answer
    isolation_violations = 0
    for i, result in enumerate(results):
        if not result.success or not result.answer:
            continue
        # Check own marker is present
        if markers[i] not in result.answer:
            isolation_violations += 1
        # Check no other markers are present
        for j, other_marker in enumerate(markers):
            if i != j and other_marker in result.answer:
                isolation_violations += 1
    
    print("\n✓ 5 Concurrent Container Tests:")
    print(f"  Successful: {len(successful)}/5")
    print(f"  Failed: {len(failed)}/5")
    print(f"  Total duration: {total_duration_ms:.0f}ms")
    print(f"  Isolation violations: {isolation_violations}")
    print(f"  Avg duration: {sum(r.duration_ms for r in successful) / len(successful) if successful else 0:.0f}ms")
    
    assert len(successful) >= 4, f"Expected at least 4 successes, got {len(successful)}"
    assert isolation_violations == 0, f"Found {isolation_violations} isolation violations"
    
    for result in failed:
        print(f"  Failed instance: {result.test_name} - {result.error}")


def test_rlm_subprocess_isolation_5_concurrent():
    """
    Test 5 concurrent RLM instances with subprocess isolation.
    
    Verifies:
    - Multiple subprocesses can run simultaneously
    - IPC works correctly under concurrency
    - No cross-contamination between instances
    """
    topics = [
        "machine learning",
        "cloud computing",
        "cybersecurity",
        "data science",
        "web development",
    ]
    markers = [f"MARKER-{i:02d}-{uuid4().hex[:6]}" for i in range(5)]
    
    from concurrent.futures import ThreadPoolExecutor
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(
                run_single_container_engine,
                i + 1,
                topics[i],
                markers[i],
                "subprocess",
            )
            for i in range(5)
        ]
        results = [f.result() for f in futures]
    
    total_duration_ms = (time.time() - start_time) * 1000
    
    # Analyze results
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    # Check isolation
    isolation_violations = 0
    for i, result in enumerate(results):
        if not result.success or not result.answer:
            continue
        if markers[i] not in result.answer:
            isolation_violations += 1
        for j, other_marker in enumerate(markers):
            if i != j and other_marker in result.answer:
                isolation_violations += 1
    
    print("\n✓ 5 Concurrent Subprocess Tests:")
    print(f"  Successful: {len(successful)}/5")
    print(f"  Failed: {len(failed)}/5")
    print(f"  Total duration: {total_duration_ms:.0f}ms")
    print(f"  Isolation violations: {isolation_violations}")
    
    assert len(successful) >= 4, f"Expected at least 4 successes, got {len(successful)}"
    assert isolation_violations == 0, f"Found {isolation_violations} isolation violations"


def test_rlm_container_isolation_10_concurrent_stress():
    """
    Stress test: 10 concurrent RLM instances with container isolation.
    
    Verifies:
    - System handles higher concurrency
    - IPC remains stable under load
    - No resource exhaustion
    """
    if not is_container_available():
        pytest.skip("Docker not available, skipping container test")
    
    topics = [
        "artificial intelligence",
        "quantum computing",
        "blockchain",
        "renewable energy",
        "space technology",
        "biotechnology",
        "nanotechnology",
        "robotics",
        "virtual reality",
        "augmented reality",
    ]
    markers = [f"MARKER-{i:02d}-{uuid4().hex[:6]}" for i in range(10)]
    
    from concurrent.futures import ThreadPoolExecutor
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(
                run_single_container_engine,
                i + 1,
                topics[i],
                markers[i],
                "container",
            )
            for i in range(10)
        ]
        results = [f.result() for f in futures]
    
    total_duration_ms = (time.time() - start_time) * 1000
    
    # Analyze results
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    # Check isolation
    isolation_violations = 0
    for i, result in enumerate(results):
        if not result.success or not result.answer:
            continue
        if markers[i] not in result.answer:
            isolation_violations += 1
        for j, other_marker in enumerate(markers):
            if i != j and other_marker in result.answer:
                isolation_violations += 1
    
    print("\n✓ 10 Concurrent Container Stress Test:")
    print(f"  Successful: {len(successful)}/10")
    print(f"  Failed: {len(failed)}/10")
    print(f"  Total duration: {total_duration_ms:.0f}ms")
    print(f"  Isolation violations: {isolation_violations}")
    if successful:
        avg_duration = sum(r.duration_ms for r in successful) / len(successful)
        print(f"  Avg duration: {avg_duration:.0f}ms")
    
    # Allow some failures in stress test (80% success rate)
    assert len(successful) >= 8, f"Expected at least 8 successes, got {len(successful)}"
    assert isolation_violations == 0, f"Found {isolation_violations} isolation violations"


# =============================================================================
# Long/Short Running Task Tests
# =============================================================================

def test_rlm_container_isolation_short_task():
    """
    Test short-running task with container isolation.
    
    Verifies:
    - Quick tasks complete successfully
    - IPC overhead is acceptable
    """
    if not is_container_available():
        pytest.skip("Docker not available, skipping container test")
    
    provider = get_provider()
    
    task = TaskSpec(
        task_id=f"container-short-{uuid4().hex[:8]}",
        input_text="What is 2+2? Just answer with the number.",
        model=MODEL,
        budget=Budget(max_tokens=50, max_total_tokens=200),
        success_criteria=SuccessCriteria(
            required_substrings=["4"],
        ),
    )
    
    engine = RLMEngine(
        isolation="container",
        max_steps=2,
        recursive_subcalls=False,  # No subcalls for short task
    )
    
    start_time = time.time()
    report = engine.run(task, provider, data="")
    duration_ms = (time.time() - start_time) * 1000
    
    assert report.success, f"RLM failed: {report.errors}"
    assert "4" in (report.answer or ""), f"Answer should contain '4': {report.answer}"
    assert duration_ms < 30000, f"Short task took too long: {duration_ms}ms"
    
    print("✓ Short task test passed")
    print(f"  Duration: {duration_ms:.0f}ms")
    print(f"  Answer: {report.answer}")


def test_rlm_container_isolation_long_task():
    """
    Test long-running task with container isolation and multiple sub-tasks.
    
    Verifies:
    - Long tasks complete successfully
    - Multiple recursive sub-tasks work
    - IPC remains stable over time
    """
    if not is_container_available():
        pytest.skip("Docker not available, skipping container test")
    
    provider = get_provider()
    
    task = TaskSpec(
        task_id=f"container-long-{uuid4().hex[:8]}",
        input_text=(
            "You need to research and analyze a complex topic. "
            "Break it down into 3 sub-tasks using llm_query: "
            "1. 'What are the main challenges in climate change?' "
            "2. 'What are potential solutions to climate change?' "
            "3. 'What is the economic impact of climate change solutions?' "
            "Then synthesize all three responses into a comprehensive final answer."
        ),
        model=MODEL,
        budget=Budget(max_tokens=800, max_total_tokens=3000),
        success_criteria=SuccessCriteria(
            min_word_count=50,
        ),
    )
    
    engine = RLMEngine(
        isolation="container",
        max_steps=8,
        recursive_subcalls=True,
        max_recursion_depth=1,
        subcall_max_steps=3,
    )
    
    start_time = time.time()
    report = engine.run(task, provider, data="Context about climate change")
    duration_ms = (time.time() - start_time) * 1000
    
    subcalls = len([s for s in report.steps if "llm_query" in str(s).lower()])
    
    assert report.success, f"RLM failed: {report.errors}"
    assert report.answer is not None, "No answer returned"
    assert len(report.answer) > 100, f"Answer should be substantial: {len(report.answer)} chars"
    assert subcalls >= 3, f"Should have made at least 3 subcalls, got {subcalls}"
    
    print("✓ Long task test passed")
    print(f"  Duration: {duration_ms:.0f}ms")
    print(f"  Subcalls: {subcalls}")
    print(f"  Answer length: {len(report.answer)} chars")


@pytest.mark.skip(reason="Long-running integration test - run manually with -m integration")
def test_rlm_subprocess_isolation_long_task():
    """
    Test long-running task with subprocess isolation and multiple sub-tasks.

    Verifies subprocess isolation handles long tasks correctly.
    """
    provider = get_provider()
    
    task = TaskSpec(
        task_id=f"subprocess-long-{uuid4().hex[:8]}",
        input_text=(
            "You need to research and analyze a complex topic. "
            "Break it down into 3 sub-tasks using llm_query: "
            "1. 'What are the benefits of microservices architecture?' "
            "2. 'What are the challenges of microservices?' "
            "3. 'When should you use microservices vs monoliths?' "
            "Then synthesize all three responses into a comprehensive final answer."
        ),
        model=MODEL,
        # Budget must account for subcall token usage including research() calls
        # which can accumulate large context. Subcalls with search tools can use
        # 10k+ tokens per call. Set limits high enough to not constrain the test.
        # Long task with 3 subcalls needs substantial budget.
        budget=Budget(max_tokens=20000, max_total_tokens=100000),
        success_criteria=SuccessCriteria(
            min_word_count=50,
        ),
    )
    
    engine = RLMEngine(
        isolation="subprocess",
        max_steps=8,
        recursive_subcalls=True,
        max_recursion_depth=1,
        subcall_max_steps=3,
    )
    
    start_time = time.time()
    report = engine.run(task, provider, data="Context about software architecture")
    duration_ms = (time.time() - start_time) * 1000
    
    subcalls = len([s for s in report.steps if "llm_query" in str(s).lower()])
    
    assert report.success, f"RLM failed: {report.errors}"
    assert report.answer is not None, "No answer returned"
    assert len(report.answer) > 100, f"Answer should be substantial: {len(report.answer)} chars"
    assert subcalls >= 3, f"Should have made at least 3 subcalls, got {subcalls}"
    
    print("✓ Subprocess long task test passed")
    print(f"  Duration: {duration_ms:.0f}ms")
    print(f"  Subcalls: {subcalls}")
    print(f"  Answer length: {len(report.answer)} chars")
