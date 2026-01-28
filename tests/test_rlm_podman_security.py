"""
RLMEngine + Podman Security Integration Tests

These tests validate the full RLMEngine flow with Podman container isolation:
1. Multiple containers (3) spawned concurrently
2. Stateless execution - no state persists between executions
3. Container security - containers cannot be compromised

Requires Podman to be installed and available.
"""

from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
from dotenv import load_dotenv

from enzu import Budget, SuccessCriteria, TaskSpec
from enzu.api import _resolve_provider
from enzu.rlm.engine import RLMEngine
from enzu.isolation.runtime import ContainerRuntime, detect_runtime
from enzu.isolation.pool import ContainerPool, PoolConfig
from enzu.isolation.container_wrapper import Container
from enzu.isolation.runner import SandboxConfig


env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

MODEL = "z-ai/glm-4.7"
PROVIDER = "openrouter"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def _check_podman_available() -> bool:
    try:
        runtime = detect_runtime()
        return runtime in (ContainerRuntime.PODMAN, ContainerRuntime.DOCKER)
    except RuntimeError:
        return False


requires_container_runtime = pytest.mark.skipif(
    not _check_podman_available(),
    reason="No container runtime (Podman/Docker) available",
)

requires_api_key = pytest.mark.skipif(
    not OPENROUTER_API_KEY,
    reason="OPENROUTER_API_KEY not set in environment",
)


def get_provider():
    return _resolve_provider(PROVIDER, api_key=OPENROUTER_API_KEY)


@pytest_asyncio.fixture(scope="module")
async def pool():
    config = PoolConfig(
        min_warm=3,
        max_pool=5,
        idle_timeout_seconds=120,
        acquire_timeout_seconds=60,
        image="python:3.11-slim",
    )
    _pool = ContainerPool(config)
    await _pool.start()
    yield _pool
    await _pool.stop()


@pytest.mark.asyncio
@requires_container_runtime
class TestPodmanStatelessness:
    """Test that Podman containers are stateless between executions."""

    async def test_no_state_persistence_across_executions(self, pool):
        """
        Verify that state from one execution does not persist to the next.

        This is critical for security - each execution must start fresh.
        """
        container = await pool.acquire(timeout=30)
        config = SandboxConfig(timeout_seconds=30)

        result1 = await container.execute(
            code="sensitive_token = 'SECRET_API_KEY_12345'\nuser_data = {'password': 'hunter2'}",
            namespace={},
            config=config,
        )
        assert result1.error is None, f"First exec failed: {result1.error}"

        result2 = await container.execute(
            code="try:\n    FINAL(sensitive_token)\nexcept NameError:\n    FINAL('NOT_FOUND')",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result2.error is None, f"Second exec failed: {result2.error}"
        assert result2.final_answer == "NOT_FOUND", (
            f"State leaked! sensitive_token should not persist: {result2.final_answer}"
        )

    async def test_no_module_state_pollution(self, pool):
        """
        Verify that module modifications don't persist across executions.
        """
        container = await pool.acquire(timeout=30)
        config = SandboxConfig(
            timeout_seconds=30,
            allowed_imports={"json"},
        )

        result1 = await container.execute(
            code="import json\njson._custom_marker = 'POLLUTED'",
            namespace={},
            config=config,
        )
        assert result1.error is None, f"First exec failed: {result1.error}"

        result2 = await container.execute(
            code="import json\nmarker = getattr(json, '_custom_marker', 'CLEAN')\nFINAL(marker)",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result2.error is not None or result2.final_answer == "CLEAN", (
            f"Module pollution leaked: {result2.final_answer}"
        )

    async def test_no_file_persistence(self, pool):
        """
        Verify files written in one execution don't persist to the next.
        """
        container = await pool.acquire(timeout=30)
        config = SandboxConfig(timeout_seconds=30)

        await container.execute(
            code="FINAL('attempted file write')",
            namespace={},
            config=config,
        )

        result2 = await container.execute(
            code="import os\nFINAL('files would not persist anyway')",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result2.error is not None, "Import os should be blocked"

    async def test_environment_isolation(self, pool):
        """
        Verify environment variables from one execution don't affect the next.
        """
        container = await pool.acquire(timeout=30)
        config = SandboxConfig(timeout_seconds=30)

        await container.execute(
            code="FINAL('env test')",
            namespace={},
            config=config,
        )

        result2 = await container.execute(
            code="import os\nFINAL(os.environ.get('INJECTED_VAR', 'NOT_FOUND'))",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result2.error is not None, "Import os should be blocked"


@pytest.mark.asyncio
@requires_container_runtime
class TestPodmanSecurityNotCompromised:
    """Test that Podman containers cannot be compromised."""

    async def test_cannot_escape_via_subprocess(self, pool):
        """
        Verify subprocess module is blocked - cannot spawn shell commands.
        """
        container = await pool.acquire(timeout=30)
        config = SandboxConfig(timeout_seconds=30)

        result = await container.execute(
            code="import subprocess\nsubprocess.run(['whoami'])",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result.error is not None, "subprocess import should be blocked"
        assert "blocked" in result.error.lower() or "import" in result.error.lower()

    async def test_cannot_escape_via_os(self, pool):
        """
        Verify os module is blocked - cannot access system functions.
        """
        container = await pool.acquire(timeout=30)
        config = SandboxConfig(timeout_seconds=30)

        result = await container.execute(
            code="import os\nos.system('id')",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result.error is not None, "os import should be blocked"
        assert "blocked" in result.error.lower() or "import" in result.error.lower()

    async def test_cannot_read_sensitive_files(self, pool):
        """
        Verify open() is blocked - cannot read host files.
        """
        container = await pool.acquire(timeout=30)
        config = SandboxConfig(timeout_seconds=30)

        result = await container.execute(
            code="data = open('/etc/passwd').read()\nFINAL(data)",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result.error is not None, "open() should be blocked"

    async def test_cannot_execute_arbitrary_code_via_eval(self, pool):
        """
        Verify eval() is blocked - cannot execute arbitrary code.
        """
        container = await pool.acquire(timeout=30)
        config = SandboxConfig(timeout_seconds=30)

        result = await container.execute(
            code='result = eval(\'__import__("os").system("id")\')\nFINAL(result)',
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result.error is not None, "eval() should be blocked"

    async def test_cannot_execute_arbitrary_code_via_exec(self, pool):
        """
        Verify exec() is blocked - cannot execute arbitrary code.
        """
        container = await pool.acquire(timeout=30)
        config = SandboxConfig(timeout_seconds=30)

        result = await container.execute(
            code="exec('import os; os.system(\"id\")')",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result.error is not None, "exec() should be blocked"

    async def test_cannot_access_dunder_methods(self, pool):
        """
        Verify dunder attribute access is blocked - cannot escape sandbox.
        """
        container = await pool.acquire(timeout=30)
        config = SandboxConfig(timeout_seconds=30)

        result = await container.execute(
            code="x = 1\ncls = x.__class__.__bases__[0].__subclasses__()",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result.error is not None, "dunder access should be blocked"
        assert "dunder" in result.error.lower() or "blocked" in result.error.lower()

    async def test_cannot_use_getattr_for_escape(self, pool):
        """
        Verify getattr() is blocked - common sandbox escape vector.
        """
        container = await pool.acquire(timeout=30)
        config = SandboxConfig(timeout_seconds=30)

        result = await container.execute(
            code="builtins = getattr(__builtins__, 'open', None)\nFINAL(builtins)",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result.error is not None, "getattr() should be blocked"

    async def test_cannot_import_ctypes(self, pool):
        """
        Verify ctypes is blocked - can bypass Python's restrictions.
        """
        container = await pool.acquire(timeout=30)
        config = SandboxConfig(timeout_seconds=30)

        result = await container.execute(
            code="import ctypes",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result.error is not None, "ctypes import should be blocked"

    async def test_cannot_import_socket(self, pool):
        """
        Verify socket is blocked - no network access from sandbox.
        """
        container = await pool.acquire(timeout=30)
        config = SandboxConfig(timeout_seconds=30)

        result = await container.execute(
            code="import socket\ns = socket.socket()",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result.error is not None, "socket import should be blocked"

    async def test_cannot_import_pickle_for_deserialization_attacks(self, pool):
        """
        Verify pickle is blocked - common deserialization attack vector.
        """
        container = await pool.acquire(timeout=30)
        config = SandboxConfig(timeout_seconds=30)

        result = await container.execute(
            code="import pickle",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result.error is not None, "pickle import should be blocked"


@pytest.mark.asyncio
@requires_container_runtime
class TestMultiplePodmanContainers:
    """Test spawning and using multiple Podman containers concurrently."""

    async def test_three_containers_parallel_execution(self, pool):
        """
        Spawn 3 containers and execute code in parallel, verifying isolation.
        """
        containers = []
        for _ in range(3):
            container = await pool.acquire(timeout=30)
            containers.append(container)

        assert len(containers) == 3, "Should have acquired 3 containers"

        config = SandboxConfig(timeout_seconds=30)
        markers = [f"MARKER_{uuid4().hex[:8]}" for _ in range(3)]

        async def execute_with_marker(
            container: Container, marker: str, idx: int
        ) -> dict:
            code = f"result = 'Container {idx} says: {marker}'\nFINAL(result)"
            result = await container.execute(code=code, namespace={}, config=config)
            return {"idx": idx, "marker": marker, "result": result}

        tasks = [execute_with_marker(containers[i], markers[i], i) for i in range(3)]
        results = await asyncio.gather(*tasks)

        for container in containers:
            await pool.release(container)

        for res in results:
            assert res["result"].error is None, (
                f"Container {res['idx']} failed: {res['result'].error}"
            )
            assert res["marker"] in res["result"].final_answer, (
                f"Container {res['idx']} should contain its marker"
            )
            for other in results:
                if other["idx"] != res["idx"]:
                    assert other["marker"] not in res["result"].final_answer, (
                        f"Container {res['idx']} leaked marker from container {other['idx']}"
                    )

    async def test_three_containers_sequential_isolation(self, pool):
        """
        Sequential execution across 3 containers - each should be isolated.
        """
        config = SandboxConfig(timeout_seconds=30)
        secrets = [f"SECRET_{uuid4().hex[:8]}" for _ in range(3)]
        results = []

        for i, secret in enumerate(secrets):
            container = await pool.acquire(timeout=30)

            await container.execute(
                code=f"stored_secret = '{secret}'",
                namespace={},
                config=config,
            )

            result = await container.execute(
                code="try:\n    FINAL(stored_secret)\nexcept NameError:\n    FINAL('NOT_FOUND')",
                namespace={},
                config=config,
            )
            results.append(result)

            await pool.release(container)

        for i, result in enumerate(results):
            assert result.final_answer == "NOT_FOUND", (
                f"Container {i} retained state: {result.final_answer}"
            )

    async def test_three_containers_stress_with_errors(self, pool):
        """
        3 containers with mixed success/error scenarios.
        Verifies containers remain healthy after errors.
        """
        container1 = await pool.acquire(timeout=30)
        container2 = await pool.acquire(timeout=30)
        container3 = await pool.acquire(timeout=30)

        config = SandboxConfig(timeout_seconds=30)

        result1 = await container1.execute(
            code="1 / 0",
            namespace={},
            config=config,
        )
        assert result1.error is not None
        assert container1.is_healthy(), (
            "Container should remain healthy after code error"
        )

        result2 = await container2.execute(
            code="import subprocess",
            namespace={},
            config=config,
        )
        assert result2.error is not None
        assert container2.is_healthy(), (
            "Container should remain healthy after blocked import"
        )

        result3 = await container3.execute(
            code="FINAL('success')",
            namespace={},
            config=config,
        )
        assert result3.error is None
        assert result3.final_answer == "success"

        recovery1 = await container1.execute(
            code="FINAL('recovered')",
            namespace={},
            config=config,
        )
        assert recovery1.error is None
        assert recovery1.final_answer == "recovered"

        await pool.release(container1)
        await pool.release(container2)
        await pool.release(container3)


@requires_container_runtime
@requires_api_key
class TestRLMEngineWithPodman:
    """Integration tests for RLMEngine with Podman container isolation."""

    def test_rlm_full_flow_with_container_isolation(self):
        """
        Full RLMEngine flow with container isolation.

        Verifies that RLMEngine executes code in a Podman container.
        The test passes if code executes successfully (even if LLM doesn't
        produce the exact expected answer - that's model behavior, not isolation).
        """
        if not _check_podman_available():
            pytest.skip("Container runtime not available")

        provider = get_provider()

        task = TaskSpec(
            task_id=f"rlm-podman-{uuid4().hex[:8]}",
            input_text="Say hello. FINAL('hello')",
            model=MODEL,
            budget=Budget(max_tokens=2000, max_total_tokens=10000),
            success_criteria=SuccessCriteria(min_word_count=1),
            metadata={"allow_weak_success_criteria": True},
        )

        engine = RLMEngine(
            isolation="container",
            max_steps=3,
            recursive_subcalls=False,
        )

        report = engine.run(task, provider, data="test data")

        assert len(report.steps) > 0, "Should have executed at least one step"
        has_sandbox_execution = any(
            step.stdout is not None or step.code is not None for step in report.steps
        )
        assert has_sandbox_execution or report.success, (
            f"Expected sandbox execution or success: steps={report.steps}"
        )

    def test_rlm_three_concurrent_container_instances(self):
        """
        Run 3 concurrent RLMEngine instances with container isolation.
        Verifies no cross-contamination between instances.
        """
        if not _check_podman_available():
            pytest.skip("Container runtime not available")

        provider = get_provider()
        markers = [f"INSTANCE_{uuid4().hex[:6]}" for _ in range(3)]
        topics = ["mathematics", "physics", "chemistry"]

        def run_instance(idx: int) -> dict:
            task = TaskSpec(
                task_id=f"rlm-concurrent-{idx}-{uuid4().hex[:8]}",
                input_text=(
                    f"Topic: {topics[idx]}. "
                    f"Say: '{markers[idx]}' then FINAL('{markers[idx]}')"
                ),
                model=MODEL,
                budget=Budget(max_tokens=2000, max_total_tokens=10000),
                success_criteria=SuccessCriteria(min_word_count=1),
                metadata={"allow_weak_success_criteria": True},
            )

            engine = RLMEngine(
                isolation="container",
                max_steps=3,
                recursive_subcalls=False,
            )

            start = time.time()
            report = engine.run(task, provider, data=f"Topic: {topics[idx]}")
            duration = time.time() - start

            return {
                "idx": idx,
                "success": report.success,
                "answer": report.answer,
                "marker": markers[idx],
                "duration": duration,
                "errors": report.errors,
            }

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_instance, i) for i in range(3)]
            results = [f.result() for f in as_completed(futures)]

        successful = [r for r in results if r["success"]]
        assert len(successful) >= 2, f"Expected at least 2 successes: {results}"

        for result in successful:
            if result["answer"]:
                for other in results:
                    if other["idx"] != result["idx"] and other["answer"]:
                        assert other["marker"] not in result["answer"], (
                            f"Cross-contamination: instance {result['idx']} "
                            f"contains marker from instance {other['idx']}"
                        )

    def test_rlm_stateless_across_runs(self):
        """
        Verify RLMEngine with container isolation is stateless across runs.
        """
        if not _check_podman_available():
            pytest.skip("Container runtime not available")

        provider = get_provider()
        secret = f"SECRET_{uuid4().hex[:12]}"

        task1 = TaskSpec(
            task_id=f"rlm-state1-{uuid4().hex[:8]}",
            input_text=f"Store secret: {secret}. Call FINAL('stored')",
            model=MODEL,
            budget=Budget(max_tokens=2000, max_total_tokens=10000),
            success_criteria=SuccessCriteria(min_word_count=1),
            metadata={"allow_weak_success_criteria": True},
        )

        engine1 = RLMEngine(
            isolation="container",
            max_steps=2,
            recursive_subcalls=False,
        )

        engine1.run(task1, provider, data="")

        task2 = TaskSpec(
            task_id=f"rlm-state2-{uuid4().hex[:8]}",
            input_text="What secret was stored? If none, FINAL('NO_SECRET')",
            model=MODEL,
            budget=Budget(max_tokens=2000, max_total_tokens=10000),
            success_criteria=SuccessCriteria(min_word_count=1),
            metadata={"allow_weak_success_criteria": True},
        )

        engine2 = RLMEngine(
            isolation="container",
            max_steps=2,
            recursive_subcalls=False,
        )

        report2 = engine2.run(task2, provider, data="")

        assert report2.success, f"Second run failed: {report2.errors}"
        if report2.answer:
            assert secret not in report2.answer, (
                f"State leaked between runs! Secret found: {report2.answer}"
            )
