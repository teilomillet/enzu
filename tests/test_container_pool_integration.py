"""
Integration tests for Container Pool with Podman.

These tests require Podman to be installed and running.
They test the full flow: pool → container → execute → release.
"""
import pytest
import pytest_asyncio

from enzu.isolation.runtime import ContainerRuntime, detect_runtime
from enzu.isolation.pool import ContainerPool, PoolConfig
from enzu.isolation.container_wrapper import Container
from enzu.isolation.runner import SandboxConfig

# Mark all tests as asyncio
pytestmark = pytest.mark.asyncio


def _check_podman_available() -> bool:
    """Check if Podman is available and working."""
    try:
        runtime = detect_runtime()
        return runtime in (ContainerRuntime.PODMAN, ContainerRuntime.DOCKER)
    except RuntimeError:
        return False


# Skip all tests if no container runtime available
requires_container_runtime = pytest.mark.skipif(
    not _check_podman_available(),
    reason="No container runtime (Podman/Docker) available"
)


@pytest_asyncio.fixture(scope="module")
async def pool():
    """Create and start a container pool for testing."""
    config = PoolConfig(
        min_warm=1,
        max_pool=2,
        idle_timeout_seconds=60,
        acquire_timeout_seconds=60,
        image="python:3.11-slim",
        # Auto-detect runtime
    )
    _pool = ContainerPool(config)
    await _pool.start()
    yield _pool
    await _pool.stop()


@requires_container_runtime
class TestRuntimeDetection:
    """Test runtime detection."""

    def test_detect_runtime_finds_container_runtime(self):
        """Verify a container runtime is detected."""
        runtime = detect_runtime()
        assert runtime in (ContainerRuntime.PODMAN, ContainerRuntime.DOCKER)


@requires_container_runtime
class TestContainerPool:
    """Test container pool lifecycle."""

    async def test_pool_starts_with_warm_containers(self, pool):
        """Pool should have min_warm containers after start."""
        # Pool started in fixture, should have at least 1 warm
        assert pool._warm.qsize() >= 1

    
    async def test_acquire_returns_container(self, pool):
        """Acquire should return a container."""
        container = await pool.acquire(timeout=10)
        assert container is not None
        assert isinstance(container, Container)
        assert container._id is not None
        # Release it back
        await pool.release(container)

    
    async def test_release_returns_container_to_pool(self, pool):
        """Release should return healthy container to pool."""
        container = await pool.acquire(timeout=10)
        initial_warm = pool._warm.qsize()

        await pool.release(container)

        # Should have one more warm container now
        assert pool._warm.qsize() == initial_warm + 1


@requires_container_runtime
class TestContainerExecution:
    """Test code execution inside containers."""

    
    async def test_simple_execution(self, pool):
        """Execute simple code and get result."""
        container = await pool.acquire(timeout=10)

        config = SandboxConfig(timeout_seconds=30)
        result = await container.execute(
            code="x = 1 + 1\nFINAL(x)",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result.error is None
        assert result.final_answer == "2"

    
    async def test_namespace_passed_correctly(self, pool):
        """Verify namespace is available in execution."""
        container = await pool.acquire(timeout=10)

        config = SandboxConfig(timeout_seconds=30)
        result = await container.execute(
            code="result = data['value'] * 2\nFINAL(result)",
            namespace={"data": {"value": 21}},
            config=config,
        )

        await pool.release(container)

        assert result.error is None
        assert result.final_answer == "42"

    
    async def test_namespace_updates_returned(self, pool):
        """Verify namespace updates are returned."""
        container = await pool.acquire(timeout=10)

        config = SandboxConfig(timeout_seconds=30)
        result = await container.execute(
            code="x = 10\ny = 20\nz = x + y",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result.error is None
        assert result.namespace_updates.get("x") == 10
        assert result.namespace_updates.get("y") == 20
        assert result.namespace_updates.get("z") == 30

    
    async def test_stdout_captured(self, pool):
        """Verify print output is captured."""
        container = await pool.acquire(timeout=10)

        config = SandboxConfig(timeout_seconds=30)
        result = await container.execute(
            code="print('hello world')",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result.error is None
        assert "hello world" in result.stdout

    
    async def test_allowed_import_works(self, pool):
        """Verify allowed imports work."""
        container = await pool.acquire(timeout=10)

        config = SandboxConfig(
            timeout_seconds=30,
            allowed_imports={"math", "json", "re"},
        )
        result = await container.execute(
            code="import math\nFINAL(math.sqrt(16))",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result.error is None
        assert result.final_answer == "4.0"


@requires_container_runtime
class TestSandboxRestrictions:
    """Test that sandbox restrictions are enforced."""

    
    async def test_blocked_import_os(self, pool):
        """Import os should be blocked."""
        container = await pool.acquire(timeout=10)

        config = SandboxConfig(timeout_seconds=30)
        result = await container.execute(
            code="import os",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result.error is not None
        assert "Import blocked" in result.error or "blocked" in result.error.lower()

    
    async def test_blocked_import_subprocess(self, pool):
        """Import subprocess should be blocked."""
        container = await pool.acquire(timeout=10)

        config = SandboxConfig(timeout_seconds=30)
        result = await container.execute(
            code="import subprocess",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result.error is not None
        assert "Import blocked" in result.error or "blocked" in result.error.lower()

    
    async def test_blocked_import_pickle(self, pool):
        """Import pickle should be blocked."""
        container = await pool.acquire(timeout=10)

        config = SandboxConfig(timeout_seconds=30)
        result = await container.execute(
            code="import pickle",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result.error is not None
        assert "Import blocked" in result.error or "blocked" in result.error.lower()

    
    async def test_blocked_eval(self, pool):
        """eval() should not be available."""
        container = await pool.acquire(timeout=10)

        config = SandboxConfig(timeout_seconds=30)
        result = await container.execute(
            code="eval('1+1')",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result.error is not None
        assert "eval" in result.error.lower() or "name" in result.error.lower()

    
    async def test_blocked_exec(self, pool):
        """exec() should not be available (as a builtin to call)."""
        container = await pool.acquire(timeout=10)

        config = SandboxConfig(timeout_seconds=30)
        result = await container.execute(
            code="exec('x=1')",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result.error is not None

    
    async def test_blocked_open(self, pool):
        """open() should not be available."""
        container = await pool.acquire(timeout=10)

        config = SandboxConfig(timeout_seconds=30)
        result = await container.execute(
            code="open('/etc/passwd')",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result.error is not None

    
    async def test_blocked_getattr(self, pool):
        """getattr() should not be available."""
        container = await pool.acquire(timeout=10)

        config = SandboxConfig(timeout_seconds=30)
        result = await container.execute(
            code="getattr(object, '__class__')",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result.error is not None

    
    async def test_blocked_dunder_access(self, pool):
        """Direct __dunder__ access should be blocked by AST."""
        container = await pool.acquire(timeout=10)

        config = SandboxConfig(timeout_seconds=30)
        result = await container.execute(
            code="x = 1\nx.__class__",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result.error is not None
        assert "dunder" in result.error.lower() or "blocked" in result.error.lower()


@requires_container_runtime
class TestProcessIsolation:
    """Test that each execution uses a fresh process."""

    
    async def test_no_state_leakage_between_executions(self, pool):
        """State from one execution should not leak to next."""
        container = await pool.acquire(timeout=10)
        config = SandboxConfig(timeout_seconds=30)

        # First execution: set a variable
        result1 = await container.execute(
            code="secret = 'user_a_secret'",
            namespace={},
            config=config,
        )
        assert result1.error is None

        # Second execution: try to access that variable (should fail)
        result2 = await container.execute(
            code="FINAL(secret)",
            namespace={},
            config=config,
        )

        await pool.release(container)

        # Should error because 'secret' is not defined in fresh process
        assert result2.error is not None
        assert "secret" in result2.error.lower() or "name" in result2.error.lower()

    
    async def test_no_module_pollution_between_executions(self, pool):
        """Module modifications should not persist."""
        container = await pool.acquire(timeout=10)
        config = SandboxConfig(
            timeout_seconds=30,
            allowed_imports={"json"},
        )

        # First execution: this would fail anyway due to restricted builtins
        # but let's verify the isolation concept
        await container.execute(
            code="import json\noriginal_dumps = json.dumps",
            namespace={},
            config=config,
        )

        # Second execution: json should be fresh
        result2 = await container.execute(
            code="import json\nFINAL(json.dumps({'a': 1}))",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result2.error is None
        assert result2.final_answer == '{"a": 1}'


@requires_container_runtime
class TestErrorHandling:
    """Test error handling scenarios."""

    
    async def test_syntax_error_reported(self, pool):
        """Syntax errors should be reported."""
        container = await pool.acquire(timeout=10)

        config = SandboxConfig(timeout_seconds=30)
        result = await container.execute(
            code="def broken(",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result.error is not None
        # Error could say "syntax" or describe the issue like "was never closed"
        assert "syntax" in result.error.lower() or "never closed" in result.error.lower()

    
    async def test_runtime_error_reported(self, pool):
        """Runtime errors should be reported."""
        container = await pool.acquire(timeout=10)

        config = SandboxConfig(timeout_seconds=30)
        result = await container.execute(
            code="1 / 0",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result.error is not None
        assert "division" in result.error.lower() or "zero" in result.error.lower()

    
    async def test_container_healthy_after_error(self, pool):
        """Container should remain healthy after code errors."""
        container = await pool.acquire(timeout=10)
        config = SandboxConfig(timeout_seconds=30)

        # Execute code that errors
        result1 = await container.execute(
            code="raise ValueError('test error')",
            namespace={},
            config=config,
        )
        assert result1.error is not None

        # Container should still be healthy
        assert container.is_healthy()

        # Should be able to execute again
        result2 = await container.execute(
            code="FINAL('ok')",
            namespace={},
            config=config,
        )

        await pool.release(container)

        assert result2.error is None
        assert result2.final_answer == "ok"
