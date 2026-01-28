"""
Container pool management.

Maintains a warm pool of containers for low-latency acquisition.
Handles scaling, health checks, and lifecycle management.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from typing import Optional, Set

from enzu.isolation.runtime import ContainerRuntime, detect_runtime, get_runtime_command
from enzu.isolation.container import DEFAULT_SECCOMP_PROFILE
from enzu.isolation.container_wrapper import Container

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    min_warm: int = 2
    max_pool: int = 10
    idle_timeout_seconds: float = 300
    acquire_timeout_seconds: float = 30
    health_check_interval: float = 30
    image: str = "python:3.11-slim"
    # Runtime will be auto-detected if None
    runtime: Optional[ContainerRuntime] = None


class ContainerPool:
    """
    Warm container pool with automatic scaling.
    """

    def __init__(self, config: PoolConfig):
        self._config = config
        self._warm: asyncio.Queue[Container] = asyncio.Queue(maxsize=config.max_pool)
        self._busy: Set[Container] = set()
        self._spawning: int = 0
        self._lock = asyncio.Lock()
        self._running = False

        self._runtime = config.runtime or detect_runtime()

        # Seccomp profile path (created lazily)
        self._seccomp_profile_path: Optional[str] = None

        # Background tasks
        self._idle_reaper_task: Optional[asyncio.Task] = None
        self._health_checker_task: Optional[asyncio.Task] = None

    def _get_seccomp_profile_path(self) -> str:
        """Get or create seccomp profile file for gov-grade isolation."""
        if self._seccomp_profile_path is None:
            fd, path = tempfile.mkstemp(suffix=".json", prefix="enzu_pool_seccomp_")
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(DEFAULT_SECCOMP_PROFILE, f)
                self._seccomp_profile_path = path
            except Exception:
                os.close(fd)
                os.unlink(path)
                raise
        return self._seccomp_profile_path

    async def start(self) -> None:
        """Initialize pool. Fails if runtime unavailable."""
        get_runtime_command(self._runtime)

        # Pull image
        await self._ensure_image()

        self._running = True
        
        # Spawn initial containers
        logger.info(f"Filling pool with {self._config.min_warm} warm containers...")
        spawn_tasks = []
        for _ in range(self._config.min_warm):
            spawn_tasks.append(self._spawn_and_add())
        
        if spawn_tasks:
            await asyncio.gather(*spawn_tasks)

        # Start background tasks
        self._idle_reaper_task = asyncio.create_task(self._idle_reaper())
        self._health_checker_task = asyncio.create_task(self._health_checker())

    async def acquire(self, timeout: float | None = None) -> Container:
        """
        Get a warm container. Blocks until available or timeout.
        """
        if not self._running:
            raise RuntimeError("Pool not running")

        timeout = timeout or self._config.acquire_timeout_seconds
        
        # Check if we need to spawn more (if pool is empty but we have capacity)
        async with self._lock:
            total_count = self._warm.qsize() + len(self._busy) + self._spawning
            if self._warm.empty() and total_count < self._config.max_pool:
                asyncio.create_task(self._spawn_and_add())

        try:
            container = await asyncio.wait_for(
                self._warm.get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"No container available after {timeout}s. "
                f"Pool status: busy={len(self._busy)}, warm={self._warm.qsize()}, "
                f"spawning={self._spawning}, max={self._config.max_pool}"
            )

        async with self._lock:
            self._busy.add(container)
            # Trigger background spawn if below min_warm
            # But only if we haven't exceeded max_pool
            total_count = self._warm.qsize() + len(self._busy) + self._spawning
            if self._warm.qsize() < self._config.min_warm and total_count < self._config.max_pool:
                asyncio.create_task(self._spawn_and_add())

        return container

    async def release(self, container: Container) -> None:
        """Return container to pool or destroy if unhealthy."""
        async with self._lock:
            if container in self._busy:
                self._busy.remove(container)

            if container.is_healthy():
                # No reset needed for V2 architecture (fresh process)
                try:
                    self._warm.put_nowait(container)
                except asyncio.QueueFull:
                    # Should not happen if logic is correct, but if it does, destroy it
                    await container.destroy()
            else:
                await container.destroy()
                # Replenish if needed
                total_count = self._warm.qsize() + len(self._busy) + self._spawning
                if total_count < self._config.min_warm:
                    asyncio.create_task(self._spawn_and_add())

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False

        if self._idle_reaper_task:
            self._idle_reaper_task.cancel()
        if self._health_checker_task:
            self._health_checker_task.cancel()

        # Destroy all containers
        while not self._warm.empty():
            try:
                container = self._warm.get_nowait()
                await container.destroy()
            except asyncio.QueueEmpty:
                break

        for container in list(self._busy):
            await container.destroy()

        # Cleanup seccomp profile temp file
        if self._seccomp_profile_path and os.path.exists(self._seccomp_profile_path):
            try:
                os.unlink(self._seccomp_profile_path)
            except OSError:
                pass
            self._seccomp_profile_path = None

    async def _spawn_container(self) -> Container:
        """Spawn a new container running 'sleep infinity' with gov-grade hardening."""
        cmd = get_runtime_command(self._runtime)
        name = f"enzu-sandbox-{uuid.uuid4().hex[:8]}"

        # Note: Seccomp profile is NOT applied at container startup because it blocks
        # execve which is needed to run "sh -c sleep infinity". The container's other
        # security measures (read-only fs, no network, dropped capabilities) provide
        # isolation for the sleep process. Seccomp restrictions are applied to the
        # exec'd code execution process via container_wrapper.py.

        # Full hardening flags matching container.py security profile
        run_cmd = [
            cmd, "run", "-d",
            "--name", name,
            # Run as non-root user (UID 1000) for defense in depth
            "--user=1000:1000",
            # Network isolation
            "--network=none",
            # Read-only rootfs
            "--read-only",
            # Drop all capabilities
            "--cap-drop=ALL",
            # No new privileges (prevent setuid binaries)
            "--security-opt=no-new-privileges",
            # Memory limit
            "--memory=512m",
            "--pids-limit=50",
            # Tmpfs for python temp files (noexec prevents executing binaries)
            "--tmpfs=/tmp:size=64m,noexec",
            self._config.image,
            "sh", "-c", "sleep infinity"
        ]
        
        proc = await asyncio.create_subprocess_exec(
            *run_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to spawn container: {stderr.decode()}")
            
        container_id = stdout.decode().strip()
        return Container(container_id, self._runtime)

    async def _spawn_and_add(self) -> None:
        """Background helper to spawn and add to queue."""
        async with self._lock:
            self._spawning += 1
            
        try:
            container = await self._spawn_container()
            await self._warm.put(container)
        except Exception as e:
            logger.error(f"Failed to spawn container: {e}")
        finally:
            async with self._lock:
                self._spawning -= 1

    async def _ensure_image(self) -> None:
        """Pull image if missing."""
        cmd = get_runtime_command(self._runtime)
        proc = await asyncio.create_subprocess_exec(
            cmd, "image", "inspect", self._config.image,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await proc.wait()
        
        if proc.returncode != 0:
            logger.info(f"Pulling image {self._config.image}...")
            pull_proc = await asyncio.create_subprocess_exec(
                cmd, "pull", self._config.image,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await pull_proc.communicate()
            if pull_proc.returncode != 0:
                raise RuntimeError(f"Failed to pull image {self._config.image}: {stderr.decode()}")

    async def _idle_reaper(self) -> None:
        """Periodically destroy idle containers if we have more than min_warm."""
        while self._running:
            await asyncio.sleep(60) # Run every minute
            
            # Implementation caveat: asyncio.Queue doesn't support easy iteration.
            # We can only reap if we pop them, check idleness, and put them back.
            # For simplicity in this version, we skip complex idle reaping logic 
            # and just rely on max_pool to cap usage.
            pass

    async def _health_checker(self) -> None:
        """Periodically ping warm containers."""
        while self._running:
            await asyncio.sleep(self._config.health_check_interval)
            
            # To check health, we must rotate the queue
            # This is tricky with concurrent acquirers.
            # Simpler approach: Check health only on acquire/release or
            # maintain a separate list of warm containers.
            # For V1 reliability, we'll keep it simple: 
            # Trust the container stays alive (sleep infinity) 
            # and fail fast on acquire()/execute() if it died.
            pass
