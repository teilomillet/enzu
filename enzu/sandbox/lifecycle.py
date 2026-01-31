"""
LifecycleManager: Automatic idle timeout and cleanup for sandbox sessions.

Part of the sandbox evolution plan (Phase 3). Manages sandbox lifecycle:
- Idle timeout: Terminate after N seconds of inactivity
- Max lifetime: Terminate after N seconds total
- Health checks: Periodic monitoring
- Cleanup: Proper resource release

Integration points:
- ContainerSandbox implements SandboxProtocol (has terminate method)
- IsolatedSandbox implements SandboxProtocol (has terminate method)
- LifecycleManager tracks registered sandboxes and terminates on timeout

Usage:
    from enzu.sandbox import LifecycleConfig, get_lifecycle_manager

    # Configure lifecycle
    config = LifecycleConfig(
        idle_timeout_seconds=300,    # 5 min idle timeout
        max_lifetime_seconds=1800,   # 30 min max lifetime
    )

    manager = get_lifecycle_manager(config)

    # Register sandbox for tracking
    managed = manager.register("sandbox-123", sandbox)

    # Mark activity (resets idle timer)
    manager.touch("sandbox-123")

    # Unregister when done
    manager.unregister("sandbox-123")
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass
class LifecycleConfig:
    """
    Configuration for sandbox lifecycle management.

    All timeouts are in seconds. The monitor thread runs every
    health_check_interval seconds to check for expired sandboxes.
    """

    # Seconds of inactivity before termination (default: 5 minutes)
    idle_timeout_seconds: float = 300.0

    # Maximum total lifetime in seconds (default: 1 hour)
    max_lifetime_seconds: float = 3600.0

    # Interval between health checks in seconds (default: 30s)
    health_check_interval: float = 30.0


class SandboxProtocol(Protocol):
    """
    Protocol for sandboxes that can be lifecycle-managed.

    Sandboxes must implement terminate() for cleanup.
    ContainerSandbox and IsolatedSandbox both satisfy this.
    """

    def terminate(self) -> None:
        """Terminate the sandbox and release resources."""
        ...


@dataclass
class ManagedSandbox:
    """
    Wrapper for a sandbox with lifecycle tracking.

    Tracks creation time and last activity for timeout calculations.
    """

    # Unique identifier for this sandbox
    sandbox_id: str

    # The actual sandbox instance
    sandbox: SandboxProtocol

    # Unix timestamp of creation
    created_at: float

    # Unix timestamp of last activity
    last_activity: float

    # Optional callback when terminated
    on_terminate: Optional[Callable[[str, str], None]] = None

    def touch(self) -> None:
        """Mark as active (reset idle timer)."""
        self.last_activity = time.time()

    def idle_seconds(self) -> float:
        """Seconds since last activity."""
        return time.time() - self.last_activity

    def lifetime_seconds(self) -> float:
        """Total seconds since creation."""
        return time.time() - self.created_at


class LifecycleManager:
    """
    Manages sandbox lifecycle: idle timeout, max lifetime, cleanup.

    Runs a background thread that periodically checks for sandboxes
    that have exceeded their idle timeout or max lifetime and terminates them.

    Thread-safe: all sandbox operations are protected by a lock.

    Usage:
        manager = LifecycleManager(config)
        manager.start()

        # Register sandboxes
        managed = manager.register("id", sandbox)

        # Mark activity
        manager.touch("id")

        # Cleanup
        manager.stop()
    """

    def __init__(self, config: LifecycleConfig) -> None:
        self._config = config
        self._sandboxes: Dict[str, ManagedSandbox] = {}
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """
        Start background lifecycle monitoring.

        Spawns a daemon thread that checks for expired sandboxes
        at the configured health_check_interval.
        """
        if self._running:
            return
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Lifecycle manager started")

    def stop(self) -> None:
        """
        Stop monitoring and terminate all sandboxes.

        Waits for the monitor thread to exit and cleans up all
        registered sandboxes.
        """
        self._running = False
        with self._lock:
            for managed in list(self._sandboxes.values()):
                self._terminate(managed, "shutdown")
            self._sandboxes.clear()
        logger.info("Lifecycle manager stopped")

    def register(
        self,
        sandbox_id: str,
        sandbox: SandboxProtocol,
        on_terminate: Optional[Callable[[str, str], None]] = None,
    ) -> ManagedSandbox:
        """
        Register a sandbox for lifecycle management.

        Args:
            sandbox_id: Unique identifier for the sandbox
            sandbox: Sandbox instance implementing SandboxProtocol
            on_terminate: Optional callback(sandbox_id, reason) on termination

        Returns:
            ManagedSandbox wrapper with lifecycle tracking
        """
        now = time.time()
        managed = ManagedSandbox(
            sandbox_id=sandbox_id,
            sandbox=sandbox,
            created_at=now,
            last_activity=now,
            on_terminate=on_terminate,
        )
        with self._lock:
            self._sandboxes[sandbox_id] = managed
        logger.debug("Registered sandbox: %s", sandbox_id)
        return managed

    def unregister(self, sandbox_id: str) -> None:
        """
        Remove sandbox from management (without terminating).

        Use this when the sandbox completes normally and cleanup
        is handled elsewhere.
        """
        with self._lock:
            self._sandboxes.pop(sandbox_id, None)

    def touch(self, sandbox_id: str) -> None:
        """
        Mark sandbox as active (reset idle timer).

        Call this on any sandbox activity (exec, IPC, etc.)
        to prevent idle timeout.
        """
        with self._lock:
            if sandbox_id in self._sandboxes:
                self._sandboxes[sandbox_id].touch()

    def get(self, sandbox_id: str) -> Optional[ManagedSandbox]:
        """Get managed sandbox by ID, or None if not found."""
        with self._lock:
            return self._sandboxes.get(sandbox_id)

    def stats(self) -> Dict[str, Any]:
        """
        Get lifecycle statistics.

        Returns:
            Dict with active_count, oldest_seconds, most_idle_seconds
        """
        with self._lock:
            if not self._sandboxes:
                return {
                    "active_count": 0,
                    "oldest_seconds": 0,
                    "most_idle_seconds": 0,
                }

            oldest = max(m.lifetime_seconds() for m in self._sandboxes.values())
            most_idle = max(m.idle_seconds() for m in self._sandboxes.values())

            return {
                "active_count": len(self._sandboxes),
                "oldest_seconds": oldest,
                "most_idle_seconds": most_idle,
            }

    def _monitor_loop(self) -> None:
        """
        Background thread: check for idle/expired sandboxes.

        Runs until self._running is False. Checks all sandboxes
        against idle_timeout and max_lifetime, terminating those
        that exceed limits.
        """
        while self._running:
            to_terminate: list[tuple[ManagedSandbox, str]] = []

            with self._lock:
                for managed in list(self._sandboxes.values()):
                    if managed.idle_seconds() > self._config.idle_timeout_seconds:
                        to_terminate.append((managed, "idle_timeout"))
                    elif managed.lifetime_seconds() > self._config.max_lifetime_seconds:
                        to_terminate.append((managed, "max_lifetime"))

            # Terminate outside lock to avoid holding it during cleanup
            for managed, reason in to_terminate:
                self._terminate(managed, reason)
                with self._lock:
                    self._sandboxes.pop(managed.sandbox_id, None)

            # Sleep in small increments for responsive shutdown
            sleep_remaining = self._config.health_check_interval
            while sleep_remaining > 0 and self._running:
                sleep_time = min(sleep_remaining, 1.0)
                time.sleep(sleep_time)
                sleep_remaining -= sleep_time

    def _terminate(self, managed: ManagedSandbox, reason: str) -> None:
        """
        Terminate a sandbox and invoke callback if set.

        Args:
            managed: The ManagedSandbox to terminate
            reason: Reason string (idle_timeout, max_lifetime, shutdown)
        """
        try:
            managed.sandbox.terminate()
            logger.info(
                "Terminated sandbox %s: %s (lifetime=%.1fs, idle=%.1fs)",
                managed.sandbox_id,
                reason,
                managed.lifetime_seconds(),
                managed.idle_seconds(),
            )
            if managed.on_terminate:
                try:
                    managed.on_terminate(managed.sandbox_id, reason)
                except Exception as e:
                    logger.warning(
                        "on_terminate callback failed for %s: %s",
                        managed.sandbox_id,
                        e,
                    )
        except Exception as e:
            logger.error("Failed to terminate %s: %s", managed.sandbox_id, e)


# Global lifecycle manager (lazy init)
_global_manager: Optional[LifecycleManager] = None
_global_lock = threading.Lock()


def get_lifecycle_manager(config: Optional[LifecycleConfig] = None) -> LifecycleManager:
    """
    Get or create global lifecycle manager.

    The global manager is shared across all sandbox usage.
    Pass config only on first call to customize settings.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        The global LifecycleManager instance
    """
    global _global_manager
    with _global_lock:
        if _global_manager is None:
            _global_manager = LifecycleManager(config or LifecycleConfig())
            _global_manager.start()
        return _global_manager


def reset_lifecycle_manager() -> None:
    """
    Stop and reset the global lifecycle manager.

    Used for testing or when reconfiguration is needed.
    """
    global _global_manager
    with _global_lock:
        if _global_manager is not None:
            _global_manager.stop()
            _global_manager = None
