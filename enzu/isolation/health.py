"""
Health checking and circuit breaker for production resilience (Phase 5).

Provides:
- HealthChecker: Proactive node monitoring with configurable intervals
- CircuitBreaker: Prevents cascading failures by stopping requests to failing nodes

Circuit breaker states:
    CLOSED: Normal operation, requests flow through
    OPEN: Node is failing, requests are rejected immediately
    HALF_OPEN: Testing recovery, single request allowed through

Integration with scheduler.py:
    The DistributedCoordinator uses HealthChecker for background monitoring
    and CircuitBreaker for per-node failure protection.

"""

from __future__ import annotations

import asyncio
import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal: requests flow through
    OPEN = "open"  # Failing: requests rejected immediately
    HALF_OPEN = "half_open"  # Recovery: testing with single request


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    # Failure threshold to trip breaker
    failure_threshold: int = 5
    # Seconds to wait before attempting recovery (OPEN -> HALF_OPEN)
    reset_timeout_seconds: float = 30.0
    # Consecutive successes needed in HALF_OPEN to close circuit
    success_threshold: int = 2
    # Time window for counting failures (sliding window)
    failure_window_seconds: float = 60.0


@dataclass
class CircuitBreakerStats:
    """Statistics for a circuit breaker."""

    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: Optional[float]
    last_success_time: Optional[float]
    total_rejected: int
    total_allowed: int


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Protects system from cascading failures by:
    1. Tracking failures per node
    2. Tripping open after threshold failures
    3. Allowing test request after timeout
    4. Closing after successful recovery

    Thread-safe for concurrent request handling.

    Usage:
        breaker = CircuitBreaker(node_id="node-1")

        if breaker.allow_request():
            try:
                result = await execute_on_node(task)
                breaker.record_success()
            except Exception:
                breaker.record_failure()
                raise
        else:
            raise CircuitBreakerOpen("Node node-1 circuit is open")
    """

    def __init__(
        self,
        node_id: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> None:
        self._node_id = node_id
        self._config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0  # For HALF_OPEN recovery
        self._failure_times: List[float] = []  # Sliding window

        self._last_failure_time: Optional[float] = None
        self._last_success_time: Optional[float] = None
        self._last_state_change: float = time.time()

        self._total_rejected = 0
        self._total_allowed = 0

        self._lock = threading.Lock()

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    def allow_request(self) -> bool:
        """
        Check if a request should be allowed through.

        Returns:
            True if request can proceed, False if circuit is open.
        """
        with self._lock:
            now = time.time()

            if self._state == CircuitState.CLOSED:
                self._total_allowed += 1
                return True

            if self._state == CircuitState.OPEN:
                # Check if reset timeout has elapsed
                elapsed = now - self._last_state_change
                if elapsed >= self._config.reset_timeout_seconds:
                    # Transition to HALF_OPEN for recovery test
                    self._state = CircuitState.HALF_OPEN
                    self._last_state_change = now
                    self._success_count = 0
                    logger.info(
                        "CircuitBreaker %s: OPEN -> HALF_OPEN (testing recovery)",
                        self._node_id,
                    )
                    self._total_allowed += 1
                    return True

                # Still open, reject
                self._total_rejected += 1
                return False

            # HALF_OPEN: allow single request for testing
            self._total_allowed += 1
            return True

    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            now = time.time()
            self._last_success_time = now

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._config.success_threshold:
                    # Recovery confirmed, close circuit
                    self._state = CircuitState.CLOSED
                    self._last_state_change = now
                    self._failure_count = 0
                    self._failure_times.clear()
                    logger.info(
                        "CircuitBreaker %s: HALF_OPEN -> CLOSED (recovery confirmed)",
                        self._node_id,
                    )
            elif self._state == CircuitState.CLOSED:
                # Clear old failures from sliding window
                self._prune_failure_window(now)

    def record_failure(self) -> None:
        """Record a failed request."""
        with self._lock:
            now = time.time()
            self._last_failure_time = now
            self._failure_times.append(now)

            # Prune old failures outside window
            self._prune_failure_window(now)

            if self._state == CircuitState.HALF_OPEN:
                # Recovery failed, go back to OPEN
                self._state = CircuitState.OPEN
                self._last_state_change = now
                logger.warning(
                    "CircuitBreaker %s: HALF_OPEN -> OPEN (recovery failed)",
                    self._node_id,
                )
            elif self._state == CircuitState.CLOSED:
                self._failure_count = len(self._failure_times)
                if self._failure_count >= self._config.failure_threshold:
                    # Threshold exceeded, trip breaker
                    self._state = CircuitState.OPEN
                    self._last_state_change = now
                    logger.warning(
                        "CircuitBreaker %s: CLOSED -> OPEN (failures=%d)",
                        self._node_id,
                        self._failure_count,
                    )

    def _prune_failure_window(self, now: float) -> None:
        """Remove failures outside the sliding window."""
        cutoff = now - self._config.failure_window_seconds
        self._failure_times = [t for t in self._failure_times if t > cutoff]
        self._failure_count = len(self._failure_times)

    def reset(self) -> None:
        """Manually reset circuit to CLOSED state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._failure_times.clear()
            self._last_state_change = time.time()
            logger.info("CircuitBreaker %s: manually reset to CLOSED", self._node_id)

    def stats(self) -> CircuitBreakerStats:
        """Get current circuit breaker statistics."""
        with self._lock:
            return CircuitBreakerStats(
                state=self._state,
                failure_count=self._failure_count,
                success_count=self._success_count,
                last_failure_time=self._last_failure_time,
                last_success_time=self._last_success_time,
                total_rejected=self._total_rejected,
                total_allowed=self._total_allowed,
            )


class CircuitBreakerOpen(Exception):
    """Raised when request is rejected due to open circuit."""

    def __init__(self, node_id: str, message: Optional[str] = None):
        self.node_id = node_id
        super().__init__(message or f"Circuit breaker open for node {node_id}")


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    node_id: str
    healthy: bool
    latency_ms: float
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class HealthCheckerConfig:
    """Configuration for health checker."""

    # Interval between health checks
    check_interval_seconds: float = 5.0
    # Timeout for individual health check
    check_timeout_seconds: float = 3.0
    # Consecutive failures before marking unhealthy
    unhealthy_threshold: int = 3
    # Consecutive successes before marking healthy again
    healthy_threshold: int = 2


class HealthChecker:
    """
    Background health checker for worker nodes.

    Monitors node health via periodic checks and updates status.
    Integrates with DistributedCoordinator to mark nodes unhealthy.

    Usage:
        checker = HealthChecker(coordinator, config=HealthCheckerConfig())
        await checker.start()

        # Later...
        await checker.stop()

    Health check protocol:
        - Local nodes: Direct function call
        - Remote nodes: HTTP GET /health endpoint
    """

    def __init__(
        self,
        on_node_unhealthy: Optional[Callable[[str, str], None]] = None,
        on_node_healthy: Optional[Callable[[str], None]] = None,
        config: Optional[HealthCheckerConfig] = None,
    ) -> None:
        """
        Args:
            on_node_unhealthy: Callback(node_id, reason) when node becomes unhealthy
            on_node_healthy: Callback(node_id) when node recovers
            config: Health checker configuration
        """
        self._on_unhealthy = on_node_unhealthy
        self._on_healthy = on_node_healthy
        self._config = config or HealthCheckerConfig()

        # Registered health check functions per node
        self._check_functions: Dict[str, Callable[[], bool]] = {}

        # Track consecutive failures/successes
        self._failure_counts: Dict[str, int] = {}
        self._success_counts: Dict[str, int] = {}
        self._node_healthy: Dict[str, bool] = {}

        # Background task
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()

        # Results history (last N per node)
        self._history: Dict[str, List[HealthCheckResult]] = {}
        self._max_history = 10

    def register_node(
        self,
        node_id: str,
        check_function: Callable[[], bool],
    ) -> None:
        """
        Register a node for health checking.

        Args:
            node_id: Unique node identifier
            check_function: Function that returns True if node is healthy
        """
        with self._lock:
            self._check_functions[node_id] = check_function
            self._failure_counts[node_id] = 0
            self._success_counts[node_id] = 0
            self._node_healthy[node_id] = True
            self._history[node_id] = []

    def unregister_node(self, node_id: str) -> None:
        """Remove node from health checking."""
        with self._lock:
            self._check_functions.pop(node_id, None)
            self._failure_counts.pop(node_id, None)
            self._success_counts.pop(node_id, None)
            self._node_healthy.pop(node_id, None)
            self._history.pop(node_id, None)

    async def start(self) -> None:
        """Start background health checking."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._check_loop())
        logger.info("HealthChecker started")

    async def stop(self) -> None:
        """Stop background health checking."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("HealthChecker stopped")

    async def _check_loop(self) -> None:
        """Background loop that runs health checks."""
        while self._running:
            try:
                await self._check_all_nodes()
                await asyncio.sleep(self._config.check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check loop error: %s", e)
                await asyncio.sleep(1.0)  # Backoff on error

    async def _check_all_nodes(self) -> None:
        """Run health check on all registered nodes."""
        with self._lock:
            nodes = list(self._check_functions.items())

        for node_id, check_func in nodes:
            await self._check_node(node_id, check_func)

    async def _check_node(
        self,
        node_id: str,
        check_func: Callable[[], bool],
    ) -> HealthCheckResult:
        """Run health check on single node."""
        start = time.time()
        error = None
        healthy = False

        try:
            # Run check with timeout
            loop = asyncio.get_event_loop()
            if asyncio.iscoroutinefunction(check_func):
                healthy = await asyncio.wait_for(
                    check_func(),
                    timeout=self._config.check_timeout_seconds,
                )
            else:
                healthy = await asyncio.wait_for(
                    loop.run_in_executor(None, check_func),
                    timeout=self._config.check_timeout_seconds,
                )
        except asyncio.TimeoutError:
            error = "health_check_timeout"
        except Exception as e:
            error = f"health_check_error: {type(e).__name__}"

        latency_ms = (time.time() - start) * 1000

        result = HealthCheckResult(
            node_id=node_id,
            healthy=healthy,
            latency_ms=latency_ms,
            error=error,
        )

        # Update state based on result
        self._update_node_state(node_id, result)

        return result

    def _update_node_state(self, node_id: str, result: HealthCheckResult) -> None:
        """Update node health state based on check result."""
        with self._lock:
            # Store in history
            if node_id in self._history:
                self._history[node_id].append(result)
                # Trim to max
                if len(self._history[node_id]) > self._max_history:
                    self._history[node_id] = self._history[node_id][
                        -self._max_history :
                    ]

            if node_id not in self._failure_counts:
                return

            was_healthy = self._node_healthy.get(node_id, True)

            if result.healthy and result.error is None:
                # Success
                self._failure_counts[node_id] = 0
                self._success_counts[node_id] += 1

                if not was_healthy:
                    if self._success_counts[node_id] >= self._config.healthy_threshold:
                        self._node_healthy[node_id] = True
                        logger.info("Node %s recovered (healthy)", node_id)
                        if self._on_healthy:
                            self._on_healthy(node_id)
            else:
                # Failure
                self._success_counts[node_id] = 0
                self._failure_counts[node_id] += 1

                if was_healthy:
                    if (
                        self._failure_counts[node_id]
                        >= self._config.unhealthy_threshold
                    ):
                        self._node_healthy[node_id] = False
                        reason = result.error or "health_check_failed"
                        logger.warning("Node %s unhealthy: %s", node_id, reason)
                        if self._on_unhealthy:
                            self._on_unhealthy(node_id, reason)

    def is_healthy(self, node_id: str) -> bool:
        """Check if node is currently considered healthy."""
        with self._lock:
            return self._node_healthy.get(node_id, False)

    def get_history(self, node_id: str) -> List[HealthCheckResult]:
        """Get recent health check history for a node."""
        with self._lock:
            return list(self._history.get(node_id, []))

    def all_healthy(self) -> bool:
        """Check if all registered nodes are healthy."""
        with self._lock:
            return all(self._node_healthy.values())

    def healthy_nodes(self) -> List[str]:
        """Get list of healthy node IDs."""
        with self._lock:
            return [nid for nid, healthy in self._node_healthy.items() if healthy]

    def unhealthy_nodes(self) -> List[str]:
        """Get list of unhealthy node IDs."""
        with self._lock:
            return [nid for nid, healthy in self._node_healthy.items() if not healthy]


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    # Maximum retry attempts
    max_retries: int = 3
    # Initial backoff delay in seconds
    initial_backoff_seconds: float = 0.1
    # Maximum backoff delay in seconds
    max_backoff_seconds: float = 10.0
    # Backoff multiplier (exponential)
    backoff_multiplier: float = 2.0
    # Jitter factor (0-1) to randomize backoff
    jitter_factor: float = 0.1
    # Retryable exception types
    retryable_exceptions: tuple = (TimeoutError, ConnectionError, OSError)


class RetryStrategy:
    """
    Retry strategy with exponential backoff.

    Provides retry logic for transient failures:
    - Exponential backoff between retries
    - Jitter to prevent thundering herd
    - Configurable retry limits

    Usage:
        retry = RetryStrategy(config=RetryConfig(max_retries=3))

        async for attempt in retry.attempts():
            try:
                result = await execute_task()
                break
            except Exception as e:
                if not retry.should_retry(e, attempt):
                    raise
    """

    def __init__(self, config: Optional[RetryConfig] = None) -> None:
        self._config = config or RetryConfig()

    def calculate_backoff(self, attempt: int) -> float:
        """
        Calculate backoff delay for attempt number.

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in seconds before next retry
        """
        import random

        # Exponential backoff
        delay = self._config.initial_backoff_seconds * (
            self._config.backoff_multiplier**attempt
        )

        # Cap at maximum
        delay = min(delay, self._config.max_backoff_seconds)

        # Add jitter
        jitter_range = delay * self._config.jitter_factor
        delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        Check if retry should be attempted.

        Args:
            exception: The exception that occurred
            attempt: Current attempt number (0-indexed)

        Returns:
            True if retry should be attempted
        """
        if attempt >= self._config.max_retries:
            return False

        return isinstance(exception, self._config.retryable_exceptions)

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute function with automatic retry.

        Args:
            func: Async function to execute
            *args, **kwargs: Arguments to pass to function

        Returns:
            Result from successful execution

        Raises:
            Last exception if all retries exhausted
        """
        last_exception = None

        for attempt in range(self._config.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None, lambda: func(*args, **kwargs)
                    )

            except Exception as e:
                last_exception = e

                if not self.should_retry(e, attempt):
                    raise

                # Calculate and apply backoff
                delay = self.calculate_backoff(attempt)
                logger.debug(
                    "Retry attempt %d after %.2fs: %s",
                    attempt + 1,
                    delay,
                    type(e).__name__,
                )
                await asyncio.sleep(delay)

        # Should not reach here, but safety raise
        raise last_exception  # type: ignore


@dataclass
class BackpressureSignal:
    """Signal for backpressure to callers."""

    # Should caller back off?
    should_backoff: bool
    # Suggested wait time in seconds
    retry_after_seconds: float
    # Current load factor (0-1)
    load_factor: float
    # Queue depth
    queue_depth: int


class BackpressureController:
    """
    Backpressure controller for graceful degradation.

    Provides signals to callers when system is overloaded:
    - Retry-After hints for 429 responses
    - Load-based rejection thresholds
    - Queue depth monitoring

    Integrates with admission controller for coordinated load shedding.
    """

    def __init__(
        self,
        warning_threshold: float = 0.7,
        critical_threshold: float = 0.9,
        base_retry_seconds: float = 1.0,
        max_retry_seconds: float = 60.0,
    ) -> None:
        self._warning_threshold = warning_threshold
        self._critical_threshold = critical_threshold
        self._base_retry = base_retry_seconds
        self._max_retry = max_retry_seconds

    def calculate_signal(
        self, load_factor: float, queue_depth: int
    ) -> BackpressureSignal:
        """
        Calculate backpressure signal based on current load.

        Args:
            load_factor: Current system load (0-1)
            queue_depth: Current queue depth

        Returns:
            BackpressureSignal with retry guidance
        """
        if load_factor < self._warning_threshold:
            return BackpressureSignal(
                should_backoff=False,
                retry_after_seconds=0,
                load_factor=load_factor,
                queue_depth=queue_depth,
            )

        # Calculate retry delay proportional to load
        if load_factor >= self._critical_threshold:
            # Critical: maximum backoff
            retry_after = self._max_retry
        else:
            # Warning zone: scale linearly
            severity = (load_factor - self._warning_threshold) / (
                self._critical_threshold - self._warning_threshold
            )
            retry_after = (
                self._base_retry + (self._max_retry - self._base_retry) * severity
            )

        return BackpressureSignal(
            should_backoff=True,
            retry_after_seconds=retry_after,
            load_factor=load_factor,
            queue_depth=queue_depth,
        )
