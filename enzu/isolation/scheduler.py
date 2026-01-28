"""
Distributed scheduler for 10K+ concurrent requests.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     COORDINATOR SERVICE                          │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
    │  │ Request Router  │  │ Node Registry   │  │ Admission Ctrl  │ │
    │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
    └───────────┼────────────────────┼────────────────────┼───────────┘
                │                    │                    │
    ┌───────────┼────────────────────┼────────────────────┼───────────┐
    │           ▼                    ▼                    ▼           │
    │  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐     │
    │  │   Node 1    │      │   Node 2    │      │   Node N    │     │
    │  │ workers: 50 │      │ workers: 50 │      │ workers: 50 │     │
    │  │ queue: 200  │      │ queue: 200  │      │ queue: 200  │     │
    │  └─────────────┘      └─────────────┘      └─────────────┘     │
    │                      WORKER NODES                               │
    └─────────────────────────────────────────────────────────────────┘

Capacity calculation:
    - 10K concurrent users
    - ~5% actively executing (most waiting on LLM I/O)
    - 500 active sandboxes needed
    - 10 nodes × 50 workers = 500 capacity
    - 10K - 500 = 9.5K queued across nodes

Usage (single coordinator, multiple workers):

    # On coordinator (e.g., API gateway)
    coordinator = DistributedCoordinator()
    coordinator.register_node("node-1", "http://node1:8000", capacity=50)
    coordinator.register_node("node-2", "http://node2:8000", capacity=50)
    
    result = await coordinator.submit(task_spec, data="...")
    
    # On worker nodes
    worker = WorkerNode(node_id="node-1", max_workers=50)
    await worker.start()

Protocol:
    - Coordinator tracks node capacity via heartbeats
    - Routes requests to least-loaded node
    - Nodes report: queued, active, total_capacity
    - Admission control: reject when total queue > threshold

Phase 5 Production Hardening:
    - Health checks: Proactive node monitoring via HealthChecker
    - Circuit breakers: Per-node failure protection
    - Retry logic: Exponential backoff for transient failures
    - Graceful degradation: Backpressure signaling with retry-after
    - Metrics: Prometheus-compatible observability

"""
from __future__ import annotations

import asyncio
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from enum import Enum
import random

# Phase 5: Import health and metrics modules for production hardening
# These are imported lazily to avoid circular imports; used when enabled
if TYPE_CHECKING:
    from enzu.isolation.health import (
        CircuitBreaker,
        HealthChecker,
        RetryStrategy,
        BackpressureController,
    )
    from enzu.isolation.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Health status of a worker node."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # responding but slow
    UNHEALTHY = "unhealthy"  # missed heartbeats
    DRAINING = "draining"  # not accepting new work


@dataclass
class NodeCapacity:
    """Capacity report from a worker node."""
    node_id: str
    endpoint: str
    max_workers: int
    active_workers: int
    queued: int
    max_queue: int
    status: NodeStatus = NodeStatus.HEALTHY
    last_heartbeat: float = field(default_factory=time.time)
    
    @property
    def available_slots(self) -> int:
        """How many more requests this node can accept."""
        queue_space = self.max_queue - self.queued
        return max(0, queue_space)
    
    @property
    def load_factor(self) -> float:
        """0.0 = empty, 1.0 = full."""
        if self.max_workers == 0:
            return 1.0
        return (self.active_workers + self.queued) / (self.max_workers + self.max_queue)
    
    def is_available(self) -> bool:
        """Can accept new requests."""
        return (
            self.status in (NodeStatus.HEALTHY, NodeStatus.DEGRADED)
            and self.available_slots > 0
        )


@dataclass
class SchedulerStats:
    """Coordinator statistics."""
    total_nodes: int
    healthy_nodes: int
    total_capacity: int  # sum of max_workers
    active_requests: int  # sum of active across nodes
    queued_requests: int  # sum of queued across nodes
    rejected_requests: int  # rejected due to capacity
    routed_requests: int  # successfully routed


@dataclass
class SubmitResult:
    """Result of submitting a request."""
    success: bool
    node_id: Optional[str] = None
    result: Any = None
    error: Optional[str] = None
    wait_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    # Phase 5: Backpressure signal for caller retry guidance
    retry_after_seconds: Optional[float] = None


@dataclass
class ProductionConfig:
    """
    Phase 5: Production hardening configuration.

    Controls health checks, circuit breakers, retry logic, and metrics.
    Disabled by default for backward compatibility.
    """
    # Enable Phase 5 production features
    enabled: bool = False

    # Health checking
    health_check_interval_seconds: float = 5.0
    health_check_timeout_seconds: float = 3.0
    unhealthy_threshold: int = 3  # Consecutive failures to mark unhealthy
    healthy_threshold: int = 2  # Consecutive successes to mark healthy

    # Circuit breaker
    circuit_failure_threshold: int = 5  # Failures to trip circuit
    circuit_reset_timeout_seconds: float = 30.0  # Time before retry
    circuit_success_threshold: int = 2  # Successes to close circuit

    # Retry logic
    max_retries: int = 2  # Retry attempts for transient failures
    initial_backoff_seconds: float = 0.1
    max_backoff_seconds: float = 5.0
    backoff_multiplier: float = 2.0

    # Backpressure
    backpressure_warning_threshold: float = 0.7  # Start warning at 70% load
    backpressure_critical_threshold: float = 0.9  # Critical at 90% load

    # Metrics
    enable_metrics: bool = True


class AdmissionController:
    """
    Controls admission of new requests based on system capacity.
    
    Policies:
    - REJECT_WHEN_FULL: Return 429 immediately when capacity exceeded
    - QUEUE_WITH_TIMEOUT: Queue and wait up to timeout, then reject
    - ALWAYS_ACCEPT: Never reject (may cause unbounded queue growth)
    """
    
    def __init__(
        self,
        max_queue_depth: int = 10000,
        queue_timeout_seconds: float = 30.0,
        rejection_threshold: float = 0.95,  # reject when 95% capacity
    ) -> None:
        self._max_queue_depth = max_queue_depth
        self._queue_timeout = queue_timeout_seconds
        self._rejection_threshold = rejection_threshold
        self._rejected_count = 0
        self._lock = threading.Lock()
    
    def should_admit(self, current_load: float, current_queue: int) -> bool:
        """
        Check if a new request should be admitted.
        
        Args:
            current_load: System load factor (0.0 to 1.0)
            current_queue: Total queued requests across all nodes
        
        Returns:
            True if request should be admitted, False to reject
        """
        if current_queue >= self._max_queue_depth:
            with self._lock:
                self._rejected_count += 1
            return False
        
        if current_load >= self._rejection_threshold:
            with self._lock:
                self._rejected_count += 1
            return False
        
        return True
    
    @property
    def rejected_count(self) -> int:
        with self._lock:
            return self._rejected_count


class DistributedCoordinator:
    """
    Coordinates work distribution across multiple worker nodes.

    Responsibilities:
    - Track node capacity via registration/heartbeats
    - Route requests to least-loaded node
    - Implement admission control
    - Handle node failures gracefully

    Phase 5 Production Hardening (when enabled):
    - Health checks: Proactive node monitoring
    - Circuit breakers: Per-node failure protection
    - Retry logic: Exponential backoff for transient failures
    - Graceful degradation: Backpressure signaling

    This is designed to run in-process (not a separate service) for simplicity.
    For production, consider extracting to a separate coordinator service.

    Example:
        coordinator = DistributedCoordinator(max_total_queue=10000)

        # Register nodes (could be local or remote)
        coordinator.register_node("node-1", capacity=50, queue_size=200)
        coordinator.register_node("node-2", capacity=50, queue_size=200)

        # Submit work
        result = await coordinator.submit(task, executor=my_executor_func)

        # Check stats
        stats = coordinator.stats()
        print(f"Active: {stats.active_requests}, Queued: {stats.queued_requests}")

    Example with Phase 5 production features:
        config = ProductionConfig(enabled=True, max_retries=3)
        coordinator = DistributedCoordinator(production_config=config)
        await coordinator.start_health_checks()  # Background monitoring
    """

    def __init__(
        self,
        max_total_queue: int = 10000,
        heartbeat_interval_seconds: float = 5.0,
        node_timeout_seconds: float = 30.0,
        admission_rejection_threshold: float = 0.95,
        production_config: Optional[ProductionConfig] = None,
    ) -> None:
        self._nodes: Dict[str, NodeCapacity] = {}
        self._nodes_lock = threading.Lock()

        self._admission = AdmissionController(
            max_queue_depth=max_total_queue,
            rejection_threshold=admission_rejection_threshold,
        )

        self._heartbeat_interval = heartbeat_interval_seconds
        self._node_timeout = node_timeout_seconds

        self._routed_count = 0
        self._stats_lock = threading.Lock()

        # Node executors (for local nodes)
        self._executors: Dict[str, Callable] = {}

        # Phase 5: Production hardening components
        self._production_config = production_config or ProductionConfig()
        self._circuit_breakers: Dict[str, "CircuitBreaker"] = {}
        self._health_checker: Optional["HealthChecker"] = None
        self._retry_strategy: Optional["RetryStrategy"] = None
        self._backpressure: Optional["BackpressureController"] = None
        self._metrics: Optional["MetricsCollector"] = None

        if self._production_config.enabled:
            self._init_production_components()

    def _init_production_components(self) -> None:
        """Initialize Phase 5 production hardening components."""
        from enzu.isolation.health import (
            HealthChecker,
            HealthCheckerConfig,
            RetryStrategy,
            RetryConfig,
            BackpressureController,
        )
        from enzu.isolation.metrics import get_metrics_collector

        cfg = self._production_config

        # Retry strategy for transient failures
        self._retry_strategy = RetryStrategy(RetryConfig(
            max_retries=cfg.max_retries,
            initial_backoff_seconds=cfg.initial_backoff_seconds,
            max_backoff_seconds=cfg.max_backoff_seconds,
            backoff_multiplier=cfg.backoff_multiplier,
        ))

        # Backpressure controller for graceful degradation
        self._backpressure = BackpressureController(
            warning_threshold=cfg.backpressure_warning_threshold,
            critical_threshold=cfg.backpressure_critical_threshold,
        )

        # Health checker with callbacks to update node status
        self._health_checker = HealthChecker(
            on_node_unhealthy=self._on_node_unhealthy,
            on_node_healthy=self._on_node_healthy,
            config=HealthCheckerConfig(
                check_interval_seconds=cfg.health_check_interval_seconds,
                check_timeout_seconds=cfg.health_check_timeout_seconds,
                unhealthy_threshold=cfg.unhealthy_threshold,
                healthy_threshold=cfg.healthy_threshold,
            ),
        )

        # Metrics collector
        if cfg.enable_metrics:
            self._metrics = get_metrics_collector()

        logger.info("Phase 5 production hardening enabled")

    def _get_or_create_circuit_breaker(self, node_id: str) -> "CircuitBreaker":
        """Get or create circuit breaker for a node."""
        if node_id not in self._circuit_breakers:
            from enzu.isolation.health import CircuitBreaker, CircuitBreakerConfig

            cfg = self._production_config
            self._circuit_breakers[node_id] = CircuitBreaker(
                node_id=node_id,
                config=CircuitBreakerConfig(
                    failure_threshold=cfg.circuit_failure_threshold,
                    reset_timeout_seconds=cfg.circuit_reset_timeout_seconds,
                    success_threshold=cfg.circuit_success_threshold,
                ),
            )
        return self._circuit_breakers[node_id]

    def _on_node_unhealthy(self, node_id: str, reason: str) -> None:
        """Callback when health checker marks node unhealthy."""
        with self._nodes_lock:
            if node_id in self._nodes:
                self._nodes[node_id].status = NodeStatus.UNHEALTHY
                logger.warning("Node %s marked UNHEALTHY: %s", node_id, reason)

        # Update metrics
        if self._metrics:
            self._metrics.set_circuit_breaker_state(node_id, "open")

        # Log audit event if audit logger available
        try:
            from enzu.isolation.audit import get_audit_logger
            get_audit_logger().log_node_unhealthy(node_id, reason)
        except Exception:
            pass  # Audit logging is optional

    def _on_node_healthy(self, node_id: str) -> None:
        """Callback when health checker marks node recovered."""
        with self._nodes_lock:
            if node_id in self._nodes:
                self._nodes[node_id].status = NodeStatus.HEALTHY
                logger.info("Node %s recovered to HEALTHY", node_id)

        # Reset circuit breaker
        if node_id in self._circuit_breakers:
            self._circuit_breakers[node_id].reset()

        # Update metrics
        if self._metrics:
            self._metrics.set_circuit_breaker_state(node_id, "closed")

    async def start_health_checks(self) -> None:
        """
        Start background health checking.

        Call after registering nodes. Health checks run until stop_health_checks().
        """
        if not self._production_config.enabled or not self._health_checker:
            logger.warning("Production features not enabled, skipping health checks")
            return

        # Register check functions for each node
        with self._nodes_lock:
            for node_id in self._nodes:
                self._register_health_check(node_id)

        await self._health_checker.start()

    async def stop_health_checks(self) -> None:
        """Stop background health checking."""
        if self._health_checker:
            await self._health_checker.stop()

    def _register_health_check(self, node_id: str) -> None:
        """Register health check function for a node."""
        if not self._health_checker:
            return

        def check_local_node() -> bool:
            """Health check for local node: verify it's in registry and responsive."""
            with self._nodes_lock:
                if node_id not in self._nodes:
                    return False
                node = self._nodes[node_id]
                # Check heartbeat freshness
                age = time.time() - node.last_heartbeat
                if age > self._node_timeout:
                    return False
                return node.status != NodeStatus.UNHEALTHY

        self._health_checker.register_node(node_id, check_local_node)

    def register_node(
        self,
        node_id: str,
        *,
        endpoint: str = "local",
        capacity: int = 50,
        queue_size: int = 200,
        executor: Optional[Callable] = None,
    ) -> None:
        """
        Register a worker node.
        
        Args:
            node_id: Unique node identifier
            endpoint: HTTP endpoint for remote nodes, "local" for in-process
            capacity: Max concurrent workers on this node
            queue_size: Max queued requests on this node
            executor: For local nodes, the function to execute tasks
        """
        with self._nodes_lock:
            self._nodes[node_id] = NodeCapacity(
                node_id=node_id,
                endpoint=endpoint,
                max_workers=capacity,
                active_workers=0,
                queued=0,
                max_queue=queue_size,
                status=NodeStatus.HEALTHY,
                last_heartbeat=time.time(),
            )
            if executor is not None:
                self._executors[node_id] = executor
        
        logger.info(
            "Registered node %s: capacity=%d, queue=%d, endpoint=%s",
            node_id, capacity, queue_size, endpoint
        )

        # Phase 5: Register with health checker and create circuit breaker
        if self._production_config.enabled:
            self._register_health_check(node_id)
            self._get_or_create_circuit_breaker(node_id)

            # Update metrics
            if self._metrics:
                self._metrics.set_queue_depth(node_id, 0)
                self._metrics.set_active_workers(node_id, 0)
                self._metrics.set_circuit_breaker_state(node_id, "closed")

    def unregister_node(self, node_id: str) -> bool:
        """Remove a node from the registry."""
        with self._nodes_lock:
            if node_id in self._nodes:
                del self._nodes[node_id]
                self._executors.pop(node_id, None)

                # Phase 5: Clean up health checker and circuit breaker
                if self._health_checker:
                    self._health_checker.unregister_node(node_id)
                self._circuit_breakers.pop(node_id, None)

                return True
            return False
    
    def update_node_capacity(
        self,
        node_id: str,
        active_workers: int,
        queued: int,
        status: NodeStatus = NodeStatus.HEALTHY,
    ) -> None:
        """
        Update node capacity (called by node heartbeats).

        Args:
            node_id: Node to update
            active_workers: Currently active worker count
            queued: Current queue depth
            status: Node health status
        """
        with self._nodes_lock:
            if node_id not in self._nodes:
                return
            node = self._nodes[node_id]
            node.active_workers = active_workers
            node.queued = queued
            node.status = status
            node.last_heartbeat = time.time()

        # Phase 5: Update metrics
        if self._metrics:
            self._metrics.set_queue_depth(node_id, queued)
            self._metrics.set_active_workers(node_id, active_workers)
    
    def _select_node(self) -> Optional[NodeCapacity]:
        """
        Select best node for next request.

        Strategy: Least-loaded with jitter to avoid thundering herd.
        Phase 5: Also checks circuit breaker state.
        """
        with self._nodes_lock:
            available = [n for n in self._nodes.values() if n.is_available()]

            # Phase 5: Filter by circuit breaker state
            if self._production_config.enabled and self._circuit_breakers:
                available = [
                    n for n in available
                    if n.node_id in self._circuit_breakers
                    and self._circuit_breakers[n.node_id].allow_request()
                ]

            if not available:
                return None

            # Sort by load factor, pick from top 3 randomly (jitter)
            available.sort(key=lambda n: n.load_factor)
            top_candidates = available[:min(3, len(available))]
            return random.choice(top_candidates)
    
    def _get_system_load(self) -> tuple[float, int]:
        """Get system-wide load factor and total queue depth."""
        with self._nodes_lock:
            total_capacity = sum(n.max_workers + n.max_queue for n in self._nodes.values())
            total_used = sum(n.active_workers + n.queued for n in self._nodes.values())
            total_queued = sum(n.queued for n in self._nodes.values())
            
            if total_capacity == 0:
                return 1.0, 0
            
            return total_used / total_capacity, total_queued
    
    async def submit(
        self,
        task: Any,
        *,
        timeout_seconds: float = 300.0,
    ) -> SubmitResult:
        """
        Submit a task for execution on a worker node.

        Args:
            task: Task to execute (passed to node executor)
            timeout_seconds: Max time to wait for result

        Returns:
            SubmitResult with success status and result/error

        Phase 5 features (when enabled):
            - Circuit breaker check before execution
            - Metrics recording for observability
            - Backpressure signaling with retry_after_seconds
        """
        start_time = time.time()

        # Admission control
        load, queue_depth = self._get_system_load()
        if not self._admission.should_admit(load, queue_depth):
            # Phase 5: Record metrics and calculate backpressure
            retry_after = None
            if self._production_config.enabled:
                if self._metrics:
                    self._metrics.record_admission(accepted=False)
                if self._backpressure:
                    signal = self._backpressure.calculate_signal(load, queue_depth)
                    retry_after = signal.retry_after_seconds

            return SubmitResult(
                success=False,
                error=f"Capacity exceeded: load={load:.1%}, queue={queue_depth}",
                retry_after_seconds=retry_after,
            )

        # Phase 5: Record accepted admission
        if self._metrics:
            self._metrics.record_admission(accepted=True)

        # Select node (Phase 5: filtered by circuit breaker)
        node = self._select_node()
        if node is None:
            return SubmitResult(
                success=False,
                error="No available nodes",
            )

        # Update queue count
        with self._nodes_lock:
            if node.node_id in self._nodes:
                self._nodes[node.node_id].queued += 1

        # Phase 5: Update queue metrics
        if self._metrics:
            self._metrics.set_queue_depth(
                node.node_id,
                self._nodes.get(node.node_id, NodeCapacity(
                    node_id=node.node_id,
                    endpoint="local",
                    max_workers=0,
                    active_workers=0,
                    queued=0,
                    max_queue=0,
                )).queued,
            )

        try:
            # Execute on node
            if node.endpoint == "local" and node.node_id in self._executors:
                # Local execution
                executor = self._executors[node.node_id]

                # Track active
                with self._nodes_lock:
                    if node.node_id in self._nodes:
                        self._nodes[node.node_id].queued -= 1
                        self._nodes[node.node_id].active_workers += 1

                exec_start = time.time()
                try:
                    if asyncio.iscoroutinefunction(executor):
                        result = await asyncio.wait_for(
                            executor(task),
                            timeout=timeout_seconds,
                        )
                    else:
                        # Run sync executor in thread pool
                        loop = asyncio.get_event_loop()
                        result = await asyncio.wait_for(
                            loop.run_in_executor(None, executor, task),
                            timeout=timeout_seconds,
                        )

                    exec_time = (time.time() - exec_start) * 1000

                    with self._stats_lock:
                        self._routed_count += 1

                    # Phase 5: Record success in circuit breaker and metrics
                    if self._production_config.enabled:
                        if node.node_id in self._circuit_breakers:
                            self._circuit_breakers[node.node_id].record_success()
                        if self._metrics:
                            self._metrics.record_request(
                                node_id=node.node_id,
                                duration_ms=exec_time,
                                success=True,
                            )

                    return SubmitResult(
                        success=True,
                        node_id=node.node_id,
                        result=result,
                        wait_time_ms=(exec_start - start_time) * 1000,
                        execution_time_ms=exec_time,
                    )

                finally:
                    with self._nodes_lock:
                        if node.node_id in self._nodes:
                            self._nodes[node.node_id].active_workers -= 1

                    # Phase 5: Update worker metrics
                    if self._metrics:
                        self._metrics.set_active_workers(
                            node.node_id,
                            self._nodes.get(node.node_id, NodeCapacity(
                                node_id=node.node_id,
                                endpoint="local",
                                max_workers=0,
                                active_workers=0,
                                queued=0,
                                max_queue=0,
                            )).active_workers,
                        )

            else:
                # Remote execution (not implemented in Phase 3)
                return SubmitResult(
                    success=False,
                    error=f"Remote execution not implemented: {node.endpoint}",
                )

        except asyncio.TimeoutError:
            # Phase 5: Record failure in circuit breaker and metrics
            exec_time = (time.time() - start_time) * 1000
            if self._production_config.enabled:
                if node.node_id in self._circuit_breakers:
                    self._circuit_breakers[node.node_id].record_failure()
                if self._metrics:
                    self._metrics.record_request(
                        node_id=node.node_id,
                        duration_ms=exec_time,
                        success=False,
                        error_type="timeout",
                    )

            return SubmitResult(
                success=False,
                node_id=node.node_id,
                error=f"Timeout after {timeout_seconds}s",
            )

        except Exception as e:
            # Phase 5: Record failure in circuit breaker and metrics
            exec_time = (time.time() - start_time) * 1000
            error_type = type(e).__name__
            if self._production_config.enabled:
                if node.node_id in self._circuit_breakers:
                    self._circuit_breakers[node.node_id].record_failure()
                if self._metrics:
                    self._metrics.record_request(
                        node_id=node.node_id,
                        duration_ms=exec_time,
                        success=False,
                        error_type=error_type,
                    )

            return SubmitResult(
                success=False,
                node_id=node.node_id,
                error=str(e),
            )
    
    def stats(self) -> SchedulerStats:
        """Get current scheduler statistics."""
        with self._nodes_lock:
            nodes = list(self._nodes.values())
        
        healthy = sum(1 for n in nodes if n.status == NodeStatus.HEALTHY)
        total_capacity = sum(n.max_workers for n in nodes)
        active = sum(n.active_workers for n in nodes)
        queued = sum(n.queued for n in nodes)
        
        return SchedulerStats(
            total_nodes=len(nodes),
            healthy_nodes=healthy,
            total_capacity=total_capacity,
            active_requests=active,
            queued_requests=queued,
            rejected_requests=self._admission.rejected_count,
            routed_requests=self._routed_count,
        )
    
    def node_capacities(self) -> List[NodeCapacity]:
        """Get capacity info for all nodes."""
        with self._nodes_lock:
            return list(self._nodes.values())


class LocalWorkerPool:
    """
    Local worker pool that integrates with the distributed coordinator.
    
    Wraps the existing TaskQueue with coordinator integration.
    
    Example:
        # Create worker pool
        pool = LocalWorkerPool(
            node_id="node-1",
            max_workers=50,
            max_queue=200,
        )
        
        # Register with coordinator
        coordinator.register_node(
            "node-1",
            capacity=50,
            queue_size=200,
            executor=pool.execute,
        )
        
        # Start processing
        await pool.start()
    """
    
    def __init__(
        self,
        node_id: str,
        *,
        provider: str = "openrouter",
        model: str,
        api_key: Optional[str] = None,
        max_workers: int = 50,
        max_queue: int = 200,
    ) -> None:
        self._node_id = node_id
        self._provider_name = provider
        self._model = model
        self._api_key = api_key
        self._max_workers = max_workers
        self._max_queue = max_queue
        
        # Task queue
        self._queue: Optional[asyncio.Queue[Any]] = None
        self._workers: List[asyncio.Task] = []
        self._running = False
        
        # Stats
        self._active = 0
        self._completed = 0
        self._failed = 0
        self._lock = asyncio.Lock()
    
    async def start(self) -> None:
        """Start the worker pool."""
        if self._running:
            return
        
        self._queue = asyncio.Queue(maxsize=self._max_queue)
        self._running = True
        
        # Spawn workers
        for _ in range(self._max_workers):
            worker = asyncio.create_task(self._worker_loop())
            self._workers.append(worker)
        
        logger.info(
            "LocalWorkerPool %s started: workers=%d, queue=%d",
            self._node_id, self._max_workers, self._max_queue
        )
    
    async def stop(self) -> None:
        """Stop the worker pool."""
        self._running = False
        
        # Cancel workers
        for worker in self._workers:
            worker.cancel()
        
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        
        self._workers.clear()
        self._queue = None
    
    async def execute(self, task: Dict[str, Any]) -> Any:
        """
        Execute a task. Called by coordinator.
        
        Args:
            task: Dict with 'input_text', 'model', 'data', etc.
        
        Returns:
            Execution result
        """
        if not self._running or self._queue is None:
            raise RuntimeError("Worker pool not started")
        
        # Create future for result
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        # Queue task
        await self._queue.put((task, future))
        
        # Wait for result
        return await future
    
    async def _worker_loop(self) -> None:
        """Worker coroutine."""
        from enzu.api import _resolve_provider, _build_task_spec
        from enzu.engine import Engine
        from enzu.rlm import RLMEngine
        
        provider = _resolve_provider(
            self._provider_name,
            api_key=self._api_key,
            use_pool=True,
        )
        
        while self._running:
            try:
                # Get task
                if self._queue is None:
                    break
                try:
                    task, future = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue
                
                async with self._lock:
                    self._active += 1
                
                try:
                    # Build spec
                    # _build_task_spec requires all parameters; provide defaults for missing ones
                    spec = _build_task_spec(
                        task.get("input_text", ""),
                        model=task.get("model", self._model),
                        tokens=task.get("max_output_tokens"),
                        seconds=task.get("seconds"),
                        cost=task.get("cost"),
                        contains=task.get("contains"),
                        matches=task.get("matches"),
                        min_words=task.get("min_words"),
                        goal=task.get("goal"),
                        limits=task.get("limits"),
                        check=task.get("check"),
                        responses=task.get("responses"),
                        temperature=task.get("temperature"),
                        is_rlm=(task.get("data") is not None),
                    )
                    
                    # Execute
                    data = task.get("data")
                    if data is not None:
                        rlm_engine = RLMEngine()
                        rlm_report = rlm_engine.run(spec, provider, data=data)
                        # RLMExecutionReport has 'answer' attribute, not 'output_text'
                        result = rlm_report.answer or ""
                    else:
                        chat_engine = Engine()
                        chat_report = chat_engine.run(spec, provider)
                        # ExecutionReport has 'output_text' attribute
                        result = chat_report.output_text or ""
                    
                    if not future.done():
                        future.set_result(result)
                    
                    async with self._lock:
                        self._completed += 1
                
                except Exception as e:
                    if not future.done():
                        future.set_exception(e)
                    
                    async with self._lock:
                        self._failed += 1
                
                finally:
                    # Mark task done (for queue.join())
                    if self._queue is not None:
                        self._queue.task_done()
                    async with self._lock:
                        self._active -= 1
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Worker error: %s", e)
    
    def capacity(self) -> NodeCapacity:
        """Get current capacity for heartbeat."""
        return NodeCapacity(
            node_id=self._node_id,
            endpoint="local",
            max_workers=self._max_workers,
            active_workers=self._active,
            queued=self._queue.qsize() if self._queue else 0,
            max_queue=self._max_queue,
            status=NodeStatus.HEALTHY if self._running else NodeStatus.UNHEALTHY,
        )


# Convenience functions for simple multi-node setup

_default_coordinator: Optional[DistributedCoordinator] = None


def get_coordinator() -> DistributedCoordinator:
    """Get or create the default coordinator."""
    global _default_coordinator
    if _default_coordinator is None:
        _default_coordinator = DistributedCoordinator()
    return _default_coordinator


def configure_coordinator(
    max_total_queue: int = 10000,
    rejection_threshold: float = 0.95,
) -> DistributedCoordinator:
    """Configure the default coordinator."""
    global _default_coordinator
    _default_coordinator = DistributedCoordinator(
        max_total_queue=max_total_queue,
        admission_rejection_threshold=rejection_threshold,
    )
    return _default_coordinator
