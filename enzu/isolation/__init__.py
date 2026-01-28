"""
Isolation module for subprocess/container-based sandbox execution and distributed scheduling.

Provides:
- OS-level isolation for code execution with resource limits (Phase 1)
- Container-based isolation with seccomp profiles (Phase 4)
- Global concurrency control for LLM API calls
- Distributed scheduler for 10K+ concurrent requests (Phase 3)
- Audit logging for government deployments
- Production hardening: health checks, circuit breakers, metrics (Phase 5)

Isolation hierarchy (weakest to strongest):
1. subprocess + resource.setrlimit - basic isolation, fastest
2. Docker + seccomp - gov-grade isolation, production ready
3. gVisor (future) - defense-in-depth
4. Firecracker microVM (future) - highest sensitivity

"""
from enzu.isolation.runner import (
    SandboxRunner,
    SandboxConfig,
    SandboxResult,
    IsolatedSandbox,
)
from enzu.isolation.concurrency import (
    ConcurrencyLimiter,
    get_global_limiter,
    configure_global_limiter,
)
from enzu.isolation.scheduler import (
    DistributedCoordinator,
    LocalWorkerPool,
    NodeCapacity,
    NodeStatus,
    SchedulerStats,
    SubmitResult,
    AdmissionController,
    ProductionConfig,
    get_coordinator,
    configure_coordinator,
)
from enzu.isolation.container import (
    ContainerSandboxRunner,
    ContainerSandbox,
    ContainerConfig,
    IsolationLevel,
    is_container_available,
)
from enzu.isolation.runtime import (
    ContainerRuntime,
    detect_runtime,
    get_runtime_command,
)
from enzu.isolation.container_wrapper import Container
from enzu.isolation.pool import (
    ContainerPool,
    PoolConfig,
)
from enzu.isolation.audit import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
    get_audit_logger,
    configure_audit_logger,
)
# Phase 5: Health checking and circuit breakers
from enzu.isolation.health import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerStats,
    CircuitBreakerOpen,
    CircuitState,
    HealthChecker,
    HealthCheckerConfig,
    HealthCheckResult,
    RetryStrategy,
    RetryConfig,
    BackpressureController,
    BackpressureSignal,
)
# Phase 5: Metrics collection
from enzu.isolation.metrics import (
    MetricsCollector,
    MetricSnapshot,
    get_metrics_collector,
    configure_metrics_collector,
)

__all__ = [
    # Subprocess isolation (Phase 1)
    "SandboxRunner",
    "SandboxConfig",
    "SandboxResult",
    "IsolatedSandbox",
    # Container isolation (Phase 4)
    "ContainerSandboxRunner",
    "ContainerSandbox",
    "ContainerConfig",
    "IsolationLevel",
    "is_container_available",
    # Container Pool (Phase 4 New)
    "ContainerRuntime",
    "detect_runtime",
    "get_runtime_command",
    "Container",
    "ContainerPool",
    "PoolConfig",
    # Concurrency control
    "ConcurrencyLimiter",
    "get_global_limiter",
    "configure_global_limiter",
    # Distributed scheduling (Phase 3)
    "DistributedCoordinator",
    "LocalWorkerPool",
    "NodeCapacity",
    "NodeStatus",
    "SchedulerStats",
    "SubmitResult",
    "AdmissionController",
    "ProductionConfig",
    "get_coordinator",
    "configure_coordinator",
    # Audit logging (Phase 4)
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
    "get_audit_logger",
    "configure_audit_logger",
    # Health checking and circuit breakers (Phase 5)
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerStats",
    "CircuitBreakerOpen",
    "CircuitState",
    "HealthChecker",
    "HealthCheckerConfig",
    "HealthCheckResult",
    "RetryStrategy",
    "RetryConfig",
    "BackpressureController",
    "BackpressureSignal",
    # Metrics collection (Phase 5)
    "MetricsCollector",
    "MetricSnapshot",
    "get_metrics_collector",
    "configure_metrics_collector",
]
