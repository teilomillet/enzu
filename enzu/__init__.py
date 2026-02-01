"""
enzu - Run LLM tasks with budgets, not open-ended conversations.

Module-level API (recommended):
    import enzu

    enzu.ask("What is 2+2?")
    enzu.run("Analyze this", data=logs)

    for chunk in enzu.stream("Write a story"):
        print(chunk, end="", flush=True)

Factory for configured instances:
    from enzu import new

    client = new("gpt-4o")
    client.run("Hello")

    # Multi-model setup
    gpt = new("gpt-4o", provider="openai")
    claude = new("claude-3-5-sonnet", provider="anthropic")

Multi-turn sessions:
    import enzu

    chat = enzu.session(max_cost_usd=5.00)
    chat.run("Find the bug", data=logs)
    chat.run("Fix it")

Advanced usage via submodules:
    from enzu.models import Budget, TaskSpec, SuccessCriteria
    from enzu.isolation import SandboxRunner, ContainerConfig
    from enzu.queue import TaskQueue, start_queue
"""

# =============================================================================
# Core API - What most users need
# =============================================================================
from enzu.client import Enzu, ask, new, stream, session, analyze  # noqa: F401
from enzu.api import resolve_provider, run  # noqa: F401
from enzu.session import Session, SessionBudgetExceeded  # noqa: F401
from enzu.rlm import RLMEngine  # noqa: F401
from enzu.engine import Engine  # noqa: F401

# =============================================================================
# Job API - Async delegation mode
# =============================================================================
from enzu.jobs import (  # noqa: F401
    submit_job as submit,
    get_job_status as status,
    cancel_job as cancel,
    list_jobs,
    cleanup_old_jobs,
)

# =============================================================================
# Essential data types - Commonly used in run() parameters
# =============================================================================
from enzu.models import (  # noqa: F401
    Budget,
    Check,
    ExecutionReport,
    Job,
    JobStatus,
    Limits,
    Outcome,
    RLMExecutionReport,
    SuccessCriteria,
)

# =============================================================================
# Typed exceptions - For structured error handling
# =============================================================================
from enzu.exceptions import (
    EnzuError,
    EnzuConfigError,
    EnzuProviderError,
    EnzuSandboxError,
    EnzuBudgetError,
    EnzuToolError,
)

# =============================================================================
# Provider registration - For custom providers
# =============================================================================
from enzu.providers.registry import list_providers, register_provider  # noqa: F401

# =============================================================================
# Everything below remains accessible but is not in __all__.
# Users who need these import from submodules directly.
# =============================================================================

# Models (full set available via `from enzu.models import ...`)
from enzu.models import (  # noqa: F401
    BudgetUsage,
    ProgressEvent,
    ProviderResult,
    RLMStep,
    TaskSpec,
    TrajectoryStep,
    VerificationResult,
)

# Contract helpers
from enzu.contract import (  # noqa: F401
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MIN_WORD_COUNT,
    apply_task_defaults,
    has_budget_limit,
    has_success_check,
    task_spec_from_payload,
)

# Token terminology documentation (import for docstrings)
from enzu import terminology  # noqa: F401

# Provider internals
from enzu.providers.base import BaseProvider  # noqa: F401
from enzu.providers.openai_compat import OpenAICompatProvider  # noqa: F401
from enzu.providers.pool import (  # noqa: F401
    CapacityExceededError,
    set_capacity_limit,
    get_capacity_stats,
    close_all_providers,
)

# Queue system
from enzu.queue import (  # noqa: F401
    TaskQueue,
    QueueStats,
    start_queue,
    stop_queue,
    submit as queue_submit,
    get_queue,
)

# Schema exports
from enzu.openresponses import openresponses_openapi_schema  # noqa: F401
from enzu.schema import (  # noqa: F401
    report_schema,
    run_payload_schema,
    schema_bundle,
    task_input_schema,
    task_spec_schema,
)

# Isolation and production infrastructure
from enzu.isolation import (  # noqa: F401
    # Subprocess isolation
    SandboxRunner,
    SandboxConfig,
    IsolatedSandbox,
    # Container isolation
    ContainerSandboxRunner,
    ContainerSandbox,
    ContainerConfig,
    IsolationLevel,
    is_container_available,
    # Concurrency control
    ConcurrencyLimiter,
    get_global_limiter,
    configure_global_limiter,
    # Distributed scheduling
    DistributedCoordinator,
    LocalWorkerPool,
    NodeCapacity,
    AdmissionController,
    ProductionConfig,
    get_coordinator,
    configure_coordinator,
    # Audit logging
    AuditLogger,
    get_audit_logger,
    configure_audit_logger,
    # Health and resilience
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitState,
    HealthChecker,
    HealthCheckerConfig,
    HealthCheckResult,
    RetryStrategy,
    RetryConfig,
    BackpressureController,
    BackpressureSignal,
    # Metrics
    MetricsCollector,
    MetricSnapshot,
    get_metrics_collector,
    configure_metrics_collector,
)

# Run metrics - p50/p95 cost/run and terminal state distributions
from enzu.metrics import (  # noqa: F401
    RunEvent,
    RunMetricsCollector,
    get_run_metrics,
    reset_run_metrics,
)

# Retry tracking - per-run retry visibility
from enzu.retries import (  # noqa: F401
    RetryReason,
    RetryTracker,
    get_retry_tracker,
    retry_tracking_context,
)

# Sandbox image building and lifecycle management
from enzu.sandbox import (  # noqa: F401
    # Image builder DSL (Phase 1)
    SandboxImage,
    BuiltImage,
    # Lifecycle management (Phase 3)
    LifecycleConfig,
    LifecycleManager,
    ManagedSandbox,
    get_lifecycle_manager,
)


# =============================================================================
# Lazy imports for optional tools - fail at use-time, not import-time
# =============================================================================
def __getattr__(name: str):
    """Lazy import handler for optional modules."""
    # Exa search tools - only load when accessed
    _EXA_TOOLS = {
        "ExaClient",
        "exa_search",
        "exa_news",
        "exa_papers",
        "exa_contents",
        "exa_similar",
    }
    if name in _EXA_TOOLS:
        from enzu.tools import exa as _exa_module

        return getattr(_exa_module, name)

    # Docling document parsing tools - only load when accessed
    _DOCLING_TOOLS = {
        "parse_document",
        "parse_documents",
        "documents_available",
    }
    if name in _DOCLING_TOOLS:
        from enzu.tools import docling_parser as _docling_module

        return getattr(_docling_module, name)

    raise AttributeError(f"module 'enzu' has no attribute {name!r}")


# =============================================================================
# Public API - Only these appear in `from enzu import *` and IDE autocomplete
# =============================================================================
__all__ = [
    # Module-level API (primary interface)
    "ask",
    "run",
    "stream",
    "session",
    "analyze",
    # Job API (async delegation)
    "submit",
    "status",
    "cancel",
    "Job",
    "JobStatus",
    # Factory function
    "new",
    # Legacy class (backward compatibility)
    "Enzu",
    # Session
    "Session",
    "SessionBudgetExceeded",
    # Typed outcome
    "Outcome",
    # Typed exceptions
    "EnzuError",
    "EnzuConfigError",
    "EnzuProviderError",
    "EnzuSandboxError",
    "EnzuBudgetError",
    "EnzuToolError",
]

# Exa search tools are available via lazy import (enzu.exa_search, etc.)
# They are not in __all__ since they require EXA_API_KEY to be configured
