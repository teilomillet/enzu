"""
Ray-inspired distributed runtime for horizontal scaling.

No external dependencies (Redis, etc.) - workers ARE the runtime.

SECURITY:
    - Remote workers require a shared secret (ENZU_WORKER_SECRET)
    - API keys are NOT sent to workers; workers use their own credentials
    - Use TLS in production (https:// endpoints or reverse proxy)
    - Workers should run in isolated networks when possible

Usage:
    from enzu import run
    from enzu.runtime import DistributedRuntime

    # Local scaling: 8 workers, 4 concurrent tasks each
    runtime = DistributedRuntime(num_workers=8, max_per_worker=4)
    result = run("Analyze this", data=logs, model="gpt-4o", runtime=runtime)

    # Async submission
    future = runtime.submit(spec=..., provider=..., data=..., options=...)
    result = future.result()

    # Remote scaling: workers on other machines
    # Set ENZU_WORKER_SECRET on both coordinator and workers
    runtime = DistributedRuntime.from_endpoints([
        "https://worker1:8080",
        "https://worker2:8080",
    ])
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Union

from enzu.models import RLMExecutionReport, TaskSpec
from enzu.runtime.local import LocalRuntime
from enzu.runtime.protocol import ProviderSpec, RuntimeOptions

logger = logging.getLogger(__name__)


class WorkerProtocol(Protocol):
    """Interface for a worker that can execute RLM tasks."""

    def run(
        self,
        spec: TaskSpec,
        provider: ProviderSpec,
        data: str,
        options: RuntimeOptions,
    ) -> RLMExecutionReport: ...


@dataclass
class WorkerStats:
    """Tracks worker load for scheduling decisions."""

    active: int = 0
    completed: int = 0
    failed: int = 0
    total_seconds: float = 0.0

    @property
    def avg_duration(self) -> float:
        if self.completed == 0:
            return 0.0
        return self.total_seconds / self.completed


@dataclass
class LocalWorker:
    """In-process worker backed by LocalRuntime."""

    runtime: LocalRuntime = field(default_factory=LocalRuntime)
    max_concurrent: int = 4
    stats: WorkerStats = field(default_factory=WorkerStats)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def run(
        self,
        spec: TaskSpec,
        provider: ProviderSpec,
        data: str,
        options: RuntimeOptions,
    ) -> RLMExecutionReport:
        start = time.monotonic()
        with self._lock:
            self.stats.active += 1
        try:
            result = self.runtime.run(
                spec=spec, provider=provider, data=data, options=options
            )
            with self._lock:
                self.stats.completed += 1
                self.stats.total_seconds += time.monotonic() - start
            return result
        except Exception:
            with self._lock:
                self.stats.failed += 1
            raise
        finally:
            with self._lock:
                self.stats.active -= 1

    @property
    def available_capacity(self) -> int:
        return max(0, self.max_concurrent - self.stats.active)

    @property
    def is_available(self) -> bool:
        return self.stats.active < self.max_concurrent


@dataclass
class RemoteWorker:
    """
    Worker on another machine, accessed via HTTP.

    Security:
        - Uses shared secret authentication (ENZU_WORKER_SECRET env var)
        - Does NOT send API keys; workers use their own credentials
        - Use https:// endpoints in production

    Streaming:
        - Use run() for blocking execution (no progress updates)
        - Use run_streaming() for SSE-based progress updates
    """

    endpoint: str
    max_concurrent: int = 4
    stats: WorkerStats = field(default_factory=WorkerStats)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _timeout: float = 600.0  # 10 min default for long RLM tasks
    _secret: Optional[str] = field(default=None)
    _stream: bool = False  # Enable SSE streaming by default

    def __post_init__(self):
        # Load shared secret from environment if not provided
        if self._secret is None:
            self._secret = os.getenv("ENZU_WORKER_SECRET")
        # Warn if using http:// without secret
        if self.endpoint.startswith("http://") and not self._secret:
            logger.warning(
                "RemoteWorker using HTTP without authentication. "
                "Set ENZU_WORKER_SECRET for security."
            )

    def _get_headers(self) -> dict:
        """Build request headers with authentication."""
        headers = {}
        if self._secret:
            headers["Authorization"] = f"Bearer {self._secret}"
        return headers

    def run(
        self,
        spec: TaskSpec,
        provider: ProviderSpec,
        data: str,
        options: RuntimeOptions,
    ) -> RLMExecutionReport:
        """Execute task. Uses streaming if on_progress is set, else blocking."""
        if options.on_progress is not None or self._stream:
            return self.run_streaming(spec, provider, data, options)
        return self._run_blocking(spec, provider, data, options)

    def _run_blocking(
        self,
        spec: TaskSpec,
        provider: ProviderSpec,
        data: str,
        options: RuntimeOptions,
    ) -> RLMExecutionReport:
        """Blocking HTTP POST execution (no progress updates)."""
        import httpx

        start = time.monotonic()
        with self._lock:
            self.stats.active += 1
        try:
            # SECURITY: Don't send API keys to workers
            # Workers resolve providers using their own credentials
            payload = _serialize_task(spec, provider, data, options)
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    f"{self.endpoint}/run",
                    json=payload,
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                result = _deserialize_report(response.json())
            with self._lock:
                self.stats.completed += 1
                self.stats.total_seconds += time.monotonic() - start
            return result
        except Exception:
            with self._lock:
                self.stats.failed += 1
            raise
        finally:
            with self._lock:
                self.stats.active -= 1

    def run_streaming(
        self,
        spec: TaskSpec,
        provider: ProviderSpec,
        data: str,
        options: RuntimeOptions,
    ) -> RLMExecutionReport:
        """
        Execute task with SSE streaming for real-time progress updates.

        Connects to /run/stream endpoint and processes Server-Sent Events.
        Progress events are forwarded to options.on_progress callback.

        Event types:
            - progress: Step/subcall updates (forwarded to on_progress)
            - complete: Final result with RLMExecutionReport
            - error: Execution failure
        """
        import httpx
        import json

        start = time.monotonic()
        with self._lock:
            self.stats.active += 1

        on_progress = options.on_progress
        result: Optional[RLMExecutionReport] = None
        error_msg: Optional[str] = None

        try:
            payload = _serialize_task(spec, provider, data, options)
            headers = self._get_headers()
            headers["Accept"] = "text/event-stream"

            with httpx.Client(timeout=None) as client:
                with client.stream(
                    "POST",
                    f"{self.endpoint}/run/stream",
                    json=payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()

                    event_type: Optional[str] = None
                    data_buffer: list[str] = []

                    for line in response.iter_lines():
                        line = line.strip()
                        if not line:
                            # Empty line = end of event
                            if event_type and data_buffer:
                                event_data = "".join(data_buffer)
                                try:
                                    parsed = json.loads(event_data)
                                except json.JSONDecodeError:
                                    parsed = event_data

                                if event_type == "progress":
                                    if on_progress and isinstance(parsed, dict):
                                        msg = parsed.get("message", str(parsed))
                                        on_progress(msg)
                                elif event_type == "step":
                                    if on_progress and isinstance(parsed, dict):
                                        step = parsed.get("step", "?")
                                        status = parsed.get("status", "")
                                        on_progress(f"Step {step}: {status}")
                                elif event_type == "subcall":
                                    if on_progress and isinstance(parsed, dict):
                                        depth = parsed.get("depth", 0)
                                        phase = parsed.get("phase", "")
                                        on_progress(f"Subcall (depth={depth}): {phase}")
                                elif event_type == "complete":
                                    result = _deserialize_report(parsed)
                                elif event_type == "error":
                                    error_msg = (
                                        parsed.get("error", str(parsed))
                                        if isinstance(parsed, dict)
                                        else str(parsed)
                                    )

                            event_type = None
                            data_buffer = []
                        elif line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:"):
                            data_buffer.append(line[5:].strip())

            if error_msg:
                raise RuntimeError(f"Remote worker error: {error_msg}")
            if result is None:
                raise RuntimeError("Remote worker did not return a result")

            with self._lock:
                self.stats.completed += 1
                self.stats.total_seconds += time.monotonic() - start
            return result

        except Exception:
            with self._lock:
                self.stats.failed += 1
            raise
        finally:
            with self._lock:
                self.stats.active -= 1

    def health_check(self) -> bool:
        """Check if worker is healthy."""
        import httpx

        try:
            headers = {}
            if self._secret:
                headers["Authorization"] = f"Bearer {self._secret}"
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.endpoint}/health", headers=headers)
                return response.status_code == 200
        except Exception:
            return False

    @property
    def available_capacity(self) -> int:
        return max(0, self.max_concurrent - self.stats.active)

    @property
    def is_available(self) -> bool:
        return self.stats.active < self.max_concurrent


# Scheduling strategies
class SchedulerProtocol(Protocol):
    """Strategy for selecting a worker."""

    def select(
        self, workers: List[Union[LocalWorker, RemoteWorker]]
    ) -> Union[LocalWorker, RemoteWorker]: ...


class LeastLoadedScheduler:
    """Pick worker with most available capacity (Ray's default)."""

    def select(
        self, workers: List[Union[LocalWorker, RemoteWorker]]
    ) -> Union[LocalWorker, RemoteWorker]:
        available = [w for w in workers if w.is_available]
        if not available:
            # All saturated - pick least loaded anyway (will queue)
            return min(workers, key=lambda w: w.stats.active)
        return max(available, key=lambda w: w.available_capacity)


class RoundRobinScheduler:
    """Simple round-robin for even distribution."""

    def __init__(self) -> None:
        self._index = 0
        self._lock = threading.Lock()

    def select(
        self, workers: List[Union[LocalWorker, RemoteWorker]]
    ) -> Union[LocalWorker, RemoteWorker]:
        with self._lock:
            worker = workers[self._index % len(workers)]
            self._index += 1
            return worker


class AdaptiveScheduler:
    """Weighted selection based on worker performance."""

    def select(
        self, workers: List[Union[LocalWorker, RemoteWorker]]
    ) -> Union[LocalWorker, RemoteWorker]:
        available = [w for w in workers if w.is_available]
        if not available:
            return min(workers, key=lambda w: w.stats.active)

        # Score: capacity * (1 / avg_duration) - favor fast workers
        def score(w: Union[LocalWorker, RemoteWorker]) -> float:
            capacity_score = w.available_capacity
            speed_score = (
                1.0 / max(w.stats.avg_duration, 0.1) if w.stats.completed > 0 else 1.0
            )
            return capacity_score * speed_score

        return max(available, key=score)


@dataclass
class BudgetLimit:
    """Coordinator-level budget limits across all tasks."""

    max_cost_usd: Optional[float] = None
    max_tasks: Optional[int] = None

    def is_set(self) -> bool:
        return self.max_cost_usd is not None or self.max_tasks is not None


@dataclass
class BudgetState:
    """Tracks cumulative spend across all tasks."""

    total_cost_usd: float = 0.0
    total_tasks: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, cost_usd: Optional[float]) -> None:
        with self._lock:
            if cost_usd is not None:
                self.total_cost_usd += cost_usd
            self.total_tasks += 1

    def check(self, limit: BudgetLimit) -> Optional[str]:
        """Returns error message if budget exceeded, None if OK."""
        with self._lock:
            if (
                limit.max_cost_usd is not None
                and self.total_cost_usd >= limit.max_cost_usd
            ):
                return f"Budget exceeded: ${self.total_cost_usd:.4f} >= ${limit.max_cost_usd:.4f}"
            if limit.max_tasks is not None and self.total_tasks >= limit.max_tasks:
                return f"Task limit exceeded: {self.total_tasks} >= {limit.max_tasks}"
            return None

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "total_cost_usd": self.total_cost_usd,
                "total_tasks": self.total_tasks,
            }


class BudgetExceededError(Exception):
    """Raised when coordinator-level budget is exceeded."""

    pass


class DistributedRuntime:
    """
    Ray-inspired distributed runtime.

    Scales RLM execution horizontally across workers without external queues.
    Workers can be local (in-process) or remote (HTTP endpoints).

    Budget tracking:
        - Per-task: Enforced by workers (max_steps, cost limits in TaskSpec)
        - Cross-task: Enforced by coordinator (max_cost_usd, max_tasks in BudgetLimit)
    """

    def __init__(
        self,
        workers: Optional[List[Union[LocalWorker, RemoteWorker]]] = None,
        *,
        num_workers: int = 0,
        max_per_worker: int = 4,
        scheduler: Optional[SchedulerProtocol] = None,
        budget: Optional[BudgetLimit] = None,
    ):
        """
        Create a distributed runtime.

        Args:
            workers: Explicit list of workers (local or remote).
            num_workers: Create this many LocalWorkers (ignored if workers provided).
            max_per_worker: Max concurrent tasks per worker.
            scheduler: Scheduling strategy (default: LeastLoadedScheduler).
            budget: Coordinator-level budget limits (total cost, task count).
        """
        if workers:
            self._workers = workers
        elif num_workers > 0:
            self._workers = [
                LocalWorker(max_concurrent=max_per_worker) for _ in range(num_workers)
            ]
        else:
            # Default: one worker per CPU core
            cpu_count = os.cpu_count() or 4
            self._workers = [
                LocalWorker(max_concurrent=max_per_worker) for _ in range(cpu_count)
            ]

        self._scheduler = scheduler or LeastLoadedScheduler()
        self._executor = ThreadPoolExecutor(
            max_workers=sum(w.max_concurrent for w in self._workers)
        )
        self._lock = threading.Lock()
        self._shutdown = False
        self._budget_limit = budget or BudgetLimit()
        self._budget_state = BudgetState()

    @classmethod
    def from_endpoints(
        cls,
        endpoints: List[str],
        *,
        max_per_worker: int = 4,
        scheduler: Optional[SchedulerProtocol] = None,
    ) -> "DistributedRuntime":
        """Create runtime with remote workers."""
        workers = [
            RemoteWorker(endpoint=ep, max_concurrent=max_per_worker) for ep in endpoints
        ]
        return cls(workers=workers, scheduler=scheduler)

    @classmethod
    def auto(cls, *, max_per_worker: int = 4) -> "DistributedRuntime":
        """Auto-configure based on environment."""
        # Check for worker endpoints in env
        endpoints_env = os.getenv("ENZU_WORKER_ENDPOINTS")
        if endpoints_env:
            endpoints = [ep.strip() for ep in endpoints_env.split(",") if ep.strip()]
            return cls.from_endpoints(endpoints, max_per_worker=max_per_worker)
        # Default to local workers
        return cls(max_per_worker=max_per_worker)

    def run(
        self,
        *,
        spec: TaskSpec,
        provider: ProviderSpec,
        data: str,
        options: RuntimeOptions,
    ) -> RLMExecutionReport:
        """Execute synchronously on a worker."""
        if self._shutdown:
            raise RuntimeError("Runtime has been shut down")

        # Check budget before starting
        if self._budget_limit.is_set():
            error = self._budget_state.check(self._budget_limit)
            if error:
                raise BudgetExceededError(error)

        worker = self._scheduler.select(self._workers)
        result = worker.run(spec, provider, data, options)

        # Track spend after completion
        cost = result.budget_usage.cost_usd if result.budget_usage else None
        self._budget_state.record(cost)

        return result

    def submit(
        self,
        *,
        spec: TaskSpec,
        provider: ProviderSpec,
        data: str,
        options: RuntimeOptions,
    ) -> Future[RLMExecutionReport]:
        """Submit task asynchronously, returns Future."""
        if self._shutdown:
            raise RuntimeError("Runtime has been shut down")

        # Check budget before starting
        if self._budget_limit.is_set():
            error = self._budget_state.check(self._budget_limit)
            if error:
                raise BudgetExceededError(error)

        worker = self._scheduler.select(self._workers)

        def run_and_track() -> RLMExecutionReport:
            result = worker.run(spec, provider, data, options)
            cost = result.budget_usage.cost_usd if result.budget_usage else None
            self._budget_state.record(cost)
            return result

        return self._executor.submit(run_and_track)

    def map(
        self,
        specs: List[TaskSpec],
        provider: ProviderSpec,
        data: str,
        options: RuntimeOptions,
        *,
        on_complete: Optional[Callable[[int, RLMExecutionReport], None]] = None,
    ) -> List[RLMExecutionReport]:
        """Execute multiple tasks in parallel, return all results."""
        futures = [
            self.submit(spec=spec, provider=provider, data=data, options=options)
            for spec in specs
        ]
        results: List[RLMExecutionReport] = []
        for i, future in enumerate(futures):
            result = future.result()
            if on_complete:
                on_complete(i, result)
            results.append(result)
        return results

    @property
    def stats(self) -> dict:
        """Aggregate stats across all workers."""
        total_active = sum(w.stats.active for w in self._workers)
        total_completed = sum(w.stats.completed for w in self._workers)
        total_failed = sum(w.stats.failed for w in self._workers)
        total_capacity = sum(w.max_concurrent for w in self._workers)
        return {
            "workers": len(self._workers),
            "capacity": total_capacity,
            "active": total_active,
            "completed": total_completed,
            "failed": total_failed,
            "utilization": total_active / total_capacity if total_capacity > 0 else 0.0,
            "budget": self._budget_state.snapshot(),
        }

    @property
    def budget(self) -> dict:
        """Current budget state."""
        return self._budget_state.snapshot()

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the runtime."""
        self._shutdown = True
        self._executor.shutdown(wait=wait)

    def __enter__(self) -> "DistributedRuntime":
        return self

    def __exit__(self, *args) -> None:
        self.shutdown(wait=True)


# Serialization helpers for remote workers
def _serialize_task(
    spec: TaskSpec,
    provider: ProviderSpec,
    data: str,
    options: RuntimeOptions,
) -> dict:
    """
    Serialize task for HTTP transport.

    SECURITY: API keys are NOT sent to workers.
    Workers must have their own credentials configured via environment variables.
    This prevents credential leakage over the network.
    """
    return {
        "spec": spec.model_dump(),
        "provider": {
            "name": provider.name,
            # SECURITY: Do NOT send api_key, referer, app_name, organization, project
            # Workers use their own credentials from environment variables
            "use_pool": provider.use_pool,
        },
        "data": data,
        "options": {
            "max_steps": options.max_steps,
            "verify_on_final": options.verify_on_final,
            "isolation": options.isolation,
            # sandbox/sandbox_factory not serializable - worker uses its own
        },
    }


def _deserialize_report(payload: dict) -> RLMExecutionReport:
    """Deserialize report from HTTP response."""
    return RLMExecutionReport.model_validate(payload)
