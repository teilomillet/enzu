"""
Job management for async delegation mode.

Provides in-memory job tracking for long-running tasks:
- submit_job(): Queue a task for background execution
- get_job_status(): Check job progress
- cancel_job(): Stop a running job

Jobs execute in background threads with proper budget enforcement.
Results are stored in memory and available for 1 hour after completion.

Usage:
    from enzu import Enzu

    client = Enzu()
    job = client.submit("Analyze this", data=large_doc, cost=5.0)

    # Poll for completion
    while job.status in (JobStatus.PENDING, JobStatus.RUNNING):
        time.sleep(1)
        job = client.status(job.job_id)

    print(job.answer)
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from enzu.models import Job, JobStatus, BudgetUsage, Outcome


# =============================================================================
# In-memory job store (production would use Redis/DB)
# =============================================================================


@dataclass
class JobState:
    """Internal job state with execution details."""

    job_id: str
    status: JobStatus = JobStatus.PENDING
    outcome: Optional[Outcome] = None
    partial: bool = False

    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    task: str = ""
    data: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    api_key: Optional[str] = None
    cost: Optional[float] = None
    tokens: Optional[int] = None
    seconds: Optional[float] = None

    answer: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[BudgetUsage] = None

    events: List[Dict[str, Any]] = field(default_factory=list)
    _cancel_requested: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def to_job(self) -> Job:
        """Convert internal state to public Job model."""
        return Job(
            job_id=self.job_id,
            status=self.status,
            outcome=self.outcome,
            partial=self.partial,
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            answer=self.answer,
            error=self.error,
            usage=self.usage,
            event_count=len(self.events),
        )


# Global job store (in-memory, single-process)
_jobs: Dict[str, JobState] = {}
_jobs_lock = threading.Lock()

# Background executor for job execution
_executor: Optional[ThreadPoolExecutor] = None


def _get_executor() -> ThreadPoolExecutor:
    """Get or create the background executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="enzu-job-")
    return _executor


def _generate_job_id() -> str:
    """Generate a unique job ID."""
    return f"job-{uuid.uuid4().hex[:12]}"


# =============================================================================
# Job execution
# =============================================================================


def _execute_job(job_id: str) -> None:
    """Execute a job in background thread."""
    with _jobs_lock:
        state = _jobs.get(job_id)
        if not state:
            return

    with state._lock:
        if state._cancel_requested:
            state.status = JobStatus.CANCELLED
            state.completed_at = time.time()
            return
        state.status = JobStatus.RUNNING
        state.started_at = time.time()

    try:
        from enzu.api import run as enzu_run
        from enzu.models import ExecutionReport, RLMExecutionReport

        result = enzu_run(
            state.task,
            model=state.model,
            provider=state.provider,
            api_key=state.api_key,
            data=state.data,
            cost=state.cost,
            tokens=state.tokens,
            seconds=state.seconds,
            return_report=True,
        )

        with state._lock:
            if state._cancel_requested:
                state.status = JobStatus.CANCELLED
                state.completed_at = time.time()
                return

            state.status = JobStatus.COMPLETED
            state.completed_at = time.time()

            if isinstance(result, RLMExecutionReport):
                state.answer = result.answer
                state.outcome = result.outcome
                state.partial = result.partial
                state.usage = result.budget_usage
            elif isinstance(result, ExecutionReport):
                state.answer = result.output_text
                state.outcome = result.outcome
                state.partial = result.partial
                state.usage = result.budget_usage
            else:
                state.answer = str(result) if result else None

    except Exception as e:
        with state._lock:
            state.status = JobStatus.FAILED
            state.completed_at = time.time()
            state.error = str(e)
            state.outcome = Outcome.PROVIDER_ERROR


# =============================================================================
# Public API
# =============================================================================


def submit_job(
    task: str,
    *,
    data: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    cost: Optional[float] = None,
    tokens: Optional[int] = None,
    seconds: Optional[float] = None,
) -> Job:
    """
    Submit a task for background execution.

    Returns immediately with a Job object containing the job_id.
    Use get_job_status() to poll for completion.
    """
    job_id = _generate_job_id()
    state = JobState(
        job_id=job_id,
        task=task,
        data=data,
        model=model,
        provider=provider,
        api_key=api_key,
        cost=cost,
        tokens=tokens,
        seconds=seconds,
    )

    with _jobs_lock:
        _jobs[job_id] = state

    _get_executor().submit(_execute_job, job_id)

    return state.to_job()


def get_job_status(job_id: str) -> Job:
    """
    Get the current status of a job.

    Raises KeyError if job not found.
    """
    with _jobs_lock:
        state = _jobs.get(job_id)
        if not state:
            raise KeyError(f"Job not found: {job_id}")
        return state.to_job()


def cancel_job(job_id: str) -> Job:
    """
    Cancel a running or pending job.

    Sets cancel flag; actual cancellation happens at next check point.
    Raises KeyError if job not found.
    """
    with _jobs_lock:
        state = _jobs.get(job_id)
        if not state:
            raise KeyError(f"Job not found: {job_id}")

    with state._lock:
        if state.status in (JobStatus.PENDING, JobStatus.RUNNING):
            state._cancel_requested = True
            if state.status == JobStatus.PENDING:
                state.status = JobStatus.CANCELLED
                state.completed_at = time.time()

    return state.to_job()


def list_jobs(
    status: Optional[JobStatus] = None,
    limit: int = 100,
) -> List[Job]:
    """List jobs, optionally filtered by status."""
    with _jobs_lock:
        jobs = list(_jobs.values())

    if status:
        jobs = [j for j in jobs if j.status == status]

    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return [j.to_job() for j in jobs[:limit]]


def cleanup_old_jobs(max_age_seconds: float = 3600) -> int:
    """Remove completed jobs older than max_age_seconds. Returns count removed."""
    cutoff = time.time() - max_age_seconds
    removed = 0

    with _jobs_lock:
        to_remove = [
            job_id
            for job_id, state in _jobs.items()
            if state.completed_at and state.completed_at < cutoff
        ]
        for job_id in to_remove:
            del _jobs[job_id]
            removed += 1

    return removed
