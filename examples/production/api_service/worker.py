"""Background job processor for async document analysis."""

import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

from .budget import BudgetController, get_budget_controller
from .models import JobResult, JobStatus


@dataclass
class Job:
    """Internal job representation."""

    job_id: str
    customer_id: str
    document: str
    task: str
    max_tokens: int
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    result: Optional[str] = None
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    error: Optional[str] = None


class JobQueue:
    """
    Thread-safe job queue with background worker.

    Features:
    - Async job submission
    - Background processing with worker threads
    - Job status tracking
    - Budget enforcement per customer
    """

    def __init__(
        self,
        num_workers: int = 2,
        budget_controller: Optional[BudgetController] = None,
    ):
        self._queue: queue.Queue[Job] = queue.Queue()
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.RLock()
        self._workers: list[threading.Thread] = []
        self._shutdown = threading.Event()
        self._budget = budget_controller or get_budget_controller()
        self._num_workers = num_workers
        self._processor: Optional[Callable] = None

        # Metrics
        self._jobs_completed = 0
        self._jobs_failed = 0
        self._total_tokens = 0
        self._total_cost = 0.0

    def set_processor(self, processor: Callable[[Job], tuple[str, int, float]]) -> None:
        """
        Set the job processor function.

        The processor should accept a Job and return (result, tokens_used, cost_usd).
        """
        self._processor = processor

    def start(self) -> None:
        """Start worker threads."""
        for i in range(self._num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"job-worker-{i}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop worker threads gracefully."""
        self._shutdown.set()
        for worker in self._workers:
            worker.join(timeout=timeout)
        self._workers.clear()

    def submit(
        self,
        customer_id: str,
        document: str,
        task: str,
        max_tokens: int = 500,
    ) -> Job:
        """
        Submit a job for processing.

        Args:
            customer_id: Customer submitting the job
            document: Document text to analyze
            task: Analysis task to perform
            max_tokens: Maximum output tokens

        Returns:
            Job object with job_id for tracking
        """
        job = Job(
            job_id=f"job-{uuid.uuid4().hex[:12]}",
            customer_id=customer_id,
            document=document,
            task=task,
            max_tokens=max_tokens,
        )

        with self._lock:
            self._jobs[job.job_id] = job

        self._queue.put(job)
        return job

    def get_status(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def get_result(self, job_id: str) -> Optional[JobResult]:
        """Get job result as API model."""
        job = self.get_status(job_id)
        if not job:
            return None

        return JobResult(
            job_id=job.job_id,
            status=job.status,
            customer_id=job.customer_id,
            created_at=job.created_at,
            completed_at=job.completed_at,
            result=job.result,
            tokens_used=job.tokens_used,
            cost_usd=job.cost_usd,
            error=job.error,
        )

    def get_customer_jobs(self, customer_id: str) -> list[Job]:
        """Get all jobs for a customer."""
        with self._lock:
            return [j for j in self._jobs.values() if j.customer_id == customer_id]

    def get_metrics(self) -> dict:
        """Get queue metrics."""
        with self._lock:
            status_counts = {}
            for job in self._jobs.values():
                status_counts[job.status.value] = status_counts.get(job.status.value, 0) + 1

            return {
                "pending": self._queue.qsize(),
                "total_jobs": len(self._jobs),
                "completed": self._jobs_completed,
                "failed": self._jobs_failed,
                "total_tokens": self._total_tokens,
                "total_cost_usd": self._total_cost,
                "by_status": status_counts,
            }

    def _worker_loop(self) -> None:
        """Worker thread main loop."""
        while not self._shutdown.is_set():
            try:
                job = self._queue.get(timeout=1.0)
                self._process_job(job)
            except queue.Empty:
                continue
            except Exception as e:
                # Log error but don't crash worker
                print(f"Worker error: {e}")

    def _process_job(self, job: Job) -> None:
        """Process a single job."""
        # Update status
        with self._lock:
            job.status = JobStatus.RUNNING

        try:
            # Check budget before processing
            if not self._budget.check_budget(job.customer_id, estimated_cost=0.01):
                job.status = JobStatus.BUDGET_EXCEEDED
                job.error = "Customer budget exceeded"
                job.completed_at = datetime.now(timezone.utc)
                with self._lock:
                    self._jobs_failed += 1
                return

            # Process with the configured processor
            if self._processor:
                result, tokens, cost = self._processor(job)
                job.result = result
                job.tokens_used = tokens
                job.cost_usd = cost
            else:
                # Fallback: simulate processing
                job.result = f"[Mock] Analysis of document: {job.task}"
                job.tokens_used = 50
                job.cost_usd = 0.001

            # Record usage
            self._budget.record_usage(job.customer_id, job.cost_usd or 0.0)

            # Update job
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)

            with self._lock:
                self._jobs_completed += 1
                self._total_tokens += job.tokens_used or 0
                self._total_cost += job.cost_usd or 0.0

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now(timezone.utc)
            with self._lock:
                self._jobs_failed += 1


# Global job queue instance
_job_queue: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    """Get the global job queue instance."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue()
    return _job_queue


def configure_job_queue(
    num_workers: int = 2,
    budget_controller: Optional[BudgetController] = None,
) -> JobQueue:
    """Configure and return the global job queue."""
    global _job_queue
    _job_queue = JobQueue(
        num_workers=num_workers,
        budget_controller=budget_controller,
    )
    return _job_queue
