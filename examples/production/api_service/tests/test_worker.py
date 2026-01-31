"""Tests for job queue and worker."""

import time

import pytest

from examples.production.api_service.budget import BudgetController
from examples.production.api_service.worker import Job, JobQueue
from examples.production.api_service.models import JobStatus


class TestJob:
    """Tests for Job dataclass."""

    def test_default_status(self):
        """Job starts as PENDING."""
        job = Job(
            job_id="test-123",
            customer_id="cust-1",
            document="Test doc",
            task="Summarize",
            max_tokens=100,
        )
        assert job.status == JobStatus.PENDING
        assert job.result is None

    def test_job_fields(self):
        """Job stores all fields."""
        job = Job(
            job_id="test-456",
            customer_id="cust-2",
            document="Long document here",
            task="Extract entities",
            max_tokens=500,
        )
        assert job.job_id == "test-456"
        assert job.customer_id == "cust-2"
        assert job.max_tokens == 500


class TestJobQueue:
    """Tests for JobQueue."""

    def test_submit_job(self):
        """Can submit a job."""
        queue = JobQueue(num_workers=0)  # No workers for this test

        job = queue.submit(
            customer_id="cust-1",
            document="Test document",
            task="Summarize",
            max_tokens=100,
        )

        assert job.job_id.startswith("job-")
        assert job.customer_id == "cust-1"
        assert job.status == JobStatus.PENDING

    def test_get_status(self):
        """Can get job status."""
        queue = JobQueue(num_workers=0)

        submitted = queue.submit(
            customer_id="cust-1",
            document="Test",
            task="Test",
        )

        retrieved = queue.get_status(submitted.job_id)
        assert retrieved is not None
        assert retrieved.job_id == submitted.job_id

    def test_get_status_not_found(self):
        """Returns None for unknown job."""
        queue = JobQueue(num_workers=0)
        assert queue.get_status("nonexistent-job") is None

    def test_get_result(self):
        """Can get job result as API model."""
        queue = JobQueue(num_workers=0)

        job = queue.submit(
            customer_id="cust-1",
            document="Test",
            task="Test",
        )

        result = queue.get_result(job.job_id)
        assert result is not None
        assert result.job_id == job.job_id
        assert result.customer_id == "cust-1"

    def test_get_customer_jobs(self):
        """Can get jobs for a customer."""
        queue = JobQueue(num_workers=0)

        queue.submit(customer_id="cust-a", document="Doc 1", task="Task")
        queue.submit(customer_id="cust-a", document="Doc 2", task="Task")
        queue.submit(customer_id="cust-b", document="Doc 3", task="Task")

        jobs_a = queue.get_customer_jobs("cust-a")
        jobs_b = queue.get_customer_jobs("cust-b")

        assert len(jobs_a) == 2
        assert len(jobs_b) == 1

    def test_metrics(self):
        """Queue provides metrics."""
        queue = JobQueue(num_workers=0)

        queue.submit(customer_id="cust-1", document="Test", task="Task")
        queue.submit(customer_id="cust-1", document="Test", task="Task")

        metrics = queue.get_metrics()
        assert metrics["total_jobs"] == 2
        assert metrics["pending"] == 2

    def test_processing_with_mock(self):
        """Worker processes jobs with custom processor."""
        budget = BudgetController(default_limit_usd=100.0)
        queue = JobQueue(num_workers=1, budget_controller=budget)

        # Set a mock processor
        def mock_processor(job):
            return f"Processed: {job.task}", 50, 0.001

        queue.set_processor(mock_processor)
        queue.start()

        try:
            job = queue.submit(
                customer_id="cust-1",
                document="Test doc",
                task="Summarize this",
            )

            # Wait for processing
            max_wait = 5.0
            start = time.time()
            while queue.get_status(job.job_id).status in (JobStatus.PENDING, JobStatus.RUNNING):
                if time.time() - start > max_wait:
                    break
                time.sleep(0.1)

            result = queue.get_status(job.job_id)
            assert result.status == JobStatus.COMPLETED
            assert result.result == "Processed: Summarize this"
            assert result.tokens_used == 50
            assert result.cost_usd == 0.001

        finally:
            queue.stop()

    def test_budget_exceeded_handling(self):
        """Job fails when budget exceeded."""
        budget = BudgetController(default_limit_usd=0.001)  # Very small budget
        budget.record_usage("broke-customer", 0.001)  # Use it all

        queue = JobQueue(num_workers=1, budget_controller=budget)
        queue.start()

        try:
            job = queue.submit(
                customer_id="broke-customer",
                document="Test",
                task="Task",
            )

            # Wait for processing
            max_wait = 5.0
            start = time.time()
            while queue.get_status(job.job_id).status in (JobStatus.PENDING, JobStatus.RUNNING):
                if time.time() - start > max_wait:
                    break
                time.sleep(0.1)

            result = queue.get_status(job.job_id)
            assert result.status == JobStatus.BUDGET_EXCEEDED

        finally:
            queue.stop()
