"""Tests for job_delegation_demo example."""

import pytest

from enzu import JobStatus
from enzu.models import Job


class TestJobDelegation:
    """Tests for the job delegation demo.

    Note: Job API uses background workers that don't work well with mock providers.
    These tests focus on data structure validation rather than full integration.
    """

    def test_job_status_enum(self):
        """Verify JobStatus enum has expected values."""
        assert hasattr(JobStatus, "PENDING")
        assert hasattr(JobStatus, "RUNNING")
        assert hasattr(JobStatus, "COMPLETED")
        assert hasattr(JobStatus, "FAILED")

    def test_job_model_structure(self):
        """Verify Job model has expected fields."""
        # Job should have these fields
        job = Job(
            job_id="test-123",
            status=JobStatus.PENDING,
            created_at=1234567890.0,
        )

        assert job.job_id == "test-123"
        assert job.status == JobStatus.PENDING
        assert job.created_at == 1234567890.0
        assert job.answer is None  # Optional field

    def test_job_with_result(self):
        """Verify Job model can hold result data."""
        from enzu.models import BudgetUsage, Outcome

        usage = BudgetUsage(
            output_tokens=50,
            total_tokens=100,
            input_tokens=50,
            elapsed_seconds=1.5,
            cost_usd=0.001,
        )

        job = Job(
            job_id="test-456",
            status=JobStatus.COMPLETED,
            created_at=1234567890.0,
            outcome=Outcome.SUCCESS,
            answer="The answer is 42",
            usage=usage,
        )

        assert job.status == JobStatus.COMPLETED
        assert job.outcome == Outcome.SUCCESS
        assert job.answer == "The answer is 42"
        assert job.usage.output_tokens == 50

    def test_job_status_transitions(self):
        """Document expected status transitions."""
        # PENDING -> RUNNING -> COMPLETED (success)
        # PENDING -> RUNNING -> FAILED (error)
        # PENDING -> CANCELLED
        # RUNNING -> CANCELLED

        valid_transitions = {
            JobStatus.PENDING: [JobStatus.RUNNING, JobStatus.CANCELLED],
            JobStatus.RUNNING: [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED],
            JobStatus.COMPLETED: [],  # Terminal
            JobStatus.FAILED: [],     # Terminal
            JobStatus.CANCELLED: [],  # Terminal
        }

        # Just verify the enum values exist
        for status in valid_transitions:
            assert isinstance(status, JobStatus)
