"""Tests for API models."""

from datetime import datetime, timezone

import pytest

from examples.production.api_service.models import (
    AnalyzeRequest,
    CustomerBudget,
    ErrorResponse,
    HealthResponse,
    JobResponse,
    JobResult,
    JobStatus,
)


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_status_values(self):
        """All expected statuses exist."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.BUDGET_EXCEEDED.value == "budget_exceeded"


class TestAnalyzeRequest:
    """Tests for AnalyzeRequest model."""

    def test_required_fields(self):
        """Document is required."""
        request = AnalyzeRequest(document="Test content")
        assert request.document == "Test content"
        assert request.task == "Summarize the key points"  # Default
        assert request.max_tokens == 500  # Default

    def test_custom_task(self):
        """Can specify custom task."""
        request = AnalyzeRequest(
            document="Contract text",
            task="Extract key terms",
            max_tokens=1000,
        )
        assert request.task == "Extract key terms"
        assert request.max_tokens == 1000


class TestJobResponse:
    """Tests for JobResponse model."""

    def test_all_fields(self):
        """JobResponse includes all fields."""
        now = datetime.now(timezone.utc)
        response = JobResponse(
            job_id="job-123",
            status=JobStatus.PENDING,
            customer_id="cust-1",
            created_at=now,
            message="Job queued",
        )
        assert response.job_id == "job-123"
        assert response.status == JobStatus.PENDING
        assert response.customer_id == "cust-1"
        assert response.message == "Job queued"


class TestJobResult:
    """Tests for JobResult model."""

    def test_pending_result(self):
        """Pending job has no result."""
        result = JobResult(
            job_id="job-456",
            status=JobStatus.PENDING,
            customer_id="cust-1",
            created_at=datetime.now(timezone.utc),
        )
        assert result.result is None
        assert result.completed_at is None

    def test_completed_result(self):
        """Completed job has result."""
        now = datetime.now(timezone.utc)
        result = JobResult(
            job_id="job-789",
            status=JobStatus.COMPLETED,
            customer_id="cust-1",
            created_at=now,
            completed_at=now,
            result="Analysis complete: ...",
            tokens_used=342,
            cost_usd=0.0023,
        )
        assert result.result == "Analysis complete: ..."
        assert result.tokens_used == 342
        assert result.cost_usd == 0.0023


class TestCustomerBudget:
    """Tests for CustomerBudget model."""

    def test_budget_status(self):
        """Budget status includes all fields."""
        now = datetime.now(timezone.utc)
        budget = CustomerBudget(
            customer_id="cust-1",
            budget_limit_usd=10.0,
            budget_used_usd=2.5,
            budget_remaining_usd=7.5,
            requests_count=15,
            period_start=now,
            period_end=now,
        )
        assert budget.budget_limit_usd == 10.0
        assert budget.budget_used_usd == 2.5
        assert budget.budget_remaining_usd == 7.5


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_health_defaults(self):
        """Health response has defaults."""
        health = HealthResponse(uptime_seconds=123.4)
        assert health.status == "healthy"
        assert health.version == "1.0.0"
        assert health.jobs_pending == 0


class TestErrorResponse:
    """Tests for ErrorResponse model."""

    def test_error_message(self):
        """Error response includes message."""
        error = ErrorResponse(
            error="Budget exceeded",
            detail="Customer budget limit reached",
            code="BUDGET_EXCEEDED",
        )
        assert error.error == "Budget exceeded"
        assert error.detail == "Customer budget limit reached"
        assert error.code == "BUDGET_EXCEEDED"
