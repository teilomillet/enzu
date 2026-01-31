"""Pydantic models for API request/response."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"


class AnalyzeRequest(BaseModel):
    """Request to analyze a document."""

    document: str = Field(..., description="Document text to analyze")
    task: str = Field(
        default="Summarize the key points",
        description="Analysis task to perform",
    )
    max_tokens: Optional[int] = Field(
        default=500,
        description="Maximum output tokens",
        ge=1,
        le=4000,
    )

    model_config = {"json_schema_extra": {"example": {
        "document": "This is a contract between...",
        "task": "Extract key terms and obligations",
        "max_tokens": 500,
    }}}


class JobResponse(BaseModel):
    """Response for job submission."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    customer_id: str = Field(..., description="Customer who owns this job")
    created_at: datetime = Field(..., description="Job creation timestamp")
    message: Optional[str] = Field(default=None, description="Status message")


class JobResult(BaseModel):
    """Full job result including output."""

    job_id: str
    status: JobStatus
    customer_id: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[str] = None
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    error: Optional[str] = None


class CustomerBudget(BaseModel):
    """Customer budget status."""

    customer_id: str
    budget_limit_usd: float = Field(..., description="Monthly budget limit")
    budget_used_usd: float = Field(..., description="Budget used this period")
    budget_remaining_usd: float = Field(..., description="Remaining budget")
    requests_count: int = Field(..., description="Total requests this period")
    period_start: datetime = Field(..., description="Budget period start")
    period_end: datetime = Field(..., description="Budget period end")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "1.0.0"
    uptime_seconds: float
    jobs_pending: int = 0
    jobs_completed: int = 0


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


class MetricsSummary(BaseModel):
    """Metrics summary for dashboard."""

    total_requests: int
    total_tokens: int
    total_cost_usd: float
    avg_latency_ms: float
    success_rate: float
    active_customers: int
    jobs_by_status: Dict[str, int]
