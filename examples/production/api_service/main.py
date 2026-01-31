#!/usr/bin/env python3
"""
Multi-Tenant API Service: FastAPI + enzu integration.

This example demonstrates:
- FastAPI REST endpoints with enzu execution
- Multi-tenant isolation with per-customer budgets
- Async job processing with background workers
- Prometheus metrics and structured logging
- Budget-aware request handling

Run:
    # Install dependencies
    pip install fastapi uvicorn

    # Set API key
    export OPENAI_API_KEY=sk-...

    # Run the server
    uvicorn examples.production.api_service.main:app --reload

    # Or run directly
    python examples/production/api_service/main.py

API Endpoints:
    POST /analyze        - Submit document for analysis
    GET  /jobs/{job_id}  - Get job status/result
    GET  /budget         - Get customer budget status
    GET  /metrics        - Prometheus metrics
    GET  /health         - Health check
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse

from .budget import BudgetExceededError, configure_budget_controller, get_budget_controller
from .metrics import get_metrics
from .models import (
    AnalyzeRequest,
    CustomerBudget,
    ErrorResponse,
    HealthResponse,
    JobResponse,
    JobResult,
    JobStatus,
)
from .worker import Job, configure_job_queue, get_job_queue

# Configuration
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PROVIDER = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "openai"
DEFAULT_BUDGET_USD = float(os.getenv("DEFAULT_BUDGET_USD", "10.0"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))


def create_processor():
    """Create the job processor function that uses enzu."""
    from enzu import Enzu, Outcome

    client = Enzu(provider=PROVIDER, model=MODEL)

    def process_job(job: Job) -> tuple[str, int, float]:
        """Process a job using enzu."""
        report = client.run(
            job.task,
            data=job.document,
            tokens=job.max_tokens,
            return_report=True,
        )

        result = getattr(report, "answer", None) or getattr(report, "output_text", "")
        tokens = report.budget_usage.output_tokens or 0
        cost = report.budget_usage.cost_usd or 0.0

        if report.outcome != Outcome.SUCCESS:
            raise Exception(f"Processing failed: {report.outcome.value}")

        return result, tokens, cost

    return process_job


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    # Startup
    budget_controller = configure_budget_controller(default_limit_usd=DEFAULT_BUDGET_USD)
    job_queue = configure_job_queue(num_workers=NUM_WORKERS, budget_controller=budget_controller)

    # Set up the processor
    try:
        processor = create_processor()
        job_queue.set_processor(processor)
    except Exception as e:
        print(f"Warning: Could not initialize enzu processor: {e}")
        print("Running in mock mode")

    job_queue.start()
    print(f"API Service started with {NUM_WORKERS} workers")

    yield

    # Shutdown
    job_queue.stop()
    print("API Service stopped")


app = FastAPI(
    title="Enzu Document Analysis API",
    description="Multi-tenant document analysis with per-customer budgets",
    version="1.0.0",
    lifespan=lifespan,
)


# Dependencies
async def get_customer_id(x_customer_id: Optional[str] = Header(default=None)) -> str:
    """Extract customer ID from header."""
    if not x_customer_id:
        raise HTTPException(
            status_code=401,
            detail="X-Customer-ID header required",
        )
    return x_customer_id


# Endpoints
@app.post(
    "/analyze",
    response_model=JobResponse,
    responses={402: {"model": ErrorResponse}, 401: {"model": ErrorResponse}},
)
async def analyze_document(
    request: Request,
    body: AnalyzeRequest,
    customer_id: str = Depends(get_customer_id),
) -> JobResponse:
    """
    Submit a document for analysis.

    Returns a job ID for tracking. Use GET /jobs/{job_id} to check status.
    """
    start_time = time.time()
    metrics = get_metrics()
    budget = get_budget_controller()

    # Check budget before accepting job
    if not budget.check_budget(customer_id, estimated_cost=0.01):
        latency = (time.time() - start_time) * 1000
        metrics.record_request(customer_id, "/analyze", 402, latency)
        raise HTTPException(
            status_code=402,
            detail="Budget exceeded. Contact support to increase your limit.",
        )

    # Submit job
    queue = get_job_queue()
    job = queue.submit(
        customer_id=customer_id,
        document=body.document,
        task=body.task,
        max_tokens=body.max_tokens or 500,
    )

    latency = (time.time() - start_time) * 1000
    metrics.record_request(customer_id, "/analyze", 202, latency)

    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        customer_id=job.customer_id,
        created_at=job.created_at,
        message="Job submitted. Poll GET /jobs/{job_id} for status.",
    )


@app.get(
    "/jobs/{job_id}",
    response_model=JobResult,
    responses={404: {"model": ErrorResponse}},
)
async def get_job(
    job_id: str,
    customer_id: str = Depends(get_customer_id),
) -> JobResult:
    """Get job status and result."""
    queue = get_job_queue()
    result = queue.get_result(job_id)

    if not result:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Verify customer owns this job
    if result.customer_id != customer_id:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return result


@app.get("/budget", response_model=CustomerBudget)
async def get_budget(
    customer_id: str = Depends(get_customer_id),
) -> CustomerBudget:
    """Get current budget status for the customer."""
    budget = get_budget_controller()
    return budget.get_status(customer_id)


@app.get("/metrics", response_class=PlainTextResponse)
async def get_prometheus_metrics() -> str:
    """Prometheus metrics endpoint."""
    metrics = get_metrics()
    queue_metrics = get_job_queue().get_metrics()

    # Combine API and queue metrics
    output = metrics.prometheus_format()
    output += "\n\n# Job Queue Metrics\n"
    output += f"# HELP enzu_jobs_pending Pending jobs in queue\n"
    output += f"# TYPE enzu_jobs_pending gauge\n"
    output += f"enzu_jobs_pending {queue_metrics['pending']}\n"
    output += f"# HELP enzu_jobs_completed_total Completed jobs\n"
    output += f"# TYPE enzu_jobs_completed_total counter\n"
    output += f"enzu_jobs_completed_total {queue_metrics['completed']}\n"
    output += f"# HELP enzu_jobs_failed_total Failed jobs\n"
    output += f"# TYPE enzu_jobs_failed_total counter\n"
    output += f"enzu_jobs_failed_total {queue_metrics['failed']}\n"

    return output


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    metrics = get_metrics()
    queue = get_job_queue()
    queue_metrics = queue.get_metrics()
    summary = metrics.get_summary()

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=summary["uptime_seconds"],
        jobs_pending=queue_metrics["pending"],
        jobs_completed=queue_metrics["completed"],
    )


# Run with: python main.py
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "examples.production.api_service.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
