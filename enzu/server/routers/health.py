"""
Health check endpoint.

Provides basic health status for load balancers and monitoring.
"""
from fastapi import APIRouter

from enzu.server.schemas import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns basic health status. Use this for load balancer health checks.
    Does not require authentication.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
    )
