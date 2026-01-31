"""
FastAPI application factory.

Usage:
    from enzu.server.app import create_app

    app = create_app()

Or run directly:
    uvicorn enzu.server:app --reload
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from enzu.server.config import get_settings
from enzu.server.exceptions import APIError
from enzu.server.middleware import RequestTrackingMiddleware, setup_audit_logging
from enzu.server.schemas import ErrorResponse, ErrorDetail
from enzu.server.routers import health, run, sessions, tasks
from enzu.server.services.session_manager import get_session_store
from enzu.server.services.enzu_service import (
    shutdown_executor,
    get_task_queue,
    shutdown_task_queue,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    setup_audit_logging()
    store = get_session_store()
    await store.start()

    # Start TaskQueue for high-concurrency standalone tasks
    await get_task_queue()

    yield

    # Shutdown
    await store.stop()
    await shutdown_task_queue()
    shutdown_executor()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    # Settings loaded for validation; app configuration is static.
    get_settings()

    app = FastAPI(
        title="Enzu API",
        description="HTTP API gateway for Enzu LLM framework",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add middleware
    app.add_middleware(RequestTrackingMiddleware)

    # Exception handlers
    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        """Handle custom API errors."""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=exc.code,
                    message=exc.message,
                    request_id=exc.request_id
                    or getattr(request.state, "request_id", None),
                )
            ).model_dump(),
            headers={"X-Request-ID": getattr(request.state, "request_id", "unknown")},
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected errors."""
        request_id = getattr(request.state, "request_id", "unknown")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="internal_error",
                    message="An internal error occurred",
                    request_id=request_id,
                )
            ).model_dump(),
            headers={"X-Request-ID": request_id},
        )

    # Include routers
    app.include_router(health.router)
    app.include_router(run.router)
    app.include_router(sessions.router)
    app.include_router(tasks.router)

    return app


# Default app instance for uvicorn
app = create_app()
