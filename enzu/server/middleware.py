"""
Request tracking middleware.

Provides:
- Request ID generation and propagation
- Audit logging integration
- Request timing
"""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from typing import Awaitable, Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from enzu.isolation.audit import get_audit_logger, configure_audit_logger
from enzu.server.config import get_settings


# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar("session_id", default=None)


def get_request_id() -> Optional[str]:
    """Get the current request ID from context."""
    return request_id_var.get()


def get_session_id() -> Optional[str]:
    """Get the current session ID from context."""
    return session_id_var.get()


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"req-{uuid.uuid4().hex[:16]}"


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request tracking and audit logging.

    Extracts request/session IDs from headers, generates if missing,
    and integrates with AuditLogger for request lifecycle tracking.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # Extract or generate request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = generate_request_id()

        # Extract session ID if present
        session_id = request.headers.get("X-Session-ID")

        # Set context variables
        request_id_token = request_id_var.set(request_id)
        session_id_token = session_id_var.set(session_id)

        # Store in request state for handlers
        request.state.request_id = request_id
        request.state.session_id = session_id

        # Get audit logger
        audit = get_audit_logger()

        # Log request submitted (skip health checks)
        if not request.url.path.startswith("/health"):
            audit.log_request_submitted(
                request_id=request_id,
                conversation_id=session_id,
            )

        start_time = time.time()

        try:
            response = await call_next(request)

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            # Log completion (skip health checks)
            if not request.url.path.startswith("/health"):
                if response.status_code < 400:
                    audit.log_request_completed(
                        request_id=request_id,
                        execution_time_ms=execution_time_ms,
                    )
                else:
                    audit.log_request_failed(
                        request_id=request_id,
                        error_category=f"http_{response.status_code}",
                        execution_time_ms=execution_time_ms,
                    )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            # Log failure
            execution_time_ms = (time.time() - start_time) * 1000
            audit.log_request_failed(
                request_id=request_id,
                error_category=type(e).__name__,
                execution_time_ms=execution_time_ms,
            )
            raise

        finally:
            # Reset context variables
            request_id_var.reset(request_id_token)
            session_id_var.reset(session_id_token)


def setup_audit_logging() -> None:
    """
    Configure audit logging based on settings.

    Call this at app startup.
    """
    settings = get_settings()
    configure_audit_logger(
        output_path=settings.audit_log_path,
        include_timestamps=True,
    )
