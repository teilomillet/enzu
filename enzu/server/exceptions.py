"""
HTTP exception types for the API server.
"""

from typing import Optional


class APIError(Exception):
    """Base exception for API errors."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, request_id: Optional[str] = None) -> None:
        self.message = message
        self.request_id = request_id
        super().__init__(message)


class AuthenticationError(APIError):
    """Invalid or missing authentication credentials."""

    status_code = 401
    code = "authentication_error"


class AuthorizationError(APIError):
    """Insufficient permissions for the requested operation."""

    status_code = 403
    code = "authorization_error"


class SessionNotFoundError(APIError):
    """Session does not exist or has expired."""

    status_code = 404
    code = "session_not_found"


class SessionBudgetExceededError(APIError):
    """Session budget (cost or tokens) has been exceeded."""

    status_code = 402
    code = "session_budget_exceeded"


class ValidationError(APIError):
    """Request validation failed."""

    status_code = 400
    code = "validation_error"


class ModelError(APIError):
    """Error from the underlying LLM provider."""

    status_code = 502
    code = "model_error"


class RateLimitError(APIError):
    """Rate limit exceeded."""

    status_code = 429
    code = "rate_limit_exceeded"


class SessionLockError(APIError):
    """Failed to acquire session lock (concurrent request conflict)."""

    status_code = 409
    code = "session_lock_error"
