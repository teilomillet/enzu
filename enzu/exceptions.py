"""
Typed exceptions for enzu.

Provides structured error handling with:
- EnzuError: Base exception for all enzu errors
- EnzuConfigError: Configuration and validation errors
- EnzuProviderError: LLM provider communication errors
- EnzuSandboxError: Sandbox execution errors
- EnzuBudgetError: Budget constraint violations

All exceptions include structured attributes for programmatic handling.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class EnzuError(Exception):
    """Base exception for all enzu errors.

    Attributes:
        message: Human-readable error description
        code: Machine-readable error code for programmatic handling
        details: Additional context as key-value pairs
    """

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize exception for logging or API responses."""
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


class EnzuConfigError(EnzuError):
    """Configuration or validation error.

    Raised when:
    - Required configuration is missing (API keys, etc.)
    - Invalid parameter values
    - Incompatible configuration combinations
    - Model validation failures

    Examples:
        EnzuConfigError("API key not configured", code="missing_api_key")
        EnzuConfigError("Invalid model name", details={"model": "gpt-999"})
    """

    pass


class EnzuProviderError(EnzuError):
    """LLM provider communication error.

    Raised when:
    - Provider API returns an error
    - Network connectivity issues
    - Authentication failures
    - Rate limiting

    Attributes:
        provider: Name of the provider that failed
        status_code: HTTP status code if available
        retry_after: Seconds to wait before retry (for rate limits)
    """

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        retry_after: Optional[float] = None,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        details = details or {}
        if provider:
            details["provider"] = provider
        if status_code:
            details["status_code"] = status_code
        if retry_after:
            details["retry_after"] = retry_after

        self.provider = provider
        self.status_code = status_code
        self.retry_after = retry_after

        super().__init__(message, code=code, details=details)

    @property
    def is_rate_limited(self) -> bool:
        """True if error is due to rate limiting."""
        return self.status_code == 429 or self.retry_after is not None

    @property
    def is_auth_error(self) -> bool:
        """True if error is due to authentication failure."""
        return self.status_code in (401, 403)


class EnzuSandboxError(EnzuError):
    """Sandbox execution error.

    Raised when:
    - Code execution fails in sandbox
    - Sandbox timeout exceeded
    - Security policy violation
    - Resource limits exceeded

    Attributes:
        sandbox_type: Type of sandbox (None, "subprocess", "container")
        stdout: Captured stdout if available
        stderr: Captured stderr if available
    """

    def __init__(
        self,
        message: str,
        *,
        sandbox_type: Optional[str] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        details = details or {}
        if sandbox_type:
            details["sandbox_type"] = sandbox_type
        if stdout:
            details["stdout"] = stdout[:1000]  # Truncate for safety
        if stderr:
            details["stderr"] = stderr[:1000]

        self.sandbox_type = sandbox_type
        self.stdout = stdout
        self.stderr = stderr

        super().__init__(message, code=code, details=details)


class EnzuBudgetError(EnzuError):
    """Budget constraint violation.

    Raised when:
    - Token limit exceeded
    - Cost limit exceeded
    - Time limit exceeded
    - Budget exhausted before completion

    Attributes:
        limit_type: Which limit was exceeded ("tokens", "cost", "time")
        limit_value: The configured limit
        current_value: The current usage when limit was hit
    """

    def __init__(
        self,
        message: str,
        *,
        limit_type: Optional[str] = None,
        limit_value: Optional[float] = None,
        current_value: Optional[float] = None,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        details = details or {}
        if limit_type:
            details["limit_type"] = limit_type
        if limit_value is not None:
            details["limit_value"] = limit_value
        if current_value is not None:
            details["current_value"] = current_value

        self.limit_type = limit_type
        self.limit_value = limit_value
        self.current_value = current_value

        super().__init__(message, code=code, details=details)


class EnzuToolError(EnzuError):
    """Tool or external service error.

    Raised when:
    - External tool (Exa, search, etc.) fails
    - Tool not configured
    - Tool API error

    Attributes:
        tool_name: Name of the tool that failed
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: Optional[str] = None,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        details = details or {}
        if tool_name:
            details["tool_name"] = tool_name

        self.tool_name = tool_name

        super().__init__(message, code=code, details=details)


__all__ = [
    "EnzuError",
    "EnzuConfigError",
    "EnzuProviderError",
    "EnzuSandboxError",
    "EnzuBudgetError",
    "EnzuToolError",
]
