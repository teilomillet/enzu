"""
API key authentication.

Uses constant-time comparison to prevent timing attacks.
"""

import hmac
from typing import Optional

from fastapi import Header, Request

from enzu.server.config import get_settings
from enzu.server.exceptions import AuthenticationError


def verify_api_key(api_key: str) -> bool:
    """
    Verify an API key against the configured key.

    Uses constant-time comparison to prevent timing attacks.

    Returns:
        True if valid, False otherwise.
    """
    settings = get_settings()
    if settings.api_key is None:
        return True  # No key configured = no auth required

    return hmac.compare_digest(api_key, settings.api_key)


def get_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> Optional[str]:
    """
    FastAPI dependency to extract and validate API key.

    Raises:
        AuthenticationError: If auth is required but key is missing or invalid.

    Returns:
        The validated API key, or None if auth not required.
    """
    settings = get_settings()

    if not settings.auth_required:
        return None

    if x_api_key is None:
        raise AuthenticationError("Missing X-API-Key header")

    if not verify_api_key(x_api_key):
        raise AuthenticationError("Invalid API key")

    return x_api_key


async def require_auth(request: Request) -> None:
    """
    Middleware-style auth check for use in app startup or custom middleware.
    """
    settings = get_settings()
    if not settings.auth_required:
        return

    api_key = request.headers.get("X-API-Key")
    if api_key is None:
        raise AuthenticationError("Missing X-API-Key header")

    if not verify_api_key(api_key):
        raise AuthenticationError("Invalid API key")
