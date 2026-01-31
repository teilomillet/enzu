"""
Server configuration from environment variables.

Usage:
    from enzu.server.config import get_settings

    settings = get_settings()
    print(settings.host, settings.port)
"""

from functools import lru_cache
from typing import Optional
import os


class Settings:
    """Server configuration loaded from environment variables."""

    def __init__(self) -> None:
        # Server
        self.host: str = os.getenv("ENZU_HOST", "0.0.0.0")
        self.port: int = int(os.getenv("ENZU_PORT", "8000"))

        # Authentication
        self.api_key: Optional[str] = os.getenv("ENZU_API_KEY")

        # LLM defaults
        self.default_model: Optional[str] = os.getenv("ENZU_DEFAULT_MODEL")
        self.default_provider: Optional[str] = os.getenv("ENZU_DEFAULT_PROVIDER")

        # Session management
        self.session_store: str = os.getenv("ENZU_SESSION_STORE", "memory")
        self.redis_url: Optional[str] = os.getenv("ENZU_REDIS_URL")
        self.session_ttl_seconds: int = int(
            os.getenv("ENZU_SESSION_TTL_DEFAULT", "3600")
        )

        # Audit logging
        self.audit_log_path: Optional[str] = os.getenv("ENZU_AUDIT_LOG_PATH")

        # Request limits
        self.max_request_size_mb: int = int(os.getenv("ENZU_MAX_REQUEST_SIZE_MB", "10"))

    @property
    def auth_required(self) -> bool:
        """Authentication is required if ENZU_API_KEY is set."""
        return self.api_key is not None


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def reset_settings() -> None:
    """Clear settings cache. For testing only."""
    get_settings.cache_clear()
