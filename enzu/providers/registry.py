from __future__ import annotations

from typing import Any, Dict, Optional
from urllib.parse import urlparse

ProviderConfig = Dict[str, Any]

PROVIDERS: Dict[str, ProviderConfig] = {
    "openai": {"supports_responses": True},
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "supports_responses": True,
    },
    "groq": {"base_url": "https://api.groq.com/openai/v1", "supports_responses": False},
    "together": {"base_url": "https://api.together.xyz/v1", "supports_responses": False},
    "mistral": {"base_url": "https://api.mistral.ai/v1", "supports_responses": False},
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "supports_responses": False,
    },
    "xai": {"base_url": "https://api.x.ai/v1", "supports_responses": False},
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "supports_responses": False,
    },
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "supports_responses": False},
    "ollama": {"base_url": "http://localhost:11434/v1", "supports_responses": False},
    "lmstudio": {"base_url": "http://localhost:1234/v1", "supports_responses": False},
}


def validate_provider_config(name: str, config: ProviderConfig) -> None:
    """Validate provider config. Raises ValueError if invalid."""
    if not name or not isinstance(name, str):
        raise ValueError("Provider name must be a non-empty string")
    
    base_url = config.get("base_url")
    if base_url:
        parsed = urlparse(base_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid base_url: {base_url}")


def register_provider(
    name: str, *, base_url: str, supports_responses: bool = False, **extra: Any
) -> None:
    """
    Register a custom provider.
    
    Args:
        name: Provider identifier
        base_url: API base URL
        supports_responses: True if provider supports Open Responses API
        **extra: Additional provider-specific config
    
    Example:
        enzu.register_provider("myapi", base_url="https://api.mycompany.com/v1", supports_responses=True)
    """
    config = {"base_url": base_url, "supports_responses": supports_responses, **extra}
    validate_provider_config(name, config)
    PROVIDERS[name] = config


def get_provider_config(name: str) -> Optional[ProviderConfig]:
    """Get config for a provider, or None if not found."""
    return PROVIDERS.get(name)


def list_providers() -> list[str]:
    """List all registered provider names."""
    return list(PROVIDERS.keys())
