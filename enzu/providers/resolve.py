from __future__ import annotations

import os
from typing import Dict, Optional, Union
from urllib.parse import urlparse

from enzu.providers.base import BaseProvider
from enzu.providers.openai_compat import OpenAICompatProvider
from enzu.providers.registry import get_provider_config


def resolve_provider(
    provider: Union[str, BaseProvider],
    *,
    api_key: Optional[str] = None,
    referer: Optional[str] = None,
    app_name: Optional[str] = None,
    organization: Optional[str] = None,
    project: Optional[str] = None,
    use_pool: bool = False,
) -> BaseProvider:
    """
    Resolve provider from string name or return BaseProvider instance.

    This lives in providers/ to avoid importing api.py in runtime modules.
    """
    if isinstance(provider, BaseProvider):
        return provider
    name = provider.lower()
    config = get_provider_config(name)
    if config is None:
        raise ValueError(
            f"Unknown provider: {provider}. Use register_provider() to add custom providers."
        )

    base_url = config.get("base_url")
    supports_responses = config.get("supports_responses", False)
    headers: Dict[str, str] = {}

    if name == "openrouter":
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        referer_val = referer or os.getenv("OPENROUTER_REFERER")
        if referer_val:
            headers["HTTP-Referer"] = referer_val
        app_name_val = app_name or os.getenv("OPENROUTER_APP_NAME")
        if app_name_val:
            headers["X-Title"] = app_name_val
    elif name == "openai":
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        organization = organization or os.getenv("OPENAI_ORG")
        project = project or os.getenv("OPENAI_PROJECT")
    else:
        api_key_env = f"{name.upper()}_API_KEY"
        api_key = api_key or os.getenv(api_key_env)
        if not api_key and _is_local_base_url(base_url):
            api_key = "local"

    if use_pool:
        from enzu.providers.pool import get_provider
        return get_provider(
            name,
            api_key=api_key,
            base_url=base_url,
            headers=headers if headers else None,
            organization=organization if name == "openai" else None,
            project=project if name == "openai" else None,
            supports_responses=supports_responses,
        )

    return OpenAICompatProvider(
        name=name,
        api_key=api_key,
        base_url=base_url,
        headers=headers if headers else None,
        organization=organization if name == "openai" else None,
        project=project if name == "openai" else None,
        supports_responses=supports_responses,
    )


def _is_local_base_url(base_url: Optional[str]) -> bool:
    if not base_url:
        return False
    parsed = urlparse(base_url)
    return parsed.hostname in {"localhost", "127.0.0.1", "::1"}
