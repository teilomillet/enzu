"""
Provider pool for connection reuse across concurrent requests.

Problem: Each enzu.run() call creates a new OpenAI client with its own httpx.Client.
Under load (N concurrent requests), this creates N connection pools, exhausting:
- File descriptors (OS limit ~1024-65535)
- TCP connections (provider rate limits)
- Memory (~50-100MB per client)

Solution: Share provider instances by (name, base_url, api_key) tuple.
One httpx.Client serves all requests to the same endpoint.

Usage:
    # Get shared provider (creates if not exists)
    provider = get_provider("openrouter", api_key="sk-...")

    # When shutting down
    close_all_providers()

Thread safety: Uses threading.Lock for provider creation.
The OpenAI client's httpx.Client is thread-safe for concurrent requests.
"""
from __future__ import annotations

import atexit
import threading
from typing import Dict, Optional, Tuple


from enzu.providers.openai_compat import OpenAICompatProvider
from enzu.providers.registry import get_provider_config


# Cache key: (provider_name, base_url, api_key_hash)
# api_key_hash avoids storing raw keys in memory longer than needed
_ProviderKey = Tuple[str, Optional[str], int]

_pool: Dict[_ProviderKey, OpenAICompatProvider] = {}
_pool_lock = threading.Lock()

# Track active request count per provider for graceful shutdown
_active_requests: Dict[_ProviderKey, int] = {}
_requests_lock = threading.Lock()

# Capacity limit: max concurrent requests before rejecting
# Set via set_capacity_limit(). None = unlimited.
_capacity_limit: Optional[int] = None
_current_total_requests: int = 0


class CapacityExceededError(Exception):
    """Raised when request capacity limit is exceeded."""
    pass


def _make_key(
    name: str,
    base_url: Optional[str],
    api_key: Optional[str],
) -> _ProviderKey:
    """Create cache key from provider params.

    Uses hash of api_key to avoid keeping raw keys in the key tuple.
    Two providers with same name/url but different keys get different entries.
    """
    key_hash = hash(api_key) if api_key else 0
    return (name.lower(), base_url, key_hash)


def get_provider(
    name: str,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    organization: Optional[str] = None,
    project: Optional[str] = None,
    supports_responses: Optional[bool] = None,
) -> OpenAICompatProvider:
    """
    Get or create a shared provider instance.

    Returns the same provider for identical (name, base_url, api_key) combinations.
    The underlying httpx.Client is thread-safe and handles connection pooling.

    Args:
        name: Provider name (e.g., "openrouter", "openai")
        api_key: API key for authentication
        base_url: Override base URL (uses registry default if None)
        headers: Extra HTTP headers
        organization: OpenAI organization ID
        project: OpenAI project ID
        supports_responses: Override responses API support flag

    Returns:
        Shared OpenAICompatProvider instance

    Raises:
        CapacityExceededError: If capacity limit is set and exceeded
        ValueError: If provider is unknown
    """
    # Check capacity before creating/returning provider
    _check_capacity()

    # Resolve base_url from registry if not provided
    config = get_provider_config(name)
    if config is None:
        raise ValueError(f"Unknown provider: {name}. Use register_provider() to add custom providers.")

    effective_base_url = base_url or config.get("base_url")
    effective_supports_responses = (
        supports_responses if supports_responses is not None
        else config.get("supports_responses", False)
    )

    key = _make_key(name, effective_base_url, api_key)

    # Fast path: provider exists
    with _pool_lock:
        if key in _pool:
            return _pool[key]

    # Slow path: create provider
    # Double-check after acquiring lock (another thread may have created it)
    with _pool_lock:
        if key in _pool:
            return _pool[key]

        provider = OpenAICompatProvider(
            name=name,
            api_key=api_key,
            base_url=effective_base_url,
            headers=headers,
            organization=organization,
            project=project,
            supports_responses=effective_supports_responses,
        )
        _pool[key] = provider
        return provider


def set_capacity_limit(limit: Optional[int]) -> None:
    """
    Set maximum concurrent requests across all providers.

    When limit is reached, get_provider() raises CapacityExceededError.
    Use this to implement backpressure in high-load scenarios.

    Args:
        limit: Max concurrent requests, or None for unlimited
    """
    global _capacity_limit
    _capacity_limit = limit


def get_capacity_stats() -> Dict[str, int]:
    """
    Get current capacity statistics.

    Returns:
        Dict with 'active_requests', 'capacity_limit', 'providers_count'
    """
    with _requests_lock:
        return {
            "active_requests": _current_total_requests,
            "capacity_limit": _capacity_limit or -1,  # -1 = unlimited
            "providers_count": len(_pool),
        }


def acquire_request_slot() -> bool:
    """
    Acquire a request slot (increment active count).

    Call this before starting a request, release_request_slot() after.
    Returns False if capacity exceeded.
    """
    global _current_total_requests
    with _requests_lock:
        if _capacity_limit is not None and _current_total_requests >= _capacity_limit:
            return False
        _current_total_requests += 1
        return True


def release_request_slot() -> None:
    """Release a request slot (decrement active count)."""
    global _current_total_requests
    with _requests_lock:
        _current_total_requests = max(0, _current_total_requests - 1)


def _check_capacity() -> None:
    """Check if capacity limit is exceeded. Raises CapacityExceededError if so."""
    if _capacity_limit is None:
        return
    with _requests_lock:
        if _current_total_requests >= _capacity_limit:
            raise CapacityExceededError(
                f"Request capacity exceeded: {_current_total_requests}/{_capacity_limit} active requests. "
                "Retry later or increase capacity via set_capacity_limit()."
            )


def close_provider(name: str, *, api_key: Optional[str] = None, base_url: Optional[str] = None) -> bool:
    """
    Close and remove a specific provider from the pool.

    Returns True if provider was found and closed, False otherwise.
    """
    key = _make_key(name, base_url, api_key)

    with _pool_lock:
        provider = _pool.pop(key, None)
        if provider is None:
            return False

        # Close the underlying httpx client
        try:
            provider._client.close()
        except Exception:
            pass  # Best effort cleanup
        return True


def close_all_providers() -> int:
    """
    Close all providers in the pool.

    Call this during application shutdown to release connections cleanly.
    Returns count of providers closed.
    """
    with _pool_lock:
        count = len(_pool)
        for provider in _pool.values():
            try:
                provider._client.close()
            except Exception:
                pass  # Best effort cleanup
        _pool.clear()
        return count


# Register cleanup on interpreter shutdown
atexit.register(close_all_providers)
