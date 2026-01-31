"""
Sandbox factory: creates sandboxes based on isolation level.

Extracted from engine.py. Centralizes sandbox creation logic for the three
isolation modes: in-process, subprocess, container.

Each mode has different trade-offs:
- In-process: fastest, full llm_query/llm_batch support, shared memory
- Subprocess: OS-level isolation, resource limits, llm callbacks via IPC
- Container: Docker + seccomp profiles, gov-grade security, llm callbacks via IPC

Integration with SandboxImage (from enzu.sandbox):
- Pass sandbox_image to use custom Docker images in container mode
- SandboxImage is built on first use and cached by content hash

Security note on namespace:
- Never pass secrets (API keys, tokens, passwords) in the namespace dict
- The namespace is serialized and sent to isolated processes/containers
- Use environment variables for secrets instead (they stay in the host process)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

from enzu.repl import PythonSandbox


# Patterns that indicate a value might be a secret
# We check namespace keys against these to prevent accidental secret leakage
SECRET_KEY_PATTERNS = frozenset(
    [
        "api_key",
        "apikey",
        "api-key",
        "secret",
        "secret_key",
        "secretkey",
        "token",
        "auth_token",
        "access_token",
        "bearer",
        "password",
        "passwd",
        "pwd",
        "credential",
        "credentials",
        "cred",
        "private_key",
        "privatekey",
        "auth",
        "authorization",
        "key",  # Generic but catches many cases like "openai_key"
    ]
)


class NamespaceSecretError(ValueError):
    """Raised when namespace contains likely secrets."""

    pass


def _check_namespace_for_secrets(namespace: Dict[str, Any], isolation: str) -> None:
    """
    Fail fast if namespace contains keys that look like secrets.

    This prevents accidental leakage of API keys, tokens, etc. into
    isolated sandboxes where they could be accessed by LLM-generated code.

    Args:
        namespace: The namespace dict being passed to the sandbox
        isolation: The isolation mode ("subprocess" or "container")

    Raises:
        NamespaceSecretError: If a key matches secret patterns

    Example of what NOT to do:
        # BAD - secrets will be pickled and sent to container
        create_sandbox(namespace={"api_key": "sk-xxx"}, isolation="container")

    Instead, use environment variables:
        # GOOD - secrets stay in host process
        import os
        os.environ["MY_API_KEY"] = "sk-xxx"
        # Your code in host process reads os.environ, sandbox never sees it
    """
    if not namespace:
        return

    suspicious_keys = []
    for key in namespace:
        key_lower = key.lower().replace("-", "_")
        # Check if any secret pattern is contained in the key
        for pattern in SECRET_KEY_PATTERNS:
            if pattern in key_lower:
                suspicious_keys.append(key)
                break

    if suspicious_keys:
        keys_str = ", ".join(f"'{k}'" for k in suspicious_keys)
        raise NamespaceSecretError(
            f"\n"
            f"══════════════════════════════════════════════════════════════════\n"
            f"  SECURITY ERROR: Namespace contains likely secrets\n"
            f"══════════════════════════════════════════════════════════════════\n"
            f"\n"
            f"  Detected keys: {keys_str}\n"
            f"\n"
            f"  The namespace is serialized and sent to the {isolation} sandbox.\n"
            f"  LLM-generated code running there could access these values.\n"
            f"\n"
            f"  HOW TO FIX:\n"
            f"  -----------\n"
            f"  Use environment variables instead of namespace for secrets:\n"
            f"\n"
            f"    # Set in your environment or .env file:\n"
            f"    export MY_API_KEY='sk-xxx'\n"
            f"\n"
            f"    # Your host-side code reads it:\n"
            f"    api_key = os.environ.get('MY_API_KEY')\n"
            f"\n"
            f"  The sandbox cannot access host environment variables.\n"
            f"  Only the host process (where your provider runs) sees them.\n"
            f"\n"
            f"  If this key is NOT a secret, rename it to avoid this check:\n"
            f"    'api_key' → 'api_identifier' or 'api_name'\n"
            f"\n"
            f"══════════════════════════════════════════════════════════════════\n"
        )


def create_sandbox(
    *,
    isolation: Optional[str],
    data: str,
    context: Any,
    namespace: Dict[str, Any],
    allowed_imports: List[str],
    output_char_limit: int,
    timeout_seconds: Optional[float],
    inject_search_tools: bool,
    enable_pip: bool,
    # Callbacks for LLM access (in-process direct, isolated via IPC)
    llm_query: Optional[Callable[[str], str]] = None,
    llm_batch: Optional[Callable[[list], list]] = None,
    # Custom container image (SandboxImage, BuiltImage, or string tag)
    sandbox_image: Optional[Any] = None,
) -> Union[PythonSandbox, Any]:
    """
    Create sandbox based on isolation level.

    Args:
        isolation: None, "subprocess", or "container"
        data: input data passed to sandbox
        context: context variable for REPL
        namespace: additional namespace bindings
        allowed_imports: list of allowed Python imports
        output_char_limit: max chars in sandbox output
        timeout_seconds: execution timeout
        inject_search_tools: whether to inject Exa search tools
        enable_pip: whether to allow pip_install()
        llm_query: LLM query callback (all modes via IPC for isolated)
        llm_batch: LLM batch callback (all modes via IPC for isolated)
        sandbox_image: Custom image for container mode (SandboxImage, BuiltImage, or string)

    Returns:
        Sandbox instance. Type depends on isolation level.
    """
    if isolation == "container":
        # Container-based isolation (Docker + seccomp).
        # llm_query/llm_batch available via IPC bridge.
        from enzu.isolation.container import ContainerSandbox, ContainerConfig

        # Security check: prevent secrets from being serialized to container
        _check_namespace_for_secrets(namespace, isolation)

        # Build config with optional custom image
        # sandbox_image can be: SandboxImage, BuiltImage, string tag, or None
        config = ContainerConfig(
            max_cpu_seconds=timeout_seconds or 30.0,
            timeout_seconds=(timeout_seconds or 30.0) + 30.0,
            allowed_imports=set(allowed_imports),
            image=sandbox_image if sandbox_image is not None else "python:3.11-slim",
        )
        # Add context to namespace for isolated sandboxes
        isolated_namespace = dict(namespace)
        isolated_namespace["context"] = context
        return ContainerSandbox(
            data=data,
            namespace=isolated_namespace,
            config=config,
            llm_query=llm_query,
            llm_batch=llm_batch,
        )

    if isolation == "subprocess":
        # Subprocess-based isolation.
        # llm_query/llm_batch available via IPC bridge.
        from enzu.isolation.runner import IsolatedSandbox, SandboxConfig

        # Security check: prevent secrets from being serialized to subprocess
        _check_namespace_for_secrets(namespace, isolation)

        sandbox_config = SandboxConfig(
            max_cpu_seconds=timeout_seconds or 30.0,
            timeout_seconds=(timeout_seconds or 30.0) + 30.0,
            allowed_imports=set(allowed_imports),
        )
        # Add context to namespace for isolated sandboxes
        isolated_namespace = dict(namespace)
        isolated_namespace["context"] = context
        return IsolatedSandbox(
            data=data,
            namespace=isolated_namespace,
            config=sandbox_config,
            llm_query=llm_query,
            llm_batch=llm_batch,
        )

    # Default: in-process sandbox (fastest, full llm_query/llm_batch).
    # PythonSandbox requires llm_query to be callable, not None
    if llm_query is None:
        raise ValueError("llm_query is required for in-process sandbox")
    return PythonSandbox(
        data=data,
        context=context,
        llm_query=llm_query,
        llm_batch=llm_batch,
        namespace=namespace,
        allowed_imports=allowed_imports,
        output_char_limit=output_char_limit,
        timeout_seconds=timeout_seconds,
        inject_safe_helpers=True,
        inject_search_tools=inject_search_tools,
        enable_pip=enable_pip,
    )
