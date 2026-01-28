"""
Sandbox module for Enzu's container image building and lifecycle management.

Provides:
- SandboxImage: Chainable DSL for building container images (Phase 1)
- LifecycleManager: Idle timeout and cleanup for sandboxes (Phase 3)

The IPC bridge for llm_query/llm_batch in containers is in enzu/isolation/ipc.py
(implemented via stdin/stdout pipes instead of Unix sockets).

Usage:
    from enzu.sandbox import SandboxImage, LifecycleConfig

    # Build custom image
    image = (
        SandboxImage.python("3.11")
        .apt("git", "curl")
        .pip("pandas", "numpy")
        .env(TOKENIZERS_PARALLELISM="false")
    )

    # Configure lifecycle
    lifecycle = LifecycleConfig(idle_timeout_seconds=300)

See docs/SANDBOX_EVOLUTION_PLAN.md for architecture details.
"""
from enzu.sandbox.image import SandboxImage, BuiltImage
from enzu.sandbox.lifecycle import (
    LifecycleConfig,
    LifecycleManager,
    ManagedSandbox,
    get_lifecycle_manager,
    reset_lifecycle_manager,
)

__all__ = [
    # Image builder (Phase 1)
    "SandboxImage",
    "BuiltImage",
    # Lifecycle management (Phase 3)
    "LifecycleConfig",
    "LifecycleManager",
    "ManagedSandbox",
    "get_lifecycle_manager",
    "reset_lifecycle_manager",
]
