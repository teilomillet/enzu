"""
Security profile configuration for enzu sandboxes.

Provides predefined security profiles:
- strict: Production default. No network, no filesystem, limited imports.
- dev: Development mode. Allows network for search tools, more imports.

Usage:
    from enzu.security import get_security_profile, SecurityProfile

    profile = get_security_profile()  # Uses ENZU_SECURITY_PROFILE env var

    # Or explicitly
    profile = get_security_profile("strict")
    profile = get_security_profile("dev")

    # Use in sandbox configuration
    sandbox = PythonSandbox(
        allowed_imports=profile.allowed_imports,
        enable_pip=profile.enable_pip,
        ...
    )
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import FrozenSet, Literal, Optional

SecurityProfileName = Literal["strict", "dev"]


@dataclass(frozen=True)
class SecurityProfile:
    """Immutable security profile configuration.

    Attributes:
        name: Profile identifier ("strict" or "dev")
        allowed_imports: Modules that can be imported in sandbox
        enable_pip: Whether pip_install() is available
        enable_network: Whether network access is allowed (for search tools)
        enable_filesystem: Whether filesystem access is allowed
        timeout_seconds: Default execution timeout
        output_char_limit: Maximum output characters
        description: Human-readable description
    """
    name: str
    allowed_imports: FrozenSet[str]
    enable_pip: bool = False
    enable_network: bool = False
    enable_filesystem: bool = False
    timeout_seconds: float = 30.0
    output_char_limit: int = 8192
    description: str = ""

    def with_imports(self, *imports: str) -> "SecurityProfile":
        """Return new profile with additional allowed imports."""
        new_imports = self.allowed_imports | frozenset(imports)
        return SecurityProfile(
            name=self.name,
            allowed_imports=new_imports,
            enable_pip=self.enable_pip,
            enable_network=self.enable_network,
            enable_filesystem=self.enable_filesystem,
            timeout_seconds=self.timeout_seconds,
            output_char_limit=self.output_char_limit,
            description=self.description,
        )


# Core safe imports - minimal set for computation
_CORE_IMPORTS: FrozenSet[str] = frozenset([
    "collections",
    "datetime",
    "functools",
    "itertools",
    "json",
    "math",
    "random",
    "re",
    "statistics",
    "string",
    "textwrap",
    "typing",
])

# Extended imports for data processing
_DATA_IMPORTS: FrozenSet[str] = frozenset([
    "csv",
    "decimal",
    "hashlib",
    "heapq",
    "operator",
    "uuid",
])

# Network-related imports (only for dev mode)
_NETWORK_IMPORTS: FrozenSet[str] = frozenset([
    "http",
    "urllib",
])


# Predefined profiles
STRICT_PROFILE = SecurityProfile(
    name="strict",
    allowed_imports=_CORE_IMPORTS,
    enable_pip=False,
    enable_network=False,
    enable_filesystem=False,
    timeout_seconds=30.0,
    output_char_limit=8192,
    description="Production default. No network, no filesystem, limited imports.",
)

DEV_PROFILE = SecurityProfile(
    name="dev",
    allowed_imports=_CORE_IMPORTS | _DATA_IMPORTS | _NETWORK_IMPORTS,
    enable_pip=True,
    enable_network=True,
    enable_filesystem=False,  # Still no raw filesystem access
    timeout_seconds=60.0,
    output_char_limit=16384,
    description="Development mode. Allows network for search tools, pip install.",
)

_PROFILES = {
    "strict": STRICT_PROFILE,
    "dev": DEV_PROFILE,
}

# Environment variable for profile selection
SECURITY_PROFILE_ENV_VAR = "ENZU_SECURITY_PROFILE"


def get_security_profile(
    name: Optional[str] = None,
    *,
    default: str = "strict",
) -> SecurityProfile:
    """Get security profile by name.

    Args:
        name: Profile name. If None, reads from ENZU_SECURITY_PROFILE env var.
        default: Default profile if env var not set.

    Returns:
        SecurityProfile instance

    Raises:
        ValueError: If profile name is invalid
    """
    if name is None:
        name = os.environ.get(SECURITY_PROFILE_ENV_VAR, default)

    name = name.lower()
    if name not in _PROFILES:
        valid = ", ".join(_PROFILES.keys())
        raise ValueError(f"Invalid security profile: {name!r}. Valid: {valid}")

    return _PROFILES[name]


def list_security_profiles() -> dict[str, str]:
    """List available security profiles with descriptions."""
    return {name: profile.description for name, profile in _PROFILES.items()}


def register_security_profile(profile: SecurityProfile) -> None:
    """Register a custom security profile.

    Use with caution - custom profiles can weaken security.
    """
    _PROFILES[profile.name] = profile


__all__ = [
    "SecurityProfile",
    "SecurityProfileName",
    "get_security_profile",
    "list_security_profiles",
    "register_security_profile",
    "STRICT_PROFILE",
    "DEV_PROFILE",
    "SECURITY_PROFILE_ENV_VAR",
]
