"""
Container runtime detection and abstraction.

Handles detection of available container runtimes (Podman vs Docker)
and provides unified command interface.
"""

from __future__ import annotations

import shutil
import subprocess
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ContainerRuntime(Enum):
    PODMAN = "podman"
    DOCKER = "docker"


def _check_podman_works() -> bool:
    """Verify podman is actually usable."""
    try:
        subprocess.run(["podman", "info"], capture_output=True, check=True, timeout=5)
        return True
    except (subprocess.SubprocessError, OSError):
        return False


def _check_docker_works() -> bool:
    """Verify docker is actually usable."""
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=5)
        return True
    except (subprocess.SubprocessError, OSError):
        return False


def detect_runtime() -> ContainerRuntime:
    """
    Detect available container runtime.

    Priority:
    1. Podman (preferred for rootless/daemonless security)
    2. Docker (fallback)

    Raises:
        RuntimeError: If no supported runtime is found/working.
    """
    if shutil.which("podman") and _check_podman_works():
        logger.info("Detected container runtime: Podman")
        return ContainerRuntime.PODMAN

    if shutil.which("docker") and _check_docker_works():
        logger.info("Detected container runtime: Docker")
        return ContainerRuntime.DOCKER

    raise RuntimeError(
        "No container runtime available. Please install Podman (preferred) or Docker."
    )


def get_runtime_command(runtime: ContainerRuntime) -> str:
    """Return the CLI command for the runtime."""
    return runtime.value
