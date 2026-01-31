"""
SandboxImage: Chainable DSL for building container images.

Part of the sandbox evolution plan (Phase 1). Provides a Modal-style
fluent API for defining container environments.

Integration points:
- ContainerConfig.image accepts either a string (Docker tag) or BuiltImage
- create_sandbox() in enzu/rlm/sandbox_factory.py uses the image
- RLMEngine accepts sandbox_image parameter

Usage:
    image = (
        SandboxImage.python("3.11")
        .apt("git", "curl")
        .pip("pandas", "numpy")
        .env(TOKENIZERS_PARALLELISM="false")
    )

    built = image.build()  # Returns BuiltImage with tag

    engine = RLMEngine(
        isolation="container",
        sandbox_image=image,
    )
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SandboxImage:
    """
    Chainable image builder for Enzu sandboxes.

    Each method returns self for chaining. Layers are recorded in order
    and converted to Dockerfile instructions via to_dockerfile().

    Layer types:
    - pip: Install Python packages
    - apt: Install system packages
    - run: Execute shell commands
    - env: Set environment variables
    - copy: Copy files into image
    - workdir: Set working directory
    """

    # Base Docker image (default: python:3.11-slim)
    _base: str = "python:3.11-slim"

    # Ordered list of (layer_type, args) tuples
    # Each layer becomes one or more Dockerfile instructions
    _layers: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)

    @classmethod
    def python(cls, version: str = "3.11") -> "SandboxImage":
        """Start with a Python base image."""
        return cls(_base=f"python:{version}-slim")

    @classmethod
    def debian(cls, python_version: str = "3.11") -> "SandboxImage":
        """Start with Debian slim + Python."""
        return cls(_base=f"python:{python_version}-slim-bookworm")

    @classmethod
    def from_image(cls, image: str) -> "SandboxImage":
        """Start from any Docker image."""
        return cls(_base=image)

    def pip(self, *packages: str, index_url: Optional[str] = None) -> "SandboxImage":
        """
        Install Python packages via pip.

        Each pip() call becomes a separate layer for better caching.
        Group related packages in one call when they change together.
        """
        self._layers.append(("pip", {"packages": packages, "index_url": index_url}))
        return self

    def apt(self, *packages: str) -> "SandboxImage":
        """
        Install system packages via apt-get.

        Includes apt-get update and cleanup in a single RUN to minimize layer size.
        """
        self._layers.append(("apt", {"packages": packages}))
        return self

    def run(self, *commands: str) -> "SandboxImage":
        """
        Run shell commands during image build.

        Each command becomes a separate RUN instruction.
        For commands that should run together, chain with && in a single string.
        """
        self._layers.append(("run", {"commands": commands}))
        return self

    def env(self, **variables: str) -> "SandboxImage":
        """
        Set environment variables.

        Variables are available during build and at runtime.
        """
        self._layers.append(("env", {"variables": variables}))
        return self

    def copy(self, src: str, dst: str) -> "SandboxImage":
        """
        Copy local files into image.

        Note: src must be within the build context (tmpdir during build).
        For absolute paths, consider using a multi-stage build or run().
        """
        self._layers.append(("copy", {"src": src, "dst": dst}))
        return self

    def workdir(self, path: str) -> "SandboxImage":
        """Set working directory for subsequent commands and runtime."""
        self._layers.append(("workdir", {"path": path}))
        return self

    def to_dockerfile(self) -> str:
        """
        Generate Dockerfile from recorded layers.

        Produces deterministic output for consistent content hashing.
        """
        lines = [f"FROM {self._base}"]

        for layer_type, args in self._layers:
            if layer_type == "pip":
                pkgs = " ".join(args["packages"])
                idx = (
                    f"--index-url {args['index_url']} " if args.get("index_url") else ""
                )
                lines.append(f"RUN pip install --no-cache-dir {idx}{pkgs}")

            elif layer_type == "apt":
                pkgs = " ".join(args["packages"])
                # Single RUN with cleanup to minimize layer size
                lines.append(
                    f"RUN apt-get update && "
                    f"apt-get install -y --no-install-recommends {pkgs} && "
                    f"rm -rf /var/lib/apt/lists/*"
                )

            elif layer_type == "run":
                for cmd in args["commands"]:
                    lines.append(f"RUN {cmd}")

            elif layer_type == "env":
                for k, v in args["variables"].items():
                    lines.append(f'ENV {k}="{v}"')

            elif layer_type == "copy":
                lines.append(f"COPY {args['src']} {args['dst']}")

            elif layer_type == "workdir":
                lines.append(f"WORKDIR {args['path']}")

        return "\n".join(lines)

    def content_hash(self) -> str:
        """
        Generate deterministic hash for caching.

        Same Dockerfile content -> same hash -> cache hit.
        Uses SHA256 truncated to 12 chars (collision-resistant for tags).
        """
        dockerfile = self.to_dockerfile()
        return hashlib.sha256(dockerfile.encode()).hexdigest()[:12]

    def build(
        self,
        tag: Optional[str] = None,
        cache: bool = True,
        quiet: bool = False,
    ) -> "BuiltImage":
        """
        Build the container image using Podman (preferred) or Docker.

        Args:
            tag: Image tag. If None, uses enzu-sandbox:{content_hash}
            cache: If True, skip build when image already exists
            quiet: If True, suppress build output

        Returns:
            BuiltImage with tag, cached flag, and dockerfile content

        Raises:
            RuntimeError: If container build fails
        """
        # Detect container runtime (Podman preferred)
        from enzu.isolation.runtime import detect_runtime, get_runtime_command

        try:
            runtime_cmd = get_runtime_command(detect_runtime())
        except RuntimeError:
            runtime_cmd = "podman"  # Default to podman

        content_hash = self.content_hash()
        tag = tag or f"enzu-sandbox:{content_hash}"

        # Check cache: does image already exist?
        if cache:
            inspect_result = subprocess.run(
                [runtime_cmd, "image", "inspect", tag],
                capture_output=True,
            )
            if inspect_result.returncode == 0:
                logger.info("Image cache hit: %s", tag)
                return BuiltImage(tag=tag, cached=True, dockerfile=self.to_dockerfile())

        # Build image
        dockerfile = self.to_dockerfile()
        logger.info("Building image: %s (using %s)", tag, runtime_cmd)

        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile_path = os.path.join(tmpdir, "Dockerfile")
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile)

            cmd = [runtime_cmd, "build", "-t", tag, "-f", dockerfile_path]
            if quiet:
                cmd.append("-q")
            cmd.append(tmpdir)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Container build failed:\n{result.stderr}")

        logger.info("Image built: %s", tag)
        return BuiltImage(tag=tag, cached=False, dockerfile=dockerfile)


@dataclass
class BuiltImage:
    """
    A built Docker image ready for use.

    Returned by SandboxImage.build(). Can be passed to ContainerConfig
    or create_sandbox() to use the custom image.
    """

    # Docker image tag (e.g., "enzu-sandbox:abc123")
    tag: str

    # True if image was found in cache (not rebuilt)
    cached: bool

    # Dockerfile content used to build (for debugging/auditing)
    dockerfile: str
