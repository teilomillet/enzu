"""
Container-based sandbox runner for gov-grade isolation (Phase 4).

Provides stronger isolation than subprocess via Docker containers with:
- seccomp profiles to restrict syscalls
- Network namespace isolation (no direct network access)
- Read-only rootfs with tmpfs scratch space
- cgroups for resource limits (CPU, memory, pids)

Isolation hierarchy (weakest to strongest):
1. subprocess + resource.setrlimit (Phase 1) - basic isolation
2. Docker + seccomp (Phase 4) - production gov-grade
3. gVisor (future) - defense-in-depth
4. Firecracker microVM (future) - highest sensitivity

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                     HOST PROCESS                             │
    │  ContainerSandboxRunner                                      │
    │     │                                                        │
    │     ├── serialize(code, namespace) ──► Docker container      │
    │     │                                      │                 │
    │     │                              ┌───────▼───────────────┐ │
    │     │                              │ CONTAINER             │ │
    │     │                              │ ┌─────────────────┐   │ │
    │     │                              │ │ seccomp filter  │   │ │
    │     │                              │ │ network: none   │   │ │
    │     │                              │ │ rootfs: ro      │   │ │
    │     │                              │ │ /tmp: tmpfs     │   │ │
    │     │                              │ └─────────────────┘   │ │
    │     │                              │ exec(code, ns)        │ │
    │     │                              │      │                │ │
    │     │                              │      ▼                │ │
    │     │                              │ result → stdout       │ │
    │     │                              └───────────────────────┘ │
    │     │                                      │                 │
    │     └── deserialize(stdout) ◄──────────────┘                 │
    └─────────────────────────────────────────────────────────────┘

Usage:
    from enzu.isolation.container import ContainerSandboxRunner, ContainerConfig
    
    runner = ContainerSandboxRunner()
    result = runner.run(
        code="FINAL(sum(data))",
        namespace={"data": [1, 2, 3]},
        config=ContainerConfig(network_mode="none"),
    )

Requirements:
    - Docker daemon running
    - Python base image available (configurable)
    - Sufficient permissions to run containers

"""
from __future__ import annotations

import json
import pickle
import subprocess
import base64
import os
import tempfile
import logging
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Callable, cast, Dict, List, Optional, Set
from enum import Enum

from enzu.isolation.runner import SandboxResult
from enzu.isolation.ipc import IPCBridge

logger = logging.getLogger(__name__)


class IsolationLevel(Enum):
    """Isolation level for container sandbox."""
    # Subprocess only (Phase 1) - fastest, weakest isolation
    SUBPROCESS = "subprocess"
    # Docker container with basic isolation
    CONTAINER = "container"
    # Docker with seccomp profile (gov-grade)
    CONTAINER_SECCOMP = "container_seccomp"
    # gVisor runtime (defense-in-depth) - future
    GVISOR = "gvisor"
    # Firecracker microVM (highest sensitivity) - future
    FIRECRACKER = "firecracker"


@dataclass
class ContainerConfig:
    """
    Configuration for container-based sandbox execution.
    
    The image field accepts either:
    - A string Docker tag (e.g., "python:3.11-slim")
    - A SandboxImage instance (from enzu.sandbox.image)
    - A BuiltImage instance (from SandboxImage.build())
    
    When SandboxImage is passed, it's built on first use and cached.
    """
    
    # Resource limits
    max_cpu_seconds: float = 30.0
    max_memory_mb: int = 512
    max_pids: int = 50
    
    # Network isolation: "none" (default, no network) or "egress_proxy" (controlled)
    network_mode: str = "none"
    
    # Allowed egress endpoints (when network_mode="egress_proxy")
    allowed_egress: List[str] = field(default_factory=list)
    
    # Wall-clock timeout
    timeout_seconds: float = 60.0
    
    # Docker image: string tag, SandboxImage, or BuiltImage
    # SandboxImage/BuiltImage from enzu.sandbox.image module
    image: Any = "python:3.11-slim"
    
    # Enable seccomp profile for syscall filtering
    enable_seccomp: bool = True
    
    # Read-only rootfs (scratch in tmpfs)
    read_only_rootfs: bool = True
    
    # Tmpfs size for scratch space
    tmpfs_size_mb: int = 64
    
    # Allowed Python imports in sandbox
    allowed_imports: Set[str] = field(default_factory=lambda: {
        "re", "math", "json", "datetime", "collections", "itertools", "functools"
    })
    
    # Custom seccomp profile path (None = use default restrictive profile)
    seccomp_profile: Optional[str] = None
    
    def get_image_tag(self) -> str:
        """
        Resolve image to Docker tag string.
        
        Handles three cases:
        1. String: return as-is
        2. BuiltImage: return .tag
        3. SandboxImage: build and return tag
        
        Returns:
            Docker image tag string
        """
        # Import here to avoid circular dependency
        # SandboxImage/BuiltImage are in enzu.sandbox.image
        if isinstance(self.image, str):
            return self.image
        
        # Check for BuiltImage (has .tag attribute)
        if hasattr(self.image, "tag") and isinstance(self.image.tag, str):
            return self.image.tag
        
        # Check for SandboxImage (has .build() method)
        if hasattr(self.image, "build") and callable(self.image.build):
            built = self.image.build()
            return built.tag
        
        # Fallback: convert to string
        return str(self.image)


# Seccomp profile for gov-grade isolation
# Blocks dangerous syscalls while allowing Python execution
DEFAULT_SECCOMP_PROFILE = {
    "defaultAction": "SCMP_ACT_ERRNO",
    "architectures": ["SCMP_ARCH_X86_64", "SCMP_ARCH_AARCH64"],
    "syscalls": [
        # Allow basic syscalls needed for Python
        {
            "names": [
                # Process
                "exit", "exit_group", "rt_sigreturn", "rt_sigaction",
                "rt_sigprocmask", "getpid", "gettid", "getuid", "geteuid",
                "getgid", "getegid", "getppid",
                # Memory
                "brk", "mmap", "munmap", "mprotect", "mremap",
                # File (read-only operations)
                "read", "pread64", "readv", "write", "writev", "pwrite64",
                "open", "openat", "close", "fstat", "newfstatat", "stat",
                "lstat", "access", "faccessat", "lseek", "dup", "dup2",
                "fcntl", "flock", "fsync", "fdatasync", "getcwd", "readlink",
                "readlinkat", "getdents", "getdents64",
                # Time
                "clock_gettime", "clock_getres", "gettimeofday", "nanosleep",
                # Misc
                "futex", "set_robust_list", "get_robust_list",
                "getrandom", "sched_yield", "sched_getaffinity",
                "arch_prctl", "set_tid_address", "prctl", "uname",
                "poll", "ppoll", "select", "pselect6", "epoll_create1",
                "epoll_ctl", "epoll_wait", "epoll_pwait", "eventfd2",
                "pipe", "pipe2", "ioctl",
            ],
            "action": "SCMP_ACT_ALLOW"
        },
        # Block network operations (EACCES)
        {
            "names": [
                "socket", "connect", "accept", "accept4", "bind", "listen",
                "sendto", "recvfrom", "sendmsg", "recvmsg", "shutdown",
                "getsockname", "getpeername", "socketpair", "setsockopt",
                "getsockopt",
            ],
            "action": "SCMP_ACT_ERRNO",
            "args": [],
            "errnoRet": 1  # EPERM
        },
        # Block fork/clone (no fork bombs)
        # Note: execve/execveat are allowed because podman run needs them to start Python.
        # Other protections (no network, read-only fs, dropped caps, no-new-privileges)
        # prevent escape even with execve.
        {
            "names": [
                "clone", "clone3", "fork", "vfork",
            ],
            "action": "SCMP_ACT_ERRNO",
            "args": [],
            "errnoRet": 1  # EPERM
        },
    ]
}


# Worker script for container execution
# Minimal code to reduce attack surface
_CONTAINER_WORKER_SCRIPT = '''
import sys
import json
import pickle
import base64
import io
import struct
from contextlib import redirect_stdout

# IPC Protocol Constants
_IPC_MAX_MESSAGE_SIZE = 10 * 1024 * 1024
_IPC_HEADER_SIZE = 4

def _ipc_encode_message(msg_type, payload):
    """Encode a message with length prefix."""
    message = {"type": msg_type, "payload": payload}
    json_bytes = json.dumps(message).encode("utf-8")
    if len(json_bytes) > _IPC_MAX_MESSAGE_SIZE:
        raise ValueError(f"Message too large: {len(json_bytes)}")
    length_prefix = struct.pack(">I", len(json_bytes))
    return length_prefix + json_bytes

def _ipc_read_message(stream):
    """Read a length-prefixed message from stream."""
    header = stream.read(_IPC_HEADER_SIZE)
    if not header or len(header) < _IPC_HEADER_SIZE:
        return None
    length = struct.unpack(">I", header)[0]
    if length > _IPC_MAX_MESSAGE_SIZE:
        raise ValueError(f"Message too large: {length}")
    data = stream.read(length)
    if len(data) < length:
        raise ValueError(f"Incomplete message")
    return json.loads(data.decode("utf-8"))

def _ipc_write_message(stream, msg_type, payload):
    """Write a length-prefixed message to stream."""
    encoded = _ipc_encode_message(msg_type, payload)
    stream.write(encoded)
    stream.flush()

def _ipc_send_result(result):
    """Send final result to host."""
    _ipc_write_message(sys.stdout.buffer, "RESULT", result)

def _ipc_send_error(error):
    """Send error to host."""
    _ipc_write_message(sys.stdout.buffer, "ERROR", {"error": str(error)})

def _make_ipc_llm_query():
    """Create llm_query function that uses IPC."""
    def llm_query(prompt):
        _ipc_write_message(sys.stdout.buffer, "LLM_QUERY", {"prompt": str(prompt)})
        response = _ipc_read_message(sys.stdin.buffer)
        if response is None:
            raise RuntimeError("IPC connection closed")
        payload = response.get("payload", {})
        if not payload.get("success", False):
            error = payload.get("error", "Unknown error")
            raise RuntimeError(f"llm_query failed: {error}")
        return payload.get("result", "")
    return llm_query

def _make_ipc_llm_batch():
    """Create llm_batch function that uses IPC."""
    def llm_batch(prompts):
        prompts_list = [str(p) for p in prompts]
        _ipc_write_message(sys.stdout.buffer, "LLM_BATCH", {"prompts": prompts_list})
        response = _ipc_read_message(sys.stdin.buffer)
        if response is None:
            raise RuntimeError("IPC connection closed")
        payload = response.get("payload", {})
        if not payload.get("success", False):
            error = payload.get("error", "Unknown error")
            raise RuntimeError(f"llm_batch failed: {error}")
        return payload.get("results", [])
    return llm_batch

def run_sandbox(code, namespace, allowed_imports, output_limit, ipc_enabled=False):
    """Execute code with restrictions."""
    import builtins
    import ast

    # Build restricted builtins
    safe_names = [
        "abs", "all", "any", "dict", "enumerate", "float", "int",
        "isinstance", "len", "list", "max", "min", "print", "range",
        "set", "sorted", "str", "sum", "tuple", "ValueError",
        "Exception", "type", "dir", "zip", "bool", "repr", "round",
        "ord", "chr", "map", "filter", "reversed", "slice",
        "NameError", "TypeError", "KeyError", "IndexError", "AttributeError",
    ]
    safe_builtins = {name: getattr(builtins, name) for name in safe_names if hasattr(builtins, name)}

    def restricted_import(name, *args, **kwargs):
        if name in allowed_imports:
            return __import__(name, *args, **kwargs)
        raise ImportError(f"Import blocked: {name}")

    safe_builtins["__import__"] = restricted_import

    # Answer state for FINAL()
    answer_state = {"content": None, "ready": False}

    def final_fn(content):
        answer_state["content"] = str(content)
        answer_state["ready"] = True

    # Build namespace
    exec_namespace = dict(namespace)
    exec_namespace["__builtins__"] = safe_builtins
    exec_namespace["FINAL"] = final_fn

    # Inject IPC-backed llm_query/llm_batch if enabled
    if ipc_enabled:
        exec_namespace["llm_query"] = _make_ipc_llm_query()
        exec_namespace["llm_batch"] = _make_ipc_llm_batch()

    # Validate code safety (block dunders)
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise RuntimeError(f"Blocked: dunder attribute access ({node.attr})")
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                if node.id not in ("__builtins__",):
                    raise RuntimeError(f"Blocked: dunder name ({node.id})")
    except SyntaxError:
        pass  # Let exec() surface syntax errors

    # Execute with stdout capture (only captures print() output, not IPC)
    stdout_buffer = io.StringIO()
    error = None

    try:
        with redirect_stdout(stdout_buffer):
            exec(code, exec_namespace, None)
    except Exception as e:
        error = str(e)

    stdout = stdout_buffer.getvalue()
    if len(stdout) > output_limit:
        stdout = stdout[:output_limit]

    # Extract picklable namespace updates
    updates = {}
    skip_keys = {"__builtins__", "FINAL", "FINAL_VAR", "llm_query", "llm_batch"}
    for key, value in exec_namespace.items():
        if key.startswith("_") or key in skip_keys:
            continue
        try:
            pickle.dumps(value)
            updates[key] = value
        except (pickle.PicklingError, TypeError, AttributeError):
            pass

    return {
        "stdout": stdout,
        "error": error,
        "final_answer": answer_state.get("content"),
        "namespace_updates": updates,
    }

def main():
    # Check for IPC mode (binary stdin will have length-prefixed message)
    # Peek at first 4 bytes to detect IPC mode
    first_bytes = sys.stdin.buffer.peek(4)[:4]
    ipc_mode = len(first_bytes) == 4 and first_bytes[0:1] != b"{"

    if ipc_mode:
        # IPC mode: read INIT message
        msg = _ipc_read_message(sys.stdin.buffer)
        if msg is None or msg.get("type") != "INIT":
            _ipc_send_error("Expected INIT message")
            return

        payload = msg.get("payload", {})
        code = payload["code"]
        namespace = pickle.loads(base64.b64decode(payload["namespace_b64"]))
        allowed_imports = set(payload["allowed_imports"])
        output_limit = payload.get("output_limit", 8192)
        ipc_enabled = payload.get("ipc_enabled", False)

        try:
            # Run sandbox with IPC enabled
            result = run_sandbox(code, namespace, allowed_imports, output_limit, ipc_enabled)

            # Serialize namespace updates
            result["namespace_updates"] = base64.b64encode(
                pickle.dumps(result["namespace_updates"])
            ).decode("ascii")

            # Send result via IPC
            _ipc_send_result(result)
        except Exception as e:
            _ipc_send_error(str(e))
    else:
        # Legacy mode: JSON on stdin, JSON on stdout
        input_data = sys.stdin.read()
        payload = json.loads(input_data)

        code = payload["code"]
        namespace = pickle.loads(base64.b64decode(payload["namespace_b64"]))
        allowed_imports = set(payload["allowed_imports"])
        output_limit = payload.get("output_limit", 8192)

        # Run sandbox (no IPC in legacy mode)
        result = run_sandbox(code, namespace, allowed_imports, output_limit, ipc_enabled=False)

        # Serialize result
        result["namespace_updates"] = base64.b64encode(
            pickle.dumps(result["namespace_updates"])
        ).decode("ascii")

        # Write to stdout
        print(json.dumps(result))

if __name__ == "__main__":
    main()
'''


def _check_container_runtime_available() -> bool:
    """Check if any container runtime (Podman or Docker) is available."""
    from enzu.isolation.runtime import _check_podman_works, _check_docker_works
    # Try Podman first (preferred for rootless/daemonless security), then Docker
    return _check_podman_works() or _check_docker_works()


def _pull_image_if_needed(image: str, timeout: float = 120, runtime_cmd: Optional[str] = None) -> bool:
    """Pull container image if not present locally. Works with Podman or Docker."""
    if runtime_cmd is None:
        # Auto-detect runtime
        from enzu.isolation.runtime import detect_runtime, get_runtime_command
        try:
            runtime_cmd = get_runtime_command(detect_runtime())
        except RuntimeError:
            runtime_cmd = "podman"  # Default to Podman
    try:
        # Check if image exists
        result = subprocess.run(
            [runtime_cmd, "image", "inspect", image],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            return True

        # Pull image
        logger.info("Pulling container image: %s (using %s)", image, runtime_cmd)
        result = subprocess.run(
            [runtime_cmd, "pull", image],
            capture_output=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.warning("Failed to pull image %s: %s", image, e)
        return False


class ContainerSandboxRunner:
    """
    Execute code in Docker container with gov-grade isolation.
    
    Each execution spawns a new container with:
    - Network isolation (network=none by default)
    - seccomp syscall filtering
    - Resource limits via cgroups (CPU, memory, pids)
    - Read-only rootfs with tmpfs scratch
    
    Example:
        runner = ContainerSandboxRunner()
        result = runner.run(
            code="x = sum(data)\\nFINAL(x)",
            namespace={"data": [1, 2, 3]},
            config=ContainerConfig(max_memory_mb=256),
        )
        print(result.final_answer)  # "6"
    
    Fallback:
        When Docker is unavailable, falls back to subprocess isolation.
        Set fallback_to_subprocess=False to require container isolation.
    """
    
    def __init__(
        self,
        output_limit: int = 8192,
        fallback_to_subprocess: bool = True,
    ) -> None:
        """
        Args:
            output_limit: Max stdout characters to capture.
            fallback_to_subprocess: If True, fall back to subprocess when Docker unavailable.
        """
        self._output_limit = output_limit
        self._fallback = fallback_to_subprocess
        self._container_available: Optional[bool] = None
        self._runtime_command: Optional[str] = None
        self._seccomp_profile_path: Optional[str] = None

    def _ensure_docker(self) -> bool:
        """Check container runtime availability (cached). Supports Podman and Docker."""
        if self._container_available is None:
            from enzu.isolation.runtime import detect_runtime, get_runtime_command
            try:
                runtime = detect_runtime()
                self._runtime_command = get_runtime_command(runtime)
                self._container_available = True
                logger.info(f"Container runtime available: {runtime.value}")
            except RuntimeError:
                self._container_available = False
                self._runtime_command = None
                logger.warning("No container runtime available (tried Podman and Docker)")
        return self._container_available
    
    def _get_seccomp_profile_path(self) -> str:
        """Get or create seccomp profile file."""
        if self._seccomp_profile_path is None:
            # Write profile to temp file
            fd, path = tempfile.mkstemp(suffix=".json", prefix="enzu_seccomp_")
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(DEFAULT_SECCOMP_PROFILE, f)
                self._seccomp_profile_path = path
            except Exception:
                os.close(fd)
                os.unlink(path)
                raise
        return self._seccomp_profile_path
    
    def run(
        self,
        code: str,
        namespace: Optional[Dict[str, Any]] = None,
        config: Optional[ContainerConfig] = None,
        llm_query: Optional[Callable] = None,
        llm_batch: Optional[Callable] = None,
    ) -> SandboxResult:
        """
        Execute code in isolated container.

        Args:
            code: Python code to execute
            namespace: Initial namespace (must be picklable)
            config: Container configuration
            llm_query: Callback for single LLM queries (enables IPC mode)
            llm_batch: Callback for batch LLM queries (enables IPC mode)

        Returns:
            SandboxResult with stdout, error, final_answer, etc.
        """
        if config is None:
            config = ContainerConfig()
        
        if namespace is None:
            namespace = {}
        
        # Check Docker availability
        if not self._ensure_docker():
            if self._fallback:
                logger.debug("Falling back to subprocess isolation")
                from enzu.isolation.runner import SandboxRunner, SandboxConfig
                subprocess_config = SandboxConfig(
                    max_cpu_seconds=config.max_cpu_seconds,
                    max_memory_mb=config.max_memory_mb,
                    max_pids=config.max_pids,
                    timeout_seconds=config.timeout_seconds,
                    allowed_imports=config.allowed_imports,
                )
                return SandboxRunner(
                    output_limit=self._output_limit,
                    llm_query=llm_query,
                    llm_batch=llm_batch,
                ).run(
                    code=code,
                    namespace=namespace,
                    config=subprocess_config,
                )
            else:
                return SandboxResult(
                    stdout="",
                    error="Docker not available and fallback disabled",
                    final_answer=None,
                    namespace_updates={},
                    resource_usage={},
                    exit_code=-1,
                )
        
        # Serialize namespace
        try:
            namespace_bytes = pickle.dumps(namespace)
            namespace_b64 = base64.b64encode(namespace_bytes).decode("ascii")
        except (pickle.PicklingError, TypeError) as e:
            return SandboxResult(
                stdout="",
                error=f"Namespace serialization failed: {e}",
                final_answer=None,
                namespace_updates={},
                resource_usage={},
                exit_code=-1,
            )
        
        # Resolve image to Docker tag (handles SandboxImage/BuiltImage)
        # get_image_tag() builds SandboxImage if needed
        image_tag = config.get_image_tag()
        
        # Ensure image is available (pull if not local)
        if not _pull_image_if_needed(image_tag, runtime_cmd=self._runtime_command):
            if self._fallback:
                logger.warning("Image pull failed, falling back to subprocess")
                from enzu.isolation.runner import SandboxRunner, SandboxConfig
                subprocess_config = SandboxConfig(
                    max_cpu_seconds=config.max_cpu_seconds,
                    max_memory_mb=config.max_memory_mb,
                    timeout_seconds=config.timeout_seconds,
                    allowed_imports=config.allowed_imports,
                )
                return SandboxRunner(
                    output_limit=self._output_limit,
                    llm_query=llm_query,
                    llm_batch=llm_batch,
                ).run(
                    code=code,
                    namespace=namespace,
                    config=subprocess_config,
                )
            return SandboxResult(
                stdout="",
                error=f"Failed to pull Docker image: {image_tag}",
                final_answer=None,
                namespace_updates={},
                resource_usage={},
                exit_code=-1,
            )
        
        # Determine if IPC mode should be used
        use_ipc = llm_query is not None or llm_batch is not None
        
        # Build input payload
        payload = {
            "code": code,
            "namespace_b64": namespace_b64,
            "allowed_imports": list(config.allowed_imports),
            "output_limit": self._output_limit,
            "ipc_enabled": use_ipc,
        }
        
        # Build container command (works with both Podman and Docker)
        runtime_cmd = self._runtime_command or "podman"
        container_args = [
            runtime_cmd, "run",
            "--rm",  # Remove container after exit
            "-i",    # Interactive (stdin)

            # Run as non-root user (UID 1000) for defense in depth
            "--user=1000:1000",

            # Resource limits (cgroups)
            f"--memory={config.max_memory_mb}m",
            f"--memory-swap={config.max_memory_mb}m",  # No swap
            f"--pids-limit={config.max_pids}",
            f"--cpus={config.max_cpu_seconds / config.timeout_seconds}",  # CPU quota

            # Network isolation
            f"--network={config.network_mode}",
        ]

        # Read-only rootfs
        if config.read_only_rootfs:
            container_args.append("--read-only")
            container_args.append(f"--tmpfs=/tmp:size={config.tmpfs_size_mb}m,noexec")

        # Seccomp profile
        if config.enable_seccomp:
            profile_path = config.seccomp_profile or self._get_seccomp_profile_path()
            container_args.append(f"--security-opt=seccomp={profile_path}")

        # Drop all capabilities
        container_args.append("--cap-drop=ALL")

        # No new privileges
        container_args.append("--security-opt=no-new-privileges")
        
        # Image and command (use resolved image_tag from above)
        container_args.extend([
            image_tag,
            "python", "-c", _CONTAINER_WORKER_SCRIPT,
        ])
        
        # Run container with IPC or legacy mode
        if use_ipc:
            return self._run_with_ipc(
                container_args=container_args,
                payload=payload,
                timeout=config.timeout_seconds,
                llm_query=llm_query,
                llm_batch=llm_batch,
            )
        else:
            return self._run_legacy(
                container_args=container_args,
                payload=payload,
                timeout=config.timeout_seconds,
            )
    
    def _run_with_ipc(
        self,
        container_args: List[str],
        payload: Dict[str, Any],
        timeout: float,
        llm_query: Optional[Callable[[str], str]],
        llm_batch: Optional[Callable[[List[str]], List[str]]],
    ) -> SandboxResult:
        """Run container with IPC bridge for llm_query/llm_batch support."""
        proc = None
        try:
            proc = subprocess.Popen(
                container_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,  # Binary mode for IPC
            )
            
            # Cast is safe: stdin/stdout are guaranteed non-None when PIPE is used
            bridge = IPCBridge(
                stdin=cast(BinaryIO, proc.stdin),
                stdout=cast(BinaryIO, proc.stdout),
                llm_query=llm_query,
                llm_batch=llm_batch,
            )
            
            result_data = bridge.run(payload, timeout=timeout)
            
            proc.wait(timeout=5)
            exit_code = proc.returncode
            
        except TimeoutError as e:
            if proc:
                proc.kill()
                proc.wait()
            return SandboxResult(
                stdout="",
                error=f"Timeout: {e}",
                final_answer=None,
                namespace_updates={},
                resource_usage={},
                exit_code=-1,
                timed_out=True,
            )
        except RuntimeError as e:
            if proc:
                proc.kill()
                proc.wait()
            stderr_content = ""
            if proc and proc.stderr:
                try:
                    stderr_content = proc.stderr.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
            error_msg = str(e)
            if stderr_content:
                error_msg = f"{error_msg}: {stderr_content}"
            return SandboxResult(
                stdout="",
                error=error_msg,
                final_answer=None,
                namespace_updates={},
                resource_usage={},
                exit_code=-1,
            )
        except Exception as e:
            if proc:
                proc.kill()
                proc.wait()
            return SandboxResult(
                stdout="",
                error=f"Container execution failed: {e}",
                final_answer=None,
                namespace_updates={},
                resource_usage={},
                exit_code=-1,
            )
        
        # Deserialize namespace updates
        try:
            updates_bytes = base64.b64decode(result_data.get("namespace_updates", ""))
            namespace_updates = pickle.loads(updates_bytes) if updates_bytes else {}
        except Exception:
            namespace_updates = {}
        
        return SandboxResult(
            stdout=result_data.get("stdout", ""),
            error=result_data.get("error"),
            final_answer=result_data.get("final_answer"),
            namespace_updates=namespace_updates,
            resource_usage={},
            exit_code=exit_code,
        )
    
    def _run_legacy(
        self,
        container_args: List[str],
        payload: Dict[str, Any],
        timeout: float,
    ) -> SandboxResult:
        """Run container with legacy JSON stdin/stdout mode."""
        try:
            proc = subprocess.Popen(
                container_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            
            stdout, stderr = proc.communicate(
                input=json.dumps(payload),
                timeout=timeout,
            )
            
            exit_code = proc.returncode
            
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return SandboxResult(
                stdout="",
                error=f"Timeout: execution exceeded {timeout}s",
                final_answer=None,
                namespace_updates={},
                resource_usage={},
                exit_code=-1,
                timed_out=True,
            )
        except Exception as e:
            return SandboxResult(
                stdout="",
                error=f"Container execution failed: {e}",
                final_answer=None,
                namespace_updates={},
                resource_usage={},
                exit_code=-1,
            )
        
        # Parse result
        if exit_code != 0:
            error_msg = f"Container exited with code {exit_code}"
            resource_exceeded = False
            
            if exit_code == 137:  # SIGKILL (OOM killer)
                error_msg = "Killed: memory limit exceeded"
                resource_exceeded = True
            elif stderr:
                error_msg = f"{error_msg}: {stderr[:500]}"
            
            return SandboxResult(
                stdout=stdout[:self._output_limit] if stdout else "",
                error=error_msg,
                final_answer=None,
                namespace_updates={},
                resource_usage={},
                exit_code=exit_code,
                resource_exceeded=resource_exceeded,
            )
        
        # Parse JSON result
        try:
            result_data = json.loads(stdout)
        except json.JSONDecodeError as e:
            return SandboxResult(
                stdout=stdout[:500],
                error=f"Result parse failed: {e}",
                final_answer=None,
                namespace_updates={},
                resource_usage={},
                exit_code=exit_code,
            )
        
        # Deserialize namespace updates
        try:
            updates_bytes = base64.b64decode(result_data.get("namespace_updates", ""))
            namespace_updates = pickle.loads(updates_bytes) if updates_bytes else {}
        except Exception:
            namespace_updates = {}
        
        return SandboxResult(
            stdout=result_data.get("stdout", ""),
            error=result_data.get("error"),
            final_answer=result_data.get("final_answer"),
            namespace_updates=namespace_updates,
            resource_usage={},
            exit_code=exit_code,
        )
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        if self._seccomp_profile_path and os.path.exists(self._seccomp_profile_path):
            try:
                os.unlink(self._seccomp_profile_path)
            except OSError:
                pass
            self._seccomp_profile_path = None


class ContainerSandbox:
    """
    Stateful sandbox using container isolation.
    
    Maintains namespace across multiple exec() calls.
    Drop-in replacement for PythonSandbox when container isolation is required.
    
    Note: Each exec() spawns a new container (expensive but isolated).
    For batch operations, accumulate code and exec() once.
    """
    
    def __init__(
        self,
        *,
        data: Any = None,
        namespace: Optional[Dict[str, Any]] = None,
        allowed_imports: Optional[Set[str]] = None,
        output_char_limit: int = 8192,
        config: Optional[ContainerConfig] = None,
        fallback_to_subprocess: bool = True,
        llm_query: Optional[Callable[[str], str]] = None,
        llm_batch: Optional[Callable[[List[str]], List[str]]] = None,
    ) -> None:
        self._runner = ContainerSandboxRunner(
            output_limit=output_char_limit,
            fallback_to_subprocess=fallback_to_subprocess,
        )
        self._config = config or ContainerConfig()
        self._llm_query = llm_query
        self._llm_batch = llm_batch
        
        if allowed_imports:
            self._config.allowed_imports = set(allowed_imports)
        
        self._namespace: Dict[str, Any] = dict(namespace or {})
        if data is not None:
            self._namespace["data"] = data
        
        self._answer: Dict[str, Any] = {"content": "", "ready": False}
    
    def exec(self, code: str) -> SandboxResult:
        """Execute code in container, updating internal namespace."""
        result = self._runner.run(
            code=code,
            namespace=self._namespace,
            config=self._config,
            llm_query=self._llm_query,
            llm_batch=self._llm_batch,
        )
        
        # Update namespace with results
        if result.namespace_updates:
            self._namespace.update(result.namespace_updates)
        
        # Update answer state
        if result.final_answer is not None:
            self._answer["content"] = result.final_answer
            self._answer["ready"] = True
        
        return result
    
    @property
    def answer(self) -> Dict[str, Any]:
        """Access answer state set by FINAL()."""
        return self._answer
    
    def get_global(self, name: str) -> Any:
        """Get variable from namespace."""
        return self._namespace.get(name)
    
    @property
    def namespace(self) -> Dict[str, Any]:
        """Current namespace state."""
        return self._namespace
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self._runner.cleanup()


def is_container_available() -> bool:
    """Check if container isolation is available (Podman or Docker)."""
    return _check_container_runtime_available()
