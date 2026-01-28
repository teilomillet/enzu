"""
Subprocess-based sandbox runner with OS-level isolation.

Problem: Current sandbox uses exec() in-process, sharing memory with all requests.
- No isolation: sandbox escape affects entire service
- No resource limits: runaway code can OOM the system
- Windows incompatibility: SIGALRM timeout doesn't work

Solution: Execute code in subprocess with resource limits.
- Separate address space: no shared memory
- OS-enforced limits: CPU time, memory, file descriptors
- Cross-platform timeout: via subprocess.Popen.communicate(timeout=)

Architecture:
    Parent Process (RLMEngine)
        │
        ├── serialize(code, namespace) ──► subprocess.Popen
        │                                      │
        │                                      ▼
        │                              Child Process
        │                              ├── setrlimit(CPU, MEM)
        │                              ├── exec(code, namespace)
        │                              └── serialize(result) ──► stdout (IPC)
        │
        └── deserialize(stdout) ◄────────────┘

Features:
- Namespace must be picklable (no lambdas, closures)
- llm_query/llm_batch available via IPC protocol
- No network isolation (Phase 4: container/microVM)
"""
from __future__ import annotations

import pickle
import subprocess
import sys
import base64
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Callable, cast, Dict, List, Optional, Set

from enzu.isolation.ipc import IPCBridge


@dataclass
class SandboxConfig:
    """Resource limits for isolated sandbox execution."""
    
    # CPU time limit in seconds (RLIMIT_CPU on Unix)
    max_cpu_seconds: float = 30.0
    
    # Memory limit in MB (RLIMIT_AS on Unix)
    max_memory_mb: int = 512
    
    # Max child processes (RLIMIT_NPROC on Unix)
    max_pids: int = 10
    
    # Max open file descriptors (RLIMIT_NOFILE on Unix)
    max_fds: int = 64
    
    # Wall-clock timeout for subprocess.communicate()
    timeout_seconds: float = 60.0
    
    # Python path (None = same as parent)
    python_executable: Optional[str] = None
    
    # Allowed imports in sandbox
    allowed_imports: Set[str] = field(default_factory=lambda: {
        "re", "math", "json", "datetime", "collections", "itertools", "functools"
    })


@dataclass
class SandboxResult:
    """Result from isolated sandbox execution."""
    
    # Captured stdout
    stdout: str
    
    # Error message if execution failed
    error: Optional[str]
    
    # FINAL() content if called
    final_answer: Optional[str]
    
    # Namespace after execution (picklable values only)
    namespace_updates: Dict[str, Any]
    
    # Resource usage (CPU time, memory)
    resource_usage: Dict[str, Any]
    
    # Exit code from subprocess
    exit_code: int
    
    # True if killed by timeout
    timed_out: bool = False
    
    # True if killed by resource limit
    resource_exceeded: bool = False


# Worker script executed in subprocess
# Minimal code to reduce attack surface
_WORKER_SCRIPT = '''
import sys
import json
import pickle
import base64
import io
import struct
from contextlib import redirect_stdout

# IPC protocol constants and functions
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

# Save original streams before any redirection
_ORIGINAL_STDIN_BUFFER = sys.stdin.buffer
_ORIGINAL_STDOUT_BUFFER = sys.stdout.buffer

def _ipc_send_result(result):
    """Send final result to host."""
    _ipc_write_message(_ORIGINAL_STDOUT_BUFFER, "RESULT", result)

def _ipc_send_error(error):
    """Send error to host."""
    _ipc_write_message(_ORIGINAL_STDOUT_BUFFER, "ERROR", {"error": str(error)})

def _make_ipc_llm_query():
    """Create llm_query function that uses IPC."""
    def llm_query(prompt):
        _ipc_write_message(_ORIGINAL_STDOUT_BUFFER, "LLM_QUERY", {"prompt": str(prompt)})
        response = _ipc_read_message(_ORIGINAL_STDIN_BUFFER)
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
        _ipc_write_message(_ORIGINAL_STDOUT_BUFFER, "LLM_BATCH", {"prompts": prompts_list})
        response = _ipc_read_message(_ORIGINAL_STDIN_BUFFER)
        if response is None:
            raise RuntimeError("IPC connection closed")
        payload = response.get("payload", {})
        if not payload.get("success", False):
            error = payload.get("error", "Unknown error")
            raise RuntimeError(f"llm_batch failed: {error}")
        return payload.get("results", [])
    return llm_batch

def set_resource_limits(config):
    """Apply OS resource limits. Unix only."""
    try:
        import resource
    except ImportError:
        return  # Windows: no resource module
    
    # CPU time (seconds)
    if config.get("max_cpu_seconds"):
        limit = int(config["max_cpu_seconds"])
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (limit, limit))
        except (ValueError, resource.error):
            pass
    
    # Address space (bytes)
    if config.get("max_memory_mb"):
        limit = config["max_memory_mb"] * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except (ValueError, resource.error):
            pass
    
    # Process count
    if config.get("max_pids"):
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (config["max_pids"], config["max_pids"]))
        except (ValueError, resource.error, AttributeError):
            pass
    
    # File descriptors
    if config.get("max_fds"):
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (config["max_fds"], config["max_fds"]))
        except (ValueError, resource.error):
            pass

def get_resource_usage():
    """Get resource usage stats. Unix only."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {
            "user_time": usage.ru_utime,
            "system_time": usage.ru_stime,
            "max_rss_kb": usage.ru_maxrss,
        }
    except ImportError:
        return {}

def run_sandbox(code, namespace, allowed_imports, output_limit, enable_llm):
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
    
    # Add llm_query and llm_batch if enabled
    if enable_llm:
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
    
    # Execute with stdout capture
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
    # Read INIT message via IPC protocol
    init_msg = _ipc_read_message(_ORIGINAL_STDIN_BUFFER)
    if init_msg is None or init_msg.get("type") != "INIT":
        _ipc_send_error("Expected INIT message")
        sys.exit(1)
    
    payload = init_msg.get("payload", {})
    
    config = payload["config"]
    code = payload["code"]
    namespace = pickle.loads(base64.b64decode(payload["namespace_b64"]))
    allowed_imports = set(payload["allowed_imports"])
    enable_llm = payload.get("enable_llm", False)
    output_limit = payload.get("output_limit", 8192)
    
    # Apply resource limits before executing untrusted code
    set_resource_limits(config)
    
    # Run sandbox
    result = run_sandbox(code, namespace, allowed_imports, output_limit, enable_llm)
    
    # Add resource usage
    result["resource_usage"] = get_resource_usage()
    
    # Serialize result
    result["namespace_updates"] = base64.b64encode(
        pickle.dumps(result["namespace_updates"])
    ).decode("ascii")
    
    # Send result via IPC
    _ipc_send_result(result)

if __name__ == "__main__":
    main()
'''


class SandboxRunner:
    """
    Execute code in isolated subprocess with resource limits.
    
    Each call spawns a new subprocess with:
    - Separate address space (no shared memory with parent)
    - OS-enforced resource limits (CPU, memory, file descriptors)
    - IPC-based communication for llm_query/llm_batch support
    
    Example:
        runner = SandboxRunner(
            llm_query=lambda p: "response",
            llm_batch=lambda ps: ["r1", "r2"],
        )
        result = runner.run(
            code="x = 1 + 1\\nFINAL(x)",
            namespace={"data": [1, 2, 3]},
            config=SandboxConfig(max_cpu_seconds=10),
        )
        print(result.final_answer)  # "2"
    """
    
    def __init__(
        self,
        output_limit: int = 8192,
        llm_query: Optional[Callable[[str], str]] = None,
        llm_batch: Optional[Callable[[List[str]], List[str]]] = None,
    ) -> None:
        """
        Args:
            output_limit: Max stdout characters to capture.
            llm_query: Callback for single LLM queries from sandbox code.
            llm_batch: Callback for batch LLM queries from sandbox code.
        """
        self._output_limit = output_limit
        self._llm_query = llm_query
        self._llm_batch = llm_batch
    
    def run(
        self,
        code: str,
        namespace: Optional[Dict[str, Any]] = None,
        config: Optional[SandboxConfig] = None,
    ) -> SandboxResult:
        """
        Execute code in isolated subprocess.
        
        Args:
            code: Python code to execute
            namespace: Initial namespace (must be picklable)
            config: Resource limits and settings
        
        Returns:
            SandboxResult with stdout, error, final_answer, etc.
        """
        if config is None:
            config = SandboxConfig()
        
        if namespace is None:
            namespace = {}
        
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
        
        # Build input payload
        enable_llm = self._llm_query is not None or self._llm_batch is not None
        payload = {
            "config": {
                "max_cpu_seconds": config.max_cpu_seconds,
                "max_memory_mb": config.max_memory_mb,
                "max_pids": config.max_pids,
                "max_fds": config.max_fds,
            },
            "code": code,
            "namespace_b64": namespace_b64,
            "allowed_imports": list(config.allowed_imports),
            "output_limit": self._output_limit,
            "enable_llm": enable_llm,
        }
        
        python_exe = config.python_executable or sys.executable
        
        # Run in subprocess with binary mode for IPC
        try:
            proc = subprocess.Popen(
                [python_exe, "-c", _WORKER_SCRIPT],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,  # Binary mode for IPC protocol
            )
            
            # Use IPCBridge for communication
            # Cast is safe: stdin/stdout are guaranteed non-None when PIPE is used
            bridge = IPCBridge(
                stdin=cast(BinaryIO, proc.stdin),
                stdout=cast(BinaryIO, proc.stdout),
                llm_query=self._llm_query,
                llm_batch=self._llm_batch,
            )
            
            result_data = bridge.run(payload, timeout=config.timeout_seconds)
            
            # Wait for process to finish
            proc.wait(timeout=5.0)
            exit_code = proc.returncode
            
        except TimeoutError:
            proc.kill()
            proc.wait()
            return SandboxResult(
                stdout="",
                error=f"Timeout: execution exceeded {config.timeout_seconds}s",
                final_answer=None,
                namespace_updates={},
                resource_usage={},
                exit_code=-1,
                timed_out=True,
            )
        except RuntimeError as e:
            # Read stderr for additional context
            stderr = b""
            try:
                if proc.stderr:
                    stderr = proc.stderr.read()
            except Exception:
                pass
            proc.kill()
            proc.wait()
            error_msg = str(e)
            if stderr:
                error_msg = f"{error_msg}: {stderr.decode('utf-8', errors='replace')[:500]}"
            return SandboxResult(
                stdout="",
                error=error_msg,
                final_answer=None,
                namespace_updates={},
                resource_usage={},
                exit_code=-1,
            )
        except Exception as e:
            proc.kill()
            proc.wait()
            return SandboxResult(
                stdout="",
                error=f"Subprocess failed: {e}",
                final_answer=None,
                namespace_updates={},
                resource_usage={},
                exit_code=-1,
            )
        
        # Check exit code for resource limit signals
        if exit_code != 0:
            resource_exceeded = False
            error_msg = f"Subprocess exited with code {exit_code}"
            
            if exit_code == -9:  # SIGKILL (OOM killer)
                error_msg = "Killed: memory limit exceeded"
                resource_exceeded = True
            elif exit_code == -24:  # SIGXCPU
                error_msg = "Killed: CPU time limit exceeded"
                resource_exceeded = True
            
            return SandboxResult(
                stdout=result_data.get("stdout", "")[:self._output_limit],
                error=error_msg,
                final_answer=None,
                namespace_updates={},
                resource_usage={},
                exit_code=exit_code,
                resource_exceeded=resource_exceeded,
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
            resource_usage=result_data.get("resource_usage", {}),
            exit_code=exit_code,
        )


class IsolatedSandbox:
    """
    Stateful sandbox using subprocess isolation.
    
    Maintains namespace across multiple exec() calls.
    Drop-in replacement for PythonSandbox when isolation is required.
    
    Supports llm_query/llm_batch via IPC protocol when callbacks are provided.
    """
    
    def __init__(
        self,
        *,
        data: Any = None,
        namespace: Optional[Dict[str, Any]] = None,
        allowed_imports: Optional[Set[str]] = None,
        output_char_limit: int = 8192,
        config: Optional[SandboxConfig] = None,
        llm_query: Optional[Callable[[str], str]] = None,
        llm_batch: Optional[Callable[[List[str]], List[str]]] = None,
    ) -> None:
        self._runner = SandboxRunner(
            output_limit=output_char_limit,
            llm_query=llm_query,
            llm_batch=llm_batch,
        )
        self._config = config or SandboxConfig()
        
        if allowed_imports:
            self._config.allowed_imports = set(allowed_imports)
        
        self._namespace: Dict[str, Any] = dict(namespace or {})
        if data is not None:
            self._namespace["data"] = data
        
        self._answer: Dict[str, Any] = {"content": "", "ready": False}
    
    def exec(self, code: str) -> SandboxResult:
        """Execute code, updating internal namespace."""
        result = self._runner.run(
            code=code,
            namespace=self._namespace,
            config=self._config,
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
