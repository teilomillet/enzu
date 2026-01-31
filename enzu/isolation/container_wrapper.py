"""
Container wrapper for warm pool.

Manages a single warm container and executes code inside it
via fresh process spawning (podman exec).
"""

from __future__ import annotations

import asyncio
import time
import json
import base64
import logging
from dataclasses import asdict
from typing import Any, Dict

import pickle

from enzu.isolation.runtime import ContainerRuntime, get_runtime_command
from enzu.isolation.runner import SandboxResult, SandboxConfig

logger = logging.getLogger(__name__)

# Script run inside the container via `podman exec`
# This runs in a fresh process for EVERY request.
#
# SECURITY: This script implements the same sandbox restrictions as
# enzu/repl/sandbox.py - restricted builtins, import allowlist, AST validation.
EXECUTOR_SCRIPT = '''
"""
One-shot executor. Runs once per request, then exits.
All state is destroyed when process exits.

SECURITY LAYERS:
1. Restricted builtins (no eval, exec, open, getattr, etc.)
2. Import allowlist (no os, subprocess, pickle, etc.)
3. AST validation (no dunder access like __class__)
"""
import sys
import json
import base64
import pickle
import ast
import io
import builtins
from contextlib import redirect_stdout

# Restricted builtins - same as enzu/repl/sandbox.py
SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "dict", "enumerate", "float", "int",
    "isinstance", "len", "list", "max", "min", "print", "range",
    "set", "sorted", "str", "sum", "tuple", "ValueError",
    "Exception", "type", "dir", "zip", "bool", "repr", "round",
    "ord", "chr", "map", "filter", "reversed", "slice",
    "NameError", "TypeError", "KeyError", "IndexError", "AttributeError",
]

# Default allowed imports
DEFAULT_ALLOWED_IMPORTS = {
    "re", "math", "json", "datetime", "collections",
    "itertools", "functools", "statistics", "string", "textwrap", "typing"
}

def build_safe_builtins(allowed_imports):
    """Build restricted builtins dict."""
    safe = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES if hasattr(builtins, name)}

    def restricted_import(name, *args, **kwargs):
        if name in allowed_imports:
            return __import__(name, *args, **kwargs)
        raise ImportError(f"Import blocked: {name}")

    safe["__import__"] = restricted_import
    return safe

def validate_code_safety(code):
    """Reject dangerous patterns via AST analysis."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return  # Let exec() surface syntax errors

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise RuntimeError(f"Blocked: dunder attribute access ({node.attr})")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            if node.id not in ("__builtins__",):
                raise RuntimeError(f"Blocked: dunder name ({node.id})")

def execute_sandboxed(code, namespace_b64, config):
    """Execute code with full sandbox restrictions."""
    # Deserialize namespace
    try:
        namespace = pickle.loads(base64.b64decode(namespace_b64))
    except Exception as e:
        return {"error": f"Namespace deserialization failed: {e}"}

    # Get allowed imports from config or use defaults
    allowed_imports = set(config.get("allowed_imports", DEFAULT_ALLOWED_IMPORTS))

    # Validate code safety (AST check)
    try:
        validate_code_safety(code)
    except RuntimeError as e:
        return {"error": str(e)}

    # Answer state for FINAL()
    answer_state = {"content": None, "ready": False}

    def final_fn(content):
        answer_state["content"] = str(content)
        answer_state["ready"] = True

    # Build sandboxed namespace with restricted builtins
    exec_namespace = dict(namespace)
    exec_namespace["__builtins__"] = build_safe_builtins(allowed_imports)
    exec_namespace["FINAL"] = final_fn

    # Capture stdout
    stdout_buffer = io.StringIO()
    error = None

    try:
        with redirect_stdout(stdout_buffer):
            exec(code, exec_namespace, None)
    except Exception as e:
        error = str(e)

    stdout = stdout_buffer.getvalue()

    # Extract picklable updates
    updates = {}
    skip_keys = {"__builtins__", "FINAL", "FINAL_VAR"}
    for key, value in exec_namespace.items():
        if key.startswith("_") or key in skip_keys:
            continue
        try:
            pickle.dumps(value)
            updates[key] = value
        except (pickle.PicklingError, TypeError, AttributeError):
            pass

    updates_b64 = base64.b64encode(pickle.dumps(updates)).decode("ascii")

    return {
        "stdout": stdout,
        "error": error,
        "final_answer": answer_state.get("content"),
        "namespace_updates": updates_b64,
    }

def main():
    try:
        input_data = sys.stdin.read()
        if not input_data:
            return

        payload = json.loads(input_data)
        code = payload["code"]
        namespace_b64 = payload["namespace_b64"]
        config = payload.get("config", {})

        result = execute_sandboxed(code, namespace_b64, config)
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": f"Executor crash: {e}"}))

if __name__ == "__main__":
    main()
'''


class Container:
    """
    Wrapper for a warm container instance.

    The container runs continuously (sleep infinity).
    Each execution spawns a NEW Python process via `podman exec`.
    This guarantees process-level isolation between execution requests.
    """

    def __init__(
        self,
        container_id: str,
        runtime: ContainerRuntime,
    ):
        self._id = container_id
        self._runtime = runtime
        self._created_at = time.monotonic()
        self._last_used = time.monotonic()
        self._exec_count = 0
        self._healthy = True

    @property
    def id(self) -> str:
        return self._id

    async def execute(
        self,
        code: str,
        namespace: Dict[str, Any],
        config: SandboxConfig,
    ) -> SandboxResult:
        """
        Execute code in a FRESH Python process inside this container.
        """
        self._last_used = time.monotonic()
        self._exec_count += 1

        cmd = get_runtime_command(self._runtime)

        # Prepare payload
        try:
            namespace_b64 = base64.b64encode(pickle.dumps(namespace)).decode("ascii")
        except Exception as e:
            return SandboxResult(
                stdout="",
                error=f"Namespace serialization failed: {e}",
                final_answer=None,
                namespace_updates={},
                resource_usage={},
                exit_code=-1,
            )

        # Convert config to JSON-serializable dict
        if hasattr(config, "__dataclass_fields__"):
            config_dict = asdict(config)
            # Convert sets to lists for JSON serialization
            if "allowed_imports" in config_dict and isinstance(
                config_dict["allowed_imports"], set
            ):
                config_dict["allowed_imports"] = list(config_dict["allowed_imports"])
        else:
            config_dict = dict(config)

        payload = {
            "code": code,
            "namespace_b64": namespace_b64,
            "config": config_dict,
        }
        payload_json = json.dumps(payload)

        # Spawn fresh Python process inside warm container
        # We pass the executor script via -c
        # Note: In production, we might want to COPY the script into the image
        # to avoid passing large args, but this works for now.
        process_cmd = [cmd, "exec", "-i", self._id, "python", "-c", EXECUTOR_SCRIPT]

        try:
            proc = await asyncio.create_subprocess_exec(
                *process_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=payload_json.encode("utf-8")),
                timeout=config.timeout_seconds,
            )

            if proc.returncode != 0:
                stderr = stderr_bytes.decode()
                # If execution failed at process level, container might be bad
                # But usually it's just a script error.
                # We assume container is fine unless `podman exec` itself failed.
                return SandboxResult(
                    stdout="",
                    error=f"Process exited {proc.returncode}: {stderr}",
                    final_answer=None,
                    namespace_updates={},
                    resource_usage={},
                    exit_code=proc.returncode or -1,
                )

            # Parse result from stdout
            output_str = stdout_bytes.decode()
            try:
                # The executor prints exactly one line of JSON
                result_data = json.loads(output_str)
            except json.JSONDecodeError:
                return SandboxResult(
                    stdout=output_str,  # Maybe it printed something else
                    error="Failed to parse executor output",
                    final_answer=None,
                    namespace_updates={},
                    resource_usage={},
                    exit_code=0,
                )

            # Deserialize updates
            updates_b64 = result_data.get("namespace_updates")
            updates = {}
            if updates_b64:
                try:
                    updates = pickle.loads(base64.b64decode(updates_b64))
                except Exception:
                    pass

            return SandboxResult(
                stdout=result_data.get("stdout", ""),
                error=result_data.get("error"),
                final_answer=result_data.get("final_answer"),
                namespace_updates=updates,
                resource_usage={},
                exit_code=0,
            )

        except asyncio.TimeoutError:
            # Process hung, kill it
            # Note: This checks `exec` process. We might want to `kill` the container if `exec` is stuck?
            # For now, we assume destroying the exec is enough, or let HealthCheck kill the container later.
            return SandboxResult(
                stdout="",
                error="Execution timed out",
                final_answer=None,
                namespace_updates={},
                resource_usage={},
                exit_code=-1,
                timed_out=True,
            )
        except Exception as e:
            self._healthy = False  # Mark as potentially tainted if system error
            return SandboxResult(
                stdout="",
                error=f"System error: {e}",
                final_answer=None,
                namespace_updates={},
                resource_usage={},
                exit_code=-1,
            )

    def is_healthy(self) -> bool:
        """Check if container is considered healthy by wrapper."""
        return self._healthy

    async def check_health(self) -> bool:
        """Active health check (ping)."""
        if not self._healthy:
            return False

        cmd = get_runtime_command(self._runtime)
        # Simple check: does exec echo work?
        proc = await asyncio.create_subprocess_exec(
            cmd,
            "exec",
            self._id,
            "echo",
            "ok",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        return proc.returncode == 0

    def idle_seconds(self) -> float:
        """Seconds since last use."""
        return time.monotonic() - self._last_used

    async def destroy(self) -> None:
        """Stop and remove container."""
        cmd = get_runtime_command(self._runtime)
        # Using force remove
        proc = await asyncio.create_subprocess_exec(
            cmd,
            "rm",
            "-f",
            self._id,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
