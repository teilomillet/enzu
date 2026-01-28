"""
RLM Python sandbox with functional core.

Design: Pure functions for execution logic, thin class wrapper for convenience.
This separation makes the sandbox easier to test and reason about.
"""
from __future__ import annotations

import ast
import builtins
import io
import signal
import types
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Set, Tuple

from enzu.repl.safe import SAFE_HELPERS
from enzu.security import get_security_profile

# Lazy import for search tools (optional dependency)
def _get_search_tools() -> Dict[str, Any]:
    """Load search tools if available (requires EXA_API_KEY)."""
    tools: Dict[str, Any] = {"__search_tools_available__": False}
    try:
        from enzu.tools.exa import SEARCH_TOOLS
        from enzu.tools.research import RESEARCH_HELPERS
        # RLM uses __search_tools_available__ to decide whether to show search guidance.
        tools["__search_tools_available__"] = True
        tools.update(SEARCH_TOOLS)
        tools.update(RESEARCH_HELPERS)
        return tools
    except (ImportError, ValueError):
        # Exa tooling is missing or unconfigured; keep context helpers and provide a clear error.
        message = "Search tools unavailable: set EXA_API_KEY to enable Exa tools."

        def _missing(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(message)

        tools.update(
            {
                "__search_tools_error__": message,
                "exa_search": _missing,
                "exa_news": _missing,
                "exa_papers": _missing,
                "exa_contents": _missing,
                "exa_similar": _missing,
                "exa_cost": _missing,
                "research": _missing,
                "explore": _missing,
                "format_sources": _missing,
            }
        )
        return tools


@dataclass
class SandboxResult:
    stdout: str
    error: Optional[str]


# Pure functions for sandbox logic

def build_safe_builtins(allowed_imports: Set[str], dynamic_imports: bool = False) -> Dict[str, Any]:
    """
    Build restricted builtins dict. Pure function.

    Args:
        allowed_imports: Static allowlist of module names
        dynamic_imports: If True, allows any import (for pip-installed packages)
    """
    safe_names = [
        "abs",
        "all",
        "any",
        "dict",
        "enumerate",
        "float",
        "int",
        "isinstance",
        "len",
        "list",
        "max",
        "min",
        "print",
        "range",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "ValueError",
        "Exception",
        "type",
        "dir",
        "zip",
    ]
    # Use builtins module so sandbox setup works even when __builtins__ is a module.
    safe = {name: getattr(builtins, name) for name in safe_names}

    def restricted_import(name: str, *args: Any, **kwargs: Any) -> types.ModuleType:
        if dynamic_imports or name in allowed_imports:
            return __import__(name, *args, **kwargs)
        raise ImportError(f"Import blocked: {name}")

    safe["__import__"] = restricted_import
    return safe


def build_namespace(
    data: Any,
    llm_query: Callable[[str], str],
    allowed_imports: Set[str],
    *,
    context: Optional[Any] = None,
    extra_namespace: Optional[Dict[str, Any]] = None,
    inject_safe_helpers: bool = True,
    inject_search_tools: bool = True,
    llm_batch: Optional[Callable[[list], list]] = None,
    enable_pip: bool = False,
) -> Dict[str, Any]:
    """
    Build initial namespace for sandbox execution. Pure function.

    The namespace contains:
    - context: the prompt stored in the REPL environment
    - data: raw context payload (kept for compatibility)
    - llm_query: sub-LLM call function (sequential)
    - llm_batch: parallel batch sub-LLM calls (optimal for latency)
    - pip_install: dynamically install packages (if enable_pip=True)
    - FINAL/FINAL_VAR: answer finalization functions
    - safe helpers: defensive patterns for error-free execution
    """
    answer_state = {"content": "", "ready": False}
    installed_packages: Set[str] = set()

    def final_fn(content: Any) -> None:
        answer_state["content"] = str(content)
        answer_state["ready"] = True

    def final_var_fn(name: Any) -> None:
        # Will be updated to use actual namespace in exec_code
        answer_state["content"] = str(name)
        answer_state["ready"] = True

    # Use explicit prompt context when provided; fall back to raw data.
    context_value = data if context is None else context

    def pip_install_fn(*packages: str) -> str:
        """
        Install pip packages dynamically during execution.
        Returns: Status message listing installed packages.
        """
        import subprocess
        import sys

        if not enable_pip:
            raise RuntimeError("pip_install is disabled. Set enable_pip=True in RLMEngine.")

        results = []
        for package in packages:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q", package],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                installed_packages.add(package)
                results.append(f"✓ {package}")
            except subprocess.CalledProcessError:
                results.append(f"✗ {package} (failed)")

        return "Installed:\n" + "\n".join(results)

    from enzu.tools.context import CONTEXT_HELPERS

    namespace: Dict[str, Any] = {
        "__builtins__": build_safe_builtins(allowed_imports, dynamic_imports=enable_pip),
        "data": data,
        "context": context_value,
        "__rlm_answer__": answer_state,
        "llm_query": llm_query,
        "query": llm_query,
        "FINAL": final_fn,
        "FINAL_VAR": final_var_fn,
        "__installed_packages__": installed_packages,
        "__search_tools_available__": False,
    }

    # Inject pip_install if enabled
    if enable_pip:
        namespace["pip_install"] = pip_install_fn

    # Inject llm_batch if provided (parallel execution)
    if llm_batch is not None:
        namespace["llm_batch"] = llm_batch
        namespace["batch_query"] = llm_batch

    # Inject safe helpers by default (error prevention)
    if inject_safe_helpers:
        namespace.update(SAFE_HELPERS)

    # Context helpers are safe; always include them for ctx_* access.
    namespace.update(CONTEXT_HELPERS)

    # Inject search tools if available (requires EXA_API_KEY)
    if inject_search_tools:
        namespace.update(_get_search_tools())

    # Add user-provided namespace
    if extra_namespace:
        namespace.update(extra_namespace)

    return namespace


def truncate_output(text: str, limit: int) -> str:
    """Truncate output to limit. Pure function."""
    if len(text) <= limit:
        return text
    return text[:limit]


def _compile_with_repl_echo(code: str) -> Any:
    """
    Compile code while echoing the final expression result like a REPL.

    If the last statement is an expression, assign it to a temp name and
    print it when it is not None.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = tree.body[-1].value
        temp_name = ast.Name(id="_rlm_last_expr", ctx=ast.Store())
        assign = ast.Assign(targets=[temp_name], value=last_expr)
        check = ast.Compare(
            left=ast.Name(id="_rlm_last_expr", ctx=ast.Load()),
            ops=[ast.IsNot()],
            comparators=[ast.Constant(value=None)],
        )
        echo = ast.Expr(
            value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=[ast.Name(id="_rlm_last_expr", ctx=ast.Load())],
                keywords=[],
            )
        )
        tree.body[-1] = assign
        tree.body.append(ast.If(test=check, body=[echo], orelse=[]))
        ast.fix_missing_locations(tree)

    return compile(tree, filename="<rlm>", mode="exec")


def exec_code(
    code: str,
    namespace: Dict[str, Any],
    output_limit: int,
    timeout_seconds: Optional[float],
    enforce_safety: bool = True,
    allowed_imports: Optional[Set[str]] = None,
) -> Tuple[SandboxResult, Dict[str, Any]]:
    """
    Execute code in namespace. Pure function (no hidden state).

    Returns (result, updated_namespace) tuple.
    The namespace is mutated by exec(), so we return it for explicit data flow.
    """
    # Rebind FINAL_VAR to capture namespace lookups
    answer_state = namespace["__rlm_answer__"]

    def final_var_fn(name: Any) -> None:
        if isinstance(name, str) and name in namespace:
            value = namespace.get(name)
        else:
            value = name
        answer_state["content"] = str(value)
        answer_state["ready"] = True

    namespace["FINAL_VAR"] = final_var_fn

    stdout_buffer = io.StringIO()
    timeout_ctx = _AlarmTimeout(timeout_seconds) if timeout_seconds else _NoopContext()

    try:
        if enforce_safety:
            # Pre-exec policy check to block known sandbox-escape patterns.
            # Allowed imports are validated at AST level for consistency.
            validate_code_safety(code, allowed_imports=allowed_imports)
        with redirect_stdout(stdout_buffer):
            with timeout_ctx:
                compiled = _compile_with_repl_echo(code)
                exec(compiled, namespace, None)
        output = stdout_buffer.getvalue()
        result = SandboxResult(
            stdout=truncate_output(output, output_limit),
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        output = stdout_buffer.getvalue()
        result = SandboxResult(
            stdout=truncate_output(output, output_limit),
            error=str(exc),
        )

    return result, namespace


# Thin class wrapper for convenience

class PythonSandbox:
    """
    Thin wrapper around functional core.
    
    Maintains namespace state across multiple exec() calls.
    Use exec_code() directly for stateless execution.
    """

    def __init__(
        self,
        *,
        data: Any,
        llm_query: Callable[[str], str],
        namespace: Optional[Dict[str, Any]] = None,
        allowed_imports: Optional[Iterable[str]] = None,
        output_char_limit: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        inject_safe_helpers: bool = True,
        inject_search_tools: bool = True,
        enforce_safety: bool = True,
        context: Optional[Any] = None,
        llm_batch: Optional[Callable[[list], list]] = None,
        enable_pip: Optional[bool] = None,
        security_profile: Optional[str] = None,
    ) -> None:
        # Apply security profile defaults for production safety
        profile = get_security_profile(security_profile)

        # Use security profile defaults when not explicitly provided
        if allowed_imports is None:
            self._allowed_imports = set(profile.allowed_imports)
        else:
            self._allowed_imports = set(allowed_imports)

        self._output_char_limit = output_char_limit or profile.output_char_limit
        # Note: timeout_seconds defaults to None (no timeout) because signal.alarm()
        # doesn't work in threaded contexts. Timeout should be opt-in.
        self._timeout_seconds = timeout_seconds
        self._enforce_safety = enforce_safety
        self._enable_pip = enable_pip if enable_pip is not None else profile.enable_pip
        self._namespace = build_namespace(
            data=data,
            llm_query=llm_query,
            allowed_imports=self._allowed_imports,
            context=context,
            extra_namespace=namespace,
            inject_safe_helpers=inject_safe_helpers,
            inject_search_tools=inject_search_tools,
            llm_batch=llm_batch,
            enable_pip=self._enable_pip,
        )

    def exec(self, code: str) -> SandboxResult:
        """Execute code, updating internal namespace."""
        result, self._namespace = exec_code(
            code=code,
            namespace=self._namespace,
            output_limit=self._output_char_limit,
            timeout_seconds=self._timeout_seconds,
            enforce_safety=self._enforce_safety,
            allowed_imports=self._allowed_imports,
        )
        return result

    @property
    def answer(self) -> Dict[str, Any]:
        """Access answer state set by FINAL/FINAL_VAR."""
        return self._namespace["__rlm_answer__"]

    def get_global(self, name: str) -> Any:
        """Get variable from namespace."""
        return self._namespace.get(name)

    @property
    def namespace(self) -> Dict[str, Any]:
        """Direct access to namespace (for testing/inspection)."""
        return self._namespace


# Timeout context managers

class _NoopContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Literal[False]:
        return False


class _AlarmTimeout:
    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._previous_handler: Any = None
        self._enabled = True

    def __enter__(self) -> None:
        if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
            # Windows does not provide SIGALRM; skip timeout enforcement.
            self._enabled = False
            return None
        self._previous_handler = signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, self._seconds)
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Literal[False]:
        if not self._enabled:
            return False
        signal.setitimer(signal.ITIMER_REAL, 0)
        if self._previous_handler:
            signal.signal(signal.SIGALRM, self._previous_handler)
        return False

    @staticmethod
    def _handle_timeout(signum: int, frame: Any) -> None:
        raise TimeoutError("Sandbox execution timed out.")
class SandboxViolation(RuntimeError):
    """Sandbox policy violation."""


def validate_code_safety(code: str, allowed_imports: Optional[Set[str]] = None) -> None:
    """Reject high-risk Python constructs before execution.

    Args:
        code: Python code to validate
        allowed_imports: Set of module names allowed for import (None = block all)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Let exec() surface syntax errors to keep feedback consistent.
        return

    allowed = allowed_imports or set()
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            # Check each imported module name against allowed list
            for alias in node.names:
                if alias.name not in allowed:
                    violations.append(f"import blocked: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            # Check the module being imported from
            if node.module and node.module not in allowed:
                violations.append(f"import blocked: {node.module}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            # Block dunder attribute access to prevent sandbox escapes.
            violations.append(f"dunder attribute blocked: {node.attr}")
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            violations.append(f"dunder name blocked: {node.id}")

    if violations:
        raise SandboxViolation("; ".join(violations))
