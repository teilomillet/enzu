from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol

from enzu.repl.sandbox import SandboxResult


class SandboxProtocol(Protocol):
    """Minimum interface the RLM step runner needs from a sandbox."""

    def exec(self, code: str) -> SandboxResult:
        """Execute code and return stdout/error."""
        ...

    @property
    def answer(self) -> Dict[str, Any]:
        """Answer state from FINAL()/FINAL_VAR()."""
        ...

    @property
    def namespace(self) -> Dict[str, Any]:
        """Execution namespace for sandbox state."""
        ...

    def get_global(self, name: str) -> Any:
        """Get a variable from the sandbox namespace."""
        ...


class SandboxFactory(Protocol):
    """Callable that creates a sandbox after llm_query/llm_batch exist."""

    def __call__(
        self,
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
        llm_query: Callable[[str], str],
        llm_batch: Optional[Callable[[list], list]] = None,
        sandbox_image: Optional[Any] = None,
    ) -> SandboxProtocol:
        """Return a sandbox instance compatible with StepRunner."""
        ...
