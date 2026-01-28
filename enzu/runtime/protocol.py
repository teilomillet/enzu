from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol

from enzu.models import RLMExecutionReport, TaskSpec
from enzu.providers.base import BaseProvider
from enzu.repl.protocol import SandboxFactory, SandboxProtocol


@dataclass
class ProviderSpec:
    """Serializable provider configuration for runtime boundaries."""

    name: Optional[str] = None
    api_key: Optional[str] = None
    referer: Optional[str] = None
    app_name: Optional[str] = None
    organization: Optional[str] = None
    project: Optional[str] = None
    use_pool: bool = False
    instance: Optional[BaseProvider] = None


@dataclass
class RuntimeOptions:
    """RLM runtime settings passed across the framework/runtime boundary."""

    max_steps: int = 8
    verify_on_final: bool = True
    isolation: Optional[str] = None
    sandbox: Optional[SandboxProtocol] = None
    sandbox_factory: Optional[SandboxFactory] = None
    on_progress: Optional[Callable[[str], None]] = None
    fallback_providers: List[ProviderSpec] = field(default_factory=list)


class RLMRuntime(Protocol):
    """Runtime interface: runs an RLM task in a chosen execution environment."""

    def run(
        self,
        *,
        spec: TaskSpec,
        provider: ProviderSpec,
        data: str,
        options: RuntimeOptions,
    ) -> RLMExecutionReport:
        ...
