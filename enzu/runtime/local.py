from __future__ import annotations

from typing import List, Optional

from enzu.models import RLMExecutionReport, TaskSpec
from enzu.providers.base import BaseProvider
from enzu.providers.resolve import resolve_provider
from enzu.rlm import RLMEngine
from enzu.runtime.protocol import ProviderSpec, RuntimeOptions


class LocalRuntime:
    """Local runtime that keeps orchestration in-process (default behavior)."""

    def run(
        self,
        *,
        spec: TaskSpec,
        provider: ProviderSpec,
        data: str,
        options: RuntimeOptions,
    ) -> RLMExecutionReport:
        provider_instance = self._resolve_provider_spec(provider)
        fallback_instances = self._resolve_fallbacks(options.fallback_providers)

        engine = RLMEngine(
            max_steps=options.max_steps,
            verify_on_final=options.verify_on_final,
            isolation=options.isolation,
        )
        # Runtime boundary: keep the framework API stable while swapping runtimes.
        return engine.run(
            spec,
            provider_instance,
            data=data,
            on_progress=options.on_progress,
            fallback_providers=fallback_instances,
            sandbox=options.sandbox,
            sandbox_factory=options.sandbox_factory,
        )

    @staticmethod
    def _resolve_provider_spec(spec: ProviderSpec) -> BaseProvider:
        if spec.instance is not None:
            return spec.instance
        if not spec.name:
            raise ValueError("provider name is required when instance is not set.")
        return resolve_provider(
            spec.name,
            api_key=spec.api_key,
            referer=spec.referer,
            app_name=spec.app_name,
            organization=spec.organization,
            project=spec.project,
            use_pool=spec.use_pool,
        )

    def _resolve_fallbacks(
        self,
        specs: List[ProviderSpec],
    ) -> Optional[List[BaseProvider]]:
        if not specs:
            return None
        return [self._resolve_provider_spec(spec) for spec in specs]
