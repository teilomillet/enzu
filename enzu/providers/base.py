from __future__ import annotations

from typing import Callable, Optional

from enzu.models import ProgressEvent, ProviderResult, TaskSpec


class BaseProvider:
    name = "base"

    def generate(self, task: TaskSpec) -> ProviderResult:
        raise NotImplementedError

    def stream(
        self,
        task: TaskSpec,
        on_progress: Optional[Callable[[ProgressEvent], None]] = None,
    ) -> ProviderResult:
        if on_progress:
            on_progress(
                ProgressEvent(
                    phase="generation",
                    message="provider_stream_not_implemented",
                    data={"provider": self.name},
                )
            )
        return self.generate(task)
