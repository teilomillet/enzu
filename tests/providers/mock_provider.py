from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

from enzu.models import ProviderResult, TaskSpec
from enzu.providers.base import BaseProvider


class MockProvider(BaseProvider):
    """
    Mock provider for testing RLM workflows.

    Supports two modes:
    1. Direct subcalls (recursive_subcalls=False): prompts starting with
       "SUBCALL:" are matched against subcall_responses.
    2. Recursive subcalls (recursive_subcalls=True): sub-RLM prompts are
       detected by context markers and auto-responded with FINAL.
    """

    name = "mock"

    def __init__(
        self,
        *,
        main_outputs: Iterable[str],
        subcall_responses: Optional[Dict[str, str]] = None,
        usage: Optional[Dict[str, float]] = None,
        subcall_fallback: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._main_outputs = list(main_outputs)
        self._subcall_responses = subcall_responses or {}
        self._usage = usage or {"output_tokens": 5, "total_tokens": 5}
        self._subcall_fallback = subcall_fallback
        # Call history keeps tests deterministic and inspectable.
        self.calls: list[str] = []

    def generate(self, task: TaskSpec) -> ProviderResult:
        prompt = task.input_text
        self.calls.append(prompt)

        # Mode 1: Direct subcalls (recursive_subcalls=False)
        if prompt.startswith("SUBCALL:"):
            key = prompt[len("SUBCALL:") :].strip()
            output_text = self._subcall_responses.get(key)
            if output_text is None:
                output_text = self._match_prefix(key)
            if output_text is None and self._subcall_fallback:
                output_text = self._subcall_fallback(key)
            if output_text is None:
                raise ValueError(f"MockProvider missing subcall response: {key}")
            return self._result(output_text, task)

        # Mode 2: Recursive sub-RLM (detect by _subcall_prompt metadata)
        subcall_prompt = task.metadata.get("_subcall_prompt")
        if subcall_prompt:
            output_text = self._handle_sub_rlm_prompt(subcall_prompt)
            return self._result(output_text, task)

        # Main RLM output
        if not self._main_outputs:
            raise ValueError("MockProvider missing main output")
        output_text = self._main_outputs.pop(0)
        return self._result(output_text, task)

    def _result(self, output_text: str, task: TaskSpec) -> ProviderResult:
        return ProviderResult(
            output_text=output_text,
            raw={"mock": True},
            usage=dict(self._usage),
            provider=self.name,
            model=task.model,
        )

    def _match_prefix(self, key: str) -> Optional[str]:
        for pattern, value in self._subcall_responses.items():
            if pattern.endswith("*") and key.startswith(pattern[:-1]):
                return value
        return None

    def _handle_sub_rlm_prompt(self, subcall_prompt: str) -> str:
        """Generate response for recursive sub-RLM calls."""
        # Extract key from SUBCALL:key pattern
        if subcall_prompt.startswith("SUBCALL:"):
            key = subcall_prompt[len("SUBCALL:"):].strip()
        else:
            key = subcall_prompt

        # Check subcall_responses for matching key
        response = self._subcall_responses.get(key)
        if response is None:
            response = self._match_prefix(key)
        if response is None and self._subcall_fallback:
            response = self._subcall_fallback(key)
        if response is None:
            response = f"sub_response_for_{key}"

        # Return code that calls FINAL with the response (use repr for safe escaping)
        return f'```python\nFINAL({repr(response)})\n```'
