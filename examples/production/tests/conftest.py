"""Shared fixtures for production example tests."""

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from enzu.models import ProviderResult, TaskSpec
from enzu.providers.base import BaseProvider


class MockProvider(BaseProvider):
    """Simple mock provider for testing examples."""

    name = "mock"

    def __init__(
        self,
        responses: Optional[Iterable[str]] = None,
        usage: Optional[Dict[str, float]] = None,
    ) -> None:
        self._responses = list(responses) if responses else ["Mock response"]
        self._usage = usage or {"output_tokens": 10, "total_tokens": 20}
        self.calls: list[str] = []

    def generate(self, task: TaskSpec) -> ProviderResult:
        self.calls.append(task.input_text)
        output = self._responses.pop(0) if self._responses else "Mock response"
        return ProviderResult(
            output_text=output,
            raw={"mock": True},
            usage=dict(self._usage),
            provider=self.name,
            model=task.model,
        )


@pytest.fixture
def mock_provider():
    """Create a mock provider with default responses."""
    return MockProvider()


@pytest.fixture
def mock_provider_factory():
    """Factory for creating mock providers with custom responses."""
    def _create(responses: list[str], usage: Optional[Dict[str, float]] = None):
        return MockProvider(responses=responses, usage=usage)
    return _create


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    return tmp_path
