"""
Verification test for Container Pool implementation (Phase 1).
"""

import pytest
from unittest.mock import patch

from enzu.isolation import (
    ContainerPool,
    PoolConfig,
    ContainerRuntime,
)


@pytest.mark.asyncio
async def test_pool_initialization():
    """Verify pool initializes with correct config."""
    config = PoolConfig(min_warm=2, max_pool=5, runtime=ContainerRuntime.PODMAN)
    pool = ContainerPool(config)
    assert pool._config.min_warm == 2
    assert pool._config.max_pool == 5
    assert pool._runtime == ContainerRuntime.PODMAN


@pytest.mark.asyncio
async def test_pool_runtime_detection():
    """Verify pool auto-detects runtime."""
    with patch("enzu.isolation.pool.detect_runtime") as mock_detect:
        mock_detect.return_value = ContainerRuntime.DOCKER

        config = PoolConfig()  # No explicit runtime
        pool = ContainerPool(config)

        assert pool._runtime == ContainerRuntime.DOCKER
        mock_detect.assert_called_once()
