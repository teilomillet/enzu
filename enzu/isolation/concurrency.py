"""
Global concurrency control for LLM API calls.

Problem: llm_batch() uses unbounded ThreadPoolExecutor, causing:
- Rate limit errors (429) when too many concurrent requests
- Connection exhaustion under load
- No fair distribution across parallel batches

Solution: Global semaphore limiting total concurrent LLM calls.

Usage:
    from enzu.isolation.concurrency import get_global_limiter
    
    limiter = get_global_limiter()
    with limiter.acquire():
        # Make LLM call
        result = provider.stream(task)

Configuration:
    from enzu.isolation.concurrency import configure_global_limiter
    configure_global_limiter(max_concurrent=100)
"""
from __future__ import annotations

import threading
import asyncio
from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager


@dataclass
class ConcurrencyStats:
    """Snapshot of concurrency limiter state."""
    max_concurrent: int
    active: int
    waiting: int
    total_acquired: int
    total_rejected: int


class ConcurrencyLimiter:
    """
    Thread-safe concurrency limiter for LLM API calls.
    
    Limits total concurrent operations across all threads/coroutines.
    Provides both sync (threading.Semaphore) and async (asyncio.Semaphore) interfaces.
    
    Design:
    - Single global instance controls all LLM calls
    - Blocking acquire prevents thundering herd
    - Stats tracking for monitoring
    """
    
    def __init__(self, max_concurrent: int = 50) -> None:
        """
        Args:
            max_concurrent: Maximum simultaneous LLM calls. 
                           50 is conservative default for most LLM APIs.
        """
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        
        self._max_concurrent = max_concurrent
        
        # Threading semaphore for sync code (llm_batch uses threads)
        self._sync_semaphore = threading.Semaphore(max_concurrent)
        
        # Async semaphore created lazily per event loop
        self._async_semaphores: dict[int, asyncio.Semaphore] = {}
        self._async_lock = threading.Lock()
        
        # Stats (atomic via lock)
        self._stats_lock = threading.Lock()
        self._active = 0
        self._waiting = 0
        self._total_acquired = 0
        self._total_rejected = 0
    
    @contextmanager
    def acquire(self, timeout: Optional[float] = None, blocking: bool = True):
        """
        Acquire a slot for making an LLM call. Context manager.
        
        Args:
            timeout: Max seconds to wait. None = wait forever.
            blocking: If False, raise immediately when no slots available.
        
        Raises:
            TimeoutError: If timeout expires before slot available.
            RuntimeError: If blocking=False and no slots available.
        
        Example:
            with limiter.acquire(timeout=30):
                result = provider.stream(task)
        """
        with self._stats_lock:
            self._waiting += 1
        
        try:
            acquired = self._sync_semaphore.acquire(blocking=blocking, timeout=timeout)
            if not acquired:
                with self._stats_lock:
                    self._total_rejected += 1
                if not blocking:
                    raise RuntimeError("No concurrency slots available (non-blocking)")
                raise TimeoutError(f"Timeout waiting for concurrency slot ({timeout}s)")
            
            with self._stats_lock:
                self._waiting -= 1
                self._active += 1
                self._total_acquired += 1
            
            try:
                yield
            finally:
                self._sync_semaphore.release()
                with self._stats_lock:
                    self._active -= 1
        except:
            with self._stats_lock:
                self._waiting -= 1
            raise
    
    async def acquire_async(self, timeout: Optional[float] = None):
        """
        Async version of acquire(). Use in async code.
        
        Example:
            async with limiter.acquire_async(timeout=30):
                result = await async_provider_call(...)
        """
        sem = self._get_async_semaphore()
        
        with self._stats_lock:
            self._waiting += 1
        
        try:
            if timeout is not None:
                try:
                    await asyncio.wait_for(sem.acquire(), timeout=timeout)
                except asyncio.TimeoutError:
                    with self._stats_lock:
                        self._waiting -= 1
                        self._total_rejected += 1
                    raise TimeoutError(f"Timeout waiting for concurrency slot ({timeout}s)")
            else:
                await sem.acquire()
            
            with self._stats_lock:
                self._waiting -= 1
                self._active += 1
                self._total_acquired += 1
            
            try:
                yield
            finally:
                sem.release()
                with self._stats_lock:
                    self._active -= 1
        except:
            with self._stats_lock:
                if self._waiting > 0:
                    self._waiting -= 1
            raise
    
    def _get_async_semaphore(self) -> asyncio.Semaphore:
        """Get or create async semaphore for current event loop."""
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
        except RuntimeError:
            raise RuntimeError("acquire_async() must be called from async context")
        
        with self._async_lock:
            if loop_id not in self._async_semaphores:
                self._async_semaphores[loop_id] = asyncio.Semaphore(self._max_concurrent)
            return self._async_semaphores[loop_id]
    
    def stats(self) -> ConcurrencyStats:
        """Current limiter state for monitoring."""
        with self._stats_lock:
            return ConcurrencyStats(
                max_concurrent=self._max_concurrent,
                active=self._active,
                waiting=self._waiting,
                total_acquired=self._total_acquired,
                total_rejected=self._total_rejected,
            )
    
    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent


# Global singleton instance
_global_limiter: Optional[ConcurrencyLimiter] = None
_global_lock = threading.Lock()


def get_global_limiter() -> ConcurrencyLimiter:
    """
    Get the global concurrency limiter.
    
    Creates with default settings (50 concurrent) if not configured.
    Thread-safe singleton.
    """
    global _global_limiter
    if _global_limiter is None:
        with _global_lock:
            if _global_limiter is None:
                _global_limiter = ConcurrencyLimiter(max_concurrent=50)
    assert _global_limiter is not None
    return _global_limiter


def configure_global_limiter(
    max_concurrent: int = 50,
    *,
    force_reconfigure: bool = False,
) -> ConcurrencyLimiter:
    """
    Configure the global concurrency limiter.
    
    Call at startup before any LLM calls. Reconfiguring after calls
    have started requires force_reconfigure=True.
    
    Args:
        max_concurrent: Maximum simultaneous LLM calls across all requests.
        force_reconfigure: Allow reconfiguring after limiter is in use.
    
    Returns:
        The configured global limiter.
    
    Example:
        # At application startup
        configure_global_limiter(max_concurrent=100)
    """
    global _global_limiter
    
    with _global_lock:
        if _global_limiter is not None and not force_reconfigure:
            stats = _global_limiter.stats()
            if stats.total_acquired > 0:
                raise RuntimeError(
                    "Cannot reconfigure limiter after use. "
                    "Set force_reconfigure=True to override."
                )
        _global_limiter = ConcurrencyLimiter(max_concurrent=max_concurrent)
        return _global_limiter


def reset_global_limiter() -> None:
    """Reset global limiter. For testing only."""
    global _global_limiter
    with _global_lock:
        _global_limiter = None
