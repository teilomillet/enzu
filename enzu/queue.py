"""
Task queue with worker pool for high-concurrency deployments.

Similar to Ray Serve: requests queue up, workers process them, auto-scaling adjusts capacity.

Architecture:
    submit() → Queue → Workers (N coroutines) → Results via Future

    Queue is bounded: when full, submit() blocks (backpressure).
    Workers pull from queue, execute via pooled provider, return via Future.
    Autoscaler monitors queue depth, adjusts worker count within [min, max].

Usage:
    import enzu
    from enzu.queue import TaskQueue

    # Start queue with worker pool
    queue = TaskQueue(
        min_workers=2,
        max_workers=20,
        queue_size=1000,
        provider="openrouter",
        model="gpt-4",
    )
    await queue.start()

    # Submit tasks (returns Future)
    future = await queue.submit("Translate: Hello")
    result = await future  # blocks until complete

    # Batch submit
    futures = await queue.submit_many(["Task 1", "Task 2", "Task 3"])
    results = await asyncio.gather(*futures)

    # Check stats
    stats = queue.stats()
    # {'queued': 5, 'active': 10, 'workers': 10, 'completed': 1000}

    # Graceful shutdown
    await queue.stop()

Thread safety:
    - Queue operations are async-safe
    - Can call submit() from multiple coroutines
    - For sync code, use submit_sync() which handles event loop

Scaling behavior:
    - Starts with min_workers
    - Scales up when queue_depth > scale_up_threshold * workers
    - Scales down when queue_depth < scale_down_threshold * workers
    - Scaling is gradual (one worker at a time per interval)
"""
from __future__ import annotations

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from enzu.models import ExecutionReport, RLMExecutionReport
from concurrent.futures import Future as SyncFuture

from enzu.models import TaskSpec
from enzu.providers.base import BaseProvider


logger = logging.getLogger(__name__)


@dataclass
class TaskRequest:
    """A queued task request."""
    task: Union[str, TaskSpec, Dict[str, Any]]
    kwargs: Dict[str, Any]
    future: asyncio.Future
    submitted_at: float = field(default_factory=time.time)


@dataclass
class QueueStats:
    """Queue statistics snapshot."""
    queued: int          # tasks waiting in queue
    active: int          # tasks being processed
    workers: int         # current worker count
    completed: int       # total completed tasks
    failed: int          # total failed tasks
    avg_latency_ms: float  # average task latency


class TaskQueue:
    """
    Async task queue with auto-scaling worker pool.

    Workers share a pooled provider (single httpx.Client per endpoint).
    Queue provides backpressure: submit() blocks when queue is full.
    Autoscaler adjusts workers based on queue depth.
    """

    def __init__(
        self,
        *,
        provider: str = "openrouter",
        model: str,
        api_key: Optional[str] = None,
        min_workers: int = 2,
        max_workers: int = 20,
        queue_size: int = 1000,
        # Scaling thresholds (ratio of queue_depth to worker_count)
        scale_up_threshold: float = 2.0,    # scale up if queue > 2x workers
        scale_down_threshold: float = 0.5,  # scale down if queue < 0.5x workers
        scale_interval: float = 1.0,        # seconds between scaling decisions
        # Default task parameters
        default_mode: Optional[str] = None,
        default_tokens: Optional[int] = None,
        default_temperature: Optional[float] = None,
    ):
        """
        Initialize task queue.

        Args:
            provider: Provider name (e.g., "openrouter", "openai")
            model: Model to use for all tasks
            api_key: Optional API key (uses env var if not provided)
            min_workers: Minimum worker count (always running)
            max_workers: Maximum worker count (scaling ceiling)
            queue_size: Max queued tasks before submit() blocks
            scale_up_threshold: Scale up when queue > threshold * workers
            scale_down_threshold: Scale down when queue < threshold * workers
            scale_interval: Seconds between autoscaler checks
            default_mode: Default mode for tasks ("chat", "rlm", "auto")
            default_tokens: Default max output tokens
            default_temperature: Default temperature
        """
        self._provider_name = provider
        self._model = model
        self._api_key = api_key
        self._min_workers = min_workers
        self._max_workers = max_workers
        self._queue_size = queue_size
        self._scale_up_threshold = scale_up_threshold
        self._scale_down_threshold = scale_down_threshold
        self._scale_interval = scale_interval

        # Default task parameters
        self._default_mode = default_mode
        self._default_tokens = default_tokens
        self._default_temperature = default_temperature

        # Runtime state (initialized in start())
        self._queue: Optional[asyncio.Queue[TaskRequest]] = None
        self._workers: List[asyncio.Task] = []
        self._scaler_task: Optional[asyncio.Task] = None
        self._running = False
        self._provider: Optional[BaseProvider] = None

        # Stats
        self._active_count = 0
        self._completed_count = 0
        self._failed_count = 0
        self._total_latency_ms = 0.0
        self._stats_lock = asyncio.Lock()

    async def start(self) -> None:
        """
        Start the queue and worker pool.

        Initializes the shared provider, starts min_workers, and begins autoscaling.
        """
        if self._running:
            return

        # Initialize queue
        self._queue = asyncio.Queue(maxsize=self._queue_size)

        # Initialize shared provider (pooled for connection reuse)
        from enzu.api import _resolve_provider
        self._provider = _resolve_provider(
            self._provider_name,
            api_key=self._api_key,
            use_pool=True,  # critical: share httpx.Client
        )

        self._running = True

        # Start minimum workers
        for _ in range(self._min_workers):
            self._spawn_worker()

        # Start autoscaler
        self._scaler_task = asyncio.create_task(self._autoscaler())

        logger.info(
            "TaskQueue started: provider=%s, model=%s, workers=%d, queue_size=%d",
            self._provider_name, self._model, self._min_workers, self._queue_size
        )

    async def stop(self, timeout: float = 30.0) -> None:
        """
        Graceful shutdown.

        Waits for queued tasks to complete (up to timeout), then cancels workers.
        """
        if not self._running:
            return

        self._running = False

        # Stop autoscaler
        if self._scaler_task:
            self._scaler_task.cancel()
            try:
                await self._scaler_task
            except asyncio.CancelledError:
                pass

        # Wait for queue to drain (with timeout)
        if self._queue and not self._queue.empty():
            try:
                await asyncio.wait_for(self._queue.join(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("Queue drain timeout, %d tasks remaining", self._queue.qsize())

        # Cancel workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)

        self._workers.clear()
        self._queue = None

        # Close provider pool connections
        from enzu.providers.pool import close_all_providers
        close_all_providers()

        logger.info("TaskQueue stopped: completed=%d, failed=%d", self._completed_count, self._failed_count)

    async def submit(
        self,
        task: Union[str, TaskSpec, Dict[str, Any]],
        **kwargs,
    ) -> asyncio.Future:
        """
        Submit a task for processing.

        Returns a Future that resolves to the task result (str or ExecutionReport).
        Blocks if queue is full (backpressure).

        Args:
            task: Task text, TaskSpec, or dict
            **kwargs: Additional arguments passed to enzu.run()

        Returns:
            Future that resolves to task result
        """
        if not self._running or self._queue is None:
            raise RuntimeError("Queue not started. Call start() first.")

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        request = TaskRequest(
            task=task,
            kwargs=kwargs,
            future=future,
        )

        # This blocks if queue is full (backpressure)
        await self._queue.put(request)

        return future

    async def submit_many(
        self,
        tasks: List[Union[str, TaskSpec, Dict[str, Any]]],
        **kwargs,
    ) -> List[asyncio.Future]:
        """
        Submit multiple tasks.

        Returns list of Futures in same order as input tasks.
        """
        return [await self.submit(task, **kwargs) for task in tasks]

    def submit_sync(
        self,
        task: Union[str, TaskSpec, Dict[str, Any]],
        **kwargs,
    ) -> SyncFuture:
        """
        Submit from synchronous code.

        Returns a concurrent.futures.Future (not asyncio.Future).
        Use result() to block and get the result.

        Example:
            future = queue.submit_sync("Translate: Hello")
            result = future.result(timeout=60)
        """
        import concurrent.futures

        sync_future: SyncFuture = concurrent.futures.Future()

        async def _submit_and_bridge():
            try:
                async_future = await self.submit(task, **kwargs)
                result = await async_future
                sync_future.set_result(result)
            except Exception as e:
                sync_future.set_exception(e)

        # Run in event loop
        try:
            asyncio.get_running_loop()
            asyncio.ensure_future(_submit_and_bridge())
        except RuntimeError:
            # No running loop - create one
            asyncio.run(_submit_and_bridge())

        return sync_future

    def stats(self) -> QueueStats:
        """Get current queue statistics."""
        queued = self._queue.qsize() if self._queue else 0
        avg_latency = (
            self._total_latency_ms / self._completed_count
            if self._completed_count > 0
            else 0.0
        )
        return QueueStats(
            queued=queued,
            active=self._active_count,
            workers=len(self._workers),
            completed=self._completed_count,
            failed=self._failed_count,
            avg_latency_ms=avg_latency,
        )

    def _spawn_worker(self) -> None:
        """Spawn a new worker coroutine."""
        worker = asyncio.create_task(self._worker_loop())
        self._workers.append(worker)

    def _remove_worker(self) -> None:
        """Remove one worker (for scale-down)."""
        if len(self._workers) <= self._min_workers:
            return
        # Cancel the last worker
        worker = self._workers.pop()
        worker.cancel()

    async def _worker_loop(self) -> None:
        """
        Worker coroutine: pulls from queue, executes, returns result via Future.

        Runs until cancelled. Handles exceptions gracefully.
        """
        from enzu.engine import Engine
        from enzu.rlm import RLMEngine
        from enzu.api import _build_task_spec

        while self._running:
            try:
                # Wait for task (with timeout to check _running flag)
                if self._queue is None:
                    break
                try:
                    request = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                # Track active count
                async with self._stats_lock:
                    self._active_count += 1

                start_time = time.time()

                try:
                    # Merge defaults with request kwargs
                    kwargs: Dict[str, Any] = {
                        "model": self._model,
                        "provider": self._provider,
                        **({"mode": self._default_mode} if self._default_mode else {}),
                        **({"tokens": self._default_tokens} if self._default_tokens else {}),
                        **({"temperature": self._default_temperature} if self._default_temperature else {}),
                        **request.kwargs,
                    }

                    # Determine mode
                    mode = kwargs.pop("mode", None) or "chat"
                    data = kwargs.pop("data", None)

                    # Build task spec
                    spec = _build_task_spec(
                        request.task,
                        model=kwargs.get("model"),
                        tokens=kwargs.get("tokens"),
                        seconds=kwargs.get("seconds"),
                        cost=kwargs.get("cost"),
                        contains=kwargs.get("contains"),
                        matches=kwargs.get("matches"),
                        min_words=kwargs.get("min_words"),
                        goal=kwargs.get("goal"),
                        limits=kwargs.get("limits"),
                        check=kwargs.get("check"),
                        responses=kwargs.get("responses"),
                        temperature=kwargs.get("temperature"),
                        is_rlm=(data is not None),
                    )

                    # Get optional progress callback
                    on_progress = kwargs.get("on_progress")

                    # Execute
                    if self._provider is None:
                        raise RuntimeError("Provider not initialized. Call start() first.")
                    if mode == "rlm" or data is not None:
                        rlm_engine = RLMEngine()
                        rlm_report = rlm_engine.run(
                            spec, 
                            self._provider, 
                            data=data or "",
                            on_progress=on_progress,
                        )
                        report: Union[ExecutionReport, RLMExecutionReport] = rlm_report
                    else:
                        chat_engine = Engine()
                        chat_report = chat_engine.run(spec, self._provider)
                        report = chat_report

                    # Return full report (caller extracts answer/usage)
                    if not request.future.done():
                        request.future.set_result(report)

                    # Update stats
                    latency_ms = (time.time() - start_time) * 1000
                    async with self._stats_lock:
                        self._completed_count += 1
                        self._total_latency_ms += latency_ms

                except Exception as e:
                    # Return error
                    if not request.future.done():
                        request.future.set_exception(e)

                    async with self._stats_lock:
                        self._failed_count += 1

                    logger.warning("Task failed: %s", e)

                finally:
                    # Mark task done (for queue.join())
                    if self._queue is not None:
                        self._queue.task_done()

                    async with self._stats_lock:
                        self._active_count -= 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Worker error: %s", e)

    async def _autoscaler(self) -> None:
        """
        Autoscaler coroutine: adjusts worker count based on queue depth.

        Runs periodically, checks queue depth vs worker count, scales accordingly.
        """
        while self._running:
            try:
                await asyncio.sleep(self._scale_interval)

                if self._queue is None:
                    continue

                queue_depth = self._queue.qsize() + self._active_count
                worker_count = len(self._workers)

                # Scale up: queue is backing up
                if queue_depth > self._scale_up_threshold * worker_count:
                    if worker_count < self._max_workers:
                        self._spawn_worker()
                        logger.debug(
                            "Scaled up: workers=%d, queue=%d",
                            len(self._workers), queue_depth
                        )

                # Scale down: queue is draining
                elif queue_depth < self._scale_down_threshold * worker_count:
                    if worker_count > self._min_workers:
                        self._remove_worker()
                        logger.debug(
                            "Scaled down: workers=%d, queue=%d",
                            len(self._workers), queue_depth
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Autoscaler error: %s", e)


# Global queue instance for simple usage
_global_queue: Optional[TaskQueue] = None


async def start_queue(
    *,
    provider: str = "openrouter",
    model: str,
    api_key: Optional[str] = None,
    min_workers: int = 2,
    max_workers: int = 20,
    queue_size: int = 1000,
    **kwargs,
) -> TaskQueue:
    """
    Start the global task queue.

    Convenience function for simple single-queue usage.
    For multiple queues, instantiate TaskQueue directly.
    """
    global _global_queue

    if _global_queue is not None:
        await _global_queue.stop()

    _global_queue = TaskQueue(
        provider=provider,
        model=model,
        api_key=api_key,
        min_workers=min_workers,
        max_workers=max_workers,
        queue_size=queue_size,
        **kwargs,
    )
    await _global_queue.start()
    return _global_queue


async def stop_queue() -> None:
    """Stop the global task queue."""
    global _global_queue

    if _global_queue is not None:
        await _global_queue.stop()
        _global_queue = None


async def submit(
    task: Union[str, TaskSpec, Dict[str, Any]],
    **kwargs,
) -> asyncio.Future:
    """Submit task to global queue. Queue must be started first."""
    if _global_queue is None:
        raise RuntimeError("Global queue not started. Call start_queue() first.")
    return await _global_queue.submit(task, **kwargs)


def get_queue() -> Optional[TaskQueue]:
    """Get the global queue instance."""
    return _global_queue
