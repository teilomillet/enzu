"""
Async task management endpoints with SSE streaming.

POST /v1/tasks - Submit async task (returns immediately with task_id)
GET /v1/tasks/{id} - Get task status/result
GET /v1/tasks/{id}/stream - SSE stream of progress events
DELETE /v1/tasks/{id} - Cancel a running task
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from enzu.server.auth import get_api_key


router = APIRouter(prefix="/v1/tasks", tags=["tasks"])


# =============================================================================
# Task State Management
# =============================================================================


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"  # Queued, waiting for worker
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Finished successfully
    FAILED = "failed"  # Finished with error
    CANCELLED = "cancelled"  # Cancelled by user


@dataclass
class ProgressEvent:
    """A progress event during task execution."""

    timestamp: float
    message: str
    event_type: str = "progress"  # progress, stream, budget, error


@dataclass
class TaskState:
    """In-memory state for an async task."""

    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Request info
    task: str = ""
    session_id: Optional[str] = None

    # Result
    answer: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None

    # Progress events (for SSE)
    events: List[ProgressEvent] = field(default_factory=list)
    _event_waiters: List[asyncio.Event] = field(default_factory=list)

    def add_event(self, message: str, event_type: str = "progress") -> None:
        """Add a progress event and notify waiters."""
        self.events.append(
            ProgressEvent(
                timestamp=time.time(),
                message=message,
                event_type=event_type,
            )
        )
        # Notify all SSE waiters
        for waiter in self._event_waiters:
            waiter.set()

    def subscribe(self) -> asyncio.Event:
        """Subscribe to events (for SSE streaming)."""
        event = asyncio.Event()
        self._event_waiters.append(event)
        return event

    def unsubscribe(self, event: asyncio.Event) -> None:
        """Unsubscribe from events."""
        if event in self._event_waiters:
            self._event_waiters.remove(event)


# In-memory task store (would use Redis in production)
_tasks: Dict[str, TaskState] = {}
_tasks_lock = asyncio.Lock()


async def get_task(task_id: str) -> Optional[TaskState]:
    """Get task by ID."""
    return _tasks.get(task_id)


async def create_task(task: str, session_id: Optional[str] = None) -> TaskState:
    """Create a new task."""
    task_id = f"task-{uuid.uuid4().hex[:12]}"
    state = TaskState(task_id=task_id, task=task, session_id=session_id)
    async with _tasks_lock:
        _tasks[task_id] = state
    return state


async def cleanup_old_tasks(max_age_seconds: int = 3600) -> int:
    """Remove completed tasks older than max_age."""
    now = time.time()
    to_remove = []
    async with _tasks_lock:
        for task_id, state in _tasks.items():
            if state.status in (
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            ):
                if state.completed_at and (now - state.completed_at) > max_age_seconds:
                    to_remove.append(task_id)
        for task_id in to_remove:
            del _tasks[task_id]
    return len(to_remove)


# =============================================================================
# Task Executor
# =============================================================================


async def execute_task(state: TaskState, request_params: Dict[str, Any]) -> None:
    """
    Execute a task in background with progress callbacks.

    This runs in a background task, updating state as it progresses.
    For session tasks, acquires lock to ensure sequential execution
    and consistent history/budget tracking.
    """
    from enzu.server.services import enzu_service
    from enzu.server.services.session_manager import get_session_store
    from enzu.server.exceptions import SessionNotFoundError, SessionLockError

    state.status = TaskStatus.RUNNING
    state.started_at = time.time()
    state.add_event("Task started", "status")

    lock_owner = state.task_id

    try:
        # Progress callback for real-time updates
        def on_progress(message: str) -> None:
            # Detect event type from message prefix
            if message.startswith("llm_stream:"):
                state.add_event(message[11:], "stream")
            elif "budget" in message.lower():
                state.add_event(message, "budget")
            else:
                state.add_event(message, "progress")

        # Execute based on whether it's a session task or standalone
        if state.session_id:
            store = get_session_store()

            # Wait for lock with retry (session tasks must run sequentially)
            max_wait = 300  # 5 minutes max wait
            wait_interval = 0.5
            waited = 0.0
            lock_acquired = False

            state.add_event("Waiting for session lock...", "progress")

            while waited < max_wait:
                try:
                    lock_acquired = await store.acquire_lock(
                        state.session_id, lock_owner
                    )
                    if lock_acquired:
                        break
                except SessionNotFoundError:
                    raise

                await asyncio.sleep(wait_interval)
                waited += wait_interval

            if not lock_acquired:
                raise RuntimeError(
                    f"Timeout waiting for session lock after {max_wait}s"
                )

            state.add_event("Session lock acquired", "progress")

            try:
                # Fetch fresh session state AFTER acquiring lock
                session = await store.get(state.session_id)

                result = await enzu_service.run_session_task(
                    session=session,
                    task=state.task,
                    data=request_params.get("data"),
                    cost=request_params.get("cost"),
                    tokens=request_params.get("tokens"),
                    seconds=request_params.get("seconds"),
                    temperature=request_params.get("temperature"),
                    max_steps=request_params.get("max_steps"),
                    contains=request_params.get("contains"),
                    matches=request_params.get("matches"),
                    min_words=request_params.get("min_words"),
                    goal=request_params.get("goal"),
                )

                # Update session in store (it's been mutated by run_session_task)
                await store.update(state.session_id, session)
            finally:
                # Always release lock
                try:
                    await store.release_lock(state.session_id, lock_owner)
                    state.add_event("Session lock released", "progress")
                except (SessionNotFoundError, SessionLockError):
                    pass
        else:
            # Standalone task: use TaskQueue for 10K+ concurrency
            state.add_event("Submitting to TaskQueue...", "progress")
            queue = await enzu_service.get_task_queue()

            # Progress callback that feeds SSE stream
            def on_queue_progress(message: str) -> None:
                if message.startswith("llm_stream:"):
                    state.add_event(message[11:], "stream")
                elif message.startswith("rlm_step:"):
                    state.add_event(message[9:], "rlm_step")
                elif message.startswith("rlm_complete:"):
                    state.add_event(message[13:], "rlm_complete")
                elif "budget" in message.lower():
                    state.add_event(message, "budget")
                else:
                    state.add_event(message, "progress")

            # Build task spec for queue submission
            task_kwargs = {
                "data": request_params.get("data"),
                "cost": request_params.get("cost"),
                "tokens": request_params.get("tokens"),
                "seconds": request_params.get("seconds"),
                "temperature": request_params.get("temperature"),
                "max_steps": request_params.get("max_steps"),
                "on_progress": on_queue_progress,
            }
            # Filter out None values (except on_progress which is always set)
            task_kwargs = {k: v for k, v in task_kwargs.items() if v is not None}

            # Submit to queue and await result
            future = await queue.submit(state.task, **task_kwargs)
            state.add_event("Task queued, waiting for worker...", "progress")

            report = await future

            # Extract result from report
            if hasattr(report, "answer"):
                answer = report.answer or ""
            elif hasattr(report, "output_text"):
                answer = report.output_text or ""
            else:
                answer = str(report)

            usage = report.budget_usage if hasattr(report, "budget_usage") else None

            result = enzu_service.RunResult(
                answer=answer,
                model=queue._model,
                total_tokens=usage.total_tokens if usage else None,
                prompt_tokens=usage.input_tokens if usage else None,
                completion_tokens=usage.output_tokens if usage else None,
                cost_usd=usage.cost_usd if usage else None,
            )

        # Success
        state.status = TaskStatus.COMPLETED
        state.completed_at = time.time()
        state.answer = result.answer
        state.usage = {
            "total_tokens": result.total_tokens,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "cost_usd": result.cost_usd,
        }
        state.add_event("Task completed", "status")

    except Exception as e:
        state.status = TaskStatus.FAILED
        state.completed_at = time.time()
        state.error = str(e)
        state.add_event(f"Task failed: {e}", "error")


# =============================================================================
# Request/Response Schemas
# =============================================================================


class SubmitTaskRequest(BaseModel):
    """Request to submit an async task."""

    task: str = Field(..., description="The task/prompt to execute")
    data: Optional[str] = Field(None, description="Context data")
    model: Optional[str] = Field(None, description="Model identifier")
    provider: Optional[str] = Field(None, description="Provider name")
    session_id: Optional[str] = Field(
        None, description="Optional session ID for context"
    )

    # Budget constraints
    cost: Optional[float] = Field(None, ge=0)
    tokens: Optional[int] = Field(None, ge=1)
    seconds: Optional[float] = Field(None, ge=0)

    # Parameters
    temperature: Optional[float] = Field(None, ge=0, le=2)
    max_steps: Optional[int] = Field(None, ge=1)


class TaskStatusResponse(BaseModel):
    """Response with task status."""

    task_id: str
    status: TaskStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Result (only if completed)
    answer: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None

    # Progress
    event_count: int = 0


class SubmitTaskResponse(BaseModel):
    """Response after submitting a task."""

    task_id: str
    status: TaskStatus
    stream_url: str = Field(..., description="URL to stream progress events")
    poll_url: str = Field(..., description="URL to poll for status")


# =============================================================================
# Endpoints
# =============================================================================


@router.post("", response_model=SubmitTaskResponse)
async def submit_task(
    request_body: SubmitTaskRequest,
    request: Request,
    api_key: str = Depends(get_api_key),
) -> SubmitTaskResponse:
    """
    Submit an async task for execution.

    Returns immediately with task_id. Use the stream_url for real-time
    progress events, or poll poll_url for status.

    Tasks execute in background - results persist for 1 hour after completion.
    """
    # Create task state
    state = await create_task(
        task=request_body.task,
        session_id=request_body.session_id,
    )

    # Start background execution
    asyncio.create_task(execute_task(state, request_body.model_dump()))

    base_url = str(request.base_url).rstrip("/")
    return SubmitTaskResponse(
        task_id=state.task_id,
        status=state.status,
        stream_url=f"{base_url}/v1/tasks/{state.task_id}/stream",
        poll_url=f"{base_url}/v1/tasks/{state.task_id}",
    )


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    api_key: str = Depends(get_api_key),
) -> TaskStatusResponse:
    """
    Get current status of a task.

    Poll this endpoint to check if task is complete.
    For real-time updates, use the /stream endpoint instead.
    """
    state = await get_task(task_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    return TaskStatusResponse(
        task_id=state.task_id,
        status=state.status,
        created_at=state.created_at,
        started_at=state.started_at,
        completed_at=state.completed_at,
        answer=state.answer,
        error=state.error,
        usage=state.usage,
        event_count=len(state.events),
    )


@router.get("/{task_id}/stream")
async def stream_task_events(
    task_id: str,
    api_key: str = Depends(get_api_key),
) -> StreamingResponse:
    """
    Stream task progress events via Server-Sent Events (SSE).

    Events are sent as they occur during execution.
    Stream ends when task completes or fails.

    Event types:
    - status: Task status changes (started, completed, failed)
    - progress: General progress updates
    - stream: LLM token streaming
    - budget: Budget-related events
    - error: Error events

    Example SSE format:
        event: progress
        data: {"message": "Starting RLM execution", "timestamp": 1234567890.123}

        event: stream
        data: {"message": "Hello", "timestamp": 1234567890.456}
    """
    state = await get_task(task_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    async def event_generator() -> AsyncGenerator[str, None]:
        import json

        # Send any existing events first
        sent_count = 0
        for event in state.events:
            yield f"event: {event.event_type}\n"
            yield f"data: {json.dumps({'message': event.message, 'timestamp': event.timestamp})}\n\n"
            sent_count += 1

        # Subscribe to new events
        waiter = state.subscribe()

        try:
            # Stream until task completes
            while state.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                # Wait for new event (with timeout to check status)
                try:
                    await asyncio.wait_for(waiter.wait(), timeout=1.0)
                    waiter.clear()
                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield ": heartbeat\n\n"
                    continue

                # Send new events
                for event in state.events[sent_count:]:
                    yield f"event: {event.event_type}\n"
                    yield f"data: {json.dumps({'message': event.message, 'timestamp': event.timestamp})}\n\n"
                    sent_count += 1

            # Send final result
            if state.status == TaskStatus.COMPLETED:
                yield "event: complete\n"
                yield f"data: {json.dumps({'answer': state.answer, 'usage': state.usage})}\n\n"
            elif state.status == TaskStatus.FAILED:
                yield "event: error\n"
                yield f"data: {json.dumps({'error': state.error})}\n\n"

        finally:
            state.unsubscribe(waiter)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.delete("/{task_id}")
async def cancel_task(
    task_id: str,
    api_key: str = Depends(get_api_key),
) -> dict:
    """
    Cancel a running task.

    Only pending or running tasks can be cancelled.
    """
    state = await get_task(task_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    if state.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel task in {state.status} status",
        )

    state.status = TaskStatus.CANCELLED
    state.completed_at = time.time()
    state.add_event("Task cancelled by user", "status")

    return {"cancelled": True, "task_id": task_id}


@router.get("")
async def list_tasks(
    status: Optional[TaskStatus] = None,
    limit: int = 100,
    api_key: str = Depends(get_api_key),
) -> List[TaskStatusResponse]:
    """
    List tasks, optionally filtered by status.

    Returns most recent tasks first.
    """
    tasks = list(_tasks.values())

    if status:
        tasks = [t for t in tasks if t.status == status]

    # Sort by created_at descending
    tasks.sort(key=lambda t: t.created_at, reverse=True)
    tasks = tasks[:limit]

    return [
        TaskStatusResponse(
            task_id=t.task_id,
            status=t.status,
            created_at=t.created_at,
            started_at=t.started_at,
            completed_at=t.completed_at,
            answer=t.answer if t.status == TaskStatus.COMPLETED else None,
            error=t.error,
            usage=t.usage,
            event_count=len(t.events),
        )
        for t in tasks
    ]
