"""
Session management endpoints.

POST /v1/sessions - Create new session
GET /v1/sessions/{id} - Get session state
POST /v1/sessions/{id}/run - Execute within session
POST /v1/sessions/{id}/run/async - Submit async task for session
GET /v1/sessions/{id}/events - SSE stream of session events
DELETE /v1/sessions/{id} - Delete session
"""
import asyncio
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse

from enzu.server.auth import get_api_key
from enzu.server.config import get_settings
from enzu.server.exceptions import SessionNotFoundError, SessionLockError
from enzu.server.schemas import (
    CreateSessionRequest,
    CreateSessionResponse,
    SessionRunRequest,
    SessionRunResponse,
    SessionStateResponse,
    ExchangeInfo,
    UsageInfo,
)
from enzu.server.services import enzu_service
from enzu.server.services.session_manager import get_session_store
from enzu.server.middleware import get_request_id


router = APIRouter(prefix="/v1/sessions", tags=["sessions"])


@router.post("", response_model=CreateSessionResponse)
async def create_session(
    request_body: CreateSessionRequest,
    api_key: str = Depends(get_api_key),
) -> CreateSessionResponse:
    """
    Create a new conversation session.

    Sessions maintain conversation history across multiple run() calls,
    enabling multi-turn conversations with context.

    Headers:
    - X-API-Key: Required if authentication is enabled

    Returns session ID and configuration.
    """
    settings = get_settings()
    store = get_session_store()

    # Create the underlying Session object
    session = enzu_service.create_session(
        model=request_body.model,
        provider=request_body.provider,
        max_cost_usd=request_body.max_cost_usd,
        max_tokens=request_body.max_tokens,
    )

    # Determine TTL
    ttl = request_body.ttl_seconds or settings.session_ttl_seconds

    # Store the session
    session_id = await store.create(session, ttl_seconds=ttl)

    provider_name = session.provider if isinstance(session.provider, str) else session.provider.name
    return CreateSessionResponse(
        session_id=session_id,
        model=session.model,
        provider=provider_name,
        created_at=session.created_at,
        max_cost_usd=session.max_cost_usd,
        max_tokens=session.max_tokens,
        ttl_seconds=ttl,
    )


@router.get("/{session_id}", response_model=SessionStateResponse)
async def get_session(
    session_id: str,
    api_key: str = Depends(get_api_key),
) -> SessionStateResponse:
    """
    Get session state and conversation history.

    Headers:
    - X-API-Key: Required if authentication is enabled

    Returns session configuration, usage totals, and conversation history.
    """
    store = get_session_store()

    try:
        stored = await store.get_stored(session_id)
        session = stored.session
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Convert exchanges to response format
    exchanges = [
        ExchangeInfo(
            user=ex.user,
            assistant=ex.assistant,
            timestamp=ex.timestamp,
            cost_usd=ex.cost_usd,
        )
        for ex in session.exchanges
    ]

    provider_name = session.provider if isinstance(session.provider, str) else session.provider.name
    return SessionStateResponse(
        session_id=session_id,
        model=session.model,
        provider=provider_name,
        created_at=session.created_at,
        total_cost_usd=session.total_cost_usd,
        total_tokens=session.total_tokens,
        max_cost_usd=session.max_cost_usd,
        max_tokens=session.max_tokens,
        remaining_cost_usd=session.remaining_budget,
        remaining_tokens=session.remaining_tokens,
        exchange_count=len(session.exchanges),
        exchanges=exchanges,
    )


@router.post("/{session_id}/run", response_model=SessionRunResponse)
async def run_in_session(
    session_id: str,
    request_body: SessionRunRequest,
    request: Request,
    api_key: str = Depends(get_api_key),
) -> SessionRunResponse:
    """
    Execute a task within an existing session.

    The session maintains conversation history, so the model has context
    from previous exchanges. Use this for multi-turn conversations.

    Headers:
    - X-API-Key: Required if authentication is enabled
    - X-Request-ID: Optional request identifier (generated if not provided)
    - X-Session-ID: Optional (session_id from URL is used)

    Returns the answer and updated session usage.
    """
    store = get_session_store()
    request_id = getattr(request.state, "request_id", None) or get_request_id() or "unknown"

    # Get session
    try:
        session = await store.get(session_id)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Try to acquire lock for concurrent request safety
    lock_acquired = await store.acquire_lock(session_id, request_id)
    if not lock_acquired:
        raise HTTPException(
            status_code=409,
            detail="Session is currently being used by another request. Try again later.",
        )

    try:
        # Run the task
        result = await enzu_service.run_session_task(
            session=session,
            task=request_body.task,
            data=request_body.data,
            cost=request_body.cost,
            tokens=request_body.tokens,
            seconds=request_body.seconds,
            temperature=request_body.temperature,
            max_steps=request_body.max_steps,
            contains=request_body.contains,
            matches=request_body.matches,
            min_words=request_body.min_words,
            goal=request_body.goal,
        )

        # Update session in store (it's been mutated by run_session_task)
        await store.update(session_id, session)

    finally:
        # Always release lock
        try:
            await store.release_lock(session_id, request_id)
        except (SessionNotFoundError, SessionLockError):
            pass  # Session may have been deleted or lock owner mismatch

    usage = None
    if result.total_tokens is not None or result.cost_usd is not None:
        usage = UsageInfo(
            total_tokens=result.total_tokens,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=result.cost_usd,
        )

    return SessionRunResponse(
        answer=result.answer,
        request_id=request_id,
        session_id=session_id,
        exchange_number=len(session.exchanges),
        usage=usage,
        session_total_cost_usd=session.total_cost_usd,
        session_total_tokens=session.total_tokens,
    )


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    api_key: str = Depends(get_api_key),
) -> dict:
    """
    Delete a session.

    Headers:
    - X-API-Key: Required if authentication is enabled

    Returns success status.
    """
    store = get_session_store()

    deleted = await store.delete(session_id)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    return {"deleted": True, "session_id": session_id}


@router.post("/{session_id}/run/async")
async def run_in_session_async(
    session_id: str,
    request_body: SessionRunRequest,
    request: Request,
    api_key: str = Depends(get_api_key),
) -> dict:
    """
    Submit an async task for execution within a session.

    Returns immediately with task_id. Use the stream_url for real-time
    progress events, or poll poll_url for status.

    Headers:
    - X-API-Key: Required if authentication is enabled
    """
    from enzu.server.routers.tasks import create_task, execute_task

    store = get_session_store()

    try:
        await store.get(session_id)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    state = await create_task(
        task=request_body.task,
        session_id=session_id,
    )

    request_params = request_body.model_dump()
    asyncio.create_task(execute_task(state, request_params))

    base_url = str(request.base_url).rstrip("/")
    return {
        "task_id": state.task_id,
        "status": state.status.value,
        "session_id": session_id,
        "stream_url": f"{base_url}/v1/tasks/{state.task_id}/stream",
        "poll_url": f"{base_url}/v1/tasks/{state.task_id}",
        "events_url": f"{base_url}/v1/sessions/{session_id}/events",
    }


@router.get("/{session_id}/events")
async def stream_session_events(
    session_id: str,
    api_key: str = Depends(get_api_key),
) -> StreamingResponse:
    """
    Stream all task events for a session via Server-Sent Events (SSE).

    This streams events from all tasks associated with this session.
    Useful for monitoring all activity within a session.

    Headers:
    - X-API-Key: Required if authentication is enabled
    """
    import json
    from enzu.server.routers.tasks import _tasks, TaskStatus

    store = get_session_store()

    try:
        await store.get(session_id)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    async def event_generator() -> AsyncGenerator[str, None]:
        sent_events: dict[str, int] = {}

        while True:
            session_tasks = [t for t in _tasks.values() if t.session_id == session_id]

            for task in session_tasks:
                sent_count = sent_events.get(task.task_id, 0)
                for event in task.events[sent_count:]:
                    yield f"event: {event.event_type}\n"
                    yield f"data: {json.dumps({'task_id': task.task_id, 'message': event.message, 'timestamp': event.timestamp})}\n\n"
                    sent_count += 1
                sent_events[task.task_id] = sent_count

            active_tasks = [t for t in session_tasks if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)]
            if not active_tasks:
                yield ": no active tasks\n\n"

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
