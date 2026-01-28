"""
Run endpoint for single-shot task execution.

POST /v1/run - Execute a task and return the answer (stateless).
"""
from fastapi import APIRouter, Depends, Request

from enzu.server.auth import get_api_key
from enzu.server.schemas import RunRequest, RunResponse, UsageInfo
from enzu.server.services import enzu_service
from enzu.server.middleware import get_request_id


router = APIRouter(prefix="/v1", tags=["run"])


@router.post("/run", response_model=RunResponse)
async def run_task(
    request_body: RunRequest,
    request: Request,
    api_key: str = Depends(get_api_key),
) -> RunResponse:
    """
    Execute a single-shot task.

    This endpoint is stateless - each request is independent.
    For multi-turn conversations, use the sessions endpoints.

    Headers:
    - X-API-Key: Required if authentication is enabled
    - X-Request-ID: Optional request identifier (generated if not provided)

    Returns the generated answer along with usage information.
    """
    request_id = getattr(request.state, "request_id", None) or get_request_id() or "unknown"

    result = await enzu_service.run_task(
        task=request_body.task,
        data=request_body.data,
        model=request_body.model,
        provider=request_body.provider,
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

    usage = None
    if result.total_tokens is not None or result.cost_usd is not None:
        usage = UsageInfo(
            total_tokens=result.total_tokens,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=result.cost_usd,
        )

    return RunResponse(
        answer=result.answer,
        request_id=request_id,
        model=result.model,
        usage=usage,
    )
