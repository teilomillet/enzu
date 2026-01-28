"""
Pydantic models for API request/response schemas.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Run Endpoint Schemas
# =============================================================================


class RunRequest(BaseModel):
    """Request body for POST /v1/run."""

    task: str = Field(..., description="The task/prompt to execute")
    data: Optional[str] = Field(None, description="Context data for the task")
    model: Optional[str] = Field(None, description="Model identifier")
    provider: Optional[str] = Field(None, description="Provider name")

    # Budget constraints
    cost: Optional[float] = Field(None, description="Max cost in USD", ge=0)
    tokens: Optional[int] = Field(None, description="Max output tokens", ge=1)
    seconds: Optional[float] = Field(None, description="Max execution time", ge=0)

    # Generation parameters
    temperature: Optional[float] = Field(None, description="Temperature (0.0-2.0)", ge=0, le=2)
    max_steps: Optional[int] = Field(None, description="Max reasoning steps (RLM mode)", ge=1)

    # Success criteria
    contains: Optional[List[str]] = Field(None, description="Output must contain these substrings")
    matches: Optional[List[str]] = Field(None, description="Output must match these regexes")
    min_words: Optional[int] = Field(None, description="Minimum word count", ge=1)
    goal: Optional[str] = Field(None, description="Goal for model self-verification")


class UsageInfo(BaseModel):
    """Token and cost usage information."""

    total_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cost_usd: Optional[float] = None


class RunResponse(BaseModel):
    """Response body for POST /v1/run."""

    answer: str = Field(..., description="The generated answer")
    request_id: str = Field(..., description="Unique request identifier")
    model: str = Field(..., description="Model used for generation")
    usage: Optional[UsageInfo] = Field(None, description="Token and cost usage")


# =============================================================================
# Session Endpoint Schemas
# =============================================================================


class CreateSessionRequest(BaseModel):
    """Request body for POST /v1/sessions."""

    model: Optional[str] = Field(None, description="Model identifier")
    provider: Optional[str] = Field(None, description="Provider name")
    max_cost_usd: Optional[float] = Field(None, description="Session cost cap", ge=0)
    max_tokens: Optional[int] = Field(None, description="Session token cap", ge=1)
    ttl_seconds: Optional[int] = Field(None, description="Session TTL in seconds", ge=1)


class CreateSessionResponse(BaseModel):
    """Response body for POST /v1/sessions."""

    session_id: str = Field(..., description="Unique session identifier")
    model: str = Field(..., description="Model for this session")
    provider: str = Field(..., description="Provider for this session")
    created_at: str = Field(..., description="ISO timestamp of creation")
    max_cost_usd: Optional[float] = None
    max_tokens: Optional[int] = None
    ttl_seconds: int = Field(..., description="Session TTL")


class SessionRunRequest(BaseModel):
    """Request body for POST /v1/sessions/{id}/run."""

    task: str = Field(..., description="The task/prompt to execute")
    data: Optional[str] = Field(None, description="Context data for the task")

    # Budget constraints (per-request, within session budget)
    cost: Optional[float] = Field(None, description="Max cost in USD for this request", ge=0)
    tokens: Optional[int] = Field(None, description="Max output tokens", ge=1)
    seconds: Optional[float] = Field(None, description="Max execution time", ge=0)

    # Generation parameters
    temperature: Optional[float] = Field(None, description="Temperature (0.0-2.0)", ge=0, le=2)
    max_steps: Optional[int] = Field(None, description="Max reasoning steps", ge=1)

    # Success criteria
    contains: Optional[List[str]] = Field(None, description="Output must contain these substrings")
    matches: Optional[List[str]] = Field(None, description="Output must match these regexes")
    min_words: Optional[int] = Field(None, description="Minimum word count", ge=1)
    goal: Optional[str] = Field(None, description="Goal for model self-verification")


class ExchangeInfo(BaseModel):
    """Single conversation exchange."""

    user: str = Field(..., description="User prompt")
    assistant: str = Field(..., description="Assistant response")
    timestamp: str = Field(..., description="ISO timestamp")
    cost_usd: Optional[float] = None


class SessionStateResponse(BaseModel):
    """Response body for GET /v1/sessions/{id}."""

    session_id: str = Field(..., description="Session identifier")
    model: str = Field(..., description="Model for this session")
    provider: str = Field(..., description="Provider for this session")
    created_at: str = Field(..., description="ISO timestamp of creation")
    total_cost_usd: float = Field(..., description="Total cost incurred")
    total_tokens: int = Field(..., description="Total tokens used")
    max_cost_usd: Optional[float] = None
    max_tokens: Optional[int] = None
    remaining_cost_usd: Optional[float] = None
    remaining_tokens: Optional[int] = None
    exchange_count: int = Field(..., description="Number of exchanges")
    exchanges: List[ExchangeInfo] = Field(..., description="Conversation history")


class SessionRunResponse(BaseModel):
    """Response body for POST /v1/sessions/{id}/run."""

    answer: str = Field(..., description="The generated answer")
    request_id: str = Field(..., description="Unique request identifier")
    session_id: str = Field(..., description="Session identifier")
    exchange_number: int = Field(..., description="Exchange number in session")
    usage: Optional[UsageInfo] = Field(None, description="Token and cost usage for this request")
    session_total_cost_usd: float = Field(..., description="Cumulative session cost")
    session_total_tokens: int = Field(..., description="Cumulative session tokens")


# =============================================================================
# Health Endpoint Schemas
# =============================================================================


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")


# =============================================================================
# Error Schemas
# =============================================================================


class ErrorDetail(BaseModel):
    """Error detail."""

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    request_id: Optional[str] = Field(None, description="Request ID for tracing")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: ErrorDetail
