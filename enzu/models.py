from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Outcome(str, Enum):
    """
    Canonical run outcomes for predictable execution.

    Budgets are enforced as hard stops - when exceeded, the run terminates
    deterministically and returns an outcome (not just an exception).
    """

    SUCCESS = "success"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    TOOL_ERROR = "tool_error"
    PROVIDER_ERROR = "provider_error"
    INVALID_REQUEST = "invalid_request"
    VERIFICATION_FAILED = "verification_failed"


class JobStatus(str, Enum):
    """Job execution status for async/delegation mode."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Budget(BaseModel):
    """
    Budget limits for task execution.

    See enzu.terminology for full documentation on token terminology.

    Key distinction:
        - Budget.max_tokens: Cumulative OUTPUT tokens across ALL API calls
        - TaskSpec.max_output_tokens: Per-call output limit sent to the API

    Attributes:
        max_tokens: Maximum cumulative output tokens (primary billing metric).
        max_total_tokens: Maximum total tokens (input + output). Advanced.
        max_seconds: Maximum execution time in seconds.
        max_cost_usd: Maximum cost in USD. **OpenRouter only**.
        fallback_providers: List of fallback provider names if primary fails.
    """

    model_config = ConfigDict(extra="forbid")

    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum cumulative output tokens (primary billing metric)",
    )
    max_total_tokens: Optional[int] = Field(default=None, ge=1)
    max_seconds: Optional[float] = Field(default=None, gt=0)
    max_cost_usd: Optional[float] = Field(
        default=None,
        gt=0,
        description="OpenRouter only - other providers don't report cost",
    )
    fallback_providers: Optional[List[str]] = Field(default=None)

    @model_validator(mode="after")
    def require_limit(self) -> "Budget":
        if not any(
            [
                self.max_tokens,
                self.max_total_tokens,
                self.max_seconds,
                self.max_cost_usd,
            ]
        ):
            raise ValueError("Budget requires at least one limit.")
        return self


class SuccessCriteria(BaseModel):
    """
    Defines when a task is considered complete.

    Two modes:
    1. Mechanical checks: required_substrings, required_regex, min_word_count
       - Predefined, deterministic verification
    2. Goal-based: goal field
       - Model self-judges whether goal is achieved
       - Used when user provides intent rather than explicit checks

    At least one of these must be specified.
    """

    model_config = ConfigDict(extra="forbid")

    required_substrings: List[str] = Field(default_factory=list)
    required_regex: List[str] = Field(default_factory=list)
    min_word_count: Optional[int] = Field(default=None, ge=1)
    case_insensitive: bool = False
    # Goal-based success: model self-judges completion.
    # When set, model verifies "Did I achieve this goal?" rather than mechanical checks.
    goal: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def require_checks(self) -> "SuccessCriteria":
        has_mechanical = (
            self.required_substrings or self.required_regex or self.min_word_count
        )
        has_goal = bool(self.goal)
        if not (has_mechanical or has_goal):
            raise ValueError("SuccessCriteria requires at least one check or a goal.")
        return self


class Limits(BaseModel):
    """
    Resource limits. All optional with sensible defaults.

    Attributes:
        tokens: Cumulative OUTPUT tokens across all API calls (primary billing metric).
        total: Cumulative TOTAL tokens (input + output) across all API calls.
        seconds: Maximum execution time in seconds.
        cost: Maximum cost in USD (OpenRouter only).
    """

    model_config = ConfigDict(extra="forbid")

    tokens: Optional[int] = Field(default=None, ge=1)
    total: Optional[int] = Field(default=None, ge=1)
    seconds: Optional[float] = Field(default=None, gt=0)
    cost: Optional[float] = Field(default=None, gt=0)


class Check(BaseModel):
    """Output verification. All optional."""

    model_config = ConfigDict(extra="forbid")

    contains: List[str] = Field(default_factory=list)
    matches: List[str] = Field(default_factory=list)
    min_words: Optional[int] = Field(default=None, ge=1)
    # Goal-based: model self-judges success. See SuccessCriteria.goal.
    goal: Optional[str] = Field(default=None)


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    input_text: str
    model: str
    responses: Dict[str, Any] = Field(
        default_factory=dict,
        description="Open Responses API request overrides (e.g., input, instructions, tools).",
    )
    budget: Budget
    success_criteria: SuccessCriteria
    max_output_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProgressEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(default_factory=utc_now)
    phase: Literal["start", "generation", "verification", "complete", "error"]
    message: str
    is_partial: bool = False
    data: Dict[str, Any] = Field(default_factory=dict)


class TrajectoryStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_index: int
    provider: str
    model: str
    request: str
    response: Optional[str]
    error: Optional[str]
    started_at: datetime
    finished_at: datetime
    usage: Dict[str, Any] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    reasons: List[str] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=utc_now)


class BudgetUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elapsed_seconds: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    cost_usd: Optional[float]
    limits_exceeded: List[str] = Field(default_factory=list)
    # System prompt overhead: tokens consumed by the RLM system prompt.
    # This is a fixed cost users should be aware of for RLM execution.
    system_prompt_tokens: Optional[int] = None


class ProviderResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_text: str
    raw: Any
    usage: Dict[str, Any] = Field(default_factory=dict)
    provider: str
    model: str


class RLMStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_index: int
    prompt: str
    model_output: str
    code: Optional[str]
    stdout: Optional[str]
    error: Optional[str]


class StepFeedback(BaseModel):
    """
    Structured feedback for RLM execution step.

    Two layers of feedback:
    1. Error recovery: hints for safe helper usage after crashes
    2. Code pattern guidance: warnings about anti-patterns (over-delegation, etc.)
    """

    model_config = ConfigDict(extra="forbid")

    # Structural violations (e.g., "multiple_blocks:3")
    violation: Optional[str] = None

    # Error recovery hint (e.g., "Use safe_get(d, key) for dict access")
    hint: Optional[str] = None

    # Available safe helpers for reference
    available_helpers: List[str] = Field(default_factory=list)

    # Code pattern warnings (e.g., "llm_query in loop without batching")
    pattern_warnings: List[str] = Field(default_factory=list)

    # Verification rejection reasons when FINAL is not accepted
    rejection_reasons: List[str] = Field(default_factory=list)

    # Execution results
    stdout: str = ""
    error: Optional[str] = None


class RLMExecutionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    outcome: Outcome = Field(
        default=Outcome.SUCCESS,
        description="Typed outcome for predictable handling",
    )
    partial: bool = Field(
        default=False,
        description="True if result is incomplete due to budget/timeout",
    )
    task_id: str
    provider: str
    model: str
    answer: Optional[str]
    steps: List[RLMStep] = Field(default_factory=list)
    budget_usage: BudgetUsage
    errors: List[str] = Field(default_factory=list)


class ExecutionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    outcome: Outcome = Field(
        default=Outcome.SUCCESS,
        description="Typed outcome for predictable handling",
    )
    partial: bool = Field(
        default=False,
        description="True if result is incomplete due to budget/timeout",
    )
    task_id: str
    provider: str
    model: str
    output_text: Optional[str]
    verification: VerificationResult
    budget_usage: BudgetUsage
    progress_events: List[ProgressEvent] = Field(default_factory=list)
    trajectory: List[TrajectoryStep] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class Job(BaseModel):
    """
    Async job status for delegation mode.

    Jobs are long-running tasks that execute in the background.
    Use submit() to create a job, status() to check progress, cancel() to stop.
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(default=JobStatus.PENDING)
    outcome: Optional[Outcome] = Field(
        default=None, description="Final outcome (only set when completed)"
    )
    partial: bool = Field(default=False)

    created_at: float = Field(..., description="Unix timestamp when job was created")
    started_at: Optional[float] = Field(
        default=None, description="Unix timestamp when execution started"
    )
    completed_at: Optional[float] = Field(
        default=None, description="Unix timestamp when job finished"
    )

    answer: Optional[str] = Field(default=None, description="Result (if completed)")
    error: Optional[str] = Field(default=None, description="Error message (if failed)")

    usage: Optional[BudgetUsage] = Field(
        default=None, description="Token/cost accounting"
    )
    event_count: int = Field(default=0, description="Number of progress events")

    stream_url: Optional[str] = Field(
        default=None, description="URL to stream progress events"
    )
    poll_url: Optional[str] = Field(default=None, description="URL to poll for status")
