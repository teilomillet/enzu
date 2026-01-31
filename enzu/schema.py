from __future__ import annotations

from typing import Any, Dict, Optional, Literal

from pydantic import BaseModel, ConfigDict, Field

from enzu.contract import DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MIN_WORD_COUNT
from enzu.models import (
    Budget,
    ExecutionReport,
    RLMExecutionReport,
    SuccessCriteria,
    TaskSpec,
)


class TaskSpecInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Input shape mirrors CLI/Python JSON payloads before defaults are applied.
    # Defaults are injected by enzu.contract.apply_task_defaults.
    task_id: str = Field(description="Client-generated task id.")
    input_text: str = Field(description="Task prompt text.")
    model: str = Field(description="Provider model identifier.")
    responses: Dict[str, Any] = Field(
        default_factory=dict,
        description="Open Responses API request overrides (input/instructions/tools/etc).",
    )
    budget: Optional[Budget] = Field(
        default=None,
        description="Optional budget. Defaults inject max_output_tokens.",
    )
    success_criteria: Optional[SuccessCriteria] = Field(
        default=None,
        description="Optional verification checks. Defaults inject min_word_count.",
    )
    max_output_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Per-call output token limit. Fallback for budget.max_tokens when budget is missing.",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Sampling temperature override.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form metadata passed to the engine.",
    )


class RunPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # CLI payload shape; automode is CLI-only and not accepted by enzu.api.run().
    # Mode list must stay in sync with CLI arg choices and schema exports.
    mode: Literal["chat", "rlm", "automode"] = Field(
        default="chat",
        description="chat|rlm|automode. RLM requires context/data.",
    )
    provider: str = Field(description="Provider name from the registry.")
    task: TaskSpecInput = Field(description="Task payload.")
    model: Optional[str] = Field(
        default=None,
        description="Optional override for task.model.",
    )
    context: Optional[str] = Field(
        default=None,
        description="RLM context string.",
    )
    context_file: Optional[str] = Field(
        default=None,
        description="CLI-only path to load context for RLM mode.",
    )
    fs_root: Optional[str] = Field(
        default=None,
        description="CLI-only root path for automode.",
    )


def task_spec_schema() -> Dict[str, Any]:
    """Schema for the fully validated TaskSpec model (post-defaults)."""
    return TaskSpec.model_json_schema()


def task_input_schema() -> Dict[str, Any]:
    """Schema for the TaskSpec input shape (pre-defaults)."""
    return TaskSpecInput.model_json_schema()


def run_payload_schema() -> Dict[str, Any]:
    """Schema for the CLI JSON payload (pre-defaults)."""
    return RunPayload.model_json_schema()


def report_schema() -> Dict[str, Any]:
    """Schema bundle for chat and RLM reports."""
    return {
        "execution_report": ExecutionReport.model_json_schema(),
        "rlm_execution_report": RLMExecutionReport.model_json_schema(),
    }


def schema_bundle() -> Dict[str, Any]:
    """Bundle used by CLI --print-schema and docs generation."""
    return {
        "meta": {
            "defaults": {
                "budget.max_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
                "success_criteria.min_word_count": DEFAULT_MIN_WORD_COUNT,
            },
            "mode_requirements": {
                "chat": {"context_required": False},
                "rlm": {"context_required": True},
                "automode": {
                    "context_required": False,
                    "fs_root_required": True,
                    "cli_only": True,
                },
            },
        },
        "task_input": task_input_schema(),
        "task_spec": task_spec_schema(),
        "run_payload": run_payload_schema(),
        "report": report_schema(),
    }
