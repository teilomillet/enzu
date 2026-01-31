"""
RunEvent: Canonical run summary schema.

A single, provider-agnostic event emitted at the end of each run.
Designed for:
- Tail behavior analysis (p95 cost/run)
- Terminal state distributions
- Integration with Prometheus, OpenTelemetry, JSON logs
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from enzu.models import BudgetUsage, Outcome


class RunEvent(BaseModel):
    """
    Canonical run summary event (single run, terminal).

    Provider-agnostic: describes *what happened*, not *how*.
    Emitted once per run at completion (success or failure).

    Attributes:
        run_id: Unique identifier for this run (for correlation).
        task_id: Optional task identifier from TaskSpec.
        provider: LLM provider used (low-cardinality, safe for metrics labels).
        model: Model used (low-cardinality, safe for metrics labels).
        outcome: Terminal state from Outcome enum.
        success: Whether the run succeeded.
        partial: True if result is incomplete due to budget/timeout.
        started_at: When the run started.
        finished_at: When the run completed.
        elapsed_seconds: Total wall-clock time.
        input_tokens: Input tokens consumed (if known).
        output_tokens: Output tokens consumed (if known).
        total_tokens: Total tokens consumed (if known).
        cost_usd: Cost in USD (if known, OpenRouter only).
        limits_exceeded: Which budget limits were hit.
        retries: Number of retries (for issue #16 tracking).
        attributes: High-cardinality attributes for logs/traces only.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    task_id: Optional[str] = None

    provider: Optional[str] = None
    model: Optional[str] = None

    outcome: Outcome
    success: bool
    partial: bool = False

    started_at: datetime
    finished_at: datetime
    elapsed_seconds: float = Field(..., ge=0)

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost_usd: Optional[float] = Field(default=None, ge=0)

    limits_exceeded: List[str] = Field(default_factory=list)

    retries: int = Field(default=0, ge=0)

    attributes: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_execution_report(
        cls,
        *,
        run_id: str,
        report: Any,
        started_at: datetime,
        finished_at: datetime,
        retries: int = 0,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "RunEvent":
        """
        Create a RunEvent from an ExecutionReport or RLMExecutionReport.

        Args:
            run_id: Unique run identifier.
            report: ExecutionReport or RLMExecutionReport.
            started_at: When the run started.
            finished_at: When the run completed.
            retries: Number of retries that occurred.
            attributes: Optional high-cardinality attributes.

        Returns:
            A new RunEvent instance.
        """
        usage: BudgetUsage = report.budget_usage

        elapsed = usage.elapsed_seconds
        if elapsed is None or elapsed == 0:
            elapsed = (finished_at - started_at).total_seconds()

        return cls(
            run_id=run_id,
            task_id=getattr(report, "task_id", None),
            provider=getattr(report, "provider", None),
            model=getattr(report, "model", None),
            outcome=report.outcome,
            success=bool(report.success),
            partial=bool(report.partial),
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=float(elapsed),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cost_usd=usage.cost_usd,
            limits_exceeded=list(usage.limits_exceeded or []),
            retries=retries,
            attributes=attributes or {},
        )

    def to_log_dict(self) -> Dict[str, Any]:
        """
        Serialize to a dict suitable for JSON logging.

        Returns a stable schema for log parsing:
        {"type": "enzu.run_event.v1", ...fields...}
        """
        data = self.model_dump(mode="json")
        data["type"] = "enzu.run_event.v1"
        return data
