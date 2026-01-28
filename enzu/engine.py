"""
Chat Engine: single-shot LLM execution with budget and verification.

Simpler than RLMEngine: no REPL loop, no code execution.
Used for direct LLM queries without the multi-step RLM scaffold.

Shares verification logic with RLMEngine via rlm/verification.py.
"""
from __future__ import annotations

import time
import os
from typing import Callable, Dict, List, Optional

from enzu.budget import count_tokens_exact
from enzu.contract import DEFAULT_MAX_OUTPUT_TOKENS
from enzu.models import (
    BudgetUsage,
    ExecutionReport,
    ProgressEvent,
    TaskSpec,
    TrajectoryStep,
    VerificationResult,
    utc_now,
)
from enzu.providers.base import BaseProvider
from enzu.usage import build_task_input_text, check_budget_limits, normalize_usage
from enzu.rlm.verification import verify_output
from enzu import telemetry


class Engine:
    def run(
        self,
        task: TaskSpec,
        provider: BaseProvider,
        on_progress: Optional[Callable[[ProgressEvent], None]] = None,
        fallback_providers: Optional[List[BaseProvider]] = None,
    ) -> ExecutionReport:
        progress_events: list[ProgressEvent] = []
        trajectory: list[TrajectoryStep] = []
        errors: list[str] = []

        def emit(event: ProgressEvent) -> None:
            progress_events.append(event)
            if on_progress:
                # Wrap callback in try/except to prevent callback failures from
                # crashing the engine. Errors are silently ignored since we don't
                # want progress reporting issues to affect task execution.
                try:
                    on_progress(event)
                except Exception:
                    pass  # Callback failures should not crash engine execution
            # Only log progress events when streaming logs are enabled.
            if telemetry.stream_enabled() or os.getenv("ENZU_LOGFIRE_PROGRESS", "").strip().lower() in {"1", "true", "yes", "on"}:
                telemetry.log(
                    "info",
                    "progress_event",
                    phase=event.phase,
                    event_message=event.message,
                    data=event.data,
                )

        with telemetry.span(
            "engine.run", task_id=task.task_id, provider=provider.name, model=task.model
        ):
            emit(
                ProgressEvent(
                    phase="start",
                    message="task_started",
                    data={"task_id": task.task_id, "provider": provider.name},
                )
            )

            # Budget limits are enforced before provider calls to keep costs bounded.
            max_output_tokens = task.max_output_tokens
            if task.budget.max_tokens is not None:
                if max_output_tokens is None:
                    max_output_tokens = task.budget.max_tokens
                elif max_output_tokens > task.budget.max_tokens:
                    errors.append("task.max_output_tokens exceeds budget.max_tokens")
                    emit(
                        ProgressEvent(
                            phase="error",
                            message="budget_limit_exceeded_preflight",
                            data={"limit": "max_output_tokens"},
                        )
                    )
                    return self._final_report(
                        task=task,
                        provider=provider,
                        output_text=None,
                        progress_events=progress_events,
                        trajectory=trajectory,
                        errors=errors,
                        started_at=None,
                        usage={},
                    )

            if max_output_tokens is None and task.budget.max_total_tokens is not None:
                max_output_tokens = min(
                    DEFAULT_MAX_OUTPUT_TOKENS,
                    task.budget.max_total_tokens,
                )

            if task.budget.max_total_tokens is not None:
                input_text = build_task_input_text(task)
                input_tokens_exact = count_tokens_exact(input_text, task.model)
                output_cap = max_output_tokens or 0
                if input_tokens_exact is None:
                    if output_cap > task.budget.max_total_tokens:
                        max_output_tokens = task.budget.max_total_tokens
                        emit(
                            ProgressEvent(
                                phase="start",
                                message="budget_output_clamped",
                                data={
                                    "limit": "max_total_tokens",
                                    "max_output_tokens": output_cap,
                                    "clamped_output_tokens": max_output_tokens,
                                },
                            )
                        )
                    emit(
                        ProgressEvent(
                            phase="start",
                            message="tokenizer_unavailable_budget_degraded",
                            data={"model": task.model},
                        )
                    )
                else:
                    remaining = task.budget.max_total_tokens - input_tokens_exact
                    if remaining <= 0:
                        errors.append("budget.max_total_tokens exhausted in preflight")
                        emit(
                            ProgressEvent(
                                phase="error",
                                message="budget_limit_exceeded_preflight",
                                data={
                                    "limit": "max_total_tokens",
                                    "input_tokens": input_tokens_exact,
                                },
                            )
                        )
                        return self._final_report(
                            task=task,
                            provider=provider,
                            output_text=None,
                            progress_events=progress_events,
                            trajectory=trajectory,
                            errors=errors,
                            started_at=None,
                            usage={},
                        )
                    if output_cap > remaining:
                        max_output_tokens = remaining
                        emit(
                            ProgressEvent(
                                phase="start",
                                message="budget_output_clamped",
                                data={
                                    "limit": "max_total_tokens",
                                    "input_tokens": input_tokens_exact,
                                    "max_output_tokens": output_cap,
                                    "clamped_output_tokens": max_output_tokens,
                                },
                            )
                        )

            task = task.model_copy(update={"max_output_tokens": max_output_tokens})

            started_at = utc_now()
            started_ts = time.time()

            # Build list of providers to try (primary + fallbacks)
            all_providers = [provider] + (fallback_providers or [])
            output_text = None
            usage: Dict[str, object] = {}
            successful_provider = provider

            for current_provider in all_providers:
                try:
                    with telemetry.span("provider.stream", provider=current_provider.name):
                        provider_result = current_provider.stream(task, on_progress=emit)
                    finished_at = utc_now()
                    trajectory.append(
                        TrajectoryStep(
                            step_index=len(trajectory),
                            provider=current_provider.name,
                            model=task.model,
                            request=build_task_input_text(task),
                            response=provider_result.output_text,
                            error=None,
                            started_at=started_at,
                            finished_at=finished_at,
                            usage=provider_result.usage,
                        )
                    )
                    output_text = provider_result.output_text
                    usage = provider_result.usage
                    successful_provider = current_provider
                    break  # Success, exit the loop
                except Exception as exc:
                    finished_at = utc_now()
                    trajectory.append(
                        TrajectoryStep(
                            step_index=len(trajectory),
                            provider=current_provider.name,
                            model=task.model,
                            request=build_task_input_text(task),
                            response=None,
                            error=str(exc),
                            started_at=started_at,
                            finished_at=finished_at,
                            usage={},
                        )
                    )
                    if current_provider == all_providers[-1]:
                        # Last provider failed, return error
                        errors.append(str(exc))
                        emit(
                            ProgressEvent(
                                phase="error",
                                message="provider_error",
                                data={"error": str(exc), "provider": current_provider.name},
                            )
                        )
                        return self._final_report(
                            task=task,
                            provider=current_provider,
                            output_text=None,
                            progress_events=progress_events,
                            trajectory=trajectory,
                            errors=errors,
                            started_at=started_ts,
                            usage={},
                        )
                    # Not the last provider, log and continue to next
                    errors.append(f"{current_provider.name}: {exc}")
                    emit(
                        ProgressEvent(
                            phase="error",
                            message="provider_fallback",
                            data={"error": str(exc), "provider": current_provider.name},
                        )
                    )
                    started_at = utc_now()  # Reset for next provider
                    continue

            emit(
                ProgressEvent(
                    phase="verification",
                    message="verification_started",
                    data={"task_id": task.task_id},
                )
            )
            # Handle None output_text: verification expects str
            verification = self._verify_output(task, output_text or "")
            elapsed_seconds = time.time() - started_ts
            budget_usage = self._budget_usage(
                task,
                usage,
                elapsed_seconds,
                output_text,
            )
            if task.budget.max_total_tokens is not None and budget_usage.total_tokens is None:
                emit(
                    ProgressEvent(
                        phase="verification",
                        message="usage_missing_total_tokens",
                        data={"provider": successful_provider.name},
                    )
                )
            budget_exceeded = bool(budget_usage.limits_exceeded)

            # Fallback errors don't count as failures if we succeeded
            final_errors = [] if output_text else errors
            success = verification.passed and not budget_exceeded and not final_errors
            emit(
                ProgressEvent(
                    phase="complete",
                    message="task_completed",
                    data={"success": success, "provider": successful_provider.name},
                )
            )
            return ExecutionReport(
                success=success,
                task_id=task.task_id,
                provider=successful_provider.name,
                model=task.model,
                output_text=output_text,
                verification=verification,
                budget_usage=budget_usage,
                progress_events=progress_events,
                trajectory=trajectory,
                errors=errors,  # Keep fallback errors for debugging
            )

    def _verify_output(self, task: TaskSpec, output_text: str) -> VerificationResult:
        """
        Verify output against success criteria (mechanical checks).

        Chat mode uses single-shot generation. goal_based_trust=False because
        chat mode has no FINAL() mechanism for model self-judgment.
        Goal-based verification requires RLM mode.

        Uses shared logic from rlm/verification.py.
        """
        # Chat mode: always run mechanical checks, don't trust goal-only.
        return verify_output(
            task.success_criteria,
            output_text,
            goal_based_trust=False,
        )

    def _budget_usage(
        self,
        task: TaskSpec,
        usage: Dict[str, object],
        elapsed_seconds: float,
        output_text: Optional[str],
    ) -> BudgetUsage:
        # Normalize usage data from provider response.
        # See usage.py for field extraction from various provider formats.
        input_text = build_task_input_text(task)
        normalized = normalize_usage(
            usage,
            input_text=input_text,
            output_text=output_text,
            model=task.model,
        )
        input_tokens = normalized.get("input_tokens")
        output_tokens = normalized.get("output_tokens")
        total_tokens = normalized.get("total_tokens")
        cost_usd = normalized.get("cost_usd")

        # Convert float tokens to int for check_budget_limits
        # normalize_usage returns Optional[float|int] but tokens are always int or None at runtime
        output_tokens_int = int(output_tokens) if isinstance(output_tokens, (int, float)) else None
        total_tokens_int = int(total_tokens) if isinstance(total_tokens, (int, float)) else None

        # Shared limit checking logic: see usage.check_budget_limits()
        limits_exceeded = check_budget_limits(
            task.budget,
            elapsed_seconds,
            output_tokens_int,
            total_tokens_int,
            cost_usd,
        )

        return BudgetUsage(
            elapsed_seconds=elapsed_seconds,
            input_tokens=input_tokens if isinstance(input_tokens, int) else None,
            output_tokens=output_tokens_int,
            total_tokens=total_tokens_int,
            cost_usd=cost_usd,
            limits_exceeded=limits_exceeded,
        )

    def _final_report(
        self,
        task: TaskSpec,
        provider: BaseProvider,
        output_text: Optional[str],
        progress_events: list[ProgressEvent],
        trajectory: list[TrajectoryStep],
        errors: list[str],
        started_at: Optional[float],
        usage: Dict[str, object],
    ) -> ExecutionReport:
        elapsed_seconds = 0.0 if started_at is None else time.time() - started_at
        budget_usage = self._budget_usage(
            task,
            usage,
            elapsed_seconds,
            output_text,
        )
        verification = VerificationResult(passed=False, reasons=["no_output"])
        return ExecutionReport(
            success=False,
            task_id=task.task_id,
            provider=provider.name,
            model=task.model,
            output_text=output_text,
            verification=verification,
            budget_usage=budget_usage,
            progress_events=progress_events,
            trajectory=trajectory,
            errors=errors,
        )
