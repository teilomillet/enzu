"""
RLM step runner: main execution loop extracted from engine.py.

Handles the iterative prompt-execute-feedback cycle with:
- Step iteration up to max_steps
- Code extraction and sandbox execution
- Feedback generation and prompt advancement
- FINAL() detection and verification

Extracted to reduce engine.py complexity and enable testing.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from enzu.models import (
    Outcome,
    RLMExecutionReport,
    RLMStep,
    TaskSpec,
)
from enzu.repl.protocol import SandboxProtocol
from enzu import telemetry

from enzu.rlm.budget import (
    TokenBudgetPool,
    BudgetTracker,
    build_budget_usage,
)
from enzu.rlm.feedback import build_feedback, extract_code, format_feedback
from enzu.rlm.verification import verify_output


def _trim_text(text: Optional[str], limit: int = 2000) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit]


def context_grew(
    before: Optional[Dict[str, Any]],
    after: Dict[str, Any],
) -> bool:
    """Return True when context store gained sources or text."""
    if not before:
        return (
            after.get("num_sources", 0) > 0
            or after.get("num_queries", 0) > 0
            or after.get("total_text_chars", 0) > 0
        )
    return (
        after.get("num_sources", 0) > before.get("num_sources", 0)
        or after.get("num_queries", 0) > before.get("num_queries", 0)
        or after.get("total_text_chars", 0) > before.get("total_text_chars", 0)
    )


class StepRunner:
    """
    Runs the RLM step loop: prompt → LLM → code → sandbox → feedback → repeat.

    Extracted from RLMEngine.run() for clarity and testability.
    """

    def __init__(
        self,
        *,
        task: TaskSpec,
        sandbox: SandboxProtocol,
        token_pool: TokenBudgetPool,
        budget_tracker: BudgetTracker,
        usage_accumulator: Dict[str, Any],
        llm_query: Callable[[str, str], str],
        emit: Callable[[str], None],
        max_steps: int,
        verify_on_final: bool,
        context_path: Optional[str] = None,
        context_before: Optional[Dict[str, Any]] = None,
        drain_budget_notices: Optional[Callable[[], Optional[str]]] = None,
        system_prompt_tokens: Optional[int] = None,
    ) -> None:
        self._task = task
        self._sandbox = sandbox
        self._token_pool = token_pool
        self._budget_tracker = budget_tracker
        self._usage_accumulator = usage_accumulator
        self._llm_query = llm_query
        self._emit = emit
        self._max_steps = max_steps
        self._verify_on_final = verify_on_final
        self._context_path = context_path
        self._context_before = context_before
        self._drain_budget_notices = drain_budget_notices
        self._system_prompt_tokens = system_prompt_tokens

    def run(
        self,
        prompt: str,
        start_ts: float,
        provider_name: str,
    ) -> RLMExecutionReport:
        """
        Execute the step loop.

        Args:
            prompt: Initial system prompt
            start_ts: Start timestamp for elapsed time calculation
            provider_name: Provider name for report

        Returns:
            RLMExecutionReport with success status, answer, steps, and errors
        """
        steps: List[RLMStep] = []
        errors: List[str] = []

        for step_index in range(self._max_steps):
            if self._token_pool.is_exhausted() or self._budget_tracker.is_exhausted():
                errors.append("budget_exceeded")
                telemetry.log(
                    "info",
                    "rlm_budget_exceeded",
                    step_index=step_index,
                    usage=self._usage_accumulator,
                )
                break

            self._emit(f"rlm_step:{step_index + 1}/{self._max_steps} started")
            with telemetry.span("rlm.step", step_index=step_index):
                self._emit(f"rlm_step:{step_index + 1} querying LLM...")
                model_output = self._llm_query(prompt, "rlm.step_query")
                telemetry.log(
                    "info",
                    "rlm_model_output",
                    step_index=step_index,
                    text=_trim_text(model_output),
                )

                code, block_count = extract_code(model_output)
                telemetry.log(
                    "info",
                    "rlm_code_extract",
                    step_index=step_index,
                    block_count=block_count,
                    code_len=len(code or ""),
                )

                stdout = None
                error = None
                budget_pct = self._budget_tracker.percentage_used()
                feedback = None
                accepted = False

                if code:
                    # Pre-extract FINAL from code in case execution fails
                    # Handles: truncated output, syntax errors before FINAL line
                    pre_extracted_final = self._extract_final_from_code(code)
                    self._emit(
                        f"rlm_step:{step_index + 1} executing code ({len(code)} chars)..."
                    )

                    with telemetry.span(
                        "sandbox.exec",
                        step_index=step_index,
                        code_len=len(code),
                    ):
                        result = self._sandbox.exec(code)
                    stdout = result.stdout
                    error = result.error
                    telemetry.log(
                        "info",
                        "sandbox_result",
                        step_index=step_index,
                        code=_trim_text(code),
                        stdout=_trim_text(stdout),
                        error=_trim_text(error),
                    )

                    # If execution failed and no FINAL captured, use pre-extracted
                    if error and not self._sandbox.answer.get("ready"):
                        if pre_extracted_final and len(pre_extracted_final) >= 10:
                            self._sandbox.answer["content"] = pre_extracted_final
                            self._sandbox.answer["ready"] = True
                            telemetry.log(
                                "info",
                                "rlm_final_salvaged_from_code",
                                step_index=step_index,
                                content_len=len(pre_extracted_final),
                            )

                    feedback = build_feedback(
                        stdout,
                        error,
                        code,
                        block_count,
                        extra_helpers=["budget_status"],
                    )
                else:
                    # No code block found - could be: (1) FINAL in text, (2) truncated code block
                    if self._final_from_output(model_output):
                        feedback = build_feedback("", None, None, 0)
                    elif self._detect_truncation(model_output):
                        # Truncated output - try to extract FINAL from incomplete block
                        # Look for code-like content after opening ```
                        partial_code_match = re.search(
                            r"```(?:python|repl)?\s*\n(.*)", model_output, re.DOTALL
                        )
                        if partial_code_match:
                            partial_code = partial_code_match.group(1)
                            early_final = self._extract_final_from_code(partial_code)
                            if early_final and len(early_final) >= 10:
                                self._sandbox.answer["content"] = early_final
                                self._sandbox.answer["ready"] = True
                                telemetry.log(
                                    "info",
                                    "rlm_final_salvaged_from_truncated",
                                    step_index=step_index,
                                    content_len=len(early_final),
                                )
                                feedback = build_feedback("", None, None, 0)
                    if feedback is None:
                        feedback = build_feedback(
                            "",
                            "no_code_block",
                            None,
                            0,
                            extra_helpers=["budget_status"],
                        )

                if feedback is not None:
                    telemetry.log(
                        "info",
                        "rlm_feedback",
                        step_index=step_index,
                        violation=feedback.violation,
                        hint=feedback.hint,
                        pattern_warnings=feedback.pattern_warnings,
                        rejection_reasons=feedback.rejection_reasons,
                        error=_trim_text(feedback.error),
                        stdout=_trim_text(feedback.stdout),
                    )

                if self._sandbox.answer.get("ready"):
                    if self._verify_on_final:
                        verification = verify_output(
                            self._task.success_criteria,
                            self._sandbox.answer.get("content") or "",
                            goal_based_trust=True,
                        )
                        if verification.passed:
                            accepted = True
                        else:
                            self._sandbox.answer["ready"] = False
                            feedback.rejection_reasons = verification.reasons
                            telemetry.log(
                                "info",
                                "rlm_final_reject",
                                step_index=step_index,
                                reasons=verification.reasons,
                            )
                    else:
                        accepted = True

                if feedback is not None:
                    budget_notice = None
                    if self._drain_budget_notices:
                        budget_notice = self._drain_budget_notices()
                    prompt = self._advance_prompt(
                        prompt,
                        model_output,
                        feedback,
                        budget_pct,
                        budget_notice=budget_notice,
                    )
                    telemetry.log(
                        "info",
                        "rlm_prompt_next",
                        step_index=step_index,
                        prompt=_trim_text(prompt),
                    )

                steps.append(
                    RLMStep(
                        step_index=step_index,
                        prompt=prompt,
                        model_output=model_output,
                        code=code,
                        stdout=stdout,
                        error=error,
                    )
                )
                if accepted:
                    self._emit(f"rlm_step:{step_index + 1} FINAL accepted")
                    telemetry.log("info", "rlm_final_accept", step_index=step_index)
                    break
                else:
                    self._emit(f"rlm_step:{step_index + 1} completed, continuing...")

        self._emit(f"rlm_complete: {len(steps)} steps executed")
        return self._finalize(steps, errors, start_ts, provider_name)

    def _final_from_output(self, model_output: str) -> bool:
        """Extract FINAL() or FINAL_VAR() from model output (no code block)."""
        match = re.search(r"FINAL\((.*?)\)", model_output, re.DOTALL)
        if match:
            content = match.group(1).strip()
            content = content.strip('"').strip("'")
            self._sandbox.answer["content"] = content
            self._sandbox.answer["ready"] = True
            return True
        match = re.search(r"FINAL_VAR\((.*?)\)", model_output, re.DOTALL)
        if match:
            var_name = match.group(1).strip()
            self._sandbox.answer["content"] = str(self._sandbox.get_global(var_name))
            self._sandbox.answer["ready"] = True
            return True
        return False

    def _detect_truncation(self, model_output: str) -> bool:
        """Detect if LLM output appears truncated.

        Truncation indicators:
        - Unclosed code block (odd number of ```)
        - Partial FINAL( without closing paren

        Used to trigger early FINAL extraction before sandbox execution fails.
        """
        # Unclosed code block: odd count of ``` fences
        if "```" in model_output:
            if model_output.count("```") % 2 != 0:
                return True
        # Partial FINAL( at end without closing paren
        if re.search(r"FINAL\([^)]*$", model_output):
            return True
        return False

    def _extract_final_from_code(self, code: str) -> Optional[str]:
        """Extract FINAL() content from code string before execution.

        Handles truncated code where FINAL() exists but sandbox can't run it.
        Tries quoted string first (most reliable), then any content in parens.

        Returns extracted content or None if no FINAL() found.
        """
        if not code:
            return None
        # Try quoted string: FINAL("content") or FINAL('content')
        match = re.search(r'FINAL\(\s*["\']([^"\']+)', code)
        if match:
            return match.group(1)
        # Try any content: FINAL(some_var) or FINAL(expression)
        match = re.search(r"FINAL\(\s*([^)]+)", code)
        if match:
            content = match.group(1).strip().strip("\"'")
            if content and len(content) >= 5:
                return content
        return None

    def _salvage_answer(self, steps: List[RLMStep]) -> Optional[str]:
        """Recover an answer from truncated execution - never lose work.

        Called when no FINAL() was found but budget was exhausted.
        Tries in order of quality:
        1. Partial FINAL( in last model output (truncated mid-call)
        2. Known variable names in sandbox (summary, report, result, answer)
        3. Last non-empty stdout that looks like a result
        4. Last model output (raw, but better than nothing)

        Returns extracted answer - always returns something if steps exist.
        """
        if not steps:
            return None

        last_output = steps[-1].model_output

        # 1. Try to extract partial FINAL( from last output
        partial_match = re.search(r'FINAL\(\s*["\']?([^"\']+)', last_output, re.DOTALL)
        if partial_match:
            content = partial_match.group(1).strip()
            if len(content) >= 20:
                telemetry.log(
                    "info", "rlm_salvage_partial_final", content_len=len(content)
                )
                return content

        # 2. Check sandbox for common variable names that hold results
        salvage_vars = ["summary", "report", "result", "answer", "output", "response"]
        for var_name in salvage_vars:
            value = self._sandbox.get_global(var_name)
            if value and isinstance(value, str) and len(value) >= 20:
                telemetry.log(
                    "info",
                    "rlm_salvage_variable",
                    var_name=var_name,
                    content_len=len(value),
                )
                return value

        # 3. Use last meaningful stdout as fallback
        for step in reversed(steps):
            stdout = step.stdout
            if stdout and len(stdout) >= 50 and not stdout.startswith("Word count:"):
                if not any(
                    x in stdout.lower() for x in ["error", "traceback", "exception"]
                ):
                    telemetry.log("info", "rlm_salvage_stdout", content_len=len(stdout))
                    return stdout

        # 4. Last resort: return raw model output (never lose the generation)
        # Strip code blocks to get the text content
        text = re.sub(r"```[\s\S]*?```", "", last_output).strip()
        if len(text) >= 10:
            telemetry.log("info", "rlm_salvage_raw_output", content_len=len(text))
            return f"[PARTIAL - budget exhausted before FINAL()]\n{text}"

        # Absolute fallback: return something
        telemetry.log("warning", "rlm_salvage_minimal", content_len=len(last_output))
        return f"[INCOMPLETE - budget exhausted]\n{last_output[:500]}"

    @staticmethod
    def _advance_prompt(
        prompt: str,
        model_output: str,
        feedback: Any,
        budget_pct: Optional[Dict[str, int]] = None,
        budget_notice: Optional[str] = None,
    ) -> str:
        """Build next prompt with feedback and budget status.

        Budget warnings escalate: 50% -> 80% -> 90%+ critical.
        At 90%+, model gets explicit instruction to call FINAL() immediately.
        """
        feedback_text = format_feedback(feedback)

        budget_line = ""
        if budget_pct:
            max_pct = max(budget_pct.values()) if budget_pct else 0
            budget_line = f"{max_pct}% of the token budget has been consumed"
            if max_pct >= 90:
                # Critical threshold: force immediate FINAL() call
                budget_line += (
                    "\n*** CRITICAL: Budget nearly exhausted. "
                    "Call FINAL() NOW with your best answer. "
                    "Do NOT write more code. Your next response will be truncated. ***"
                )
            elif max_pct >= 80:
                budget_line += " - WRAP UP SOON"
            elif max_pct >= 50:
                budget_line += " - be efficient"
            budget_line += "\n"

        notice_text = f"{budget_notice}\n" if budget_notice else ""

        return (
            f"{prompt}\n"
            "---\n"
            f"{budget_line}"
            f"{notice_text}"
            "Model output:\n"
            f"{model_output}\n"
            "---\n"
            f"{feedback_text}\n"
            "---\n"
            "Write another ```repl``` block or call FINAL()/FINAL_VAR().\n"
        )

    def _finalize(
        self,
        steps: List[RLMStep],
        errors: List[str],
        start_ts: float,
        provider_name: str,
    ) -> RLMExecutionReport:
        """Build final report after step loop completes.

        If no FINAL() was called but budget was exhausted, attempts salvage
        to recover partial work from truncated output or sandbox variables.
        """
        elapsed_seconds = time.time() - start_ts
        budget_usage = build_budget_usage(
            self._task.budget,
            self._usage_accumulator,
            elapsed_seconds,
            system_prompt_tokens=self._system_prompt_tokens,
        )
        answer = self._sandbox.answer.get("content")

        # Salvage attempt: if no answer but steps exist, try to recover work
        # Triggers on: budget exhaustion, truncation, max_steps reached, errors
        # Key principle: never lose generation work
        salvaged = False
        if not answer and steps:
            salvaged_answer = self._salvage_answer(steps)
            if salvaged_answer:
                answer = salvaged_answer
                salvaged = True
                telemetry.log(
                    "info",
                    "rlm_answer_salvaged",
                    answer_len=len(answer),
                    budget_exceeded="budget_exceeded" in errors,
                    limits_exceeded=bool(budget_usage.limits_exceeded),
                )

        verification = verify_output(
            self._task.success_criteria,
            answer or "",
            goal_based_trust=True,
        )
        if not verification.passed:
            errors.extend(
                [f"verification_failed:{reason}" for reason in verification.reasons]
            )
        telemetry.log(
            "info",
            "rlm_verification",
            passed=verification.passed,
            reasons=verification.reasons,
            answer=_trim_text(answer),
            salvaged=salvaged,
        )

        success = (
            bool(answer)
            and not errors
            and not budget_usage.limits_exceeded
            and verification.passed
        )
        # Salvaged answers can still succeed if they pass verification
        if salvaged and answer and verification.passed:
            # Remove budget_exceeded error since we recovered
            errors = [e for e in errors if e != "budget_exceeded"]
            success = not errors and not budget_usage.limits_exceeded

        if not answer:
            errors.append("no_answer")
            telemetry.log("info", "rlm_no_answer")

        # Add explicit error message when budget was too low for task complexity
        if "budget_exceeded" in errors or budget_usage.limits_exceeded:
            # Calculate how much budget was used vs available
            budget = self._task.budget
            used_total = budget_usage.total_tokens or 0
            used_output = budget_usage.output_tokens or 0
            max_total = budget.max_total_tokens
            max_output = budget.max_tokens

            # Determine which limit was hit
            limit_details = []
            if max_total and used_total >= max_total:
                limit_details.append(f"total_tokens: used {used_total}/{max_total}")
            if max_output and used_output >= max_output:
                limit_details.append(f"output_tokens: used {used_output}/{max_output}")

            # Check if task completed very few steps (indicates budget too low)
            if len(steps) <= 2 and not success:
                budget_error = (
                    f"budget_insufficient: Task failed after {len(steps)} step(s) due to budget exhaustion. "
                    f"Budget limits hit: {', '.join(limit_details) if limit_details else 'unknown'}. "
                    f"Consider increasing budget for this task complexity."
                )
                if budget_error not in errors:
                    errors.append(budget_error)
                telemetry.log(
                    "warning",
                    "rlm_budget_insufficient",
                    steps_completed=len(steps),
                    used_total=used_total,
                    max_total=max_total,
                    used_output=used_output,
                    max_output=max_output,
                )

        # Persist context if it grew during execution
        if self._context_path:
            try:
                from enzu.tools.context import ctx_save, ctx_stats

                after = ctx_stats()
                if context_grew(self._context_before, after):
                    Path(self._context_path).parent.mkdir(parents=True, exist_ok=True)
                    ctx_save(self._context_path)
                    telemetry.log(
                        "info",
                        "context_saved",
                        path=str(self._context_path),
                        stats=after,
                    )
            except Exception:
                pass

        # Determine typed outcome
        # Note: budget_exceeded error may have been removed if answer was salvaged
        if success:
            outcome = Outcome.SUCCESS
        elif "budget_exceeded" in errors or bool(budget_usage.limits_exceeded):
            outcome = Outcome.BUDGET_EXCEEDED
        elif "timeout" in " ".join(errors).lower():
            outcome = Outcome.TIMEOUT
        elif "no_answer" in errors:
            outcome = Outcome.VERIFICATION_FAILED
        elif any("provider" in e.lower() for e in errors):
            outcome = Outcome.PROVIDER_ERROR
        elif any("tool" in e.lower() or "sandbox" in e.lower() for e in errors):
            outcome = Outcome.TOOL_ERROR
        else:
            outcome = Outcome.PROVIDER_ERROR

        # Partial if answer was salvaged (budget exhausted before FINAL)
        partial = salvaged and bool(answer)

        telemetry.log(
            "info",
            "rlm_run_complete",
            success=success,
            outcome=outcome.value,
            partial=partial,
            errors=errors,
            budget_used=budget_usage.limits_exceeded,
            answer=_trim_text(answer),
        )
        return RLMExecutionReport(
            success=success,
            outcome=outcome,
            partial=partial,
            task_id=self._task.task_id,
            provider=provider_name,
            model=self._task.model,
            answer=answer,
            steps=steps,
            budget_usage=budget_usage,
            errors=errors,
        )
