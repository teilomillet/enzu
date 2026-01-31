"""
Enzu client wrapper for HTTP context.

Wraps the Enzu client to run in async context and integrate with request tracking.
Uses TaskQueue for high-concurrency (10K+) deployments with auto-scaling workers.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

from enzu.client import Enzu
from enzu.session import Session, SessionBudgetExceeded
from enzu.models import ExecutionReport, RLMExecutionReport
from enzu.queue import TaskQueue
from enzu.server.config import get_settings
from enzu.server.exceptions import ModelError, SessionBudgetExceededError


@dataclass
class RunResult:
    """Result from an enzu run operation."""

    answer: str
    model: str
    total_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cost_usd: Optional[float] = None


# Thread pool for running sync Enzu operations (session tasks only)
_executor: Optional[ThreadPoolExecutor] = None

# TaskQueue for high-concurrency standalone tasks
_task_queue: Optional[TaskQueue] = None
_task_queue_lock = asyncio.Lock()


def get_executor() -> ThreadPoolExecutor:
    """Get thread pool executor for sync operations."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=50, thread_name_prefix="enzu-session-"
        )
    return _executor


async def get_task_queue() -> TaskQueue:
    """
    Get or create the global TaskQueue for standalone tasks.

    Auto-scales workers based on queue depth:
    - Min: 10 workers (baseline capacity)
    - Max: 200 workers (10K concurrent with queue buffering)
    - Queue: 10000 tasks (backpressure when full)
    """
    global _task_queue
    async with _task_queue_lock:
        if _task_queue is None:
            settings = get_settings()
            _task_queue = TaskQueue(
                provider=settings.default_provider or "openrouter",
                model=settings.default_model or "openai/gpt-4o-mini",
                min_workers=10,
                max_workers=200,
                queue_size=10000,
                scale_up_threshold=2.0,
                scale_down_threshold=0.5,
            )
            await _task_queue.start()
        return _task_queue


async def shutdown_task_queue() -> None:
    """Shutdown the task queue. Call on app shutdown."""
    global _task_queue
    async with _task_queue_lock:
        if _task_queue is not None:
            await _task_queue.stop()
            _task_queue = None


def shutdown_executor() -> None:
    """Shutdown the executor. Call on app shutdown."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False)
        _executor = None


def _extract_usage(report: ExecutionReport | RLMExecutionReport) -> dict:
    """Extract usage info from a report."""
    usage = report.budget_usage
    return {
        "total_tokens": usage.total_tokens,
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "cost_usd": usage.cost_usd,
    }


def _extract_answer(report: ExecutionReport | RLMExecutionReport) -> str:
    """Extract answer from a report."""
    if isinstance(report, RLMExecutionReport):
        return report.answer or ""
    return report.output_text or ""


def _is_malformed_response(
    answer: str, expected_contains: Optional[List[str]] = None
) -> bool:
    """
    Detect if a response is malformed/truncated/invalid.

    Signs of malformed response:
    - Empty or very short (< 10 chars for non-code responses)
    - Contains partial FINAL( without proper closure
    - Contains raw instruction text mixed with code
    - Just the word "answer" (common RLM failure mode)
    - Expected content not found (if specified)
    """
    if not answer:
        return True

    answer_stripped = answer.strip()

    # Very short responses are likely failures
    if len(answer_stripped) < 10:
        return True

    # Common RLM failure patterns - model outputs meta-text instead of answer
    failure_words = ["answer", "response", "output", "result"]
    if answer_stripped.lower() in failure_words:
        return True

    # Check for failure words with punctuation/dots (e.g., "answer...", "answer.")
    answer_base = answer_stripped.lower().rstrip(".!?:;,")
    if answer_base in failure_words:
        return True

    # Responses starting with problematic fragments
    malformed_starts = [
        ")",  # Incomplete parenthesis
        "(",  # Incomplete parenthesis
        "answer",  # Meta-word leaked
        ") with",  # Instruction fragment
        ") now",  # Instruction fragment
        "```",  # Code block leaked
        "Since the",  # LLM explaining
        "The output",  # LLM explaining
        "Goal achieved",  # LLM meta-commentary
        "The goal",  # LLM meta-commentary
        "I'll output",  # LLM reasoning
        "Here is",  # LLM preamble
        "Here's",  # LLM preamble
        "This is straightforward",  # LLM meta-commentary
        "This is the",  # LLM preamble
        "Let me",  # LLM reasoning
        "I need to",  # LLM reasoning
        "I just need",  # LLM reasoning
        "The requested",  # LLM preamble
        "As requested",  # LLM preamble
    ]
    for prefix in malformed_starts:
        if answer_stripped.lower().startswith(prefix.lower()):
            return True

    # Check for partial FINAL artifacts that leaked into answer (short responses)
    short_malformed_patterns = [
        "FINAL(",  # Incomplete FINAL call
        "```repl",  # Code block leaked
        "with this exact code",  # Instruction text leaked
        "[PARTIAL",  # Salvage marker
        "let me check",  # Model thinking out loud
        "Let me first",  # Model thinking out loud
        "I should",  # Model thinking out loud
    ]

    for pattern in short_malformed_patterns:
        if pattern.lower() in answer.lower() and len(answer) < 200:
            return True

    # Patterns that indicate malformed response regardless of length
    # These are clear signs the LLM is outputting reasoning/code instead of answer
    always_malformed_patterns = [
        "FINAL_VAR",  # Code artifact leaked
        ")/FINAL",  # Partial FINAL call
        "The context contains",  # LLM explaining its reasoning
        "the answer I need",  # LLM meta-commentary
        "first element of the context",  # LLM parsing explanation
        "I need to output",  # LLM reasoning
        "I will output",  # LLM reasoning
        "Just output the code",  # Instruction text leaked
        "Context chunks:",  # LLM analyzing context
        "Context length:",  # LLM analyzing context
        "chunk length:",  # LLM analyzing context
        "chars\nChunk",  # LLM parsing chunks
        "answer) when",  # Instruction fragment leaked
        ") with the answer",  # Instruction fragment leaked
        "when I believe",  # Instruction fragment leaked
        "I believe the goal",  # LLM meta-reasoning
        "the goal is achieved",  # Instruction text leaked
        "achieved the goal",  # LLM meta-reasoning
        "previous response",  # LLM referencing history
        "already achieved",  # LLM meta-reasoning
    ]

    for pattern in always_malformed_patterns:
        if pattern.lower() in answer.lower():
            return True

    # If expected content specified, check for it
    if expected_contains:
        for expected in expected_contains:
            if expected not in answer:
                return True

    return False


async def run_task(
    task: str,
    *,
    data: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    cost: Optional[float] = None,
    tokens: Optional[int] = None,
    seconds: Optional[float] = None,
    temperature: Optional[float] = None,
    max_steps: Optional[int] = None,
    contains: Optional[List[str]] = None,
    matches: Optional[List[str]] = None,
    min_words: Optional[int] = None,
    goal: Optional[str] = None,
) -> RunResult:
    """
    Run a single-shot task asynchronously.

    Args:
        task: The task/prompt to execute.
        data: Optional context data.
        model: Model identifier (uses default if not specified).
        provider: Provider name (uses default if not specified).
        cost: Max cost in USD.
        tokens: Max output tokens.
        seconds: Max execution time.
        temperature: Temperature (0.0-2.0).
        max_steps: Max reasoning steps (RLM mode).
        contains: Output must contain these substrings.
        matches: Output must match these regexes.
        min_words: Minimum word count.
        goal: Goal for model self-verification.

    Returns:
        RunResult with answer and usage info.

    Raises:
        ModelError: If the underlying LLM provider returns an error.
    """
    settings = get_settings()
    model = model or settings.default_model
    provider = provider or settings.default_provider

    def _run() -> RunResult:
        try:
            client = Enzu(model, provider=provider)
            report = client.run(
                task,
                data=data,
                cost=cost,
                tokens=tokens,
                seconds=seconds,
                temperature=temperature,
                max_steps=max_steps,
                contains=contains,
                matches=matches,
                min_words=min_words,
                goal=goal,
                return_report=True,
            )

            # _extract_answer and _extract_usage expect ExecutionReport or RLMExecutionReport
            # When return_report=True, client.run always returns a report, never str
            if isinstance(report, (ExecutionReport, RLMExecutionReport)):
                answer = _extract_answer(report)
                usage = _extract_usage(report)
            else:
                # Fallback: should not happen when return_report=True
                answer = str(report) if report else ""
                usage = {}

            return RunResult(
                answer=answer,
                model=client.model,
                **usage,
            )
        except Exception as e:
            raise ModelError(str(e))

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(get_executor(), _run)


async def run_session_task(
    session: Session,
    task: str,
    *,
    data: Optional[str] = None,
    cost: Optional[float] = None,
    tokens: Optional[int] = None,
    seconds: Optional[float] = None,
    temperature: Optional[float] = None,
    max_steps: Optional[int] = None,
    contains: Optional[List[str]] = None,
    matches: Optional[List[str]] = None,
    min_words: Optional[int] = None,
    goal: Optional[str] = None,
    max_retries: int = 3,
) -> RunResult:
    """
    Run a task within an existing session.

    The session maintains conversation history across calls.
    Automatically retries on malformed responses (up to max_retries).

    Args:
        session: The Session object.
        task: The task/prompt to execute.
        data: Optional context data for this turn.
        cost: Max cost for this request.
        tokens: Max output tokens.
        seconds: Max execution time.
        temperature: Temperature (0.0-2.0).
        max_steps: Max reasoning steps.
        contains: Output must contain these substrings.
        matches: Output must match these regexes.
        min_words: Minimum word count.
        goal: Goal for model self-verification.
        max_retries: Max retry attempts for malformed responses.

    Returns:
        RunResult with answer and usage info.

    Raises:
        SessionBudgetExceededError: If session budget is exceeded.
        ModelError: If the underlying LLM provider returns an error.
    """

    def _run_single() -> RunResult:
        """Execute a single run attempt."""
        report = session.run(
            task,
            data=data,
            cost=cost,
            tokens=tokens,
            seconds=seconds,
            temperature=temperature,
            max_steps=max_steps,
            contains=contains,
            matches=matches,
            min_words=min_words,
            goal=goal,
            return_report=True,
        )

        # _extract_answer and _extract_usage expect ExecutionReport or RLMExecutionReport
        # When return_report=True, session.run always returns a report, never str
        if isinstance(report, (ExecutionReport, RLMExecutionReport)):
            answer = _extract_answer(report)
            usage = _extract_usage(report)
        else:
            # Fallback: should not happen when return_report=True
            answer = str(report) if report else ""
            usage = {}

        return RunResult(
            answer=answer,
            model=session.model,
            **usage,
        )

    def _run_with_retry() -> RunResult:
        """Run with retry logic for malformed responses."""
        last_error = None
        last_result = None

        for attempt in range(max_retries + 1):
            try:
                result = _run_single()

                # Check for malformed response
                if _is_malformed_response(result.answer):
                    if attempt < max_retries:
                        # Remove the malformed exchange from history before retry
                        if session.exchanges:
                            session.exchanges.pop()
                        continue
                    # Last attempt - return what we got
                    last_result = result
                else:
                    return result

            except SessionBudgetExceeded as e:
                raise SessionBudgetExceededError(str(e))
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    continue
                raise ModelError(str(e))

        # Return last result even if malformed
        if last_result:
            return last_result
        if last_error:
            raise ModelError(str(last_error))
        raise ModelError("Unexpected error in run_session_task")

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(get_executor(), _run_with_retry)


def create_session(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    max_cost_usd: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Session:
    """
    Create a new Session object.

    Args:
        model: Model identifier (uses default if not specified).
        provider: Provider name (uses default if not specified).
        max_cost_usd: Session cost cap.
        max_tokens: Session token cap.

    Returns:
        A new Session object.
    """
    settings = get_settings()
    model = model or settings.default_model
    provider = provider or settings.default_provider

    # If no model specified and no default, auto-detect
    if model is None:
        client = Enzu()  # Auto-detects
        model = client.model
        provider = client.provider

    return Session(
        model=model,
        provider=provider,
        max_cost_usd=max_cost_usd,
        max_tokens=max_tokens,
    )
