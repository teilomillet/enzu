"""
LLM execution: direct queries, recursive subcalls, and parallel batch.

Extracted from engine.py. Encapsulates LLM call logic with:
- Provider fallback on failure
- Token budget reservation/commit
- Usage accumulation
- Telemetry spans

Thread-safe for concurrent llm_batch execution.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from enzu.models import TaskSpec, RLMExecutionReport
from enzu.providers.base import BaseProvider
from enzu.budget import count_tokens_exact, estimate_tokens_conservative
from enzu.contract import DEFAULT_MAX_OUTPUT_TOKENS
from enzu import telemetry

from enzu.rlm.budget import TokenBudgetPool, BudgetTracker, accumulate_usage
from enzu.rlm.budget import parse_budget_critical_threshold


def _trim_text(text: Optional[str], limit: int = 2000) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit]


class LLMExecutorConfig:
    """Configuration for LLM executor."""

    def __init__(
        self,
        *,
        task: TaskSpec,
        provider: BaseProvider,
        fallback_providers: Optional[List[BaseProvider]] = None,
        usage_accumulator: Dict[str, Any],
        token_pool: TokenBudgetPool,
        budget_tracker: BudgetTracker,
        emit: Callable[[str], None],
        errors: List[str],
        add_budget_notice: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.task = task
        self.provider = provider
        self.fallback_providers = fallback_providers or []
        self.usage_accumulator = usage_accumulator
        self.token_pool = token_pool
        self.budget_tracker = budget_tracker
        self.emit = emit
        self.errors = errors
        self.add_budget_notice = add_budget_notice


class LLMExecutor:
    """
    Executes LLM queries with budget management and provider fallback.
    
    Extracted from RLMEngine.run() to enable testing and reuse.
    """

    # Max threads per llm_batch call. Global limiter controls total concurrency.
    MAX_BATCH_WORKERS = 20

    def __init__(self, config: LLMExecutorConfig) -> None:
        self._task = config.task
        self._provider = config.provider
        self._fallback_providers = config.fallback_providers
        self._usage_accumulator = config.usage_accumulator
        self._token_pool = config.token_pool
        self._budget_tracker = config.budget_tracker
        self._emit = config.emit
        self._errors = config.errors
        self._add_budget_notice = config.add_budget_notice
        self._lock = threading.Lock()
        self._critical_threshold = parse_budget_critical_threshold(self._task.metadata)

        # Subcall support (set by RLMEngine when recursive mode enabled)
        # Signature: (prompt, max_output_tokens, max_total_tokens) -> RLMExecutionReport
        self._subcall_runner: Optional[Callable[[str, Optional[int], Optional[int]], RLMExecutionReport]] = None

    def set_subcall_runner(
        self,
        runner: Callable[[str, Optional[int], Optional[int]], RLMExecutionReport],
    ) -> None:
        """Set callback for recursive subcalls. Called by RLMEngine."""
        self._subcall_runner = runner

    def _requested_output_tokens(self) -> int:
        """Calculate max output tokens for current request.
        
        Uses adaptive scaling based on budget consumption:
        - At 90%+ usage: caps at 256 tokens (enough for FINAL call)
        - At 80%+ usage: reduces to 50% of base
        - At 50%+ usage: reduces to 75% of base
        
        This prevents truncation by reducing output size as budget depletes.
        """
        base_output_tokens = (
            self._task.max_output_tokens
            or self._task.budget.max_tokens
            or DEFAULT_MAX_OUTPUT_TOKENS
        )
        if self._task.budget.max_total_tokens is not None:
            base_output_tokens = min(base_output_tokens, self._task.budget.max_total_tokens)
        
        # Apply adaptive scaling based on budget usage
        remaining_output = self._budget_tracker.remaining_output_tokens()
        remaining_total = self._token_pool.snapshot().get("remaining_tokens")
        return self._budget_tracker.adaptive_max_output_tokens(
            base_output_tokens,
            remaining_output_tokens=remaining_output,
            remaining_total_tokens=remaining_total,
        )

    def _budget_status(self) -> tuple[Optional[int], Optional[int], Optional[int]]:
        pct = self._budget_tracker.percentage_used()
        max_pct = max(pct.values()) if pct else None
        remaining_total = self._token_pool.snapshot().get("remaining_tokens")
        remaining_output = self._budget_tracker.remaining_output_tokens()
        return max_pct, remaining_output, remaining_total

    def _critical_budget_reached(self) -> bool:
        threshold = self._critical_threshold
        max_pct, remaining_output, remaining_total = self._budget_status()
        if threshold <= 100:
            return max_pct is not None and max_pct >= threshold
        candidates = [
            value for value in (remaining_total, remaining_output) if isinstance(value, int)
        ]
        if candidates:
            return min(candidates) <= threshold
        return max_pct is not None and max_pct >= 90

    def _critical_budget_message(self) -> str:
        max_pct, remaining_output, remaining_total = self._budget_status()
        if max_pct is not None:
            pct_text = f"{max_pct}% of the token budget has been consumed"
        else:
            pct_text = "token budget consumption unknown"
        remaining_parts = []
        if remaining_total is not None:
            remaining_parts.append(f"remaining_total_tokens={remaining_total}")
        if remaining_output is not None:
            remaining_parts.append(f"remaining_output_tokens={remaining_output}")
        remaining_text = ", ".join(remaining_parts) if remaining_parts else "remaining tokens unknown"
        threshold = self._critical_threshold
        if threshold <= 100:
            threshold_text = f"critical threshold={threshold}%"
        else:
            threshold_text = f"critical threshold={threshold} tokens remaining"
        return (
            "Sub-LLM calls disabled: critical budget threshold reached. "
            f"{pct_text}; {remaining_text}; {threshold_text}. "
            "Call FINAL() now."
        )

    def _reserve_budget(
        self,
        prompt: str,
        requested_output_tokens: int,
    ) -> tuple[int, int]:
        """
        Reserve tokens before LLM call.
        
        Returns (output_cap, reserved).
        Raises RuntimeError if budget exhausted or prompt too large.
        """
        prompt_tokens = count_tokens_exact(prompt, self._task.model)
        
        # If exact count unavailable and we have a total token budget,
        # use conservative estimate to prevent overspend for non-mock models
        # (Mock models are for testing and return controlled usage)
        model_lower = (self._task.model or "").lower()
        is_mock = "mock" in model_lower or "test" in model_lower
        if prompt_tokens is None and self._task.budget.max_total_tokens is not None and not is_mock:
            prompt_tokens = estimate_tokens_conservative(prompt)
            telemetry.log(
                "info",
                "using_conservative_token_estimate",
                estimated_tokens=prompt_tokens,
                prompt_len=len(prompt),
            )
        
        # HARD ENFORCEMENT: If the prompt alone exceeds remaining budget, fail immediately
        if prompt_tokens is not None and self._task.budget.max_total_tokens is not None:
            pool_snapshot = self._token_pool.snapshot()
            remaining = pool_snapshot.get("remaining_tokens")
            if remaining is not None and prompt_tokens >= remaining:
                telemetry.log(
                    "warning",
                    "budget_exceeded_prompt_too_large",
                    prompt_tokens=prompt_tokens,
                    remaining_tokens=remaining,
                )
                raise RuntimeError(
                    f"budget_exhausted_preflight: prompt requires ~{prompt_tokens} tokens "
                    f"but only {remaining} tokens remaining"
                )
        
        output_cap, reserved = self._token_pool.reserve(prompt_tokens, requested_output_tokens)
        if output_cap <= 0:
            raise RuntimeError("budget_exhausted_preflight")
        if output_cap < requested_output_tokens:
            reduction = 1 - (output_cap / max(requested_output_tokens, 1))
            reduction_pct = round(100 * reduction)
            if reduction_pct >= 50:
                if self._add_budget_notice:
                    self._add_budget_notice(
                        f"Budget clamp: output capped to {output_cap} tokens "
                        f"(requested {requested_output_tokens}, -{reduction_pct}%). "
                        "Consider simplifying."
                    )
                telemetry.log(
                    "info",
                    "budget_output_clamped",
                    requested_output_tokens=requested_output_tokens,
                    clamped_output_tokens=output_cap,
                )
            self._emit(f"budget_output_clamped:{output_cap}")
        return output_cap, reserved

    def direct_query(self, prompt: str, *, span_name: str = "rlm.llm_query") -> str:
        """
        Execute direct LLM query with provider fallback.
        
        Used by RLMEngine for step queries and as fallback for sandbox_llm_query.
        """
        if self._budget_tracker.is_exhausted():
            raise RuntimeError("budget_exhausted")
        requested_output_tokens = self._requested_output_tokens()
        output_cap, reserved = self._reserve_budget(prompt, requested_output_tokens)

        all_providers = [self._provider] + self._fallback_providers
        last_exception = None
        for current_provider in all_providers:
            try:
                with telemetry.span(span_name, prompt_len=len(prompt), provider=current_provider.name):
                    result = current_provider.stream(
                        self._task.model_copy(update={
                            "input_text": prompt,
                            "max_output_tokens": output_cap,
                        }),
                        on_progress=lambda event: self._emit(f"llm_stream:{event.message}"),
                    )
                with self._lock:
                    normalized_usage = accumulate_usage(
                        self._usage_accumulator,
                        result.usage,
                        prompt=prompt,
                        output_text=result.output_text,
                        model=self._task.model,
                    )
                    self._budget_tracker.consume(normalized_usage)
                    if self._task.budget.max_total_tokens is not None and normalized_usage.get("total_tokens") is None:
                        self._emit("usage_missing_total_tokens")
                    self._token_pool.commit(reserved, normalized_usage.get("total_tokens"))
                    
                    # Track if budget exhausted after this call
                    budget_exhausted_after = self._budget_tracker.is_exhausted()
                    if budget_exhausted_after:
                        telemetry.log(
                            "warning",
                            "budget_exceeded_after_call",
                            reserved=reserved,
                            actual_total=normalized_usage.get("total_tokens"),
                        )
                        
                telemetry.log(
                    "info",
                    "llm_query_done",
                    prompt=_trim_text(prompt),
                    output=_trim_text(result.output_text),
                    output_len=len(result.output_text),
                    provider=current_provider.name,
                )
                return result.output_text
            except Exception as exc:
                last_exception = exc
                if current_provider == all_providers[-1]:
                    self._token_pool.release(reserved)
                    raise
                self._errors.append(f"{current_provider.name}: {exc}")
                telemetry.log(
                    "warning",
                    "llm_query_fallback",
                    provider=current_provider.name,
                    error=str(exc),
                )
                continue
        if last_exception:
            raise last_exception
        raise RuntimeError("No providers available")

    def sandbox_query(self, prompt: str) -> str:
        """
        LLM query for sandbox use. Supports recursive subcalls if configured.
        
        Falls back to direct_query if subcall runner not set.
        
        For recursive subcalls, reserves the ENTIRE remaining budget for this subcall
        to ensure hard enforcement - the subcall cannot exceed what was reserved.
        """
        if self._critical_budget_reached():
            raise RuntimeError(self._critical_budget_message())
        if self._budget_tracker.is_exhausted():
            raise RuntimeError("budget_exhausted")
        if self._subcall_runner is not None:
            # Get remaining budget and reserve it ALL for this subcall
            # This ensures hard enforcement: subcall cannot exceed reserved amount
            pool_snapshot = self._token_pool.snapshot()
            remaining = pool_snapshot.get("remaining_tokens")
            
            if remaining is None:
                # No token limit set - use unlimited
                max_total_tokens = None
                reserved_total = 0
            elif remaining <= 0:
                raise RuntimeError("budget_exhausted_preflight")
            else:
                # Reserve all remaining tokens for this subcall
                # For sequential llm_query, we can give it all remaining budget
                reserved_total = self._token_pool.reserve_total(remaining)
                if reserved_total <= 0:
                    raise RuntimeError("budget_exhausted_preflight")
                max_total_tokens = reserved_total
            
            # Calculate adaptive output cap based on reserved budget
            requested_output_tokens = self._requested_output_tokens()
            output_cap = min(requested_output_tokens, max_total_tokens or requested_output_tokens)
            
            try:
                with telemetry.span("rlm.llm_query_recursive", prompt_len=len(prompt)):
                    sub_report = self._subcall_runner(prompt, output_cap, max_total_tokens)
            except Exception:
                if reserved_total > 0:
                    self._token_pool.release(reserved_total)
                raise
            
            sub_usage = {
                "input_tokens": sub_report.budget_usage.input_tokens,
                "output_tokens": sub_report.budget_usage.output_tokens,
                "total_tokens": sub_report.budget_usage.total_tokens,
                "cost_usd": sub_report.budget_usage.cost_usd,
            }
            
            # Verify subcall didn't exceed its reserved budget (belt and suspenders)
            actual_total = sub_usage.get("total_tokens")
            if actual_total is not None and reserved_total > 0 and actual_total > reserved_total:
                telemetry.log(
                    "warning",
                    "budget_violation_subcall_overuse",
                    reserved=reserved_total,
                    actual=actual_total,
                )
                # Clamp to reserved to maintain accounting integrity
                sub_usage["total_tokens"] = reserved_total
            
            with self._lock:
                normalized_usage = accumulate_usage(self._usage_accumulator, sub_usage)
                self._budget_tracker.consume(normalized_usage)
                if self._task.budget.max_total_tokens is not None and normalized_usage.get("total_tokens") is None:
                    self._emit("usage_missing_total_tokens")
                # Commit using reserved_total (not the old prompt+output reservation)
                self._token_pool.commit(reserved_total, normalized_usage.get("total_tokens"))
            
            telemetry.log(
                "info",
                "llm_query_recursive_done",
                prompt=_trim_text(prompt),
                output=_trim_text(sub_report.answer),
                output_len=len(sub_report.answer or ""),
                reserved_total=reserved_total,
            )
            return sub_report.answer or ""
        return self.direct_query(prompt, span_name="rlm.llm_query")

    def batch_query(self, prompts: List[str]) -> List[str]:
        """
        Execute multiple LLM queries in parallel.
        
        Uses thread pool for concurrent execution with global rate limiting
        via enzu.isolation.concurrency.
        """
        from enzu.isolation.concurrency import get_global_limiter

        if self._critical_budget_reached():
            raise RuntimeError(self._critical_budget_message())
        if self._budget_tracker.is_exhausted():
            raise RuntimeError("budget_exhausted")
        if not prompts:
            return []

        limiter = get_global_limiter()

        if self._subcall_runner is not None:
            return self._batch_subcalls(prompts, limiter)
        return self._batch_direct(prompts, limiter)

    def _batch_subcalls(self, prompts: List[str], limiter: Any) -> List[str]:
        """Batch execution via recursive subcalls.
        
        Splits remaining budget evenly across all subcalls upfront to ensure
        parallel subcalls cannot oversubscribe the total budget.
        """
        # Pre-calculate budget split BEFORE spawning threads
        pool_snapshot = self._token_pool.snapshot()
        remaining = pool_snapshot.get("remaining_tokens")
        
        if remaining is None:
            # No token limit - each subcall gets unlimited
            per_subcall_budget = None
        elif remaining <= 0:
            raise RuntimeError("budget_exhausted_preflight")
        else:
            # Split budget evenly across all prompts
            per_subcall_budget = max(1, remaining // len(prompts))
            telemetry.log(
                "info",
                "llm_batch_budget_split",
                total_remaining=remaining,
                num_prompts=len(prompts),
                per_subcall=per_subcall_budget,
            )
        
        def query_one(prompt: str, index: int) -> Tuple[int, str]:
            with limiter.acquire():
                if self._budget_tracker.is_exhausted():
                    raise RuntimeError("budget_exhausted")

                if self._subcall_runner is None:
                    raise RuntimeError("Subcall runner not configured")
                
                # Reserve this subcall's share of the budget
                if per_subcall_budget is None:
                    reserved_total = 0
                    max_total_tokens = None
                else:
                    reserved_total = self._token_pool.reserve_total(per_subcall_budget)
                    if reserved_total <= 0:
                        raise RuntimeError("budget_exhausted_preflight")
                    max_total_tokens = reserved_total
                
                # Calculate output cap within the subcall's budget
                requested_output_tokens = self._requested_output_tokens()
                output_cap = min(requested_output_tokens, max_total_tokens or requested_output_tokens)
                
                with telemetry.span(
                    "rlm.llm_batch.subcall",
                    prompt_len=len(prompt),
                    batch_index=index,
                    reserved_total=reserved_total,
                ):
                    try:
                        sub_report = self._subcall_runner(prompt, output_cap, max_total_tokens)
                    except Exception:
                        if reserved_total > 0:
                            self._token_pool.release(reserved_total)
                        raise
                
                sub_usage = {
                    "input_tokens": sub_report.budget_usage.input_tokens,
                    "output_tokens": sub_report.budget_usage.output_tokens,
                    "total_tokens": sub_report.budget_usage.total_tokens,
                    "cost_usd": sub_report.budget_usage.cost_usd,
                }
                
                # Verify subcall didn't exceed its reserved budget
                actual_total = sub_usage.get("total_tokens")
                if actual_total is not None and reserved_total > 0 and actual_total > reserved_total:
                    telemetry.log(
                        "warning",
                        "budget_violation_batch_subcall_overuse",
                        batch_index=index,
                        reserved=reserved_total,
                        actual=actual_total,
                    )
                    sub_usage["total_tokens"] = reserved_total
                
                with self._lock:
                    normalized_usage = accumulate_usage(self._usage_accumulator, sub_usage)
                    self._budget_tracker.consume(normalized_usage)
                    if self._task.budget.max_total_tokens is not None and normalized_usage.get("total_tokens") is None:
                        self._emit("usage_missing_total_tokens")
                    self._token_pool.commit(reserved_total, normalized_usage.get("total_tokens"))
                
                telemetry.log(
                    "info",
                    "llm_batch_subcall_done",
                    batch_index=index,
                    prompt=_trim_text(prompt),
                    output=_trim_text(sub_report.answer),
                    output_len=len(sub_report.answer or ""),
                    reserved_total=reserved_total,
                )
                return (index, sub_report.answer or "")

        with telemetry.span("rlm.llm_batch", batch_size=len(prompts)):
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_BATCH_WORKERS) as executor:
                futures = [executor.submit(query_one, p, i) for i, p in enumerate(prompts)]
                results = [f.result() for f in futures]
        return [output for _, output in sorted(results)]

    def _batch_direct(self, prompts: List[str], limiter: Any) -> List[str]:
        """Batch execution via direct provider calls."""
        def query_one(prompt: str, index: int) -> Tuple[int, str]:
            with limiter.acquire():
                requested_output_tokens = self._requested_output_tokens()
                output_cap, reserved = self._reserve_budget(prompt, requested_output_tokens)
                all_providers = [self._provider] + self._fallback_providers
                for current_provider in all_providers:
                    try:
                        with telemetry.span(
                            "rlm.llm_batch.query",
                            prompt_len=len(prompt),
                            batch_index=index,
                            provider=current_provider.name,
                        ):
                            result = current_provider.stream(
                                self._task.model_copy(update={
                                    "input_text": prompt,
                                    "max_output_tokens": output_cap,
                                }),
                                on_progress=lambda event: self._emit(f"llm_stream:batch[{index}]:{event.message}"),
                            )
                        with self._lock:
                            normalized_usage = accumulate_usage(
                                self._usage_accumulator,
                                result.usage,
                                prompt=prompt,
                                output_text=result.output_text,
                                model=self._task.model,
                            )
                            self._budget_tracker.consume(normalized_usage)
                            if self._task.budget.max_total_tokens is not None and normalized_usage.get("total_tokens") is None:
                                self._emit("usage_missing_total_tokens")
                            self._token_pool.commit(reserved, normalized_usage.get("total_tokens"))
                        telemetry.log(
                            "info",
                            "llm_batch_query_done",
                            batch_index=index,
                            prompt=_trim_text(prompt),
                            output=_trim_text(result.output_text),
                            output_len=len(result.output_text),
                            provider=current_provider.name,
                        )
                        return (index, result.output_text)
                    except Exception as exc:
                        if current_provider == all_providers[-1]:
                            self._token_pool.release(reserved)
                            raise
                        telemetry.log(
                            "warning",
                            "llm_batch_query_fallback",
                            batch_index=index,
                            provider=current_provider.name,
                            error=str(exc),
                        )
                        continue
                raise RuntimeError("No providers available")

        async def run_batch() -> List[str]:
            with telemetry.span("rlm.llm_batch", batch_size=len(prompts)):
                loop = asyncio.get_event_loop()
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_BATCH_WORKERS)
                try:
                    futures = [loop.run_in_executor(executor, query_one, p, i) for i, p in enumerate(prompts)]
                    results = await asyncio.gather(*futures)
                    return [output for _, output in sorted(results)]
                finally:
                    executor.shutdown(wait=False)

        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, run_batch())
                return future.result()  # type: ignore[return-value]
        except RuntimeError:
            return asyncio.run(run_batch())
