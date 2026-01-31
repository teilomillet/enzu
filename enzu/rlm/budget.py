"""
RLM budget tracking: token pools, usage accumulation, limit enforcement.

Extracted from engine.py. Manages token budgets across multi-step RLM execution
where each step consumes tokens and multiple subcalls may run concurrently.

Thread-safe: _TokenBudgetPool uses locks for concurrent llm_batch calls.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from enzu.models import Budget, BudgetUsage
from enzu.usage import check_budget_limits, normalize_usage

CRITICAL_THRESHOLD_METADATA_KEY = "budget_critical_threshold"


class TokenBudgetPool:
    """
    Thread-safe token budget pool for multi-step execution.

    Tracks max_total_tokens across concurrent operations:
    - reserve(): claim tokens before LLM call
    - commit(): record actual usage after call completes
    - release(): return unused reservation on error

    Used by llm_batch() where multiple LLM calls run in parallel.
    """

    def __init__(self, max_total_tokens: Optional[int]) -> None:
        self._max_total_tokens = max_total_tokens
        self._used_tokens = 0
        self._reserved_tokens = 0
        self._lock = threading.Lock()
        self._exact_usage = True

    def reserve(
        self,
        prompt_tokens: Optional[int],
        requested_output_tokens: int,
    ) -> tuple[int, int]:
        """
        Reserve tokens for an LLM call.

        Returns (output_cap, reserved_total):
        - output_cap: max output tokens allowed for this call
        - reserved_total: total tokens reserved (prompt + output)

        If tokenizer unavailable (prompt_tokens=None), uses conservative estimate.
        """
        if self._max_total_tokens is None:
            return requested_output_tokens, 0
        with self._lock:
            remaining = (
                self._max_total_tokens - self._used_tokens - self._reserved_tokens
            )
            if remaining <= 0:
                return 0, 0
            prompt = prompt_tokens or 0
            if prompt_tokens is None:
                self._exact_usage = False
                available_for_output = remaining
            else:
                available_for_output = remaining - prompt
            if available_for_output <= 0:
                return 0, 0
            output_cap = min(requested_output_tokens, available_for_output)
            reserved = prompt + output_cap
            self._reserved_tokens += reserved
            return output_cap, reserved

    def commit(self, reserved: int, actual_total_tokens: Optional[int]) -> None:
        """
        Commit usage after LLM call completes.

        Releases reservation and records actual token usage.
        If actual_total_tokens unavailable, uses reservation as estimate.

        Defensive: negative values are clamped to 0.
        """
        if self._max_total_tokens is None:
            return
        with self._lock:
            # Defensive: ensure reserved is non-negative
            reserved = max(0, reserved)
            self._reserved_tokens = max(0, self._reserved_tokens - reserved)
            if isinstance(actual_total_tokens, int):
                # Defensive: clamp negative actual usage to 0
                self._used_tokens += max(0, actual_total_tokens)
            else:
                self._exact_usage = False
                self._used_tokens += reserved

    def release(self, reserved: int) -> None:
        """Release reservation on error (call did not complete)."""
        if self._max_total_tokens is None:
            return
        with self._lock:
            self._reserved_tokens = max(0, self._reserved_tokens - reserved)

    def is_exhausted(self) -> bool:
        """Check if budget is fully consumed."""
        if self._max_total_tokens is None:
            return False
        with self._lock:
            return self._used_tokens + self._reserved_tokens >= self._max_total_tokens

    def reserve_total(self, total_tokens: int) -> int:
        """
        Reserve a fixed total token budget for an entire operation (e.g., subcall).

        Unlike reserve() which estimates based on prompt+output, this reserves
        a specific total amount for operations that may have multiple internal steps.

        Returns the amount actually reserved (may be less than requested if budget low).
        Returns 0 if no budget available.
        """
        if self._max_total_tokens is None:
            return total_tokens  # No limit, grant full request
        with self._lock:
            remaining = (
                self._max_total_tokens - self._used_tokens - self._reserved_tokens
            )
            if remaining <= 0:
                return 0
            granted = min(total_tokens, remaining)
            self._reserved_tokens += granted
            return granted

    def snapshot(self) -> Dict[str, Any]:
        """Return current budget state for debugging/logging."""
        with self._lock:
            max_tokens = self._max_total_tokens
            used = self._used_tokens
            reserved = self._reserved_tokens
            remaining = (
                None if max_tokens is None else max(0, max_tokens - used - reserved)
            )
            return {
                "max_total_tokens": max_tokens,
                "used_tokens": used,
                "reserved_tokens": reserved,
                "remaining_tokens": remaining,
                "exact_usage": self._exact_usage,
            }


class BudgetTracker:
    """
    Accumulated usage tracker for budget limit checks.

    Tracks cumulative output_tokens, total_tokens, cost_usd across steps.
    Used to determine when to stop execution due to budget exhaustion.
    """

    def __init__(self, budget: Budget) -> None:
        self._budget = budget
        self._output_tokens = 0
        self._total_tokens = 0
        self._cost_usd = 0.0

    def consume(self, usage: Dict[str, Any]) -> None:
        """Add usage from an LLM call to the running total.

        Defensive: negative values are clamped to 0.
        """
        normalized = normalize_usage(usage)
        output_tokens = normalized.get("output_tokens")
        total_tokens = normalized.get("total_tokens")
        cost_usd = normalized.get("cost_usd")
        # Defensive: clamp negative values to 0
        if isinstance(output_tokens, int):
            self._output_tokens += max(0, output_tokens)
        if isinstance(total_tokens, int):
            self._total_tokens += max(0, total_tokens)
        if isinstance(cost_usd, (int, float)):
            self._cost_usd += max(0.0, float(cost_usd))
            # Round to prevent floating-point accumulation errors
            self._cost_usd = round(self._cost_usd, 10)

    def is_exhausted(self) -> bool:
        """Check if any budget limit is reached."""
        if self._budget.max_tokens and self._output_tokens >= self._budget.max_tokens:
            return True
        if (
            self._budget.max_total_tokens
            and self._total_tokens >= self._budget.max_total_tokens
        ):
            return True
        if self._budget.max_cost_usd and self._cost_usd >= self._budget.max_cost_usd:
            return True
        return False

    def percentage_used(self) -> Dict[str, int]:
        """Return percentage used for each budget dimension (for UI/logging)."""
        pct: Dict[str, int] = {}
        if self._budget.max_tokens:
            pct["output_tokens"] = round(
                100 * self._output_tokens / self._budget.max_tokens
            )
        if self._budget.max_total_tokens:
            pct["total_tokens"] = round(
                100 * self._total_tokens / self._budget.max_total_tokens
            )
        if self._budget.max_cost_usd:
            pct["cost_usd"] = round(100 * self._cost_usd / self._budget.max_cost_usd)
        return pct

    def remaining_output_tokens(self) -> Optional[int]:
        if self._budget.max_tokens is None:
            return None
        return max(0, self._budget.max_tokens - self._output_tokens)

    def remaining_total_tokens(self) -> Optional[int]:
        if self._budget.max_total_tokens is None:
            return None
        return max(0, self._budget.max_total_tokens - self._total_tokens)

    def adaptive_max_output_tokens(
        self,
        base_max: int,
        *,
        remaining_output_tokens: Optional[int] = None,
        remaining_total_tokens: Optional[int] = None,
    ) -> int:
        """Compute max_output_tokens for next step, reduced as budget depletes.

        Scales down output token limit based on remaining budget to prevent
        truncation, then clamps to remaining budget if available.

        Emits telemetry when adaptive scaling triggers.

        Args:
            base_max: The default max_output_tokens for the task
            remaining_output_tokens: Optional remaining output token budget
            remaining_total_tokens: Optional remaining total token budget

        Returns:
            Adjusted max_output_tokens for the next LLM call (never exceeds base_max)
        """
        from enzu import telemetry

        pct = self.percentage_used()
        if not pct:
            max_allowed = base_max
        else:
            max_pct = max(pct.values())
            if max_pct >= 80:
                max_allowed = base_max // 2
                telemetry.log(
                    "info",
                    "budget_adaptive_scaling",
                    threshold_pct=80,
                    scale_factor=0.5,
                    base_tokens=base_max,
                    scaled_tokens=max_allowed,
                    budget_used_pct=max_pct,
                )
            elif max_pct >= 50:
                max_allowed = (base_max * 3) // 4
                telemetry.log(
                    "info",
                    "budget_adaptive_scaling",
                    threshold_pct=50,
                    scale_factor=0.75,
                    base_tokens=base_max,
                    scaled_tokens=max_allowed,
                    budget_used_pct=max_pct,
                )
            else:
                max_allowed = base_max

        if isinstance(remaining_output_tokens, int):
            max_allowed = min(max_allowed, remaining_output_tokens)
        if isinstance(remaining_total_tokens, int):
            max_allowed = min(max_allowed, remaining_total_tokens)
        return max_allowed


def accumulate_usage(
    accumulator: Dict[str, Any],
    usage: Dict[str, Any],
    *,
    prompt: Optional[str] = None,
    output_text: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Normalize and accumulate usage from an LLM call.

    Returns normalized usage dict. Mutates accumulator in place.
    """
    normalized = normalize_usage(
        usage,
        input_text=prompt,
        output_text=output_text,
        model=model,
    )
    input_tokens = normalized.get("input_tokens")
    output_tokens = normalized.get("output_tokens")
    total_tokens = normalized.get("total_tokens")
    cost_usd = normalized.get("cost_usd")

    if isinstance(input_tokens, int):
        accumulator["input_tokens"] = accumulator.get("input_tokens", 0) + input_tokens
    if isinstance(output_tokens, int):
        accumulator["output_tokens"] = (
            accumulator.get("output_tokens", 0) + output_tokens
        )
    if isinstance(total_tokens, int):
        accumulator["total_tokens"] = accumulator.get("total_tokens", 0) + total_tokens
    if isinstance(cost_usd, (int, float)):
        new_cost = accumulator.get("cost_usd", 0.0) + float(cost_usd)
        # Round to prevent floating-point accumulation errors
        accumulator["cost_usd"] = round(new_cost, 10)

    return normalized


def parse_budget_critical_threshold(
    metadata: Dict[str, Any],
    *,
    default: int = 90,
) -> int:
    value = metadata.get(CRITICAL_THRESHOLD_METADATA_KEY, default)
    try:
        threshold = int(value)
    except (TypeError, ValueError):
        return default
    if threshold <= 0:
        return default
    return threshold


def build_budget_usage(
    budget: Budget,
    usage: Dict[str, Any],
    elapsed_seconds: float,
    system_prompt_tokens: Optional[int] = None,
) -> BudgetUsage:
    """Build BudgetUsage from accumulated usage dict.

    Args:
        budget: The Budget model with limits
        usage: Accumulated usage dict from LLM calls
        elapsed_seconds: Total elapsed time
        system_prompt_tokens: Estimated tokens for the RLM system prompt.
            This is overhead cost users should be aware of.
    """
    normalized = normalize_usage(usage)
    input_tokens = normalized.get("input_tokens")
    output_tokens = normalized.get("output_tokens")
    total_tokens = normalized.get("total_tokens")
    cost_usd = normalized.get("cost_usd")

    # Convert float tokens to int for check_budget_limits
    output_tokens_int = (
        int(output_tokens) if isinstance(output_tokens, (int, float)) else None
    )
    total_tokens_int = (
        int(total_tokens) if isinstance(total_tokens, (int, float)) else None
    )

    limits_exceeded = check_budget_limits(
        budget, elapsed_seconds, output_tokens_int, total_tokens_int, cost_usd
    )
    return BudgetUsage(
        elapsed_seconds=elapsed_seconds,
        input_tokens=input_tokens if isinstance(input_tokens, int) else None,
        output_tokens=output_tokens_int,
        total_tokens=total_tokens_int,
        cost_usd=cost_usd,
        limits_exceeded=limits_exceeded,
        system_prompt_tokens=system_prompt_tokens,
    )
