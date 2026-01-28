"""
Budget control for cost and token limits.

Provides pre-call checks and post-call enforcement to guarantee
budget limits are never exceeded.

Architecture:
- BudgetController: Tracks cumulative usage, enforces limits
- Token counting via tiktoken for precise pre-call estimation
- Hard stop after limit exceeded (no more calls allowed)
- Audit log for every transaction
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# tiktoken for accurate token counting (same tokenizer as OpenAI)
try:
    import tiktoken
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False


@dataclass
class BudgetEvent:
    """Single budget transaction for audit trail."""
    timestamp: str
    event_type: str  # "call", "exceeded", "blocked"
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    cumulative_cost_usd: Optional[float] = None
    cumulative_tokens: Optional[int] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "cumulative_cost_usd": self.cumulative_cost_usd,
            "cumulative_tokens": self.cumulative_tokens,
            "details": self.details,
        }


class BudgetExceeded(Exception):
    """Raised when budget limit is exceeded. No more calls allowed."""

    def __init__(
        self,
        limit_type: str,
        limit_value: float,
        current_value: float,
        message: Optional[str] = None,
    ):
        self.limit_type = limit_type
        self.limit_value = limit_value
        self.current_value = current_value
        msg = message or f"Budget exceeded: {limit_type} limit {limit_value}, current {current_value}"
        super().__init__(msg)


class BudgetController:
    """
    Controls budget for a session or task.

    See enzu.terminology for full documentation on token terminology.

    Guarantees:
        - Pre-call: Blocks if estimated output tokens would exceed limit
        - Post-call: Records actual usage, blocks future calls if exceeded
        - Thread-safe: All operations are atomic
        - Auditable: Every transaction logged

    Token limits:
        - max_tokens: Cumulative OUTPUT tokens (primary billing metric)
        - max_total_tokens: Cumulative total (input + output) - advanced
        - max_input_tokens: Cumulative input tokens - advanced

    Usage:
        controller = BudgetController(max_cost_usd=1.00, max_tokens=10000)

        # Before call: check if allowed (max_output_tokens is per-call estimate)
        controller.pre_call_check(input_tokens=500, max_output_tokens=1000)

        # After call: record usage
        controller.record_usage(input_tokens=500, output_tokens=200, cost_usd=0.01)

        # Check remaining output budget
        print(controller.remaining_tokens)  # 9800
    """

    def __init__(
        self,
        max_cost_usd: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
        max_input_tokens: Optional[int] = None,
    ):
        self._lock = threading.Lock()

        # Limits (None = unlimited)
        # max_tokens = cumulative output tokens (primary limit)
        self.max_cost_usd = max_cost_usd
        self.max_tokens = max_tokens
        self.max_total_tokens = max_total_tokens
        self.max_input_tokens = max_input_tokens

        # Cumulative usage
        self._total_cost_usd: float = 0.0
        self._total_tokens: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

        # State
        self._exceeded: bool = False
        self._exceeded_reason: Optional[str] = None

        # Audit log
        self._events: List[BudgetEvent] = []

    @property
    def is_exceeded(self) -> bool:
        """True if any limit has been exceeded. No more calls allowed."""
        with self._lock:
            return self._exceeded

    @property
    def total_cost_usd(self) -> float:
        with self._lock:
            return self._total_cost_usd

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return self._total_tokens

    @property
    def remaining_cost(self) -> Optional[float]:
        """Remaining cost budget. None if unlimited."""
        with self._lock:
            if self.max_cost_usd is None:
                return None
            return max(0.0, self.max_cost_usd - self._total_cost_usd)

    @property
    def remaining_tokens(self) -> Optional[int]:
        """Remaining output token budget. None if unlimited."""
        with self._lock:
            if self.max_tokens is None:
                return None
            return max(0, self.max_tokens - self._total_output_tokens)

    @property
    def remaining_total_tokens(self) -> Optional[int]:
        """Remaining total (input+output) token budget. None if unlimited."""
        with self._lock:
            if self.max_total_tokens is None:
                return None
            return max(0, self.max_total_tokens - self._total_tokens)

    @property
    def audit_log(self) -> List[Dict[str, Any]]:
        """Full audit trail of all budget events."""
        with self._lock:
            return [e.to_dict() for e in self._events]

    def pre_call_check(
        self,
        input_tokens: int,
        max_output_tokens: Optional[int] = None,
    ) -> None:
        """
        Check if call is allowed before making it.

        Raises BudgetExceeded if:
        - Budget already exceeded from previous calls
        - Estimated tokens would exceed limit

        Args:
            input_tokens: Number of input tokens (count with count_tokens())
            max_output_tokens: Maximum output tokens for this call
        """
        with self._lock:
            # Hard stop if already exceeded
            if self._exceeded:
                self._log_event("blocked", details={"reason": self._exceeded_reason})
                raise BudgetExceeded(
                    limit_type=self._exceeded_reason or "unknown",
                    limit_value=0,
                    current_value=0,
                    message=f"Budget already exceeded: {self._exceeded_reason}. No more calls allowed.",
                )

            # Check input tokens
            if self.max_input_tokens is not None:
                projected = self._total_input_tokens + input_tokens
                if projected > self.max_input_tokens:
                    self._log_event("blocked", input_tokens=input_tokens,
                                   details={"reason": "input_tokens_exceeded"})
                    raise BudgetExceeded(
                        limit_type="max_input_tokens",
                        limit_value=self.max_input_tokens,
                        current_value=projected,
                    )

            # Check output tokens (max_tokens = cumulative output tokens limit)
            if self.max_tokens is not None:
                output_estimate = max_output_tokens or 0
                projected_output = self._total_output_tokens + output_estimate
                if projected_output > self.max_tokens:
                    self._log_event("blocked", input_tokens=input_tokens,
                                   details={"reason": "output_tokens_exceeded",
                                           "projected": projected_output})
                    raise BudgetExceeded(
                        limit_type="max_tokens",
                        limit_value=self.max_tokens,
                        current_value=projected_output,
                    )

            # Check total tokens (advanced: input + output)
            if self.max_total_tokens is not None:
                output_estimate = max_output_tokens or 0
                projected_total = self._total_tokens + input_tokens + output_estimate
                if projected_total > self.max_total_tokens:
                    self._log_event("blocked", input_tokens=input_tokens,
                                   details={"reason": "total_tokens_exceeded",
                                           "projected": projected_total})
                    raise BudgetExceeded(
                        limit_type="max_total_tokens",
                        limit_value=self.max_total_tokens,
                        current_value=projected_total,
                    )

    def record_usage(
        self,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
    ) -> None:
        """
        Record usage after a call completes.

        If this causes any limit to be exceeded, marks budget as exceeded
        and blocks all future calls.

        Negative values are clamped to 0 for defensive robustness.
        """
        with self._lock:
            # Defensive: clamp negative values to 0 (should not happen but be robust)
            if input_tokens is not None:
                input_tokens = max(0, input_tokens)
                self._total_input_tokens += input_tokens
                self._total_tokens += input_tokens
            if output_tokens is not None:
                output_tokens = max(0, output_tokens)
                self._total_output_tokens += output_tokens
                self._total_tokens += output_tokens
            if cost_usd is not None:
                cost_usd = max(0.0, cost_usd)
                self._total_cost_usd += cost_usd
                # Round to prevent floating-point accumulation errors
                self._total_cost_usd = round(self._total_cost_usd, 10)

            # Log the call
            self._log_event(
                "call",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
            )

            # Check if any limit now exceeded
            self._check_limits_after_call()

    def _check_limits_after_call(self) -> None:
        """Check all limits and mark exceeded if any breached."""
        if self.max_cost_usd is not None and self._total_cost_usd >= self.max_cost_usd:
            self._mark_exceeded("max_cost_usd")
        # max_tokens enforces cumulative output tokens
        if self.max_tokens is not None and self._total_output_tokens >= self.max_tokens:
            self._mark_exceeded("max_tokens")
        # max_total_tokens enforces cumulative total (input + output)
        if self.max_total_tokens is not None and self._total_tokens >= self.max_total_tokens:
            self._mark_exceeded("max_total_tokens")
        if self.max_input_tokens is not None and self._total_input_tokens >= self.max_input_tokens:
            self._mark_exceeded("max_input_tokens")

    def _mark_exceeded(self, reason: str) -> None:
        """Mark budget as exceeded. No more calls allowed after this."""
        if not self._exceeded:
            self._exceeded = True
            self._exceeded_reason = reason
            self._log_event("exceeded", details={"reason": reason})

    def _log_event(
        self,
        event_type: str,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add event to audit log."""
        self._events.append(BudgetEvent(
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            event_type=event_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            cumulative_cost_usd=self._total_cost_usd,
            cumulative_tokens=self._total_tokens,
            details=details,
        ))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize controller state for persistence or reporting."""
        with self._lock:
            return {
                "max_cost_usd": self.max_cost_usd,
                "max_tokens": self.max_tokens,
                "max_total_tokens": self.max_total_tokens,
                "max_input_tokens": self.max_input_tokens,
                "total_cost_usd": self._total_cost_usd,
                "total_tokens": self._total_tokens,
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "is_exceeded": self._exceeded,
                "exceeded_reason": self._exceeded_reason,
                "event_count": len(self._events),
            }


# Token counting utilities

# Default encoding for models (cl100k_base covers GPT-4, GPT-3.5-turbo)
_DEFAULT_ENCODING = "cl100k_base"

# Cache encodings to avoid repeated initialization
_encoding_cache: Dict[str, Any] = {}


def count_tokens(text: str, model: Optional[str] = None) -> int:
    """
    Count tokens in text using tiktoken.

    Uses the same tokenizer as OpenAI models for accurate counts.
    Falls back to approximate count (chars/4) if tiktoken unavailable.

    Args:
        text: Text to tokenize
        model: Model name for encoding selection (optional)

    Returns:
        Token count
    """
    if not _HAS_TIKTOKEN:
        # Fallback: ~4 chars per token is a reasonable approximation
        return len(text) // 4

    encoding_name = _get_encoding_for_model(model)

    if encoding_name not in _encoding_cache:
        try:
            _encoding_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
        except Exception:
            # Model-specific encoding failed, use default
            if _DEFAULT_ENCODING not in _encoding_cache:
                _encoding_cache[_DEFAULT_ENCODING] = tiktoken.get_encoding(_DEFAULT_ENCODING)
            encoding_name = _DEFAULT_ENCODING

    encoding = _encoding_cache[encoding_name]
    return len(encoding.encode(text))


def _supports_exact_count(model: Optional[str]) -> bool:
    if not model:
        return False
    model_lower = model.lower()
    if model_lower.startswith("openai/") or "/gpt-" in model_lower:
        return True
    return any(key in model_lower for key in ("gpt-4", "gpt-3.5", "gpt-4o", "o1", "o3", "o4"))


def count_tokens_exact(text: str, model: Optional[str] = None) -> Optional[int]:
    """
    Count tokens only when we can do so accurately.

    Returns None when tiktoken is unavailable or the model tokenizer is unknown.
    """
    if not _HAS_TIKTOKEN:
        return None
    if not _supports_exact_count(model):
        return None
    try:
        encoding_name = _get_encoding_for_model(model)
        if encoding_name not in _encoding_cache:
            _encoding_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
        encoding = _encoding_cache[encoding_name]
        return len(encoding.encode(text))
    except Exception:
        return None


def estimate_tokens_conservative(text: str) -> int:
    """
    Estimate tokens using a conservative character-based approximation.
    
    Uses ~3 characters per token (conservative - typical is ~4 chars/token).
    This ensures we don't underestimate and exceed budgets.
    
    Use when exact token counting is unavailable.
    """
    if not text:
        return 0
    # Conservative estimate: 3 chars per token (overestimates slightly)
    return max(1, len(text) // 3)


def _get_encoding_for_model(model: Optional[str]) -> str:
    """Get tiktoken encoding name for a model."""
    if model is None:
        return _DEFAULT_ENCODING

    model_lower = model.lower()

    # GPT-4o and O-series use o200k_base (check before gpt-4 to avoid substring match)
    if "gpt-4o" in model_lower or "o1" in model_lower or "o3" in model_lower:
        return "o200k_base"

    # GPT-4 and GPT-3.5-turbo use cl100k_base
    if "gpt-4" in model_lower or "gpt-3.5" in model_lower:
        return "cl100k_base"

    # Claude, Llama, etc. - use cl100k_base as approximation
    # (tiktoken doesn't have native encodings for these)
    return _DEFAULT_ENCODING


def count_message_tokens(messages: List[Dict[str, str]], model: Optional[str] = None) -> int:
    """
    Count tokens in a chat message list.

    Accounts for message structure overhead (role, separators).

    Args:
        messages: List of {"role": ..., "content": ...} dicts
        model: Model name for encoding selection

    Returns:
        Total token count including overhead
    """
    total = 0
    for msg in messages:
        # ~4 tokens per message for structure (role, separators)
        total += 4
        content = msg.get("content", "")
        if content:
            total += count_tokens(content, model)
    # Final separator
    total += 2
    return total
