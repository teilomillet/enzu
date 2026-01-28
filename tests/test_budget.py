"""
Comprehensive tests for the budget system.

Tests cover:
1. BudgetController - core budget enforcement
2. Token counting utilities
3. TokenBudgetPool - RLM token reservation system
4. BudgetTracker - multi-dimensional budget tracking
5. Budget model validation
6. Engine preflight checks
7. Edge cases and robustness
8. Thread safety
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from enzu.budget import (
    BudgetController,
    BudgetEvent,
    BudgetExceeded,
    count_tokens,
    count_tokens_exact,
    count_message_tokens,
    _get_encoding_for_model,
    _HAS_TIKTOKEN,
)
from enzu.models import Budget, BudgetUsage, Limits, TaskSpec, SuccessCriteria
from enzu.rlm.budget import TokenBudgetPool, BudgetTracker
from enzu.engine import Engine
from enzu.rlm import RLMEngine
from tests.providers.mock_provider import MockProvider


# =============================================================================
# BudgetController Tests
# =============================================================================

class TestBudgetControllerInit:
    """Test BudgetController initialization."""

    def test_init_with_cost_limit(self) -> None:
        controller = BudgetController(max_cost_usd=10.0)
        assert controller.max_cost_usd == 10.0
        assert controller.max_tokens is None
        assert controller.is_exceeded is False
        assert controller.total_cost_usd == 0.0

    def test_init_with_token_limit(self) -> None:
        controller = BudgetController(max_tokens=1000)
        assert controller.max_tokens == 1000
        assert controller.max_cost_usd is None
        assert controller.total_tokens == 0

    def test_init_with_all_limits(self) -> None:
        controller = BudgetController(
            max_cost_usd=5.0,
            max_tokens=10000,
            max_total_tokens=20000,
            max_input_tokens=5000,
        )
        assert controller.max_cost_usd == 5.0
        assert controller.max_tokens == 10000  # output tokens limit
        assert controller.max_total_tokens == 20000
        assert controller.max_input_tokens == 5000

    def test_init_no_limits(self) -> None:
        """Controller can be created without limits (unlimited)."""
        controller = BudgetController()
        assert controller.max_cost_usd is None
        assert controller.max_tokens is None
        assert controller.remaining_cost is None
        assert controller.remaining_tokens is None


class TestBudgetControllerPreCallCheck:
    """Test pre-call validation."""

    def test_pre_call_passes_when_under_limit(self) -> None:
        controller = BudgetController(max_tokens=1000)
        # Should not raise
        controller.pre_call_check(input_tokens=100, max_output_tokens=200)

    def test_pre_call_blocks_when_output_projected_exceeds(self) -> None:
        """max_tokens enforces cumulative output tokens."""
        controller = BudgetController(max_tokens=1000)
        with pytest.raises(BudgetExceeded) as exc_info:
            # 1500 output would exceed 1000 limit
            controller.pre_call_check(input_tokens=800, max_output_tokens=1500)
        assert exc_info.value.limit_type == "max_tokens"
        assert exc_info.value.limit_value == 1000
        assert exc_info.value.current_value == 1500  # output only

    def test_pre_call_blocks_when_total_projected_exceeds(self) -> None:
        """max_total_tokens enforces input + output."""
        controller = BudgetController(max_total_tokens=1000)
        with pytest.raises(BudgetExceeded) as exc_info:
            controller.pre_call_check(input_tokens=800, max_output_tokens=500)
        assert exc_info.value.limit_type == "max_total_tokens"
        assert exc_info.value.limit_value == 1000
        assert exc_info.value.current_value == 1300  # 800 + 500

    def test_pre_call_blocks_when_already_exceeded(self) -> None:
        controller = BudgetController(max_tokens=100)
        # 110 output tokens exceeds max_tokens=100 (output limit)
        controller.record_usage(input_tokens=50, output_tokens=110)
        assert controller.is_exceeded is True

        with pytest.raises(BudgetExceeded) as exc_info:
            controller.pre_call_check(input_tokens=10)
        assert "already exceeded" in str(exc_info.value)

    def test_pre_call_blocks_input_tokens_limit(self) -> None:
        controller = BudgetController(max_input_tokens=500)
        with pytest.raises(BudgetExceeded) as exc_info:
            controller.pre_call_check(input_tokens=600)
        assert exc_info.value.limit_type == "max_input_tokens"

    def test_pre_call_no_output_estimate(self) -> None:
        """Pre-call without output estimate only checks input."""
        controller = BudgetController(max_tokens=1000)
        # 800 input, no output estimate = 800 projected, under 1000
        controller.pre_call_check(input_tokens=800)  # Should pass


class TestBudgetControllerRecordUsage:
    """Test usage recording and limit checking."""

    def test_record_usage_updates_totals(self) -> None:
        controller = BudgetController(max_tokens=1000)
        controller.record_usage(input_tokens=100, output_tokens=50, cost_usd=0.01)

        assert controller.total_tokens == 150
        assert controller.total_cost_usd == 0.01

    def test_record_usage_marks_exceeded_on_cost(self) -> None:
        controller = BudgetController(max_cost_usd=1.0)
        controller.record_usage(cost_usd=1.5)

        assert controller.is_exceeded is True
        assert controller.total_cost_usd == 1.5

    def test_record_usage_marks_exceeded_on_tokens(self) -> None:
        """max_tokens enforces cumulative output tokens."""
        controller = BudgetController(max_tokens=100)
        controller.record_usage(input_tokens=60, output_tokens=110)

        assert controller.is_exceeded is True
        assert controller.total_tokens == 170  # total is still input + output

    def test_record_usage_accumulates(self) -> None:
        controller = BudgetController(max_tokens=1000)
        controller.record_usage(input_tokens=100, output_tokens=50)
        controller.record_usage(input_tokens=200, output_tokens=100)

        assert controller.total_tokens == 450

    def test_record_usage_partial_fields(self) -> None:
        """Can record only some usage fields."""
        controller = BudgetController(max_cost_usd=10.0)
        controller.record_usage(cost_usd=1.0)  # Only cost
        controller.record_usage(input_tokens=100)  # Only input

        assert controller.total_cost_usd == 1.0
        assert controller.total_tokens == 100


class TestBudgetControllerRemaining:
    """Test remaining budget calculation."""

    def test_remaining_cost(self) -> None:
        controller = BudgetController(max_cost_usd=10.0)
        controller.record_usage(cost_usd=3.5)
        assert controller.remaining_cost == 6.5

    def test_remaining_tokens(self) -> None:
        """remaining_tokens is based on output tokens (max_tokens limit)."""
        controller = BudgetController(max_tokens=1000)
        controller.record_usage(input_tokens=300, output_tokens=200)
        assert controller.remaining_tokens == 800  # 1000 - 200 output

    def test_remaining_never_negative(self) -> None:
        controller = BudgetController(max_tokens=100)
        controller.record_usage(output_tokens=150)
        assert controller.remaining_tokens == 0  # max(0, -50)

    def test_remaining_unlimited(self) -> None:
        controller = BudgetController()  # No limits
        assert controller.remaining_cost is None
        assert controller.remaining_tokens is None


class TestBudgetControllerAuditLog:
    """Test audit log functionality."""

    def test_audit_log_records_calls(self) -> None:
        controller = BudgetController(max_tokens=1000)
        controller.record_usage(input_tokens=100, output_tokens=50)
        controller.record_usage(input_tokens=200, output_tokens=100)

        log = controller.audit_log
        assert len(log) == 2
        assert log[0]["event_type"] == "call"
        assert log[0]["input_tokens"] == 100
        assert log[1]["input_tokens"] == 200

    def test_audit_log_records_exceeded(self) -> None:
        controller = BudgetController(max_tokens=100)
        controller.record_usage(output_tokens=150)  # exceed output limit

        log = controller.audit_log
        # Should have call + exceeded events
        event_types = [e["event_type"] for e in log]
        assert "call" in event_types
        assert "exceeded" in event_types

    def test_audit_log_records_blocked(self) -> None:
        controller = BudgetController(max_tokens=100)
        controller.record_usage(output_tokens=150)  # Exceed output limit

        try:
            controller.pre_call_check(input_tokens=10)
        except BudgetExceeded:
            pass

        log = controller.audit_log
        event_types = [e["event_type"] for e in log]
        assert "blocked" in event_types


class TestBudgetControllerSerialization:
    """Test state serialization."""

    def test_to_dict(self) -> None:
        controller = BudgetController(max_cost_usd=10.0, max_tokens=1000)
        controller.record_usage(input_tokens=100, cost_usd=0.5)

        state = controller.to_dict()
        assert state["max_cost_usd"] == 10.0
        assert state["max_tokens"] == 1000
        assert state["total_cost_usd"] == 0.5
        assert state["total_tokens"] == 100
        assert state["is_exceeded"] is False


# =============================================================================
# Token Counting Tests
# =============================================================================

class TestTokenCounting:
    """Test token counting utilities."""

    def test_count_tokens_basic(self) -> None:
        text = "Hello, world!"
        tokens = count_tokens(text)
        assert tokens > 0
        # With tiktoken, "Hello, world!" is typically 4 tokens
        if _HAS_TIKTOKEN:
            assert tokens < 10  # Sanity check

    def test_count_tokens_empty_string(self) -> None:
        assert count_tokens("") == 0

    def test_count_tokens_long_text(self) -> None:
        text = "word " * 1000  # ~1000 words
        tokens = count_tokens(text)
        # Should be roughly 1000-2000 tokens
        assert 500 < tokens < 3000

    def test_count_tokens_with_model(self) -> None:
        text = "Hello, world!"
        tokens_gpt4 = count_tokens(text, model="gpt-4")
        tokens_gpt35 = count_tokens(text, model="gpt-3.5-turbo")
        # Both use cl100k_base, should be same
        assert tokens_gpt4 == tokens_gpt35

    @pytest.mark.skipif(not _HAS_TIKTOKEN, reason="tiktoken not installed")
    def test_count_tokens_exact_with_supported_model(self) -> None:
        text = "Hello, world!"
        tokens = count_tokens_exact(text, model="gpt-4")
        assert tokens is not None
        assert tokens > 0

    def test_count_tokens_exact_unsupported_model(self) -> None:
        text = "Hello, world!"
        tokens = count_tokens_exact(text, model="llama-70b")
        # Unsupported model returns None
        assert tokens is None

    def test_count_tokens_exact_no_model(self) -> None:
        text = "Hello, world!"
        tokens = count_tokens_exact(text, model=None)
        assert tokens is None

    def test_count_message_tokens(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        tokens = count_message_tokens(messages)
        assert tokens > 0
        # ~4 tokens overhead per message + content + final separator
        assert tokens > 10

    def test_count_message_tokens_empty(self) -> None:
        tokens = count_message_tokens([])
        # Just final separator
        assert tokens == 2


class TestEncodingSelection:
    """Test model to encoding mapping."""

    def test_gpt4_uses_cl100k(self) -> None:
        assert _get_encoding_for_model("gpt-4") == "cl100k_base"
        assert _get_encoding_for_model("gpt-4-turbo") == "cl100k_base"

    def test_gpt35_uses_cl100k(self) -> None:
        assert _get_encoding_for_model("gpt-3.5-turbo") == "cl100k_base"

    def test_gpt4o_uses_o200k(self) -> None:
        # GPT-4o uses o200k_base encoding
        assert _get_encoding_for_model("gpt-4o") == "o200k_base"
        assert _get_encoding_for_model("gpt-4o-mini") == "o200k_base"

    def test_o1_uses_o200k(self) -> None:
        # o1 models should use o200k_base
        assert _get_encoding_for_model("o1") == "o200k_base"
        assert _get_encoding_for_model("o1-preview") == "o200k_base"

    def test_unknown_model_uses_default(self) -> None:
        assert _get_encoding_for_model("llama-70b") == "cl100k_base"
        assert _get_encoding_for_model("claude-3") == "cl100k_base"

    def test_none_model_uses_default(self) -> None:
        assert _get_encoding_for_model(None) == "cl100k_base"


# =============================================================================
# TokenBudgetPool Tests
# =============================================================================

class TestTokenBudgetPool:
    """Test RLM token reservation system."""

    def test_init_with_limit(self) -> None:
        pool = TokenBudgetPool(max_total_tokens=1000)
        snapshot = pool.snapshot()
        assert snapshot["max_total_tokens"] == 1000
        assert snapshot["used_tokens"] == 0
        assert snapshot["reserved_tokens"] == 0
        assert snapshot["remaining_tokens"] == 1000

    def test_init_unlimited(self) -> None:
        pool = TokenBudgetPool(max_total_tokens=None)
        snapshot = pool.snapshot()
        assert snapshot["max_total_tokens"] is None
        assert snapshot["remaining_tokens"] is None

    def test_reserve_basic(self) -> None:
        pool = TokenBudgetPool(max_total_tokens=1000)
        output_cap, reserved = pool.reserve(prompt_tokens=100, requested_output_tokens=200)

        assert output_cap == 200  # Full request granted
        assert reserved == 300  # prompt + output

        snapshot = pool.snapshot()
        assert snapshot["reserved_tokens"] == 300
        assert snapshot["remaining_tokens"] == 700

    def test_reserve_capped_by_remaining(self) -> None:
        pool = TokenBudgetPool(max_total_tokens=500)
        output_cap, reserved = pool.reserve(prompt_tokens=400, requested_output_tokens=200)

        # Only 100 tokens remaining after prompt
        assert output_cap == 100
        assert reserved == 500  # 400 + 100

    def test_reserve_no_room_for_output(self) -> None:
        pool = TokenBudgetPool(max_total_tokens=500)
        output_cap, reserved = pool.reserve(prompt_tokens=500, requested_output_tokens=200)

        assert output_cap == 0
        assert reserved == 0

    def test_reserve_unlimited(self) -> None:
        pool = TokenBudgetPool(max_total_tokens=None)
        output_cap, reserved = pool.reserve(prompt_tokens=1000, requested_output_tokens=500)

        assert output_cap == 500  # Full request
        assert reserved == 0  # No reservation needed when unlimited

    def test_commit_updates_used(self) -> None:
        pool = TokenBudgetPool(max_total_tokens=1000)
        _, reserved = pool.reserve(prompt_tokens=100, requested_output_tokens=200)
        pool.commit(reserved, actual_total_tokens=250)

        snapshot = pool.snapshot()
        assert snapshot["used_tokens"] == 250
        assert snapshot["reserved_tokens"] == 0

    def test_commit_fallback_to_reserved(self) -> None:
        pool = TokenBudgetPool(max_total_tokens=1000)
        _, reserved = pool.reserve(prompt_tokens=100, requested_output_tokens=200)
        pool.commit(reserved, actual_total_tokens=None)  # No actual count

        snapshot = pool.snapshot()
        assert snapshot["used_tokens"] == reserved  # Uses reserved as fallback
        assert snapshot["exact_usage"] is False

    def test_release_without_commit(self) -> None:
        pool = TokenBudgetPool(max_total_tokens=1000)
        _, reserved = pool.reserve(prompt_tokens=100, requested_output_tokens=200)
        pool.release(reserved)

        snapshot = pool.snapshot()
        assert snapshot["reserved_tokens"] == 0
        assert snapshot["used_tokens"] == 0
        assert snapshot["remaining_tokens"] == 1000

    def test_is_exhausted(self) -> None:
        pool = TokenBudgetPool(max_total_tokens=100)
        assert pool.is_exhausted() is False

        _, reserved = pool.reserve(prompt_tokens=50, requested_output_tokens=60)
        # Reserved 100, at limit
        assert pool.is_exhausted() is True

    def test_is_exhausted_unlimited(self) -> None:
        pool = TokenBudgetPool(max_total_tokens=None)
        assert pool.is_exhausted() is False
        # Stays false even after operations
        pool.reserve(prompt_tokens=10000, requested_output_tokens=10000)
        assert pool.is_exhausted() is False

    def test_multiple_reserves(self) -> None:
        pool = TokenBudgetPool(max_total_tokens=1000)
        _, r1 = pool.reserve(prompt_tokens=100, requested_output_tokens=100)
        _, r2 = pool.reserve(prompt_tokens=100, requested_output_tokens=100)

        snapshot = pool.snapshot()
        assert snapshot["reserved_tokens"] == 400  # 200 + 200

    def test_reserve_with_none_prompt_tokens(self) -> None:
        """When prompt_tokens is None, pool marks usage as inexact."""
        pool = TokenBudgetPool(max_total_tokens=1000)
        output_cap, _ = pool.reserve(prompt_tokens=None, requested_output_tokens=200)

        # Should still work, uses full remaining for output calculation
        assert output_cap == 200
        assert pool.snapshot()["exact_usage"] is False


# =============================================================================
# BudgetTracker Tests
# =============================================================================

class TestBudgetTracker:
    """Test multi-dimensional budget tracking."""

    def test_init(self) -> None:
        budget = Budget(max_tokens=100)
        tracker = BudgetTracker(budget)
        assert tracker.is_exhausted() is False

    def test_consume_output_tokens(self) -> None:
        budget = Budget(max_tokens=100)
        tracker = BudgetTracker(budget)
        tracker.consume({"output_tokens": 50})
        assert tracker.is_exhausted() is False
        tracker.consume({"output_tokens": 60})
        assert tracker.is_exhausted() is True

    def test_consume_total_tokens(self) -> None:
        budget = Budget(max_total_tokens=200)
        tracker = BudgetTracker(budget)
        tracker.consume({"total_tokens": 100})
        assert tracker.is_exhausted() is False
        tracker.consume({"total_tokens": 150})
        assert tracker.is_exhausted() is True

    def test_consume_cost(self) -> None:
        budget = Budget(max_cost_usd=1.0)
        tracker = BudgetTracker(budget)
        tracker.consume({"cost_usd": 0.5})
        assert tracker.is_exhausted() is False
        tracker.consume({"cost_usd": 0.6})
        assert tracker.is_exhausted() is True

    def test_consume_multiple_dimensions(self) -> None:
        budget = Budget(max_tokens=100, max_cost_usd=1.0)
        tracker = BudgetTracker(budget)

        # Under both limits
        tracker.consume({"output_tokens": 50, "cost_usd": 0.3})
        assert tracker.is_exhausted() is False

        # Exceed cost but not tokens
        tracker.consume({"output_tokens": 10, "cost_usd": 0.8})
        assert tracker.is_exhausted() is True

    def test_percentage_used(self) -> None:
        budget = Budget(max_tokens=100, max_total_tokens=1000)
        tracker = BudgetTracker(budget)
        tracker.consume({"output_tokens": 50, "total_tokens": 250})

        pct = tracker.percentage_used()
        assert pct["output_tokens"] == 50
        assert pct["total_tokens"] == 25

    def test_consume_with_normalized_keys(self) -> None:
        """Tracker should handle provider-specific key variants."""
        budget = Budget(max_tokens=100)
        tracker = BudgetTracker(budget)
        # completion_tokens is an alias for output_tokens
        tracker.consume({"completion_tokens": 50})
        assert tracker._output_tokens == 50


# =============================================================================
# Budget Model Validation Tests
# =============================================================================

class TestBudgetModelValidation:
    """Test Budget model validation rules."""

    def test_budget_requires_at_least_one_limit(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            Budget()

    def test_budget_with_output_tokens(self) -> None:
        budget = Budget(max_tokens=100)
        assert budget.max_tokens == 100

    def test_budget_with_total_tokens(self) -> None:
        budget = Budget(max_total_tokens=1000)
        assert budget.max_total_tokens == 1000

    def test_budget_with_seconds(self) -> None:
        budget = Budget(max_seconds=60.0)
        assert budget.max_seconds == 60.0

    def test_budget_with_cost(self) -> None:
        budget = Budget(max_cost_usd=5.0)
        assert budget.max_cost_usd == 5.0

    def test_budget_positive_values_only(self) -> None:
        with pytest.raises(ValueError):
            Budget(max_tokens=0)

        with pytest.raises(ValueError):
            Budget(max_tokens=-1)

        with pytest.raises(ValueError):
            Budget(max_seconds=0)

        with pytest.raises(ValueError):
            Budget(max_cost_usd=-0.01)

    def test_budget_all_limits(self) -> None:
        budget = Budget(
            max_tokens=100,
            max_total_tokens=500,
            max_seconds=30.0,
            max_cost_usd=1.0,
        )
        assert budget.max_tokens == 100
        assert budget.max_total_tokens == 500
        assert budget.max_seconds == 30.0
        assert budget.max_cost_usd == 1.0


class TestLimitsModel:
    """Test Limits model (alternative to Budget)."""

    def test_limits_basic(self) -> None:
        limits = Limits(tokens=100)
        assert limits.tokens == 100

    def test_limits_all_fields(self) -> None:
        limits = Limits(tokens=100, total=500, seconds=30, cost=1.0)
        assert limits.tokens == 100
        assert limits.total == 500
        assert limits.seconds == 30
        assert limits.cost == 1.0


# =============================================================================
# Engine Preflight Tests
# =============================================================================

class TestEnginePreflight:
    """Test Engine budget preflight checks."""

    def test_chat_engine_rejects_output_exceeds_budget(self) -> None:
        provider = MockProvider(main_outputs=["test"])
        task = TaskSpec(
            task_id="test",
            input_text="test",
            model="mock",
            budget=Budget(max_tokens=50),
            success_criteria=SuccessCriteria(min_word_count=1),
            max_output_tokens=100,  # Exceeds budget
        )
        engine = Engine()
        report = engine.run(task, provider)

        assert report.success is False
        assert any("exceeds budget" in err for err in report.errors)

    def test_rlm_engine_rejects_output_exceeds_budget(self) -> None:
        provider = MockProvider(main_outputs=['```python\nFINAL("test")\n```'])
        task = TaskSpec(
            task_id="test",
            input_text="test",
            model="mock",
            budget=Budget(max_tokens=50),
            success_criteria=SuccessCriteria(required_substrings=["test"]),
            max_output_tokens=100,  # Exceeds budget
        )
        engine = RLMEngine(max_steps=1)
        report = engine.run(task, provider, data="context")

        assert report.success is False
        assert any("exceeds budget" in err for err in report.errors)


# =============================================================================
# Integration Tests
# =============================================================================

class TestBudgetIntegration:
    """Integration tests for budget enforcement."""

    def test_rlm_exhausts_budget_mid_task(self) -> None:
        """RLM should stop when budget exhausted during execution."""
        model_output = '''
```python
first = llm_query("SUBCALL:first")
print(first)
second = llm_query("SUBCALL:second")
print(second)
```
'''.strip()
        provider = MockProvider(
            main_outputs=[model_output],
            subcall_responses={"first": "ok"},
            usage={"output_tokens": 6, "total_tokens": 6},
        )
        task = TaskSpec(
            task_id="budget-test",
            input_text="exhaust budget",
            model="mock",
            budget=Budget(max_total_tokens=12),  # Tight budget
            success_criteria=SuccessCriteria(required_substrings=["unused"]),
        )
        engine = RLMEngine(max_steps=1)
        report = engine.run(task, provider, data="context")

        assert report.success is False
        assert report.steps[0].error is not None
        assert "budget_exhausted" in report.steps[0].error

    def test_chat_engine_tracks_usage(self) -> None:
        """Chat engine should track token usage in report."""
        provider = MockProvider(
            main_outputs=["The answer is 42."],
            usage={"output_tokens": 10, "total_tokens": 20},
        )
        task = TaskSpec(
            task_id="usage-test",
            input_text="What is the answer?",
            model="mock",
            budget=Budget(max_tokens=100),
            success_criteria=SuccessCriteria(min_word_count=1),
        )
        engine = Engine()
        report = engine.run(task, provider)

        assert report.success is True
        assert report.budget_usage.output_tokens == 10
        assert report.budget_usage.total_tokens == 20


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestBudgetThreadSafety:
    """Test thread safety of budget components."""

    def test_budget_controller_concurrent_record(self) -> None:
        """Multiple threads recording usage should not corrupt state."""
        controller = BudgetController(max_tokens=100000)

        def record_batch():
            for _ in range(100):
                controller.record_usage(input_tokens=1, output_tokens=1)

        threads = [threading.Thread(target=record_batch) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 10 threads * 100 iterations * 2 tokens = 2000 tokens
        assert controller.total_tokens == 2000

    def test_token_pool_concurrent_reserve_commit(self) -> None:
        """Multiple threads reserving/committing should not corrupt state."""
        pool = TokenBudgetPool(max_total_tokens=100000)
        committed_total = [0]
        lock = threading.Lock()

        def reserve_commit_batch():
            for _ in range(50):
                output_cap, reserved = pool.reserve(
                    prompt_tokens=10, requested_output_tokens=10
                )
                if reserved > 0:
                    pool.commit(reserved, actual_total_tokens=reserved)
                    with lock:
                        committed_total[0] += reserved

        threads = [threading.Thread(target=reserve_commit_batch) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snapshot = pool.snapshot()
        assert snapshot["used_tokens"] == committed_total[0]
        assert snapshot["reserved_tokens"] == 0


# =============================================================================
# Edge Cases and Robustness
# =============================================================================

class TestBudgetEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_exact_limit_boundary(self) -> None:
        """Exactly hitting limit should mark exceeded."""
        controller = BudgetController(max_tokens=100)
        controller.record_usage(output_tokens=100)  # max_tokens = output limit
        assert controller.is_exceeded is True
        assert controller.remaining_tokens == 0

    def test_very_large_values(self) -> None:
        """Should handle very large token counts."""
        controller = BudgetController(max_tokens=10**12)
        controller.record_usage(input_tokens=10**11, output_tokens=10**11)
        assert controller.total_tokens == 2 * 10**11
        assert controller.is_exceeded is False

    def test_very_small_cost(self) -> None:
        """Should handle very small costs."""
        controller = BudgetController(max_cost_usd=0.001)
        controller.record_usage(cost_usd=0.0001)
        assert controller.total_cost_usd == 0.0001
        assert controller.is_exceeded is False

    def test_zero_input_tokens(self) -> None:
        """Pre-call with zero input should work."""
        controller = BudgetController(max_tokens=100)
        controller.pre_call_check(input_tokens=0, max_output_tokens=50)
        # Should not raise

    def test_pool_exhausted_returns_zero(self) -> None:
        """Exhausted pool should return 0 for output_cap."""
        pool = TokenBudgetPool(max_total_tokens=100)
        # Reserve 50 prompt + 50 output = 100 tokens
        output_cap, reserved = pool.reserve(prompt_tokens=50, requested_output_tokens=50)
        assert reserved == 100  # Pool now fully reserved

        # Second reserve should fail - no remaining capacity
        output_cap2, reserved2 = pool.reserve(prompt_tokens=10, requested_output_tokens=50)
        assert output_cap2 == 0
        assert reserved2 == 0

    def test_tracker_with_missing_fields(self) -> None:
        """Tracker should handle missing usage fields gracefully."""
        budget = Budget(max_tokens=100)
        tracker = BudgetTracker(budget)
        tracker.consume({})  # Empty usage
        tracker.consume({"unknown_field": 123})  # Unknown field
        assert tracker.is_exhausted() is False

    def test_budget_event_serialization(self) -> None:
        """BudgetEvent should serialize all fields."""
        event = BudgetEvent(
            timestamp="2024-01-01T00:00:00Z",
            event_type="call",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            cumulative_cost_usd=0.01,
            cumulative_tokens=150,
            details={"model": "gpt-4"},
        )
        d = event.to_dict()
        assert d["timestamp"] == "2024-01-01T00:00:00Z"
        assert d["event_type"] == "call"
        assert d["details"] == {"model": "gpt-4"}


class TestBudgetUsageModel:
    """Test BudgetUsage model."""

    def test_budget_usage_basic(self) -> None:
        usage = BudgetUsage(
            elapsed_seconds=1.5,
            output_tokens=100,
            total_tokens=200,
            cost_usd=0.01,
        )
        assert usage.elapsed_seconds == 1.5
        assert usage.output_tokens == 100
        assert usage.limits_exceeded == []

    def test_budget_usage_with_limits_exceeded(self) -> None:
        usage = BudgetUsage(
            elapsed_seconds=60.0,
            output_tokens=500,
            total_tokens=1000,
            cost_usd=0.50,
            limits_exceeded=["max_seconds", "max_tokens"],
        )
        assert "max_seconds" in usage.limits_exceeded
        assert "max_tokens" in usage.limits_exceeded

    def test_budget_usage_optional_input_tokens(self) -> None:
        """input_tokens is optional."""
        usage = BudgetUsage(
            elapsed_seconds=1.0,
            input_tokens=None,
            output_tokens=100,
            total_tokens=100,
            cost_usd=0.01,
        )
        assert usage.input_tokens is None


# =============================================================================
# Robustness Tests
# =============================================================================

class TestBudgetRobustness:
    """Robustness tests for enterprise deployment."""

    def test_rapid_fire_operations(self) -> None:
        """Handle rapid successive operations."""
        controller = BudgetController(max_tokens=1000000)
        for i in range(1000):
            controller.record_usage(input_tokens=100, output_tokens=50)
        assert controller.total_tokens == 150000

    def test_pool_stress_test(self) -> None:
        """Stress test token pool with many operations."""
        pool = TokenBudgetPool(max_total_tokens=1000000)
        for _ in range(100):
            _, reserved = pool.reserve(prompt_tokens=100, requested_output_tokens=100)
            pool.commit(reserved, actual_total_tokens=200)

        snapshot = pool.snapshot()
        assert snapshot["used_tokens"] == 20000  # 100 * 200

    def test_concurrent_mixed_operations(self) -> None:
        """Multiple threads doing different operations."""
        controller = BudgetController(max_tokens=1000000, max_cost_usd=1000.0)

        def record_tokens():
            for _ in range(100):
                controller.record_usage(input_tokens=10)

        def record_cost():
            for _ in range(100):
                controller.record_usage(cost_usd=0.01)

        def check_remaining():
            for _ in range(100):
                _ = controller.remaining_tokens
                _ = controller.remaining_cost

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(record_tokens),
                executor.submit(record_tokens),
                executor.submit(record_cost),
                executor.submit(record_cost),
                executor.submit(check_remaining),
                executor.submit(check_remaining),
            ]
            for f in futures:
                f.result()

        # Verify final state is consistent
        assert controller.total_tokens == 2000  # 2 threads * 100 * 10
        assert abs(controller.total_cost_usd - 2.0) < 0.001  # 2 threads * 100 * 0.01

    def test_audit_log_under_load(self) -> None:
        """Audit log should handle many events."""
        controller = BudgetController(max_tokens=1000000)
        for _ in range(1000):
            controller.record_usage(input_tokens=10)

        log = controller.audit_log
        assert len(log) == 1000
        # All events should have timestamps
        assert all(e["timestamp"] for e in log)
