"""Targeted chaos probes for suspicious code patterns in enzu.

Each test targets a specific code pattern that looks like it could
harbor a bug. These are more like adversarial fuzzing than property
testing.
"""

from __future__ import annotations

import uuid

from hypothesis import given, settings, strategies as st

from enzu.budget import BudgetController, BudgetExceeded
from enzu.engine import Engine
from enzu.models import (
    Budget,
    Outcome,
    ProviderResult,
    SuccessCriteria,
    TaskSpec,
)
from enzu.providers.base import BaseProvider
from enzu.session import Exchange, Session, _format_history


# ============================================================================
# Probe: _format_history with single huge exchange > max_chars
# ============================================================================


@settings(max_examples=200)
@given(
    max_chars=st.integers(min_value=0, max_value=50),
    text_len=st.integers(min_value=100, max_value=500),
)
def test_format_history_huge_exchange(max_chars: int, text_len: int):
    """When a single exchange exceeds max_chars, history should be empty
    or contain at most the header."""
    big_text = "x" * text_len
    exchanges = [Exchange(user=big_text, assistant=big_text)]
    result = _format_history(exchanges, max_chars)
    # The function should not crash, and output should be bounded
    assert isinstance(result, str)
    # With a small max_chars and huge exchange, result should be empty
    # because the exchange doesn't fit
    if max_chars < 50:
        assert result == "", (
            f"Expected empty history for max_chars={max_chars}, "
            f"exchange_size={text_len}, got len={len(result)}"
        )


# ============================================================================
# Probe: Budget at exact boundary (cost == limit)
# ============================================================================


def test_budget_exact_boundary():
    """When cost exactly equals the limit, budget should be marked exceeded."""
    controller = BudgetController(max_cost_usd=1.0, max_tokens=100)

    # Record exactly the limit
    controller.record_usage(cost_usd=1.0, output_tokens=100)

    assert controller.is_exceeded, "Budget should be exceeded at exact limit"
    assert controller.remaining_cost == 0.0
    assert controller.remaining_tokens == 0

    # Further calls should be blocked
    try:
        controller.pre_call_check(input_tokens=0, max_output_tokens=0)
        assert False, "Should have raised BudgetExceeded"
    except BudgetExceeded:
        pass


def test_budget_one_below_boundary():
    """When cost is one tick below the limit, budget should NOT be exceeded."""
    controller = BudgetController(max_cost_usd=1.0)

    controller.record_usage(cost_usd=0.9999999999)
    # 0.9999999999 < 1.0, so should not be exceeded
    assert not controller.is_exceeded, "Budget should not be exceeded just below limit"
    assert controller.remaining_cost > 0


# ============================================================================
# Probe: Session raise_cost_cap at exact current cap
# ============================================================================


def test_session_raise_cap_exact_current():
    """raise_cost_cap with new_cap == current cap should raise ValueError."""
    session = Session(model="gpt-4o-mini", provider="mock", max_cost_usd=5.0)
    try:
        session.raise_cost_cap(5.0)
        assert False, "Should reject equal cap"
    except ValueError:
        pass


def test_session_raise_token_cap_exact_current():
    """raise_token_cap with new_cap == current cap should raise ValueError."""
    session = Session(model="gpt-4o-mini", provider="mock", max_tokens=100)
    try:
        session.raise_token_cap(100)
        assert False, "Should reject equal cap"
    except ValueError:
        pass


# ============================================================================
# Probe: Engine with same provider instance in primary and fallback
# ============================================================================


class _CountingProvider(BaseProvider):
    name = "counting"

    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, task: TaskSpec) -> ProviderResult:
        self.call_count += 1
        if self.call_count <= 1:
            raise RuntimeError("First call fails")
        return ProviderResult(
            output_text="recovered",
            raw={},
            usage={"output_tokens": 5, "total_tokens": 10},
            provider=self.name,
            model=task.model,
        )


def test_engine_same_provider_in_fallback():
    """What happens if the same provider instance is both primary and fallback?

    The engine compares `current_provider == all_providers[-1]` to detect
    the last provider. If the same instance appears twice, the identity
    check `==` uses object identity by default, which should work.
    """
    engine = Engine()
    provider = _CountingProvider()

    task = TaskSpec(
        task_id=str(uuid.uuid4()),
        input_text="test",
        model="gpt-4o-mini",
        budget=Budget(max_tokens=500),
        success_criteria=SuccessCriteria(goal="test"),
    )

    # Same instance as both primary and fallback
    report = engine.run(task, provider, fallback_providers=[provider])

    # The first call fails, the second succeeds (same instance, count incremented)
    assert report.success, "Should succeed on second call to same provider"
    assert provider.call_count == 2
    assert len(report.trajectory) == 2


# ============================================================================
# Probe: BudgetController negative values defense
# ============================================================================


@settings(max_examples=200)
@given(
    input_tokens=st.integers(min_value=-1000, max_value=1000),
    output_tokens=st.integers(min_value=-1000, max_value=1000),
    cost=st.floats(min_value=-100.0, max_value=100.0).filter(lambda x: x == x),
)
def test_budget_negative_values_clamped(
    input_tokens: int, output_tokens: int, cost: float
):
    """record_usage should clamp negative values to 0."""
    controller = BudgetController(max_cost_usd=1000.0, max_tokens=10000)
    controller.record_usage(
        input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost
    )

    # Totals should never be negative
    assert controller.total_cost_usd >= 0, (
        f"Cost went negative: {controller.total_cost_usd}"
    )
    assert controller.total_tokens >= 0, (
        f"Tokens went negative: {controller.total_tokens}"
    )

    remaining = controller.remaining_cost
    assert remaining is not None and remaining >= 0


# ============================================================================
# Probe: Engine with all providers failing produces correct outcome
# ============================================================================


class _AlwaysFailProvider(BaseProvider):
    def __init__(self, name: str, error_msg: str) -> None:
        self.name = name
        self._error = error_msg

    def generate(self, task: TaskSpec) -> ProviderResult:
        raise RuntimeError(self._error)


def test_engine_all_providers_fail_outcome():
    """When all providers fail, outcome should be PROVIDER_ERROR."""
    engine = Engine()
    task = TaskSpec(
        task_id=str(uuid.uuid4()),
        input_text="test",
        model="gpt-4o-mini",
        budget=Budget(max_tokens=500),
        success_criteria=SuccessCriteria(goal="test"),
    )

    primary = _AlwaysFailProvider("p1", "provider 1 failed")
    fallbacks = [
        _AlwaysFailProvider("p2", "provider 2 failed"),
        _AlwaysFailProvider("p3", "provider 3 failed"),
    ]

    report = engine.run(task, primary, fallback_providers=fallbacks)

    assert not report.success
    assert report.outcome == Outcome.PROVIDER_ERROR
    assert len(report.trajectory) == 3
    assert len(report.errors) >= 1
    # Last provider's name should be in the report
    assert report.provider == "p3"


# ============================================================================
# Probe: Session serialization with edge-case float costs
# ============================================================================


@settings(max_examples=100)
@given(
    cost=st.one_of(
        st.just(0.0),
        st.just(float("inf")),
        st.just(float("-inf")),
        st.just(float("nan")),
        st.just(1e308),
        st.just(5e-324),
        st.floats(allow_nan=True, allow_infinity=True),
    ),
)
def test_session_serialize_edge_costs(cost: float):
    """Session serialization should handle or reject edge-case float costs."""
    session = Session(model="gpt-4o-mini", provider="mock")
    session.total_cost_usd = cost if cost == cost else 0.0  # skip NaN for total
    session.exchanges.append(
        Exchange(user="q", assistant="a", cost_usd=cost if cost == cost else None)
    )

    d = session.to_dict()
    assert isinstance(d, dict)
    assert "exchanges" in d

    # Roundtrip should not crash
    try:
        import json

        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        assert loaded["model"] == "gpt-4o-mini"
    except (ValueError, OverflowError):
        pass  # inf/nan may not be JSON-serializable, that's a known limitation
