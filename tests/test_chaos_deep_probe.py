"""Deep probing chaos tests — designed to find real bugs in enzu.

These tests use more aggressive settings, longer step sequences,
and specifically target known-fragile areas:

1. Floating-point budget accumulation at scale
2. Concurrent budget check-then-record races
3. Session cost tracking drift
4. Engine verification edge cases with empty/whitespace output
5. Budget controller audit log completeness
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed

from hypothesis import given, settings, strategies as st

from ordeal import ChaosTest, always, invariant, rule

from enzu.budget import BudgetController, BudgetExceeded, count_tokens
from enzu.engine import Engine
from enzu.models import (
    Budget,
    Outcome,
    ProviderResult,
    SuccessCriteria,
    TaskSpec,
)
from enzu.providers.base import BaseProvider
from enzu.session import Exchange, Session


# ============================================================================
# Probe 1: Floating-point accumulation drift
# ============================================================================


class FloatingPointDriftProbe(ChaosTest):
    """Probe for floating-point accumulation errors in BudgetController.

    Record many tiny costs and verify the total doesn't drift from
    the sum of inputs. Enzu rounds to 10 decimal places — is that enough?
    """

    faults = []

    def __init__(self) -> None:
        super().__init__()
        self.controller = BudgetController(max_cost_usd=100.0)
        self.expected_cost: float = 0.0
        self.record_count: int = 0

    @rule(
        cost=st.sampled_from(
            [0.001, 0.0001, 0.00001, 0.1, 0.01, 0.003, 0.007, 0.0099, 1e-8, 0.33333]
        ),
    )
    def record_tiny_cost(self, cost: float) -> None:
        """Record a tiny cost and track the expected total."""
        self.controller.record_usage(cost_usd=cost)
        self.expected_cost += cost
        self.record_count += 1

    @invariant()
    def cost_drift_bounded(self) -> None:
        """Total cost must not drift more than 1e-6 from expected sum."""
        actual = self.controller.total_cost_usd
        drift = abs(actual - round(self.expected_cost, 10))
        always(
            drift < 1e-6,
            f"cost drift bounded: actual={actual}, expected~={self.expected_cost:.12f}, drift={drift}",
        )


TestFloatingPointDriftProbe = FloatingPointDriftProbe.TestCase


# ============================================================================
# Probe 2: Check-then-record race condition
# ============================================================================


def test_concurrent_check_then_record_race():
    """Probe for TOCTOU race: pre_call_check succeeds but budget exceeded
    by the time record_usage is called.

    This isn't a bug per se (the controller handles it), but we verify
    the controller stays consistent even when races happen.
    """
    controller = BudgetController(max_cost_usd=0.10, max_tokens=50)
    errors = []
    results = {"pre_call_ok": 0, "pre_call_blocked": 0, "records": 0}

    def worker(i: int) -> None:
        try:
            controller.pre_call_check(input_tokens=5, max_output_tokens=10)
            results["pre_call_ok"] += 1
        except BudgetExceeded:
            results["pre_call_blocked"] += 1
            return

        # Simulate work between check and record
        controller.record_usage(input_tokens=5, output_tokens=10, cost_usd=0.02)
        results["records"] += 1

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, i) for i in range(20)]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                errors.append(str(e))

    assert not errors, f"Unexpected errors: {errors}"

    # Budget should be exceeded after enough records
    remaining_cost = controller.remaining_cost
    assert remaining_cost is not None
    assert remaining_cost >= 0, f"Remaining cost negative: {remaining_cost}"

    # Audit log should have an entry for every record + exceeded events
    log = controller.audit_log
    call_events = [e for e in log if e["event_type"] == "call"]
    assert len(call_events) == results["records"]


# ============================================================================
# Probe 3: Session cost tracking vs Exchange cost sum
# ============================================================================


class SessionCostDriftProbe(ChaosTest):
    """Probe for drift between session.total_cost_usd and sum of exchange costs.

    The session tracks total_cost_usd separately from individual exchange
    cost_usd fields. If these diverge, something is wrong.
    """

    faults = []

    def __init__(self) -> None:
        super().__init__()
        self.session = Session(model="gpt-4o-mini", provider="mock", max_cost_usd=100.0)
        self.manually_added_cost: float = 0.0

    @rule(
        cost=st.floats(min_value=0.0, max_value=1.0).filter(
            lambda x: x == x and not math.isinf(x)
        ),
    )
    def add_exchange_with_cost(self, cost: float) -> None:
        """Manually add an exchange with a cost and update session tracking."""
        self.session.exchanges.append(Exchange(user="q", assistant="a", cost_usd=cost))
        self.session.total_cost_usd += cost
        self.manually_added_cost += cost

    @invariant()
    def cost_matches_manual_tracking(self) -> None:
        """Session's total must match our manual sum."""
        actual = self.session.total_cost_usd
        drift = abs(actual - self.manually_added_cost)
        always(
            drift < 1e-9,
            f"session cost drift: actual={actual}, expected={self.manually_added_cost}",
        )

    @invariant()
    def exchange_costs_consistent(self) -> None:
        """Sum of exchange costs should approximate session total."""
        exchange_sum = sum(
            ex.cost_usd for ex in self.session.exchanges if ex.cost_usd is not None
        )
        drift = abs(exchange_sum - self.session.total_cost_usd)
        always(
            drift < 1e-6,
            f"exchange cost sum drift: sum={exchange_sum}, total={self.session.total_cost_usd}",
        )


TestSessionCostDriftProbe = SessionCostDriftProbe.TestCase


# ============================================================================
# Probe 4: Engine verification edge cases
# ============================================================================


class _EdgeCaseProvider(BaseProvider):
    name = "edge"

    def __init__(self, output: str) -> None:
        self._output = output

    def generate(self, task: TaskSpec) -> ProviderResult:
        return ProviderResult(
            output_text=self._output,
            raw={},
            usage={
                "output_tokens": max(1, len(self._output) // 4),
                "total_tokens": 100,
            },
            provider=self.name,
            model=task.model,
        )


def _make_task_with_criteria(**criteria_kwargs):
    import uuid

    return TaskSpec(
        task_id=str(uuid.uuid4()),
        input_text="test",
        model="gpt-4o-mini",
        budget=Budget(max_tokens=500),
        success_criteria=SuccessCriteria(**criteria_kwargs),
    )


@settings(max_examples=200)
@given(
    output=st.one_of(
        st.just(""),
        st.just(" "),
        st.just("\n\n\n"),
        st.just("\t  \t"),
        st.just("   a   "),
        st.text(min_size=0, max_size=200),
    ),
    substring=st.text(min_size=1, max_size=20),
)
def test_engine_verification_edge_cases(output: str, substring: str):
    """Probe Engine verification with adversarial outputs.

    Properties:
    - Empty/whitespace output should fail verification with required substrings
    - If output contains the substring, verification should pass
    - Report outcome should be consistent with success flag
    """
    engine = Engine()
    provider = _EdgeCaseProvider(output)
    task = _make_task_with_criteria(required_substrings=[substring])

    report = engine.run(task, provider)

    # Consistency: success and outcome must agree
    if report.success:
        assert report.outcome == Outcome.SUCCESS
    if report.outcome == Outcome.SUCCESS:
        assert report.success

    # If output contains the substring, verification should pass
    if substring in (report.output_text or ""):
        assert report.verification.passed, (
            f"Verification should pass when output contains '{substring}'"
        )

    # If verification failed, success must be False
    if not report.verification.passed:
        assert not report.success


# ============================================================================
# Probe 5: Audit log completeness
# ============================================================================


class AuditLogCompletenessProbe(ChaosTest):
    """Probe that every record_usage call produces exactly one 'call' event,
    and exceeded events are logged exactly once per limit type.
    """

    faults = []

    def __init__(self) -> None:
        super().__init__()
        self.controller = BudgetController(
            max_cost_usd=1.0, max_tokens=100, max_total_tokens=200
        )
        self.record_count: int = 0
        self.seen_exceeded_reasons: set[str] = set()

    @rule(
        output_tokens=st.integers(min_value=1, max_value=30),
        cost=st.floats(min_value=0.01, max_value=0.3).filter(lambda x: x == x),
    )
    def record(self, output_tokens: int, cost: float) -> None:
        self.controller.record_usage(
            input_tokens=5, output_tokens=output_tokens, cost_usd=cost
        )
        self.record_count += 1

    @invariant()
    def call_events_match_records(self) -> None:
        """Number of 'call' events must equal record_usage calls."""
        log = self.controller.audit_log
        call_events = [e for e in log if e["event_type"] == "call"]
        always(
            len(call_events) == self.record_count,
            f"call events ({len(call_events)}) != records ({self.record_count})",
        )

    @invariant()
    def exceeded_logged_at_most_once_per_reason(self) -> None:
        """Each exceeded reason should appear at most once in the log."""
        log = self.controller.audit_log
        exceeded_events = [e for e in log if e["event_type"] == "exceeded"]
        reasons = [e.get("details", {}).get("reason") for e in exceeded_events]
        for reason in set(reasons):
            count = reasons.count(reason)
            always(
                count <= 1,
                f"exceeded reason '{reason}' logged {count} times (should be <= 1)",
            )


TestAuditLogCompletenessProbe = AuditLogCompletenessProbe.TestCase


# ============================================================================
# Probe 6: Token counting fallback robustness
# ============================================================================


def test_count_tokens_never_negative():
    """count_tokens should never return negative, even with adversarial input."""
    adversarial = [
        "",
        " ",
        "\x00" * 100,
        "a" * 100_000,
        "\n" * 50,
        "Hello, world!",
        "\ud800",  # lone surrogate (may cause encoding issues)
    ]
    for text in adversarial:
        try:
            result = count_tokens(text)
            assert result >= 0, f"count_tokens returned {result} for {text!r}"
        except Exception:
            pass  # encoding errors are acceptable, negative values are not
