"""Per-customer budget tracking and enforcement."""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

from .models import CustomerBudget


@dataclass
class BudgetState:
    """Internal budget state for a customer."""

    customer_id: str
    limit_usd: float
    used_usd: float = 0.0
    requests_count: int = 0
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BudgetExceededError(Exception):
    """Raised when customer budget is exceeded."""

    def __init__(self, customer_id: str, limit: float, used: float):
        self.customer_id = customer_id
        self.limit = limit
        self.used = used
        super().__init__(
            f"Budget exceeded for customer {customer_id}: "
            f"${used:.4f} used of ${limit:.2f} limit"
        )


class BudgetController:
    """
    Per-customer budget tracking with thread-safe operations.

    Features:
    - Per-customer budget limits
    - Automatic period reset (monthly)
    - Pre-flight budget checks
    - Atomic usage recording
    """

    def __init__(self, default_limit_usd: float = 10.0, reset_period_days: int = 30):
        self._budgets: Dict[str, BudgetState] = {}
        self._lock = threading.RLock()
        self._default_limit = default_limit_usd
        self._reset_period_days = reset_period_days

    def get_or_create(self, customer_id: str, limit_usd: Optional[float] = None) -> BudgetState:
        """Get or create budget state for a customer."""
        with self._lock:
            if customer_id not in self._budgets:
                self._budgets[customer_id] = BudgetState(
                    customer_id=customer_id,
                    limit_usd=limit_usd or self._default_limit,
                )
            return self._budgets[customer_id]

    def set_limit(self, customer_id: str, limit_usd: float) -> None:
        """Set budget limit for a customer."""
        with self._lock:
            state = self.get_or_create(customer_id)
            state.limit_usd = limit_usd

    def check_budget(self, customer_id: str, estimated_cost: float = 0.0) -> bool:
        """
        Check if customer has sufficient budget.

        Args:
            customer_id: Customer identifier
            estimated_cost: Estimated cost of the operation

        Returns:
            True if budget is available, False otherwise
        """
        with self._lock:
            state = self.get_or_create(customer_id)
            self._maybe_reset(state)
            remaining = state.limit_usd - state.used_usd
            return remaining >= estimated_cost

    def reserve(self, customer_id: str, estimated_cost: float) -> bool:
        """
        Reserve budget for an operation (pre-flight check).

        Returns True if reservation successful, False if insufficient budget.
        Does NOT deduct from budget - use record_usage after operation completes.
        """
        with self._lock:
            state = self.get_or_create(customer_id)
            self._maybe_reset(state)
            remaining = state.limit_usd - state.used_usd
            return remaining >= estimated_cost

    def record_usage(self, customer_id: str, cost_usd: float) -> None:
        """
        Record actual usage after operation completes.

        Args:
            customer_id: Customer identifier
            cost_usd: Actual cost incurred
        """
        with self._lock:
            state = self.get_or_create(customer_id)
            state.used_usd += cost_usd
            state.requests_count += 1

    def get_status(self, customer_id: str) -> CustomerBudget:
        """Get current budget status for a customer."""
        with self._lock:
            state = self.get_or_create(customer_id)
            self._maybe_reset(state)

            period_end = datetime.fromtimestamp(
                state.period_start.timestamp() + (self._reset_period_days * 86400),
                tz=timezone.utc,
            )

            return CustomerBudget(
                customer_id=customer_id,
                budget_limit_usd=state.limit_usd,
                budget_used_usd=state.used_usd,
                budget_remaining_usd=max(0, state.limit_usd - state.used_usd),
                requests_count=state.requests_count,
                period_start=state.period_start,
                period_end=period_end,
            )

    def get_all_customers(self) -> list[str]:
        """Get list of all customers with budgets."""
        with self._lock:
            return list(self._budgets.keys())

    def _maybe_reset(self, state: BudgetState) -> None:
        """Reset budget if period has elapsed."""
        now = datetime.now(timezone.utc)
        days_elapsed = (now - state.period_start).days
        if days_elapsed >= self._reset_period_days:
            state.used_usd = 0.0
            state.requests_count = 0
            state.period_start = now
            state.last_reset = now


# Global budget controller instance
_budget_controller: Optional[BudgetController] = None


def get_budget_controller() -> BudgetController:
    """Get the global budget controller instance."""
    global _budget_controller
    if _budget_controller is None:
        _budget_controller = BudgetController()
    return _budget_controller


def configure_budget_controller(
    default_limit_usd: float = 10.0,
    reset_period_days: int = 30,
) -> BudgetController:
    """Configure and return the global budget controller."""
    global _budget_controller
    _budget_controller = BudgetController(
        default_limit_usd=default_limit_usd,
        reset_period_days=reset_period_days,
    )
    return _budget_controller
