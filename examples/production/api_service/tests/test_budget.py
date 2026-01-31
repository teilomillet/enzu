"""Tests for budget controller."""

import pytest

from examples.production.api_service.budget import (
    BudgetController,
    BudgetExceededError,
    BudgetState,
)


class TestBudgetState:
    """Tests for BudgetState dataclass."""

    def test_default_values(self):
        """BudgetState has sensible defaults."""
        state = BudgetState(customer_id="cust-1", limit_usd=10.0)
        assert state.used_usd == 0.0
        assert state.requests_count == 0

    def test_with_usage(self):
        """BudgetState tracks usage."""
        state = BudgetState(
            customer_id="cust-1",
            limit_usd=10.0,
            used_usd=5.0,
            requests_count=10,
        )
        assert state.used_usd == 5.0
        assert state.requests_count == 10


class TestBudgetController:
    """Tests for BudgetController."""

    def test_get_or_create(self):
        """Creates new budget for unknown customer."""
        controller = BudgetController(default_limit_usd=20.0)
        state = controller.get_or_create("new-customer")

        assert state.customer_id == "new-customer"
        assert state.limit_usd == 20.0
        assert state.used_usd == 0.0

    def test_check_budget_available(self):
        """Returns True when budget is available."""
        controller = BudgetController(default_limit_usd=10.0)
        assert controller.check_budget("cust-1", estimated_cost=5.0) is True

    def test_check_budget_exceeded(self):
        """Returns False when budget would be exceeded."""
        controller = BudgetController(default_limit_usd=10.0)
        controller.record_usage("cust-1", 9.0)  # Use most of budget

        assert controller.check_budget("cust-1", estimated_cost=5.0) is False

    def test_record_usage(self):
        """Records usage correctly."""
        controller = BudgetController(default_limit_usd=10.0)

        controller.record_usage("cust-1", 2.5)
        controller.record_usage("cust-1", 1.5)

        status = controller.get_status("cust-1")
        assert status.budget_used_usd == 4.0
        assert status.requests_count == 2

    def test_set_limit(self):
        """Can set custom limit for customer."""
        controller = BudgetController(default_limit_usd=10.0)
        controller.set_limit("premium-customer", 100.0)

        status = controller.get_status("premium-customer")
        assert status.budget_limit_usd == 100.0

    def test_get_status(self):
        """Returns complete status."""
        controller = BudgetController(default_limit_usd=10.0)
        controller.record_usage("cust-1", 3.0)

        status = controller.get_status("cust-1")
        assert status.customer_id == "cust-1"
        assert status.budget_limit_usd == 10.0
        assert status.budget_used_usd == 3.0
        assert status.budget_remaining_usd == 7.0

    def test_isolation_between_customers(self):
        """Customers have isolated budgets."""
        controller = BudgetController(default_limit_usd=10.0)

        controller.record_usage("cust-a", 5.0)
        controller.record_usage("cust-b", 2.0)

        status_a = controller.get_status("cust-a")
        status_b = controller.get_status("cust-b")

        assert status_a.budget_used_usd == 5.0
        assert status_b.budget_used_usd == 2.0

    def test_get_all_customers(self):
        """Lists all customers."""
        controller = BudgetController()
        controller.get_or_create("cust-1")
        controller.get_or_create("cust-2")
        controller.get_or_create("cust-3")

        customers = controller.get_all_customers()
        assert len(customers) == 3
        assert "cust-1" in customers
        assert "cust-2" in customers
        assert "cust-3" in customers
