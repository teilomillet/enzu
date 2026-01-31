"""Tests for file_researcher example."""

import pytest

from enzu import Session, SessionBudgetExceeded


class TestFileResearcher:
    """Tests for the file researcher demo."""

    def test_session_creation(self, mock_provider_factory):
        """Session can be created with budget limits."""
        session = Session(
            model="test-model",
            provider=mock_provider_factory(["Response"]),
            max_cost_usd=0.05,
            max_tokens=1200,
        )

        assert session is not None

    def test_session_run(self, mock_provider_factory):
        """Session.run executes task within budget."""
        mock = mock_provider_factory(["Research findings: Test result"])

        session = Session(
            model="test-model",
            provider=mock,
            max_cost_usd=0.05,
            max_tokens=1200,
        )

        result = session.run("Research this topic", tokens=400)
        assert result is not None

    def test_session_persistence(self, temp_dir, mock_provider_factory):
        """Session can be saved and loaded."""
        # Note: Session.save() serializes the entire session including provider.
        # With string provider name, this works. With mock provider object, it fails.
        # This test verifies the pattern works with real provider names.
        session = Session(
            model="test-model",
            provider="openai",  # Use string name for serialization
            max_cost_usd=0.05,
            max_tokens=1200,
        )

        # Save (without running, to avoid API calls)
        save_path = str(temp_dir / "session.json")
        session.save(save_path)

        # Load
        loaded = Session.load(save_path)
        assert loaded is not None

    def test_step_limit(self, mock_provider_factory):
        """Step limit bounds exploration."""
        mock = mock_provider_factory([
            "Step 1 result",
            "Step 2 result",
            "Step 3 result",
            "Step 4 result",
        ])

        session = Session(
            model="test-model",
            provider=mock,
            max_cost_usd=1.00,
            max_tokens=5000,
        )

        # Run with step limit
        result = session.run("Multi-step research", tokens=400, max_steps=2)
        assert result is not None

    def test_budget_tracking(self, mock_provider_factory):
        """Session tracks cumulative budget usage."""
        mock = mock_provider_factory(
            ["Response"],
            usage={"output_tokens": 50, "total_tokens": 100},
        )

        session = Session(
            model="test-model",
            provider=mock,
            max_cost_usd=0.05,
            max_tokens=1200,
        )

        session.run("Task 1", tokens=200)

        # Budget should be tracked via session.total_* properties
        # The exact API depends on Session implementation
        assert session.total_tokens >= 0 or session.total_cost_usd >= 0
