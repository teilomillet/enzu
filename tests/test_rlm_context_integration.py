"""Integration tests for RLM context metrics auto-population."""

from enzu.models import Budget, SuccessCriteria, TaskSpec
from enzu.rlm import RLMEngine
from tests.providers.mock_provider import MockProvider


class TestRLMContextIntegration:
    """Test that RLM engine automatically populates context_breakdown."""

    def test_context_breakdown_populated(self):
        """Verify context_breakdown is automatically populated in reports."""
        task = TaskSpec(
            task_id="test-context-metrics",
            input_text="What is 2+2? Just answer with the number.",
            model="gpt-4o-mini",
            budget=Budget(max_total_tokens=1000, max_seconds=30),
            success_criteria=SuccessCriteria(required_substrings=["4"]),
            max_output_tokens=50,
        )

        provider = MockProvider(main_outputs=["```repl\nFINAL('4')\n```"])

        engine = RLMEngine(max_steps=2)
        report = engine.run(task, provider, data="")

        # Verify context_breakdown exists
        assert report.context_breakdown is not None

        # Verify key metrics are present
        assert "system_prompt_chars" in report.context_breakdown
        assert "task_prompt_chars" in report.context_breakdown
        assert "total_steps" in report.context_breakdown
        assert "llm_invocations" in report.context_breakdown

    def test_context_breakdown_tracks_steps(self):
        """Verify context_breakdown tracks trajectory steps."""
        task = TaskSpec(
            task_id="test-steps",
            input_text="Count to 3",
            model="gpt-4o-mini",
            budget=Budget(max_total_tokens=1000),
            success_criteria=SuccessCriteria(required_substrings=["3"]),
            max_output_tokens=100,
        )

        # Provider returns 2 responses before FINAL
        provider = MockProvider(
            main_outputs=[
                "```repl\nprint('step 1')\n```",
                "```repl\nFINAL('1, 2, 3')\n```",
            ]
        )

        engine = RLMEngine(max_steps=5)
        report = engine.run(task, provider, data="")

        # Should track steps
        assert report.context_breakdown is not None
        assert report.context_breakdown["total_steps"] >= 1

        # Should track LLM invocations (at least as many as steps)
        assert report.context_breakdown["llm_invocations"] >= report.context_breakdown["total_steps"]

    def test_context_breakdown_system_prompt(self):
        """Verify system prompt size is tracked."""
        task = TaskSpec(
            task_id="test-system-prompt",
            input_text="Test",
            model="gpt-4o-mini",
            budget=Budget(max_total_tokens=2000),  # RLM system prompt ~781 tokens
            success_criteria=SuccessCriteria(required_substrings=["test"]),
        )

        provider = MockProvider(main_outputs=["```repl\nFINAL('test')\n```"])

        engine = RLMEngine(max_steps=2)
        report = engine.run(task, provider, data="")

        assert report.context_breakdown is not None
        # System prompt should be non-zero (RLM adds substantial system prompt)
        assert report.context_breakdown["system_prompt_chars"] > 0

    def test_context_breakdown_task_chars(self):
        """Verify task prompt size matches input."""
        task_text = "What is the capital of France? Answer in one word."
        task = TaskSpec(
            task_id="test-task-chars",
            input_text=task_text,
            model="gpt-4o-mini",
            budget=Budget(max_total_tokens=2000),
            success_criteria=SuccessCriteria(required_substrings=["Paris"]),
        )

        provider = MockProvider(main_outputs=["```repl\nFINAL('Paris')\n```"])

        engine = RLMEngine(max_steps=2)
        report = engine.run(task, provider, data="")

        assert report.context_breakdown is not None
        # Task prompt chars should match input length
        assert report.context_breakdown["task_prompt_chars"] == len(task_text)

    def test_context_breakdown_no_symbolic_context(self):
        """Verify symbolic_context flag when no file data provided."""
        task = TaskSpec(
            task_id="test-no-symbolic",
            input_text="Test",
            model="gpt-4o-mini",
            budget=Budget(max_total_tokens=2000),
            success_criteria=SuccessCriteria(required_substrings=["test"]),
        )

        provider = MockProvider(main_outputs=["```repl\nFINAL('test')\n```"])

        engine = RLMEngine(max_steps=2)
        report = engine.run(task, provider, data="")

        assert report.context_breakdown is not None
        # No file context provided
        assert report.context_breakdown["used_symbolic_context"] is False
        assert report.context_breakdown["file_data_chars"] == 0

    def test_subcall_count_tracked(self):
        """Verify subcall count is tracked in context breakdown."""
        task = TaskSpec(
            task_id="test-subcalls",
            input_text="Test",
            model="gpt-4o-mini",
            budget=Budget(max_total_tokens=1000),
            success_criteria=SuccessCriteria(required_substrings=["test"]),
        )

        provider = MockProvider(main_outputs=["```repl\nFINAL('test')\n```"])

        engine = RLMEngine(max_steps=2, recursive_subcalls=True)
        report = engine.run(task, provider, data="")

        assert report.context_breakdown is not None
        # Subcall count should be present (may be 0 if no subcalls made)
        assert "subcalls" in report.context_breakdown
        assert report.context_breakdown["subcalls"] >= 0

    def test_context_breakdown_failure_graceful(self):
        """Verify context breakdown failure doesn't break execution."""
        # Even if context metrics fail, run should complete
        task = TaskSpec(
            task_id="test-graceful",
            input_text="Test",
            model="gpt-4o-mini",
            budget=Budget(max_total_tokens=2000),
            success_criteria=SuccessCriteria(required_substrings=["test"]),
        )

        provider = MockProvider(main_outputs=["```repl\nFINAL('test')\n```"])

        engine = RLMEngine(max_steps=2)
        report = engine.run(task, provider, data="")

        # Report should exist even if context_breakdown has issues
        assert report is not None
        assert report.success is True
        # context_breakdown should be populated
        assert report.context_breakdown is not None
