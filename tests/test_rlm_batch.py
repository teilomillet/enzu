"""
Test llm_batch parallel execution functionality and pip_install feature.

This test verifies that:
1. llm_batch is available in the sandbox namespace
2. System prompts guide usage of llm_batch for parallel queries
3. pip_install is conditionally available based on enable_pip flag
"""

from enzu import Budget, SuccessCriteria, TaskSpec
from enzu.rlm.engine import RLMEngine
from tests.providers.mock_provider import MockProvider


def test_llm_batch_in_system_prompt() -> None:
    """Test that llm_batch is mentioned in system prompts."""
    provider = MockProvider(
        main_outputs=["```python\nprint('checking prompt')\nFINAL('done')\n```"],
    )

    input_data = "test data"

    task = TaskSpec(
        task_id="test-prompt",
        input_text="Test task",
        model="test-model",
        budget=Budget(max_tokens=200),
        success_criteria=SuccessCriteria(required_substrings=["done"]),
    )

    engine = RLMEngine(max_steps=2, prompt_style="extended")
    report = engine.run(task, provider, data=input_data)

    # Check that the system prompt mentions llm_batch
    assert "llm_batch" in report.steps[0].prompt, (
        "System prompt should mention llm_batch"
    )

    # Verify guidance about parallel execution
    assert "parallel" in report.steps[0].prompt.lower(), (
        "Should mention parallel execution"
    )
    assert "llm_query" in report.steps[0].prompt, "Should still mention llm_query"


def test_llm_batch_namespace_injection() -> None:
    """Test that llm_batch function is available in sandbox."""
    provider = MockProvider(
        main_outputs=[
            "```python\n# Verify llm_batch exists by trying to reference it\ntry:\n    ref = llm_batch\n    exists = True\nexcept NameError:\n    exists = False\nFINAL(f'llm_batch exists: {exists}')\n```"
        ],
    )

    input_data = "test"

    task = TaskSpec(
        task_id="test-namespace",
        input_text="Check namespace",
        model="test-model",
        budget=Budget(max_tokens=200),
        success_criteria=SuccessCriteria(required_substrings=["True"]),
    )

    engine = RLMEngine(max_steps=1, prompt_style="extended")
    report = engine.run(task, provider, data=input_data)

    # llm_batch should exist in namespace
    assert report.answer is not None and "True" in report.answer, (
        f"llm_batch should exist in namespace. Answer: {report.answer}"
    )


def test_llm_batch_empty_list_handling() -> None:
    """Test that llm_batch([]) returns empty list without errors."""
    provider = MockProvider(
        main_outputs=[
            "```python\nresult = llm_batch([])\nFINAL(f'Empty batch result: {result}')\n```"
        ],
    )

    input_data = "test"

    task = TaskSpec(
        task_id="test-empty-batch",
        input_text="Test empty llm_batch",
        model="test-model",
        budget=Budget(max_tokens=200),
        success_criteria=SuccessCriteria(required_substrings=["Empty"]),
    )

    engine = RLMEngine(max_steps=2)
    report = engine.run(task, provider, data=input_data)

    # Should handle empty list gracefully
    assert not report.errors, f"Should not error on empty batch: {report.errors}"
    assert report.answer is not None and "[]" in report.answer, (
        f"Should return empty list. Answer: {report.answer}"
    )


def test_pip_install_guidance_when_enabled() -> None:
    """Test that pip_install guidance appears when enable_pip=True."""
    provider = MockProvider(
        main_outputs=["```python\nprint('test')\nFINAL('done')\n```"],
    )

    input_data = "test"

    task = TaskSpec(
        task_id="test-pip-enabled",
        input_text="Test with pip enabled",
        model="test-model",
        budget=Budget(max_tokens=200),
        success_criteria=SuccessCriteria(required_substrings=["done"]),
    )

    # Enable pip installation
    engine = RLMEngine(max_steps=1, enable_pip=True, prompt_style="extended")
    report = engine.run(task, provider, data=input_data)

    # Verify pip_install guidance was in the prompt
    assert "pip_install" in report.steps[0].prompt, (
        "System prompt should include pip_install when enabled"
    )
    assert "Dynamic Package Installation" in report.steps[0].prompt, (
        "Should have pip section header"
    )


def test_pip_install_not_in_prompt_by_default() -> None:
    """Test that pip_install guidance is NOT shown by default."""
    provider = MockProvider(
        main_outputs=["```python\nFINAL('completed')\n```"],
    )

    input_data = "test"

    task = TaskSpec(
        task_id="test-no-pip",
        input_text="Test without pip",
        model="test-model",
        budget=Budget(max_tokens=200),
        success_criteria=SuccessCriteria(required_substrings=["completed"]),
    )

    # Default: enable_pip=False
    engine = RLMEngine(max_steps=1, prompt_style="extended")
    report = engine.run(task, provider, data=input_data)

    # Verify pip_install guidance was NOT in the prompt
    assert "pip_install" not in report.steps[0].prompt, (
        "pip_install should not be in prompt by default"
    )
    assert "Dynamic Package Installation" not in report.steps[0].prompt


def test_llm_batch_vs_llm_query_guidance() -> None:
    """Test that guidance emphasizes llm_batch for multiple queries."""
    provider = MockProvider(
        main_outputs=["```python\nFINAL('done')\n```"],
    )

    task = TaskSpec(
        task_id="test-guidance",
        input_text="Process multiple items",
        model="test-model",
        budget=Budget(max_tokens=200),
        success_criteria=SuccessCriteria(required_substrings=["done"]),
    )

    engine = RLMEngine(max_steps=1)
    report = engine.run(task, provider, data="item1\nitem2\nitem3")

    prompt = report.steps[0].prompt

    # Should have clear guidance about when to use each
    assert "llm_query" in prompt, "Should explain llm_query"
    assert "llm_batch" in prompt, "Should explain llm_batch"
    assert "MUCH FASTER" in prompt or "faster" in prompt.lower(), (
        "Should emphasize speed benefit"
    )
