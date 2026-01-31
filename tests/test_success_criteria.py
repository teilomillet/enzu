"""
Comprehensive tests for success_criteria verification.

Tests cover:
1. Empty output handling
2. Goal + mechanical criteria combinations
3. Invalid regex patterns
4. Case sensitivity
5. min_word_count edge cases
6. Substring edge cases
7. RLM vs Chat Engine behavior differences
8. Sub-RLM weak criteria handling
"""

from __future__ import annotations

import random
from typing import Any, Dict

from enzu.models import Budget, SuccessCriteria, TaskSpec, VerificationResult
from enzu.engine import Engine
from enzu.rlm.engine import RLMEngine, _has_strong_success_criteria
from enzu.rlm.verification import verify_output


def _rlm_verify(task: TaskSpec, output_text: str) -> VerificationResult:
    """Wrapper matching old RLMEngine._verify_output API for tests."""
    return verify_output(task.success_criteria, output_text, goal_based_trust=True)


def _make_task(
    *,
    required_substrings: list[str] | None = None,
    required_regex: list[str] | None = None,
    min_word_count: int | None = None,
    goal: str | None = None,
    case_insensitive: bool = False,
) -> TaskSpec:
    """Helper to create TaskSpec with specific success criteria."""
    criteria_kwargs: Dict[str, Any] = {"case_insensitive": case_insensitive}
    if required_substrings:
        criteria_kwargs["required_substrings"] = required_substrings
    if required_regex:
        criteria_kwargs["required_regex"] = required_regex
    if min_word_count:
        criteria_kwargs["min_word_count"] = min_word_count
    if goal:
        criteria_kwargs["goal"] = goal

    # Ensure at least one criterion exists
    if not any([required_substrings, required_regex, min_word_count, goal]):
        criteria_kwargs["min_word_count"] = 1

    return TaskSpec(
        task_id="test",
        input_text="test prompt",
        model="test-model",
        budget=Budget(max_tokens=100),
        success_criteria=SuccessCriteria(**criteria_kwargs),
    )


# =============================================================================
# Empty Output Handling
# =============================================================================


class TestEmptyOutputHandling:
    """Empty output must always fail verification."""

    def test_empty_string_fails(self) -> None:
        task = _make_task(min_word_count=1)
        result = _rlm_verify(task, "")
        assert result.passed is False
        assert "no_output" in result.reasons

    def test_whitespace_only_fails(self) -> None:
        task = _make_task(min_word_count=1)
        result = _rlm_verify(task, "   ")
        assert result.passed is False
        assert "no_output" in result.reasons

    def test_newlines_only_fails(self) -> None:
        task = _make_task(min_word_count=1)
        result = _rlm_verify(task, "\n\n\n")
        assert result.passed is False
        assert "no_output" in result.reasons

    def test_tabs_only_fails(self) -> None:
        task = _make_task(min_word_count=1)
        result = _rlm_verify(task, "\t\t")
        assert result.passed is False
        assert "no_output" in result.reasons

    def test_mixed_whitespace_fails(self) -> None:
        task = _make_task(min_word_count=1)
        result = _rlm_verify(task, " \n\t \n ")
        assert result.passed is False
        assert "no_output" in result.reasons

    def test_empty_with_goal_fails(self) -> None:
        """Even goal-based verification requires non-empty output."""
        task = _make_task(goal="Find the answer")
        result = _rlm_verify(task, "")
        assert result.passed is False
        assert "no_output" in result.reasons

    def test_chat_engine_empty_fails(self) -> None:
        """Chat Engine also rejects empty output."""
        task = _make_task(min_word_count=1)
        engine = Engine()
        result = engine._verify_output(task, "")
        assert result.passed is False
        assert "no_output" in result.reasons


# =============================================================================
# Goal-Only Verification (RLM trusts model judgment)
# =============================================================================


class TestGoalOnlyVerification:
    """Goal-only: model self-judges, any non-empty output passes."""

    def test_goal_only_passes_any_output(self) -> None:
        task = _make_task(goal="Find the root cause")
        result = _rlm_verify(task, "I found it")
        assert result.passed is True
        assert result.reasons == []

    def test_goal_only_passes_single_word(self) -> None:
        task = _make_task(goal="Summarize")
        result = _rlm_verify(task, "Done")
        assert result.passed is True

    def test_goal_only_passes_long_output(self) -> None:
        task = _make_task(goal="Analyze thoroughly")
        output = "This is a very long analysis " * 100
        result = _rlm_verify(task, output)
        assert result.passed is True

    def test_goal_with_min_word_count_1_is_goal_only(self) -> None:
        """min_word_count=1 doesn't count as mechanical criteria."""
        task = _make_task(goal="Find answer", min_word_count=1)
        result = _rlm_verify(task, "x")  # 1 char, 1 word
        assert result.passed is True


# =============================================================================
# Goal + Mechanical Criteria (both must pass)
# =============================================================================


class TestGoalPlusMechanical:
    """When goal AND mechanical criteria set, both must pass."""

    def test_goal_plus_substring_both_pass(self) -> None:
        task = _make_task(goal="Find Einstein", required_substrings=["Einstein"])
        result = _rlm_verify(task, "Albert Einstein was a physicist")
        assert result.passed is True

    def test_goal_plus_substring_missing_fails(self) -> None:
        task = _make_task(goal="Find Einstein", required_substrings=["Einstein"])
        result = _rlm_verify(task, "Albert was a physicist")
        assert result.passed is False
        assert "missing_substring:Einstein" in result.reasons

    def test_goal_plus_regex_both_pass(self) -> None:
        task = _make_task(goal="Find year", required_regex=[r"\d{4}"])
        result = _rlm_verify(task, "The year was 1921")
        assert result.passed is True

    def test_goal_plus_regex_missing_fails(self) -> None:
        task = _make_task(goal="Find year", required_regex=[r"\d{4}"])
        result = _rlm_verify(task, "The year was unknown")
        assert result.passed is False
        assert "missing_regex:" in result.reasons[0]

    def test_goal_plus_min_words_both_pass(self) -> None:
        task = _make_task(goal="Write summary", min_word_count=5)
        result = _rlm_verify(task, "This is a five word summary")
        assert result.passed is True

    def test_goal_plus_min_words_insufficient_fails(self) -> None:
        task = _make_task(goal="Write summary", min_word_count=10)
        result = _rlm_verify(task, "Too short")
        assert result.passed is False
        assert "min_word_count:10" in result.reasons

    def test_goal_plus_multiple_mechanical_all_must_pass(self) -> None:
        task = _make_task(
            goal="Complete analysis",
            required_substrings=["Einstein", "1921"],
            required_regex=[r"physicist"],
            min_word_count=5,
        )
        result = _rlm_verify(task, "Einstein was a physicist in 1921 period")
        assert result.passed is True

    def test_goal_plus_multiple_mechanical_partial_fail(self) -> None:
        task = _make_task(
            goal="Complete analysis",
            required_substrings=["Einstein", "1921"],
            required_regex=[r"physicist"],
        )
        # Missing "1921"
        result = _rlm_verify(task, "Einstein was a physicist")
        assert result.passed is False
        assert "missing_substring:1921" in result.reasons


# =============================================================================
# Invalid Regex Handling
# =============================================================================


class TestInvalidRegex:
    """Invalid regex patterns should fail gracefully, not crash."""

    def test_unterminated_character_class(self) -> None:
        task = _make_task(required_regex=["[abc"])
        result = _rlm_verify(task, "some output")
        assert result.passed is False
        assert any("invalid_regex" in r for r in result.reasons)

    def test_invalid_escape_sequence(self) -> None:
        task = _make_task(required_regex=[r"\k"])  # Invalid escape
        result = _rlm_verify(task, "some output")
        assert result.passed is False
        assert any("invalid_regex" in r for r in result.reasons)

    def test_multiple_invalid_patterns(self) -> None:
        task = _make_task(required_regex=["[invalid", "(unclosed"])
        result = _rlm_verify(task, "some output")
        assert result.passed is False
        # Both should be reported
        invalid_reasons = [r for r in result.reasons if "invalid_regex" in r]
        assert len(invalid_reasons) == 2

    def test_valid_and_invalid_mixed(self) -> None:
        task = _make_task(required_regex=[r"\d+", "[invalid"])
        result = _rlm_verify(task, "123")
        assert result.passed is False
        # Valid pattern passes, invalid fails
        assert any("invalid_regex" in r for r in result.reasons)

    def test_chat_engine_invalid_regex(self) -> None:
        """Chat Engine also handles invalid regex gracefully."""
        task = _make_task(required_regex=["[invalid"])
        engine = Engine()
        result = engine._verify_output(task, "some output")
        assert result.passed is False
        assert any("invalid_regex" in r for r in result.reasons)


# =============================================================================
# Case Sensitivity
# =============================================================================


class TestCaseSensitivity:
    """Test case_insensitive flag for substrings and regex."""

    def test_substring_case_sensitive_by_default(self) -> None:
        task = _make_task(required_substrings=["Einstein"])
        result = _rlm_verify(task, "einstein was smart")
        assert result.passed is False
        assert "missing_substring:Einstein" in result.reasons

    def test_substring_case_insensitive(self) -> None:
        task = _make_task(
            required_substrings=["Einstein"],
            case_insensitive=True,
        )
        result = _rlm_verify(task, "EINSTEIN was smart")
        assert result.passed is True

    def test_regex_case_sensitive_by_default(self) -> None:
        task = _make_task(required_regex=[r"Einstein"])
        result = _rlm_verify(task, "EINSTEIN")
        assert result.passed is False

    def test_regex_case_insensitive(self) -> None:
        task = _make_task(
            required_regex=[r"Einstein"],
            case_insensitive=True,
        )
        result = _rlm_verify(task, "EINSTEIN")
        assert result.passed is True

    def test_mixed_case_substrings(self) -> None:
        task = _make_task(
            required_substrings=["ABC", "xyz"],
            case_insensitive=True,
        )
        result = _rlm_verify(task, "abc XYZ")
        assert result.passed is True


# =============================================================================
# min_word_count Edge Cases
# =============================================================================


class TestMinWordCount:
    """Test min_word_count boundary conditions."""

    def test_exact_word_count_passes(self) -> None:
        task = _make_task(min_word_count=3)
        result = _rlm_verify(task, "one two three")
        assert result.passed is True

    def test_one_below_word_count_fails(self) -> None:
        task = _make_task(min_word_count=3)
        result = _rlm_verify(task, "one two")
        assert result.passed is False
        assert "min_word_count:3" in result.reasons

    def test_one_above_word_count_passes(self) -> None:
        task = _make_task(min_word_count=3)
        result = _rlm_verify(task, "one two three four")
        assert result.passed is True

    def test_punctuation_not_counted_as_words(self) -> None:
        """Punctuation attached to words doesn't increase count."""
        task = _make_task(min_word_count=3)
        # "one," "two." "three!" = 3 words
        result = _rlm_verify(task, "one, two. three!")
        assert result.passed is True

    def test_multiple_spaces_dont_create_words(self) -> None:
        task = _make_task(min_word_count=3)
        result = _rlm_verify(task, "one    two")
        assert result.passed is False  # Only 2 words

    def test_newlines_separate_words(self) -> None:
        task = _make_task(min_word_count=3)
        result = _rlm_verify(task, "one\ntwo\nthree")
        assert result.passed is True


def test_min_word_count_fuzz() -> None:
    # Covers min_word_count behavior across randomized word counts.
    rng = random.Random(0)
    for _ in range(50):
        min_words = rng.randint(1, 6)
        word_count = rng.randint(0, 8)
        words = ["w"] * word_count
        output = " ".join(words)
        task = _make_task(min_word_count=min_words)
        result = _rlm_verify(task, output)
        if not output.strip():
            assert result.passed is False
            assert "no_output" in result.reasons
        else:
            assert result.passed == (word_count >= min_words)


# =============================================================================
# Substring Edge Cases
# =============================================================================


class TestSubstringEdgeCases:
    """Test substring matching edge cases."""

    def test_multiple_substrings_all_required(self) -> None:
        task = _make_task(required_substrings=["A", "B", "C"])
        result = _rlm_verify(task, "A and B and C")
        assert result.passed is True

    def test_multiple_substrings_one_missing(self) -> None:
        task = _make_task(required_substrings=["A", "B", "C"])
        result = _rlm_verify(task, "A and B")
        assert result.passed is False
        assert "missing_substring:C" in result.reasons

    def test_substring_at_start(self) -> None:
        task = _make_task(required_substrings=["START"])
        result = _rlm_verify(task, "START of output")
        assert result.passed is True

    def test_substring_at_end(self) -> None:
        task = _make_task(required_substrings=["END"])
        result = _rlm_verify(task, "output at END")
        assert result.passed is True

    def test_substring_with_special_chars(self) -> None:
        task = _make_task(required_substrings=["$100", "50%"])
        result = _rlm_verify(task, "Cost is $100, discount 50%")
        assert result.passed is True

    def test_unicode_substring(self) -> None:
        task = _make_task(required_substrings=["caf\u00e9", "\u4e2d\u6587"])
        result = _rlm_verify(task, "Visit the caf\u00e9 for \u4e2d\u6587 food")
        assert result.passed is True

    def test_empty_substring_always_matches(self) -> None:
        """Empty string is found in any output."""
        task = _make_task(required_substrings=[""])
        result = _rlm_verify(task, "anything")
        assert result.passed is True


# =============================================================================
# Regex Edge Cases
# =============================================================================


class TestRegexEdgeCases:
    """Test regex matching edge cases."""

    def test_multiline_regex(self) -> None:
        task = _make_task(required_regex=[r"^start"])
        # MULTILINE flag is set, ^ matches line starts
        result = _rlm_verify(task, "first\nstart of line")
        assert result.passed is True

    def test_special_regex_chars(self) -> None:
        task = _make_task(required_regex=[r"\$\d+\.\d{2}"])
        result = _rlm_verify(task, "Price: $19.99")
        assert result.passed is True

    def test_multiple_regex_all_required(self) -> None:
        task = _make_task(required_regex=[r"\d+", r"[A-Z]+"])
        result = _rlm_verify(task, "ABC 123")
        assert result.passed is True

    def test_multiple_regex_one_missing(self) -> None:
        task = _make_task(required_regex=[r"\d+", r"[A-Z]+"])
        result = _rlm_verify(task, "abc 123")
        assert result.passed is False

    def test_empty_regex_matches_everything(self) -> None:
        task = _make_task(required_regex=[""])
        result = _rlm_verify(task, "anything")
        assert result.passed is True


# =============================================================================
# Strong Criteria Detection
# =============================================================================


class TestStrongCriteriaDetection:
    """Test _has_strong_success_criteria function."""

    def test_min_word_count_1_is_weak(self) -> None:
        criteria = SuccessCriteria(min_word_count=1)
        assert _has_strong_success_criteria(criteria) is False

    def test_min_word_count_2_is_strong(self) -> None:
        criteria = SuccessCriteria(min_word_count=2)
        assert _has_strong_success_criteria(criteria) is True

    def test_required_substrings_is_strong(self) -> None:
        criteria = SuccessCriteria(required_substrings=["x"])
        assert _has_strong_success_criteria(criteria) is True

    def test_required_regex_is_strong(self) -> None:
        criteria = SuccessCriteria(required_regex=[r"\d"])
        assert _has_strong_success_criteria(criteria) is True

    def test_goal_is_strong(self) -> None:
        criteria = SuccessCriteria(goal="Find answer")
        assert _has_strong_success_criteria(criteria) is True

    def test_goal_plus_weak_mechanical_is_strong(self) -> None:
        """Goal makes it strong even with min_word_count=1."""
        criteria = SuccessCriteria(goal="Find answer", min_word_count=1)
        assert _has_strong_success_criteria(criteria) is True


# =============================================================================
# Chat Engine vs RLM Engine Differences
# =============================================================================


class TestEnginesDifferences:
    """Test differences between Chat and RLM Engine verification."""

    def test_chat_engine_ignores_goal(self) -> None:
        """Chat Engine doesn't have FINAL mechanism, so goal is ignored."""
        task = _make_task(goal="Find answer", required_substrings=["must-have"])
        engine = Engine()
        # Chat Engine only checks mechanical criteria
        result = engine._verify_output(task, "output without the required string")
        assert result.passed is False
        assert "missing_substring:must-have" in result.reasons

    def test_rlm_with_goal_only_trusts_model(self) -> None:
        """RLM with goal-only trusts model's FINAL judgment."""
        task = _make_task(goal="Find answer")
        result = _rlm_verify(task, "any output works")
        assert result.passed is True

    def test_both_engines_check_empty_output(self) -> None:
        """Both engines reject empty output."""
        task = _make_task(min_word_count=1)

        chat_result = Engine()._verify_output(task, "")
        rlm_result = _rlm_verify(task, "")

        assert chat_result.passed is False
        assert rlm_result.passed is False
        assert "no_output" in chat_result.reasons
        assert "no_output" in rlm_result.reasons

    def test_both_engines_handle_invalid_regex(self) -> None:
        """Both engines handle invalid regex gracefully."""
        task = _make_task(required_regex=["[invalid"])

        chat_result = Engine()._verify_output(task, "output")
        rlm_result = _rlm_verify(task, "output")

        assert chat_result.passed is False
        assert rlm_result.passed is False
        assert any("invalid_regex" in r for r in chat_result.reasons)
        assert any("invalid_regex" in r for r in rlm_result.reasons)


# =============================================================================
# Integration Tests with Full Task Execution
# =============================================================================


class TestIntegrationVerification:
    """Integration tests using actual Engine.run() and RLMEngine.run()."""

    def test_chat_engine_verification_failure_in_report(self) -> None:
        """Chat Engine reports verification failure correctly."""
        from tests.providers.mock_provider import MockProvider

        provider = MockProvider(main_outputs=["wrong output"])
        task = TaskSpec(
            task_id="test",
            input_text="test",
            model="mock",
            budget=Budget(max_tokens=100),
            success_criteria=SuccessCriteria(required_substrings=["must-have"]),
        )
        engine = Engine()
        report = engine.run(task, provider)

        assert report.success is False
        assert report.verification.passed is False
        assert "missing_substring:must-have" in report.verification.reasons

    def test_rlm_goal_based_success(self) -> None:
        """RLM with goal-only succeeds when FINAL is called."""
        from tests.providers.mock_provider import MockProvider

        provider = MockProvider(main_outputs=['```python\nFINAL("done")\n```'])
        task = TaskSpec(
            task_id="test",
            input_text="do something",
            model="mock",
            budget=Budget(max_tokens=100),
            success_criteria=SuccessCriteria(goal="Complete the task"),
        )
        engine = RLMEngine(max_steps=1)
        report = engine.run(task, provider, data="context")

        assert report.success is True
        assert report.answer == "done"

    def test_rlm_rejects_empty_final(self) -> None:
        """RLM rejects FINAL with empty content."""
        from tests.providers.mock_provider import MockProvider

        # First output: FINAL with empty string (rejected)
        # Second output: FINAL with content (accepted)
        provider = MockProvider(
            main_outputs=[
                '```python\nFINAL("")\n```',
                '```python\nFINAL("actual answer")\n```',
            ]
        )
        task = TaskSpec(
            task_id="test",
            input_text="answer",
            model="mock",
            budget=Budget(max_tokens=100),
            success_criteria=SuccessCriteria(required_substrings=["actual"]),
        )
        engine = RLMEngine(max_steps=2)
        report = engine.run(task, provider, data="context")

        assert report.success is True
        assert report.answer == "actual answer"


# =============================================================================
# Robustness Tests
# =============================================================================


class TestRobustness:
    """Tests for edge cases and potential attack vectors."""

    def test_very_long_output(self) -> None:
        """Verification handles very long output."""
        task = _make_task(required_substrings=["needle"])
        output = "x" * 1_000_000 + "needle" + "y" * 1_000_000
        result = _rlm_verify(task, output)
        assert result.passed is True

    def test_output_with_null_bytes(self) -> None:
        """Verification handles output with null bytes."""
        task = _make_task(required_substrings=["test"])
        result = _rlm_verify(task, "test\x00value")
        assert result.passed is True

    def test_output_with_control_characters(self) -> None:
        """Verification handles control characters."""
        task = _make_task(min_word_count=2)
        # Control chars don't separate words in Python split()
        # "word1\x07\x08word2" is one token, need space to separate
        result = _rlm_verify(task, "word1\x07 \x08word2")
        assert result.passed is True

    def test_regex_dos_prevention(self) -> None:
        """Regex doesn't cause catastrophic backtracking."""
        # This pattern could cause ReDoS with naive implementation
        task = _make_task(required_regex=[r"(a+)+b"])
        # Input that would cause backtracking
        result = _rlm_verify(task, "a" * 30 + "c")
        assert result.passed is False  # Should complete quickly, not hang

    def test_many_substrings(self) -> None:
        """Verification handles many required substrings."""
        substrings = [f"item{i}" for i in range(100)]
        task = _make_task(required_substrings=substrings)
        output = " ".join(substrings)
        result = _rlm_verify(task, output)
        assert result.passed is True

    def test_many_regex_patterns(self) -> None:
        """Verification handles many regex patterns."""
        patterns = [rf"pattern{i}" for i in range(100)]
        task = _make_task(required_regex=patterns)
        output = " ".join([f"pattern{i}" for i in range(100)])
        result = _rlm_verify(task, output)
        assert result.passed is True

    def test_unicode_edge_cases(self) -> None:
        """Verification handles various Unicode edge cases."""
        task = _make_task(
            required_substrings=[
                "\U0001f600",  # Emoji
                "\u200b",  # Zero-width space
                "\ufeff",  # BOM
            ]
        )
        result = _rlm_verify(task, "\U0001f600\u200b\ufeff")
        assert result.passed is True

    def test_combining_characters(self) -> None:
        """Verification handles combining characters correctly."""
        # e + combining acute = \u00e9
        task = _make_task(required_substrings=["caf\u00e9"])
        result = _rlm_verify(task, "I love caf\u00e9")
        assert result.passed is True
