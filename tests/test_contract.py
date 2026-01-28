from __future__ import annotations

import pytest

from enzu.contract import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MIN_WORD_COUNT,
    apply_task_defaults,
    has_budget_limit,
    has_success_check,
    task_spec_from_payload,
)
from enzu.models import Budget, SuccessCriteria, TaskSpec


def test_has_budget_limit_accepts_budget_objects() -> None:
    budget = Budget(max_tokens=10)
    assert has_budget_limit(budget) is True


def test_has_budget_limit_accepts_dicts() -> None:
    assert has_budget_limit({"max_total_tokens": 5}) is True
    assert has_budget_limit({"max_tokens": 0}) is False
    assert has_budget_limit({}) is False


def test_has_success_check_accepts_success_criteria_objects() -> None:
    criteria = SuccessCriteria(min_word_count=1)
    assert has_success_check(criteria) is True


def test_has_success_check_accepts_dicts() -> None:
    assert has_success_check({"required_substrings": ["ok"]}) is True
    assert has_success_check({"min_word_count": 0}) is False
    assert has_success_check({}) is False


def test_apply_task_defaults_sets_budget_and_criteria() -> None:
    # Covers default application for missing budget and success_criteria.
    task = {"input_text": "hi", "model": "m"}
    updated = apply_task_defaults(task)

    assert updated["budget"]["max_tokens"] == DEFAULT_MAX_OUTPUT_TOKENS
    assert updated["success_criteria"]["min_word_count"] == DEFAULT_MIN_WORD_COUNT


def test_task_spec_from_payload_uses_model_override() -> None:
    payload = {
        "task": {
            "task_id": "t",
            "input_text": "hi",
            "budget": {"max_tokens": 3},
            "success_criteria": {"min_word_count": 1},
        },
        "model": "ignored",
    }
    spec = task_spec_from_payload(payload, model_override="override")
    assert isinstance(spec, TaskSpec)
    assert spec.model == "override"


def test_task_spec_from_payload_uses_top_level_model() -> None:
    payload = {
        "task": {
            "task_id": "t",
            "input_text": "hi",
            "budget": {"max_tokens": 3},
            "success_criteria": {"min_word_count": 1},
        },
        "model": "top-level",
    }
    spec = task_spec_from_payload(payload)
    assert spec.model == "top-level"


def test_task_spec_from_payload_requires_model() -> None:
    payload = {
        "task": {
            "task_id": "t",
            "input_text": "hi",
        }
    }
    with pytest.raises(ValueError, match="model is required"):
        task_spec_from_payload(payload)
