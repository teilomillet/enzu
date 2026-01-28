from __future__ import annotations

from typing import Any, Dict, Optional

from enzu.models import Budget, SuccessCriteria, TaskSpec

# Shared defaults for CLI and api.run() payloads.
DEFAULT_MAX_OUTPUT_TOKENS = 256
DEFAULT_MIN_WORD_COUNT = 1


def has_budget_limit(budget: Any) -> bool:
    if isinstance(budget, Budget):
        return True
    if not isinstance(budget, dict):
        return False
    return any(
        budget.get(key)
        for key in ("max_tokens", "max_total_tokens", "max_seconds", "max_cost_usd")
    )


def has_success_check(criteria: Any) -> bool:
    if isinstance(criteria, SuccessCriteria):
        return True
    if not isinstance(criteria, dict):
        return False
    return any(
        [
            criteria.get("required_substrings"),
            criteria.get("required_regex"),
            criteria.get("min_word_count"),
            criteria.get("goal"),  # Goal-based success
        ]
    )


def apply_task_defaults(task_data: Dict[str, Any]) -> Dict[str, Any]:
    if not has_budget_limit(task_data.get("budget")):
        task_data["budget"] = {
            "max_tokens": task_data.get("max_output_tokens")
            or DEFAULT_MAX_OUTPUT_TOKENS
        }
    if not has_success_check(task_data.get("success_criteria")):
        task_data["success_criteria"] = {"min_word_count": DEFAULT_MIN_WORD_COUNT}
    return task_data


def task_spec_from_payload(
    payload: Dict[str, Any],
    *,
    model_override: Optional[str] = None,
) -> TaskSpec:
    # Normalize JSON payloads so CLI and Python share the same TaskSpec rules.
    task_data = payload.get("task", payload)
    if not isinstance(task_data, dict):
        raise ValueError("task must be a JSON object.")
    task_data = dict(task_data)
    if model_override is not None:
        task_data["model"] = model_override
    elif "model" not in task_data and payload.get("model"):
        task_data["model"] = payload["model"]
    if "model" not in task_data:
        raise ValueError("model is required in task payload or via model=.")
    apply_task_defaults(task_data)
    return TaskSpec.model_validate(task_data)
