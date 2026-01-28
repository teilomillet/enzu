from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from enzu.budget import count_tokens_exact

if TYPE_CHECKING:
    from enzu.models import Budget, TaskSpec


_INPUT_KEYS = ("input_tokens", "prompt_tokens")
_OUTPUT_KEYS = ("output_tokens", "completion_tokens")
_TOTAL_KEYS = ("total_tokens",)
_COST_KEYS = ("cost_usd", "cost", "total_cost", "total_cost_usd")


def _read_int(usage: Dict[str, Any], keys: tuple[str, ...]) -> Optional[int]:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return None


def _read_float(usage: Dict[str, Any], keys: tuple[str, ...]) -> Optional[float]:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _extract_text_from_content_part(part: Any) -> str:
    if part is None:
        return ""
    if isinstance(part, str):
        return part

    part_type = _get_field(part, "type")
    if part_type in {"input_text", "output_text", "summary_text", "text"}:
        text = _get_field(part, "text", "")
        return text if isinstance(text, str) else ""
    if part_type == "refusal":
        refusal = _get_field(part, "refusal", "")
        return refusal if isinstance(refusal, str) else ""
    if part_type in {"input_image", "input_file", "input_video"}:
        return f"[{part_type}]"

    text = _get_field(part, "text")
    if isinstance(text, str):
        return text
    refusal = _get_field(part, "refusal")
    if isinstance(refusal, str):
        return refusal

    output = _get_field(part, "output")
    if output is not None:
        return _extract_text_from_content(output)
    return ""


def _extract_text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    chunks = []
    for part in _as_list(content):
        text = _extract_text_from_content_part(part)
        if text:
            chunks.append(text)
    return "".join(chunks)


def _extract_text_from_item(item: Any) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item

    item_type = _get_field(item, "type")
    if item_type == "message":
        return _extract_text_from_content(_get_field(item, "content"))
    if item_type == "function_call":
        name = _get_field(item, "name") or ""
        arguments = _get_field(item, "arguments")
        if isinstance(arguments, str):
            return f"{name}({arguments})" if name else arguments
        if arguments is not None:
            return f"{name}({arguments})" if name else str(arguments)
        return name
    if item_type == "function_call_output":
        return _extract_text_from_content(_get_field(item, "output"))
    if item_type == "reasoning":
        return _extract_text_from_content(_get_field(item, "summary"))

    content = _get_field(item, "content")
    if content is not None:
        return _extract_text_from_content(content)
    text = _get_field(item, "text")
    return text if isinstance(text, str) else ""


def _extract_text_from_openresponses_input(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks = []
        for item in value:
            text = _extract_text_from_item(item)
            if text:
                chunks.append(text)
        return "\n".join(chunks)
    if isinstance(value, dict):
        return _extract_text_from_item(value)
    return str(value)


def build_task_input_text(task: TaskSpec) -> str:
    """
    Build the full input text sent to providers for token accounting.

    Mirrors OpenAICompatProvider._build_input to keep estimates aligned
    with actual requests.
    """
    responses_input = None
    instructions = None
    if isinstance(task.responses, dict) and task.responses:
        responses_input = task.responses.get("input")
        instructions = task.responses.get("instructions")

    include_success_criteria = True
    if responses_input is not None:
        content = _extract_text_from_openresponses_input(responses_input)
        include_success_criteria = False
        if not content:
            content = task.input_text
    else:
        content = task.input_text

    if isinstance(instructions, str) and instructions:
        content = f"{instructions}\n{content}" if content else instructions
    criteria_lines = []
    criteria = task.success_criteria

    if criteria.required_substrings:
        criteria_lines.append(
            "Required substrings: " + ", ".join(criteria.required_substrings)
        )
    if criteria.required_regex:
        criteria_lines.append(
            "Required regex: " + ", ".join(criteria.required_regex)
        )
    if criteria.min_word_count:
        criteria_lines.append(f"Minimum word count: {criteria.min_word_count}")
    if criteria.case_insensitive:
        criteria_lines.append("Case-insensitive checks: true")

    if include_success_criteria and criteria_lines:
        criteria_text = "\n".join(criteria_lines)
        content = f"{content}\n\nSuccess criteria:\n{criteria_text}"

    return content


def normalize_usage(
    usage: Dict[str, Any],
    *,
    input_text: Optional[str] = None,
    output_text: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Optional[float | int]]:
    """
    Normalize usage fields and backfill missing token counts when possible.

    Returns a dict with keys: input_tokens, output_tokens, total_tokens, cost_usd.
    Token fields are ints when available. cost_usd is a float when available.
    """
    if not isinstance(usage, dict):
        usage = {}

    input_tokens = _read_int(usage, _INPUT_KEYS)
    output_tokens = _read_int(usage, _OUTPUT_KEYS)
    total_tokens = _read_int(usage, _TOTAL_KEYS)
    cost_usd = _read_float(usage, _COST_KEYS)

    if input_tokens is None and input_text is not None:
        input_tokens = count_tokens_exact(input_text, model)
    if output_tokens is None and output_text is not None:
        output_tokens = count_tokens_exact(output_text, model)

    if total_tokens is None:
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
    }


def check_budget_limits(
    budget: "Budget",
    elapsed_seconds: float,
    output_tokens: Optional[int],
    total_tokens: Optional[int],
    cost_usd: Optional[float],
) -> List[str]:
    """
    Check which budget limits have been exceeded.

    Returns list of exceeded limit names: "max_seconds", "max_output_tokens",
    "max_total_tokens", "max_cost_usd".

    Consolidated from engine.py and rlm/engine.py to avoid duplication.
    """
    limits_exceeded: List[str] = []
    if budget.max_seconds and elapsed_seconds > budget.max_seconds:
        limits_exceeded.append("max_seconds")
    if budget.max_tokens and isinstance(output_tokens, int):
        if output_tokens > budget.max_tokens:
            limits_exceeded.append("max_output_tokens")
    if budget.max_total_tokens and isinstance(total_tokens, int):
        if total_tokens > budget.max_total_tokens:
            limits_exceeded.append("max_total_tokens")
    if budget.max_cost_usd and isinstance(cost_usd, (int, float)):
        if cost_usd > budget.max_cost_usd:
            limits_exceeded.append("max_cost_usd")
    return limits_exceeded
