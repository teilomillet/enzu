from __future__ import annotations

from enzu.usage import normalize_usage


def test_normalize_usage_sums_prompt_and_completion() -> None:
    usage = {"prompt_tokens": 10, "completion_tokens": 5}
    normalized = normalize_usage(usage)
    assert normalized["input_tokens"] == 10
    assert normalized["output_tokens"] == 5
    assert normalized["total_tokens"] == 15


def test_normalize_usage_prefers_total_tokens() -> None:
    usage = {"total_tokens": 7, "prompt_tokens": 3, "completion_tokens": 4}
    normalized = normalize_usage(usage)
    assert normalized["total_tokens"] == 7


def test_normalize_usage_backfills_from_text() -> None:
    normalized = normalize_usage({}, input_text="hello", output_text="world")
    input_tokens = normalized["input_tokens"]
    output_tokens = normalized["output_tokens"]
    if isinstance(input_tokens, int) and isinstance(output_tokens, int):
        assert normalized["total_tokens"] == input_tokens + output_tokens


def test_normalize_usage_reads_cost_variants() -> None:
    normalized = normalize_usage({"cost": 0.12})
    assert normalized["cost_usd"] == 0.12
