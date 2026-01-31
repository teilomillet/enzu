"""
RLM feedback generation: error hints, code analysis, structured feedback.

Extracted from engine.py for modularity. Single responsibility:
translate execution results into actionable guidance for the model.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from enzu.models import StepFeedback
from enzu.repl import SAFE_HELPERS


# Error pattern → actionable hint mapping
ERROR_HINTS = {
    "KeyError": "Use safe_get(d, key, default) for dict access",
    "'NoneType' object is not subscriptable": "Use safe_get(d, key, default) for dict access",
    "'NoneType' object is not iterable": "Use safe_rows(context) to extract list safely",
    "object has no attribute": "Use safe_rows(context) to handle None/missing attributes",
    "not supported between instances": "Use safe_sort(context, key) for type-safe sorting",
    "'NoneType' object has no attribute": "Check for None before accessing attributes",
    "list index out of range": "Check list length before indexing",
    "Search tools unavailable": "Set EXA_API_KEY to enable Exa search tools",
}


def classify_error(error: Optional[str]) -> Optional[str]:
    """Map error message to actionable hint."""
    if not error:
        return None
    for pattern, hint in ERROR_HINTS.items():
        if pattern in error:
            return hint
    return None


def analyze_code_patterns(code: str) -> List[str]:
    """
    Detect anti-patterns in code.

    Returns warnings about patterns that waste compute or miss opportunities.
    """
    if not code:
        return []

    warnings = []

    # Over-delegation: llm_query inside a loop
    has_loop = re.search(r"(for|while)\s+.+:", code)
    if has_loop:
        post_loop = code[has_loop.end() :]
        if "llm_query" in post_loop:
            warnings.append(
                "llm_query inside loop (called N times at runtime). "
                "Batch chunks: llm_query(f'Process:\\n{{chunk1}}\\n{{chunk2}}')"
            )

    # Passing full context to llm_query without filtering
    if re.search(r"llm_query\([^)]*\b(context|data)\b[^)]*\)", code):
        if not re.search(r"(context|data)\[", code) and "for" not in code:
            warnings.append(
                "Passing full context to llm_query. "
                "Filter/chunk first: chunks = [context[1][i:i+1000] for i in range(0, len(context[1]), 1000)]"
            )

    # Code-doable tasks delegated to llm_query
    code_doable = ["count", "filter", "sort", "format", "join", "split", "len"]
    for keyword in code_doable:
        if re.search(rf"llm_query\([^)]*\b{keyword}\b[^)]*\)", code, re.I):
            warnings.append(
                f"'{keyword}' can be done in code. Reserve llm_query for semantic tasks "
                "(classification, summarization, interpretation)."
            )

    return warnings


def extract_code(model_output: str) -> Tuple[Optional[str], int]:
    """
    Extract code from model output.

    Returns (first_block, total_block_count).
    Takes first block because it usually contains setup code.
    """
    matches = re.findall(r"```(?:python|repl)?\s*\n(.*?)```", model_output, re.DOTALL)
    if not matches:
        return None, 0
    return matches[0].strip(), len(matches)


def build_feedback(
    stdout: str,
    error: Optional[str],
    code: Optional[str],
    block_count: int,
    extra_helpers: Optional[List[str]] = None,
) -> StepFeedback:
    """Build structured feedback from execution result."""
    violation = f"multiple_blocks:{block_count}" if block_count > 1 else None
    hint = classify_error(error)
    pattern_warnings = analyze_code_patterns(code) if code else []
    helpers = list(SAFE_HELPERS.keys())
    if extra_helpers:
        helpers.extend(extra_helpers)

    return StepFeedback(
        violation=violation,
        hint=hint,
        available_helpers=helpers,
        pattern_warnings=pattern_warnings,
        stdout=stdout,
        error=error,
    )


def format_feedback(feedback: StepFeedback) -> str:
    """Format StepFeedback into prompt section."""
    lines = []

    if feedback.violation:
        if feedback.violation.startswith("multiple_blocks"):
            count = feedback.violation.split(":")[1]
            lines.append(
                f"[VIOLATION] You wrote {count} code blocks. "
                "Only the first was executed. Consolidate into one block."
            )

    if feedback.rejection_reasons:
        reasons = ", ".join(feedback.rejection_reasons)
        lines.append(f"[REJECTED] Answer failed verification: {reasons}")

    if feedback.stdout:
        lines.append(f"[OUTPUT]\n{feedback.stdout}")

    if feedback.error:
        lines.append(f"[ERROR]\n{feedback.error}")

    if feedback.hint:
        lines.append(f"[HINT] {feedback.hint}")

    for warning in feedback.pattern_warnings:
        lines.append(f"[PATTERN] {warning}")

    return "\n".join(lines)
