"""
Output verification: shared logic for chat Engine and RLMEngine.

Extracted to deduplicate verification between engine.py and rlm/engine.py.
Single source of truth for success criteria checking.
"""

from __future__ import annotations

import re
import signal
from contextlib import contextmanager
from typing import Generator

from enzu.models import SuccessCriteria, VerificationResult

_REGEX_TIMEOUT_SECONDS = 2


@contextmanager
def _regex_timeout(
    seconds: int = _REGEX_TIMEOUT_SECONDS,
) -> Generator[None, None, None]:
    """Guard against catastrophic regex backtracking via alarm signal."""

    def _handler(signum: int, frame: object) -> None:
        raise TimeoutError("regex timed out")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def verify_output(
    criteria: SuccessCriteria,
    output_text: str,
    *,
    goal_based_trust: bool = False,
) -> VerificationResult:
    """
    Verify output against success criteria.

    Args:
        criteria: The success criteria to check against
        output_text: The output to verify
        goal_based_trust: If True and criteria has goal but no mechanical checks,
                         trust the model's judgment (RLM mode).
                         If False, always run mechanical checks (chat mode).

    Three modes:
    1. Goal-only (RLM): model's FINAL() call is the verification. Trust model.
    2. Mechanical-only: check required_substrings, required_regex, min_word_count.
    3. Goal + Mechanical: both must pass. Goal = soft target, mechanical = hard.

    Empty output always fails.
    """
    reasons: list[str] = []
    passed = True

    # Empty output is always a failure.
    if not output_text or not output_text.strip():
        return VerificationResult(passed=False, reasons=["no_output"])

    # Check if mechanical criteria exist.
    has_mechanical = bool(
        criteria.required_substrings
        or criteria.required_regex
        or (criteria.min_word_count and criteria.min_word_count > 1)
    )

    # Goal-only mode: model self-judges. FINAL() with non-empty output = success.
    # Only applies when goal_based_trust=True (RLM mode).
    if goal_based_trust and criteria.goal and not has_mechanical:
        return VerificationResult(passed=True, reasons=[])

    # Mechanical verification: predefined checks.
    case_insensitive = criteria.case_insensitive
    check_text = output_text.casefold() if case_insensitive else output_text

    for required in criteria.required_substrings:
        needle = required.casefold() if case_insensitive else required
        if needle not in check_text:
            passed = False
            reasons.append(f"missing_substring:{required}")

    regex_flags = re.MULTILINE | (re.IGNORECASE if case_insensitive else 0)
    for pattern in criteria.required_regex:
        try:
            with _regex_timeout():
                if re.search(pattern, output_text, regex_flags) is None:
                    passed = False
                    reasons.append(f"missing_regex:{pattern}")
        except re.error as e:
            # Invalid regex should fail verification, not crash.
            passed = False
            reasons.append(f"invalid_regex:{pattern}:{e}")
        except TimeoutError:
            passed = False
            reasons.append(f"regex_timeout:{pattern}")

    if criteria.min_word_count:
        word_count = len(output_text.split())
        if word_count < criteria.min_word_count:
            passed = False
            reasons.append(f"min_word_count:{criteria.min_word_count}")

    return VerificationResult(passed=passed, reasons=reasons)
