from __future__ import annotations

from enzu import Check, Limits
from enzu.api import _merge_check, _merge_limits


def test_limits_defaults() -> None:
    limits = Limits()
    assert limits.tokens is None
    assert limits.total is None
    assert limits.seconds is None
    assert limits.cost is None


def test_limits_with_values() -> None:
    limits = Limits(tokens=100, seconds=30.0, cost=0.01)
    assert limits.tokens == 100
    assert limits.seconds == 30.0
    assert limits.cost == 0.01


def test_check_defaults() -> None:
    check = Check()
    assert check.contains == []
    assert check.matches == []
    assert check.min_words is None


def test_check_with_values() -> None:
    check = Check(contains=["hello"], matches=[r"\d+"], min_words=5)
    assert check.contains == ["hello"]
    assert check.matches == [r"\d+"]
    assert check.min_words == 5


def test_merge_limits_inline_wins() -> None:
    base = Limits(tokens=100, seconds=30.0)
    merged = _merge_limits(base, tokens=200, seconds=None, cost=0.05)
    assert merged.tokens == 200
    assert merged.seconds == 30.0
    assert merged.cost == 0.05


def test_merge_limits_no_base() -> None:
    merged = _merge_limits(None, tokens=500, seconds=60.0, cost=None)
    assert merged.tokens == 500
    assert merged.seconds == 60.0
    assert merged.cost is None


def test_merge_check_inline_wins() -> None:
    base = Check(contains=["a"], min_words=10)
    merged = _merge_check(base, contains=["b", "c"], matches=None, min_words=None)
    assert merged.contains == ["b", "c"]
    assert merged.min_words == 10


def test_merge_check_no_base() -> None:
    merged = _merge_check(None, contains=["x"], matches=[r"test"], min_words=3)
    assert merged.contains == ["x"]
    assert merged.matches == [r"test"]
    assert merged.min_words == 3
