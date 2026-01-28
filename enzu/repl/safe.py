"""Optional safe helpers for RLM sandbox. Import and add to namespace if wanted."""
from __future__ import annotations

from typing import Any, Optional


def safe_get(d: Any, key: str, default: Any = None) -> Any:
    """Dict access that never crashes."""
    if d is None or not isinstance(d, dict):
        return default
    return d.get(key, default)


def safe_rows(data: Any) -> list:
    """Extract list from any structure. Never crashes."""
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("rows", "items", "data", "results"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


def safe_sort(data: Any, key: Optional[str] = None, reverse: bool = True) -> list:
    """Sort safely. Never crashes."""
    rows = safe_rows(data)
    if not rows or not key:
        return rows
    try:
        return sorted(rows, key=lambda x: float(safe_get(x, key, 0)), reverse=reverse)
    except (TypeError, ValueError):
        return rows


SAFE_HELPERS = {
    "safe_get": safe_get,
    "safe_rows": safe_rows,
    "safe_sort": safe_sort,
}
