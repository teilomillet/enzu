from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from enzu.tools import exa
from enzu.tools.context import ctx_add, ctx_clear, ctx_get, ctx_sources, ctx_stats
from enzu.tools.exa import ExaResult, exa_contents, exa_cost, exa_reset_cost, exa_search, exa_similar
from enzu.tools.research import research

# Get the actual module object (not the function that shadows it in __init__.py)
research_module = sys.modules["enzu.tools.research"]


@dataclass
class _Call:
    name: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]


class _FakeExaClient:
    def __init__(self) -> None:
        self.calls: List[_Call] = []

    def search(self, *args: Any, **kwargs: Any) -> List[ExaResult]:
        self.calls.append(_Call("search", args, kwargs))
        return [
            ExaResult(
                url="https://example.com/a",
                title="Example A",
                text="alpha",
                score=0.9,
                published_date="2025-01-01",
            )
        ]

    def get_contents(self, urls: List[str], **kwargs: Any) -> List[ExaResult]:
        self.calls.append(_Call("contents", (urls,), kwargs))
        return [
            ExaResult(
                url=urls[0],
                title="Example C",
                text="content",
                score=0.7,
                published_date="2025-01-02",
            )
        ]

    def find_similar(self, url: str, **kwargs: Any) -> List[ExaResult]:
        self.calls.append(_Call("similar", (url,), kwargs))
        return [
            ExaResult(
                url="https://example.com/similar",
                title="Example S",
                text="similar",
                score=0.6,
                published_date="2025-01-03",
            )
        ]


def _install_fake_client(monkeypatch) -> _FakeExaClient:
    fake = _FakeExaClient()
    # Route Exa wrappers through a fake client to verify ctx_add behavior.
    monkeypatch.setattr(exa, "_get_client", lambda: fake)
    return fake


def test_exa_search_auto_accumulates_and_no_default_limit(monkeypatch) -> None:
    ctx_clear()
    fake = _install_fake_client(monkeypatch)

    exa_search("test query", num_results=1)

    sources = ctx_sources()
    stats = ctx_stats()

    # Low-level exa_* results should land in the shared context store.
    assert len(sources) == 1
    assert sources[0]["url"] == "https://example.com/a"
    assert sources[0]["published_date"] == "2025-01-01"
    assert "test query" in stats["queries"]

    # Default max_characters is None, so no per-result cap is imposed.
    assert fake.calls[0].kwargs.get("max_characters") is None


def test_exa_contents_auto_accumulates(monkeypatch) -> None:
    ctx_clear()
    _install_fake_client(monkeypatch)

    exa_contents(["https://example.com/c"])

    sources = ctx_sources()
    stats = ctx_stats()

    # Contents fetch should still populate the context store for ctx_get().
    assert len(sources) == 1
    assert sources[0]["url"] == "https://example.com/c"
    assert stats["queries"] == []


def test_exa_similar_auto_accumulates(monkeypatch) -> None:
    ctx_clear()
    _install_fake_client(monkeypatch)

    exa_similar("https://example.com/seed")

    sources = ctx_sources()
    stats = ctx_stats()

    # Similar search should add sources and tag the seed URL as the query.
    assert len(sources) == 1
    assert sources[0]["url"] == "https://example.com/similar"
    assert "https://example.com/seed" in stats["queries"]


def test_research_auto_accumulates_and_published_in_ctx_get(monkeypatch) -> None:
    ctx_clear()

    def _fake_search(*_args: Any, **_kwargs: Any) -> List[Dict[str, Any]]:
        return [
            {
                "url": "https://example.com/r",
                "title": "Example R",
                "text": "research text",
                "score": 0.8,
                "published_date": "2025-01-05",
            }
        ]

    # Force research() to use a stub search source for deterministic context behavior.
    monkeypatch.setattr(research_module, "_search_fn", _fake_search)
    monkeypatch.setattr(research_module, "_news_fn", None)
    monkeypatch.setattr(research_module, "_papers_fn", None)
    monkeypatch.setattr(research_module, "_similar_fn", None)

    research("topic", include_news=False, include_papers=False)

    sources = ctx_sources()
    stats = ctx_stats()
    context_text = ctx_get(max_chars=2000, max_chars_per_source=None)

    assert len(sources) == 1
    assert sources[0]["published_date"] == "2025-01-05"
    assert "topic" in stats["queries"]
    assert "Published: 2025-01-05" in context_text


def test_ctx_get_no_truncation_when_none() -> None:
    ctx_clear()
    long_text = "x" * 2048 + "END"

    # Context output must include full text when no per-source cap is set.
    ctx_add(
        [
            {
                "url": "https://example.com/full",
                "title": "Full Text",
                "text": long_text,
                "score": 0.9,
                "published_date": "2025-01-06",
            }
        ],
        query="full-text",
    )

    context_text = ctx_get(max_chars=10000, max_chars_per_source=None)

    assert "END" in context_text


def test_research_handles_none_scores(monkeypatch) -> None:
    ctx_clear()

    def _fake_search(*_args: Any, **_kwargs: Any) -> List[Dict[str, Any]]:
        return [
            {
                "url": "https://example.com/none",
                "title": "No Score",
                "text": "text",
                "score": None,
                "published_date": "2025-01-07",
            }
        ]

    monkeypatch.setattr(research_module, "_search_fn", _fake_search)
    monkeypatch.setattr(research_module, "_news_fn", None)
    monkeypatch.setattr(research_module, "_papers_fn", None)
    monkeypatch.setattr(research_module, "_similar_fn", None)

    result = research("none-score", include_news=False, include_papers=False, min_score=0.0)

    assert result["stats"]["kept"] == 1
    sources = ctx_sources()
    assert sources[0]["score"] == 0.0


def test_context_dedupes_across_research_and_exa(monkeypatch) -> None:
    ctx_clear()
    _install_fake_client(monkeypatch)

    def _fake_search(*_args: Any, **_kwargs: Any) -> List[Dict[str, Any]]:
        return [
            {
                "url": "https://example.com/a",
                "title": "Example A",
                "text": "alpha",
                "score": 0.9,
                "published_date": "2025-01-01",
            }
        ]

    # Use the same URL in research() and exa_search() to verify dedupe.
    monkeypatch.setattr(research_module, "_search_fn", _fake_search)
    monkeypatch.setattr(research_module, "_news_fn", None)
    monkeypatch.setattr(research_module, "_papers_fn", None)
    monkeypatch.setattr(research_module, "_similar_fn", None)

    research("dup", include_news=False, include_papers=False)
    exa_search("dup query", num_results=1)

    sources = ctx_sources()
    stats = ctx_stats()

    assert len(sources) == 1
    assert sources[0]["url"] == "https://example.com/a"
    assert "dup" in stats["queries"]
    assert "dup query" in stats["queries"]


def test_exa_cost_tracking_advances_with_searches(monkeypatch) -> None:
    ctx_clear()
    exa_reset_cost()
    _install_fake_client(monkeypatch)

    exa_search("cost query", num_results=1)
    exa_contents(["https://example.com/c"])

    costs = exa_cost()

    # Each wrapper call increments session search_count.
    assert costs["search_count"] == 2
