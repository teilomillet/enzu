"""
Context accumulator for RLM research sessions.

Manages accumulated research data across RLM steps and can persist
for later retrieval (useful for actu.me to avoid re-searching).

The context store tracks:
- Sources found (with deduplication)
- Accumulated text content
- Search queries performed (to avoid repeats)
- Timestamps for freshness
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import telemetry

@dataclass
class Source:
    """A single source with metadata."""
    url: str
    title: str
    text: str
    score: float
    published_date: Optional[str] = None
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())
    query: Optional[str] = None  # The query that found this source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "score": self.score,
            "published_date": self.published_date,
            "fetched_at": self.fetched_at,
            "query": self.query,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Source":
        raw_score = d.get("score")
        safe_score = raw_score if isinstance(raw_score, (int, float)) else 0.0
        return cls(
            url=d.get("url", ""),
            title=d.get("title", ""),
            text=d.get("text", ""),
            score=safe_score,
            published_date=d.get("published_date"),
            fetched_at=d.get("fetched_at", datetime.now().isoformat()),
            query=d.get("query"),
        )


@dataclass
class ContextStore:
    """
    Accumulates research context across RLM steps.

    Usage in sandbox:
        # Add sources from research
        ctx.add_sources(result["sources"], query="AI agents")

        # Get accumulated context for synthesis
        context = ctx.get_context(max_chars=50000)

        # Check what we have
        print(ctx.stats())

        # Save for later (actu.me persistence)
        ctx.save("research_session_123.json")
    """
    sources: List[Source] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    topic: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    search_cost_usd: float = 0.0

    def _url_hash(self, url: str) -> str:
        """Hash URL for deduplication."""
        return hashlib.md5(url.encode()).hexdigest()[:12]

    def _seen_urls(self) -> set:
        """Get set of seen URL hashes."""
        return {self._url_hash(s.url) for s in self.sources}

    def add_sources(
        self,
        sources: List[Dict[str, Any]],
        query: Optional[str] = None,
    ) -> int:
        """
        Add sources to the store (with deduplication).

        Args:
            sources: List of source dicts from research() or exa_search()
            query: The query that found these sources

        Returns:
            Number of new sources added
        """
        if query and query not in self.queries:
            self.queries.append(query)

        seen = self._seen_urls()
        added = 0

        for s in sources:
            url = s.get("url", "")
            url_hash = self._url_hash(url)

            if url_hash not in seen:
                raw_score = s.get("score")
                safe_score = raw_score if isinstance(raw_score, (int, float)) else 0.0
                self.sources.append(Source(
                    url=url,
                    title=s.get("title", ""),
                    text=s.get("text", ""),
                    score=safe_score,
                    published_date=s.get("published_date"),
                    query=query,
                ))
                seen.add(url_hash)
                added += 1

        return added

    def add_cost(self, cost_usd: float) -> None:
        """Track search API costs."""
        self.search_cost_usd += cost_usd

    def get_sources(
        self,
        min_score: float = 0.0,
        max_sources: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get sources as list of dicts, filtered and sorted."""
        filtered = [s for s in self.sources if s.score >= min_score]
        filtered.sort(key=lambda x: x.score, reverse=True)
        if max_sources:
            filtered = filtered[:max_sources]
        return [s.to_dict() for s in filtered]

    def get_context(
        self,
        max_chars: int = 50000,
        max_chars_per_source: Optional[int] = None,
        min_score: float = 0.0,
    ) -> str:
        """
        Get accumulated context as formatted text for llm_query.

        Args:
            max_chars: Total context length limit
        max_chars_per_source: Text limit per source (None = no per-source truncation)
        min_score: Minimum score filter

        Returns:
            Formatted context string
        """
        sources = self.get_sources(min_score=min_score)
        parts = []
        total_chars = 0

        for s in sources:
            if s["text"]:
                text = s["text"] if max_chars_per_source is None else s["text"][:max_chars_per_source]
            else:
                text = ""
            published = s.get("published_date") or "unknown"
            part = (
                f"## {s['title']}\n"
                f"Source: {s['url']}\n"
                f"Published: {published}\n"
                f"Score: {s['score']:.2f}\n"
                f"{text}"
            )

            if total_chars + len(part) > max_chars:
                break

            parts.append(part)
            total_chars += len(part)

        return "\n\n---\n\n".join(parts)

    def get_urls(self) -> List[str]:
        """Get all accumulated URLs."""
        return [s.url for s in self.sources]

    def has_query(self, query: str) -> bool:
        """Check if a query was already performed."""
        return query in self.queries

    def stats(self) -> Dict[str, Any]:
        """Get statistics about accumulated context."""
        total_text = sum(len(s.text) for s in self.sources)
        return {
            "num_sources": len(self.sources),
            "num_queries": len(self.queries),
            "total_text_chars": total_text,
            "search_cost_usd": self.search_cost_usd,
            "avg_score": sum(s.score for s in self.sources) / len(self.sources) if self.sources else 0,
            "queries": self.queries,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for persistence."""
        return {
            "topic": self.topic,
            "created_at": self.created_at,
            "search_cost_usd": self.search_cost_usd,
            "queries": self.queries,
            "sources": [s.to_dict() for s in self.sources],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContextStore":
        """Load from dict."""
        store = cls(
            topic=d.get("topic"),
            created_at=d.get("created_at", datetime.now().isoformat()),
            search_cost_usd=d.get("search_cost_usd", 0.0),
            queries=d.get("queries", []),
        )
        for s in d.get("sources", []):
            store.sources.append(Source.from_dict(s))
        return store

    def save(self, path: str) -> None:
        """Save to JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> "ContextStore":
        """Load from JSON file."""
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    def clear(self) -> None:
        """Clear all accumulated data."""
        self.sources = []
        self.queries = []
        self.search_cost_usd = 0.0


# Singleton for sandbox use
_context_store: Optional[ContextStore] = None


def get_context_store() -> ContextStore:
    """Get or create the session context store."""
    global _context_store
    if _context_store is None:
        _context_store = ContextStore()
    return _context_store


def reset_context_store() -> ContextStore:
    """Reset and return a fresh context store."""
    global _context_store
    _context_store = ContextStore()
    return _context_store


# Sandbox-friendly functions
def ctx_add(sources: List[Dict[str, Any]], query: Optional[str] = None) -> int:
    """Add sources to context store. Returns count of new sources added."""
    added = get_context_store().add_sources(sources, query)
    telemetry.log(
        "info",
        "ctx_add",
        added=added,
        query=query,
    )
    return added


def ctx_get(
    max_chars: int = 50000,
    min_score: float = 0.0,
    max_chars_per_source: Optional[int] = None,
) -> str:
    """Get accumulated context as formatted text."""
    return get_context_store().get_context(
        max_chars=max_chars,
        min_score=min_score,
        max_chars_per_source=max_chars_per_source,
    )


def ctx_stats() -> Dict[str, Any]:
    """Get context store statistics."""
    return get_context_store().stats()


def ctx_sources(min_score: float = 0.0, max_sources: int = 20) -> List[Dict[str, Any]]:
    """Get accumulated sources as list."""
    return get_context_store().get_sources(min_score=min_score, max_sources=max_sources)


def ctx_save(path: str) -> None:
    """Save context store to file for later use."""
    get_context_store().save(path)
    telemetry.log("info", "ctx_save", path=path)


def ctx_load(path: str) -> Dict[str, Any]:
    """Load context store from file. Returns stats."""
    global _context_store
    _context_store = ContextStore.load(path)
    telemetry.log("info", "ctx_load", path=path)
    return _context_store.stats()


def ctx_clear() -> None:
    """Clear accumulated context."""
    get_context_store().clear()


def ctx_has_query(query: str) -> bool:
    """Check if query was already performed (avoid re-searching)."""
    return get_context_store().has_query(query)


# Export for sandbox
CONTEXT_HELPERS = {
    "ctx_add": ctx_add,
    "ctx_get": ctx_get,
    "ctx_stats": ctx_stats,
    "ctx_sources": ctx_sources,
    "ctx_save": ctx_save,
    "ctx_load": ctx_load,
    "ctx_clear": ctx_clear,
    "ctx_has_query": ctx_has_query,
}
