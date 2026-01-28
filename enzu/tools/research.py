"""
Research helpers for RLM sandbox.

These functions simplify common research patterns, making it easier
for the model to gather and process web content effectively.

Key feature: All research functions AUTO-ACCUMULATE results to the context
store, so the RLM context grows as the model searches. No manual ctx_add() needed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .. import telemetry

# Will be populated when search tools are available
_search_fn = None
_news_fn = None
_papers_fn = None
_similar_fn = None

# None keeps full text; ctx_get controls how much enters prompts.
DEFAULT_MAX_CHARACTERS: Optional[int] = None


def _ensure_search():
    """Ensure search functions are loaded."""
    global _search_fn, _news_fn, _papers_fn, _similar_fn
    if _search_fn is None:
        from enzu.tools.exa import exa_search, exa_news, exa_papers, exa_similar
        _search_fn = exa_search
        _news_fn = exa_news
        _papers_fn = exa_papers
        _similar_fn = exa_similar


def _auto_accumulate(sources: List[Dict[str, Any]], query: Optional[str] = None) -> int:
    """Auto-add sources to context store. Returns count added."""
    try:
        from enzu.tools.context import ctx_add
        return ctx_add(sources, query=query)
    except Exception:
        # Context store not available, skip accumulation
        return 0


def research(
    topic: str,
    *,
    num_results: int = 10,
    min_score: float = 0.5,
    max_chars_per_source: Optional[int] = None,
    include_news: bool = True,
    include_papers: bool = False,
    days_back: int = 7,  # Default to 7 days for newsletter freshness
    max_characters: Optional[int] = DEFAULT_MAX_CHARACTERS,
) -> Dict[str, Any]:
    """
    Research a topic and return structured results.

    This is a high-level helper that:
    1. Searches for content on the topic
    2. Optionally adds news and/or papers
    3. Filters by quality score
    4. Returns structured data ready for synthesis

    Args:
        topic: The topic to research
        num_results: Number of results per search (default 10)
        min_score: Minimum quality score to keep (default 0.5)
        max_chars_per_source: Limit text per source (None = no per-source truncation)
        include_news: Also search news (default True)
        include_papers: Also search papers (default False)
        days_back: Limit to recent content (default 7 days for freshness)
        max_characters: Max text length fetched per result (None = no limit)

    Returns:
        Dict with:
        - sources: List of {title, url, text, score}
        - context: Combined text ready for llm_query
        - stats: {total_found, kept, context_length}

    Example:
        result = research("AI agents", include_news=True, days_back=7)
        print(result["stats"])
        summary = llm_query(f"Summarize:\\n{result['context']}")
    """
    _ensure_search()
    telemetry.log(
        "info",
        "research_start",
        topic=topic,
        num_results=num_results,
        min_score=min_score,
        include_news=include_news,
        include_papers=include_papers,
        days_back=days_back,
    )

    all_results = []

    # Main search
    if _search_fn is not None:
        try:
            # Spans keep research subcalls visible in Logfire traces.
            with telemetry.span(
                "research.search",
                topic=topic,
                num_results=num_results,
                days_back=days_back,
            ):
                results = _search_fn(
                    topic,
                    num_results=num_results,
                    days_back=days_back,
                    max_characters=max_characters,
                )
            all_results.extend(results)
        except Exception as e:
            print(f"Search error: {e}")

    # News search
    if include_news and _news_fn is not None:
        try:
            with telemetry.span(
                "research.news",
                topic=topic,
                num_results=num_results // 2,
                days_back=days_back,
            ):
                news = _news_fn(
                    topic,
                    num_results=num_results // 2,
                    days_back=days_back,
                    max_characters=max_characters,
                )
            all_results.extend(news)
        except Exception as e:
            print(f"News search error: {e}")

    # Papers search
    if include_papers and _papers_fn is not None:
        try:
            with telemetry.span(
                "research.papers",
                topic=topic,
                num_results=num_results // 2,
            ):
                papers = _papers_fn(
                    topic,
                    num_results=num_results // 2,
                    max_characters=max_characters,
                )
            all_results.extend(papers)
        except Exception as e:
            print(f"Papers search error: {e}")

    # Dedupe by URL
    seen_urls = set()
    unique_results = []
    for r in all_results:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)

    # Filter by score when scores are available.
    # Exa Auto search (default) deprecated scores in July 2025.
    # When scores are None, keep all results and preserve API order.
    has_scores = any(
        isinstance(r.get("score"), (int, float)) for r in unique_results
    )

    if has_scores:
        good_results = [
            r for r in unique_results
            if isinstance(r.get("score"), (int, float)) and r["score"] >= min_score
        ]
        good_results.sort(
            key=lambda x: x.get("score", 0.0),
            reverse=True,
        )
    else:
        # No scores returned (Auto search). Keep all results in API order.
        good_results = unique_results

    # Build sources and context
    sources = []
    context_parts = []
    for r in good_results:
        text = r.get("text") or ""
        truncated = text if max_chars_per_source is None else text[:max_chars_per_source]
        published_date = r.get("published_date")

        sources.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "text": truncated,
            "score": r.get("score", 0),
            "published_date": published_date,
        })

        date_line = f"Published: {published_date}" if published_date else "Published: unknown"
        context_parts.append(
            f"## {r.get('title', 'Untitled')}\n"
            f"Source: {r.get('url', '')}\n"
            f"{date_line}\n"
            f"{truncated}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Auto-accumulate into context store (best-effort)
    _auto_accumulate(sources, query=topic)

    telemetry.log(
        "info",
        "research_done",
        total_found=len(all_results),
        unique=len(unique_results),
        kept=len(good_results),
        context_length=len(context),
    )
    return {
        "sources": sources,
        "context": context,
        "stats": {
            "total_found": len(all_results),
            "unique": len(unique_results),
            "kept": len(good_results),
            "context_length": len(context),
        },
    }


def explore(
    seed_url: str,
    *,
    num_similar: int = 5,
    min_score: float = 0.5,
    max_chars_per_source: Optional[int] = None,
    max_characters: Optional[int] = DEFAULT_MAX_CHARACTERS,
) -> Dict[str, Any]:
    """
    Explore from a seed URL to find related content.

    Args:
        seed_url: Starting URL to explore from
        num_similar: Number of similar pages to find (default 5)
        min_score: Minimum quality score (default 0.5)
        max_chars_per_source: Text limit per source (None = no per-source truncation)
        max_characters: Max text length fetched per result (None = no limit)

    Returns:
        Same structure as research()

    Example:
        result = explore("https://arxiv.org/abs/...")
        print(f"Found {result['stats']['kept']} related articles")
    """
    _ensure_search()
    telemetry.log(
        "info",
        "explore_start",
        seed_url=seed_url,
        num_similar=num_similar,
        min_score=min_score,
    )

    if _similar_fn is None:
        return {"sources": [], "context": "", "stats": {"error": "Search functions not initialized"}}
    try:
        results = _similar_fn(
            seed_url,
            num_results=num_similar,
            max_characters=max_characters,
        )
    except Exception as e:
        print(f"Explore error: {e}")
        return {"sources": [], "context": "", "stats": {"error": str(e)}}

    # Filter by score when scores are available.
    # Exa similar search may not return scores in Auto mode.
    has_scores = any(isinstance(r.get("score"), (int, float)) for r in results)

    if has_scores:
        good_results = [
            r for r in results
            if isinstance(r.get("score"), (int, float)) and r["score"] >= min_score
        ]
    else:
        good_results = results

    # Build sources and context
    sources = []
    context_parts = []
    for r in good_results:
        text = r.get("text") or ""
        truncated = text if max_chars_per_source is None else text[:max_chars_per_source]
        published_date = r.get("published_date")

        sources.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "text": truncated,
            "score": r.get("score", 0),
            "published_date": published_date,
        })

        date_line = f"Published: {published_date}" if published_date else "Published: unknown"
        context_parts.append(
            f"## {r.get('title', 'Untitled')}\n"
            f"Source: {r.get('url', '')}\n"
            f"{date_line}\n"
            f"{truncated}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Auto-accumulate into context store (best-effort)
    _auto_accumulate(sources, query=seed_url)

    telemetry.log(
        "info",
        "explore_done",
        total_found=len(results),
        kept=len(good_results),
        context_length=len(context),
    )
    return {
        "sources": sources,
        "context": context,
        "stats": {
            "total_found": len(results),
            "kept": len(good_results),
            "context_length": len(context),
        },
    }


def format_sources(sources: List[Dict[str, Any]], max_sources: int = 10) -> str:
    """
    Format sources as a reference list for citations.

    Args:
        sources: List from research() or explore()
        max_sources: Max sources to include

    Returns:
        Formatted string with numbered references
    """
    lines = ["## Sources\n"]
    for i, s in enumerate(sources[:max_sources], 1):
        title = s.get("title", "Untitled")
        url = s.get("url", "")
        date = s.get("published_date", "")
        date_str = f" ({date[:10]})" if date else ""
        lines.append(f"{i}. [{title}]({url}){date_str}")
    return "\n".join(lines)


# Export for sandbox injection
RESEARCH_HELPERS = {
    "research": research,
    "explore": explore,
    "format_sources": format_sources,
}
