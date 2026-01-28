"""
Exa API client for web search and content retrieval.

Provides search capabilities for the RLM sandbox, allowing models to:
1. Search the web with semantic queries
2. Retrieve clean markdown content from results
3. Get highlights (key excerpts) relevant to the query
4. Filter by category, domain, date, and text content

Usage in sandbox:
    # Basic search
    results = exa_search("AI agents 2025", num_results=5)

    # Search news only, last 7 days
    results = exa_search("AI breakthrough", category="news", days_back=7)

    # Search specific domains
    results = exa_search("transformers", include_domains=["arxiv.org", "github.com"])

    # Find research papers
    results = exa_search("attention mechanism", category="research paper")
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional

import httpx

from .. import telemetry

# Valid categories for filtering
Category = Literal[
    "company",
    "research paper",
    "news",
    "pdf",
    "github",
    "tweet",
    "personal site",
    "financial report",
    "people",
    "linkedin profile",
]

# Valid search types
SearchType = Literal["auto", "neural", "fast", "deep"]

# Valid livecrawl options
LivecrawlOption = Literal["always", "fallback", "never", "preferred"]

# Default to full text; ctx_get controls prompt size.
DEFAULT_MAX_CHARACTERS: Optional[int] = None


@dataclass
class ExaResult:
    """Single search result with content."""
    url: str
    title: str
    text: Optional[str] = None
    highlights: Optional[List[str]] = None
    score: Optional[float] = None
    published_date: Optional[str] = None
    author: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for sandbox use."""
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "highlights": self.highlights,
            "score": self.score,
            "published_date": self.published_date,
            "author": self.author,
        }


class ExaClient:
    """
    Exa API client for search and content retrieval.

    Supports multiple search types:
    - auto: Let Exa choose best method (default)
    - neural: Semantic search using embeddings
    - fast: Streamlined, low-latency search
    - deep: Comprehensive search with query expansion
    """

    BASE_URL = "https://api.exa.ai"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or os.environ.get("EXA_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Exa API key required. Set EXA_API_KEY env var or pass api_key."
            )
        self._client = httpx.Client(
            base_url=self.BASE_URL,
            headers={
                "x-api-key": self._api_key,
                "Content-Type": "application/json",
            },
            timeout=60.0,  # Increased for deep searches
        )

    def search(
        self,
        query: str,
        *,
        num_results: int = 10,
        search_type: SearchType = "auto",
        category: Optional[Category] = None,
        text: bool = True,
        highlights: bool = False,
        max_characters: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        start_crawl_date: Optional[str] = None,
        end_crawl_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        livecrawl: LivecrawlOption = "fallback",
        use_autoprompt: bool = False,
    ) -> List[ExaResult]:
        """
        Search and retrieve content in one call.

        Args:
            query: Search query (can be natural language)
            num_results: Number of results to return (default 10, max 100)
            search_type: "auto", "neural", "fast", or "deep"
            category: Filter by content type (news, research paper, github, etc.)
            text: Include full text content as markdown (default True)
            highlights: Include key excerpts (default False)
            max_characters: Limit text length per result
            include_domains: Only search these domains (max 1200)
            exclude_domains: Exclude these domains (max 1200)
            start_published_date: Filter by publish date (ISO format)
            end_published_date: Filter by publish date (ISO format)
            start_crawl_date: Filter by crawl/discovery date (ISO format)
            end_crawl_date: Filter by crawl date (ISO format)
            include_text: Required strings in page text (max 1 string, 5 words)
            exclude_text: Excluded strings (max 1 string)
            livecrawl: "always", "fallback", "never", or "preferred"
            use_autoprompt: Let Exa enhance the query (default False)

        Returns:
            List of ExaResult with content
        """
        payload: Dict[str, Any] = {
            "query": query,
            "numResults": num_results,
            "type": search_type,
            "contents": {},
        }

        # Category filter
        if category:
            payload["category"] = category

        # Content options
        if text:
            text_opts: Dict[str, Any] = {}
            if max_characters:
                text_opts["maxCharacters"] = max_characters
            payload["contents"]["text"] = text_opts if text_opts else True

        if highlights:
            payload["contents"]["highlights"] = True

        # Livecrawl setting
        if livecrawl != "fallback":
            payload["contents"]["livecrawl"] = livecrawl

        # Domain filters
        if include_domains:
            payload["includeDomains"] = include_domains
        if exclude_domains:
            payload["excludeDomains"] = exclude_domains

        # Date filters (published)
        if start_published_date:
            payload["startPublishedDate"] = start_published_date
        if end_published_date:
            payload["endPublishedDate"] = end_published_date

        # Date filters (crawl)
        if start_crawl_date:
            payload["startCrawlDate"] = start_crawl_date
        if end_crawl_date:
            payload["endCrawlDate"] = end_crawl_date

        # Text filters
        if include_text:
            payload["includeText"] = include_text
        if exclude_text:
            payload["excludeText"] = exclude_text

        # Autoprompt
        if use_autoprompt:
            payload["useAutoprompt"] = True

        response = self._client.post("/search", json=payload)
        response.raise_for_status()
        data = response.json()

        return self._parse_results(data)

    def get_contents(
        self,
        urls: List[str],
        *,
        text: bool = True,
        highlights: bool = False,
        max_characters: Optional[int] = None,
        livecrawl: LivecrawlOption = "fallback",
    ) -> List[ExaResult]:
        """
        Get content from specific URLs (without searching).

        Useful when you already have URLs and just need clean content.
        """
        payload: Dict[str, Any] = {
            "urls": urls,
            "contents": {},
        }

        if text:
            text_opts: Dict[str, Any] = {}
            if max_characters:
                text_opts["maxCharacters"] = max_characters
            payload["contents"]["text"] = text_opts if text_opts else True

        if highlights:
            payload["contents"]["highlights"] = True

        if livecrawl != "fallback":
            payload["contents"]["livecrawl"] = livecrawl

        response = self._client.post("/contents", json=payload)
        response.raise_for_status()
        data = response.json()

        return self._parse_results(data)

    def find_similar(
        self,
        url: str,
        *,
        num_results: int = 10,
        text: bool = True,
        highlights: bool = False,
        max_characters: Optional[int] = None,
        exclude_source_domain: bool = True,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        category: Optional[Category] = None,
    ) -> List[ExaResult]:
        """
        Find pages similar to a given URL.

        Great for exploration: given one good source, find more like it.
        """
        payload: Dict[str, Any] = {
            "url": url,
            "numResults": num_results,
            "excludeSourceDomain": exclude_source_domain,
            "contents": {},
        }

        if text:
            text_opts: Dict[str, Any] = {}
            if max_characters:
                text_opts["maxCharacters"] = max_characters
            payload["contents"]["text"] = text_opts if text_opts else True

        if highlights:
            payload["contents"]["highlights"] = True

        if include_domains:
            payload["includeDomains"] = include_domains
        if exclude_domains:
            payload["excludeDomains"] = exclude_domains
        if category:
            payload["category"] = category

        response = self._client.post("/findSimilar", json=payload)
        response.raise_for_status()
        data = response.json()

        return self._parse_results(data)

    @staticmethod
    def _parse_results(data: Dict[str, Any]) -> List[ExaResult]:
        """Parse API response into ExaResult objects."""
        results = []
        for item in data.get("results", []):
            results.append(ExaResult(
                url=item.get("url", ""),
                title=item.get("title", ""),
                text=item.get("text"),
                highlights=item.get("highlights"),
                score=item.get("score"),
                published_date=item.get("publishedDate"),
                author=item.get("author"),
            ))
        return results


# Singleton client for sandbox use
_client: Optional[ExaClient] = None

# Cost tracking (per session)
# Exa pricing: https://exa.ai/pricing
# - Search (1-25 results): $5/1000 = $0.005 per search
# - Search (26-100 results): $25/1000 = $0.025 per search
# - Deep search: $15/1000 = $0.015 per search
# - Text content: charged per 1000 tokens retrieved
_session_costs = {
    "search_count": 0,
    "total_cost_usd": 0.0,
}


def _estimate_search_cost(num_results: int, search_type: str) -> float:
    """Estimate cost for a search request."""
    if search_type == "deep":
        return 0.015  # $15/1000 requests
    elif num_results <= 25:
        return 0.005  # $5/1000 requests
    else:
        return 0.025  # $25/1000 requests (26-100 results)


def _track_cost(cost: float) -> None:
    """Track cost for this session."""
    _session_costs["search_count"] += 1
    _session_costs["total_cost_usd"] += cost


def exa_cost() -> dict:
    """Get Exa API cost tracking for this session."""
    return {
        "search_count": _session_costs["search_count"],
        "total_cost_usd": round(_session_costs["total_cost_usd"], 4),
    }


def exa_reset_cost() -> None:
    """Reset cost tracking."""
    _session_costs["search_count"] = 0
    _session_costs["total_cost_usd"] = 0.0


def _get_client() -> ExaClient:
    """Get or create singleton Exa client."""
    global _client
    if _client is None:
        _client = ExaClient()
    return _client


def _auto_accumulate(sources: List[Dict[str, Any]], query: Optional[str] = None) -> None:
    """Best-effort context accumulation for RLM ctx_get()."""
    try:
        from enzu.tools.context import ctx_add
        # Low-level searches should still populate the shared context store.
        ctx_add(sources, query=query)
    except Exception:
        # Context store not available, skip accumulation.
        return


def _days_back_to_date(days: int) -> str:
    """Convert days_back to ISO date string."""
    return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def exa_search(
    query: str,
    *,
    num_results: int = 5,
    max_characters: Optional[int] = DEFAULT_MAX_CHARACTERS,
    highlights: bool = True,
    search_type: SearchType = "auto",
    category: Optional[Category] = None,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days_back: int = 7,  # Default to last 7 days for freshness
    include_text: Optional[List[str]] = None,
    exclude_text: Optional[List[str]] = None,
    livecrawl: LivecrawlOption = "fallback",
) -> List[Dict[str, Any]]:
    """
    Search the web and get clean content. Sandbox-friendly function.

    Args:
        query: Natural language search query
        num_results: Number of results (default 5, max 100)
        max_characters: Max text length per result (None = no limit)
        highlights: Include key excerpts (default True)
        search_type: "auto", "neural", "fast", or "deep"
        category: Filter by type - "news", "research paper", "github",
                  "company", "pdf", "tweet", "personal site", "financial report"
        include_domains: Only search these domains (e.g., ["arxiv.org"])
        exclude_domains: Exclude these domains
        start_date: Filter by publish date (ISO format: "2024-01-01")
        end_date: Filter by publish date
        days_back: Only results from last N days (default 7 for freshness)
        include_text: Required text in results (e.g., ["python"])
        exclude_text: Excluded text
        livecrawl: "always" (fresh), "fallback", "never", "preferred"

    Returns:
        List of dicts with: url, title, text, highlights, score, published_date
        Results are auto-added to the context store when available.

    Examples:
        # Basic search
        results = exa_search("AI agents 2025", num_results=3)

        # News from last week
        results = exa_search("OpenAI", category="news", days_back=7)

        # Research papers only
        results = exa_search("transformer architecture", category="research paper")

        # Specific domains
        results = exa_search("LLM tutorial", include_domains=["github.com"])

        # Fresh content
        results = exa_search("breaking news AI", livecrawl="always")
    """
    client = _get_client()
    telemetry.log(
        "info",
        "exa_search",
        query=query,
        num_results=num_results,
        category=category,
        days_back=days_back,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
    )

    # Track cost
    _track_cost(_estimate_search_cost(num_results, search_type))

    # Handle days_back - default 7 days for freshness
    effective_start_date = start_date
    if days_back and not start_date:
        effective_start_date = _days_back_to_date(days_back)

    # Span to capture Exa search latency in Logfire traces.
    with telemetry.span(
        "exa.search",
        query=query,
        num_results=num_results,
        category=category,
        days_back=days_back,
    ):
        results = client.search(
            query,
            num_results=num_results,
            search_type=search_type,
            category=category,
            text=True,
            highlights=highlights,
            max_characters=max_characters,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            start_published_date=effective_start_date,
            end_published_date=end_date,
            include_text=include_text,
            exclude_text=exclude_text,
            livecrawl=livecrawl,
        )
    payload = [r.to_dict() for r in results]
    telemetry.log("info", "exa_search_done", count=len(payload))
    _auto_accumulate(payload, query=query)
    return payload


def exa_news(
    query: str,
    *,
    num_results: int = 5,
    max_characters: Optional[int] = DEFAULT_MAX_CHARACTERS,
    days_back: int = 7,
    highlights: bool = True,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Search news articles. Convenience wrapper for exa_search with category="news".

    Args:
        query: News search query
        num_results: Number of results (default 5)
        max_characters: Max text per result (None = no limit)
        days_back: Only results from last N days (default 7)
        highlights: Include key excerpts (default True)
        include_domains: Limit to specific news sites
        exclude_domains: Exclude sites

    Example:
        news = exa_news("AI regulation", days_back=3)
    """
    return exa_search(
        query,
        num_results=num_results,
        max_characters=max_characters,
        highlights=highlights,
        category="news",
        days_back=days_back,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
    )


def exa_papers(
    query: str,
    *,
    num_results: int = 5,
    max_characters: Optional[int] = DEFAULT_MAX_CHARACTERS,
    highlights: bool = True,
    include_domains: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Search research papers. Convenience wrapper with category="research paper".

    Args:
        query: Research topic query
        num_results: Number of results (default 5)
        max_characters: Max text per result (None = no limit)
        highlights: Include key excerpts (default True)
        include_domains: Limit to specific sites (e.g., ["arxiv.org"])

    Example:
        papers = exa_papers("attention mechanism transformers")
    """
    return exa_search(
        query,
        num_results=num_results,
        max_characters=max_characters,
        highlights=highlights,
        category="research paper",
        include_domains=include_domains,
    )


def exa_contents(
    urls: List[str],
    *,
    max_characters: Optional[int] = DEFAULT_MAX_CHARACTERS,
    highlights: bool = False,
    livecrawl: LivecrawlOption = "fallback",
) -> List[Dict[str, Any]]:
    """
    Get clean content from specific URLs. Sandbox-friendly function.

    Args:
        urls: List of URLs to fetch
        max_characters: Max text length per result (None = no limit)
        highlights: Include key excerpts (default False)
        livecrawl: "always" for fresh, "fallback" for cached (default)

    Returns:
        List of dicts with: url, title, text, highlights
        Results are auto-added to the context store when available.
    """
    client = _get_client()
    telemetry.log("info", "exa_contents", count=len(urls))
    # Contents fetch - estimate similar cost to search
    _track_cost(0.005 * len(urls))  # ~$5/1000 per URL
    # Span to capture content fetch latency in Logfire traces.
    with telemetry.span("exa.contents", count=len(urls), livecrawl=livecrawl):
        results = client.get_contents(
            urls,
            text=True,
            highlights=highlights,
            max_characters=max_characters,
            livecrawl=livecrawl,
        )
    payload = [r.to_dict() for r in results]
    telemetry.log("info", "exa_contents_done", count=len(payload))
    _auto_accumulate(payload, query=None)
    return payload


def exa_similar(
    url: str,
    *,
    num_results: int = 5,
    max_characters: Optional[int] = DEFAULT_MAX_CHARACTERS,
    category: Optional[Category] = None,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Find pages similar to a given URL. Sandbox-friendly function.

    Great for exploration: given one good source, find more like it.

    Args:
        url: Source URL to find similar content for
        num_results: Number of results (default 5)
        max_characters: Max text length per result (None = no limit)
        category: Filter similar results by type
        include_domains: Limit to specific domains
        exclude_domains: Exclude domains

    Returns:
        List of dicts with: url, title, text, score
        Results are auto-added to the context store when available.

    Example:
        # Found a good article, find more like it
        similar = exa_similar("https://arxiv.org/abs/2301.00000")

        # Find similar but only from specific sites
        similar = exa_similar(url, include_domains=["nature.com", "science.org"])
    """
    client = _get_client()
    telemetry.log("info", "exa_similar", url=url, num_results=num_results)
    # Track cost
    _track_cost(_estimate_search_cost(num_results, "auto"))
    # Span to capture similarity search latency in Logfire traces.
    with telemetry.span("exa.similar", url=url, num_results=num_results):
        results = client.find_similar(
            url,
            num_results=num_results,
            text=True,
            max_characters=max_characters,
            category=category,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )
    payload = [r.to_dict() for r in results]
    telemetry.log("info", "exa_similar_done", count=len(payload))
    _auto_accumulate(payload, query=url)
    return payload


# Export for sandbox injection
SEARCH_TOOLS = {
    "exa_search": exa_search,
    "exa_news": exa_news,
    "exa_papers": exa_papers,
    "exa_contents": exa_contents,
    "exa_similar": exa_similar,
    "exa_cost": exa_cost,
}
