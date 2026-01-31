"""
Financial Researcher - enzu-powered SEC filing analysis.

Loads SEC filings into enzu's context and provides an interface
for asking questions about financial data, risks, and trends.
"""

from dataclasses import dataclass

from enzu import Enzu

try:
    from .fetcher import SECFetcher, Filing, FilingType
except ImportError:
    from fetcher import SECFetcher, Filing, FilingType


@dataclass
class ResearchResult:
    """Result from a research query."""

    answer: str
    tokens_used: int
    cost_usd: float | None
    filings_analyzed: list[str]


class FinancialResearcher:
    """Research assistant for SEC filings."""

    def __init__(
        self,
        model: str = "gpt-4o",
        provider: str = "openai",
        max_tokens: int = 2000,
    ):
        self.client = Enzu(model=model, provider=provider)
        self.fetcher = SECFetcher()
        self.max_tokens = max_tokens
        self.filings: list[Filing] = []
        self._context: str = ""

    def load_company(
        self,
        ticker: str,
        filing_types: list[FilingType] = ["10-K"],
        years: int = 2,
    ) -> None:
        """Load SEC filings for a company."""
        print(f"Loading {ticker} filings...")

        new_filings = self.fetcher.fetch_company(
            ticker,
            filing_types=filing_types,
            limit_per_type=years,
        )

        self.filings.extend(new_filings)
        self._rebuild_context()

        print(f"  Loaded {len(new_filings)} filings ({len(self._context):,} chars)")

    def load_multiple(
        self,
        tickers: list[str],
        filing_types: list[FilingType] = ["10-K"],
    ) -> None:
        """Load filings for multiple companies."""
        for ticker in tickers:
            self.load_company(ticker, filing_types, years=1)

    def _rebuild_context(self) -> None:
        """Rebuild the combined context from all filings."""
        sections = []

        for filing in self.filings:
            header = f"{'=' * 60}\n{filing.ticker} - {filing.filing_type} (Filed: {filing.filed_date})\n{'=' * 60}"

            # Truncate very long filings to most relevant sections
            content = filing.content
            if len(content) > 100_000:
                # Keep first 50k (business description, risk factors)
                # and last 20k (MD&A often near end)
                content = (
                    content[:50_000] + "\n\n[...truncated...]\n\n" + content[-20_000:]
                )

            sections.append(f"{header}\n\n{content}")

        self._context = "\n\n".join(sections)

    def ask(self, question: str, tokens: int | None = None) -> ResearchResult:
        """Ask a question about the loaded filings."""
        if not self.filings:
            raise ValueError("No filings loaded. Call load_company() first.")

        tokens = tokens or self.max_tokens

        # Build the prompt
        system_context = """You are a financial analyst assistant. Analyze the SEC filings 
provided and answer questions accurately. When citing information:
- Reference specific filings (e.g., "AAPL 10-K 2024")
- Quote relevant passages when helpful
- Note if information is missing or unclear
- Provide quantitative data when available"""

        result = self.client.run(
            f"{system_context}\n\nQuestion: {question}",
            data=self._context,
            tokens=tokens,
            return_report=True,
        )

        filing_refs = [
            f"{f.ticker} {f.filing_type} ({f.filed_date})" for f in self.filings
        ]

        return ResearchResult(
            answer=result.answer or "",
            tokens_used=result.budget_usage.total_tokens,
            cost_usd=result.budget_usage.cost_usd,
            filings_analyzed=filing_refs,
        )

    def summarize(self, focus: str = "business and risks") -> ResearchResult:
        """Generate a summary of the loaded filings."""
        return self.ask(
            f"Provide a comprehensive summary focusing on: {focus}. "
            "Include key business segments, risk factors, and recent developments."
        )

    def compare(self, aspects: list[str] | None = None) -> ResearchResult:
        """Compare multiple companies or time periods."""
        aspects = aspects or ["revenue", "growth", "risks", "strategy"]
        aspects_str = ", ".join(aspects)

        return self.ask(
            f"Compare the companies/periods in these filings across: {aspects_str}. "
            "Create a structured comparison highlighting key differences."
        )

    def extract_metrics(self) -> ResearchResult:
        """Extract key financial metrics."""
        return self.ask(
            "Extract key financial metrics from these filings:\n"
            "- Revenue (total and by segment)\n"
            "- Net income\n"
            "- Gross margin\n"
            "- Operating expenses\n"
            "- Cash and equivalents\n"
            "- Debt levels\n"
            "Present as a table if multiple periods/companies."
        )

    def find_risks(self) -> ResearchResult:
        """Identify and categorize risk factors."""
        return self.ask(
            "Analyze the Risk Factors sections and categorize the main risks:\n"
            "1. Operational risks\n"
            "2. Financial risks\n"
            "3. Regulatory/legal risks\n"
            "4. Market/competitive risks\n"
            "5. Geopolitical risks\n"
            "For each, note severity and any mitigations mentioned."
        )

    def clear(self) -> None:
        """Clear all loaded filings."""
        self.filings = []
        self._context = ""


if __name__ == "__main__":
    # Quick test
    researcher = FinancialResearcher()
    researcher.load_company("AAPL", filing_types=["10-K"], years=1)

    result = researcher.ask("What are Apple's main revenue segments?")
    print(f"\nAnswer:\n{result.answer}")
    print(f"\nTokens: {result.tokens_used}, Cost: ${result.cost_usd:.4f}")
