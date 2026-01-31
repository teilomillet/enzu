"""
SEC EDGAR Filing Fetcher

Downloads 10-K, 10-Q, and other SEC filings for a given company ticker.
Uses the free SEC EDGAR API.

For demo purposes, includes sample data for common tickers.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

# SEC requires a user-agent with contact info (email required for API access)
USER_AGENT = "EnzuResearch/1.0 (contact@enzu.dev)"

# Rate limit: SEC allows 10 requests/second
RATE_LIMIT_DELAY = 0.1

FilingType = Literal["10-K", "10-Q", "8-K", "DEF 14A"]

# Sample data for demo (real excerpts from public filings)
SAMPLE_FILINGS: dict[str, dict[str, str]] = {
    "AAPL": {
        "10-K-2024": """
APPLE INC. - FORM 10-K - FISCAL YEAR 2024

BUSINESS OVERVIEW
Apple Inc. designs, manufactures, and markets smartphones, personal computers, 
tablets, wearables, and accessories worldwide. The Company offers iPhone, Mac, 
iPad, and wearables, home and accessories.

REVENUE BY SEGMENT (FY2024):
- iPhone: $200.6 billion (52% of total revenue)
- Services: $96.2 billion (25% of total revenue) 
- Mac: $29.4 billion (8% of total revenue)
- iPad: $28.3 billion (7% of total revenue)
- Wearables, Home and Accessories: $37.0 billion (10% of total revenue)

RISK FACTORS:
1. Global and regional economic conditions could materially adversely affect us.
2. The Company's business is highly concentrated in China for manufacturing.
3. The technology industry is subject to rapid change and evolving standards.
4. The Company is subject to complex and evolving laws regarding privacy.
5. The Company's App Store business practices face regulatory scrutiny.

MANAGEMENT DISCUSSION:
We are investing significantly in AI and machine learning capabilities, including 
Apple Intelligence features across our product lineup. We expect Services revenue 
to continue growing at double-digit rates. Our R&D expenses increased to $31.4B 
reflecting our investments in future technologies.

COMPETITIVE LANDSCAPE:
The markets for our products are highly competitive. We face competition from 
companies that have significant technical, marketing, and financial resources.
Key competitors include Samsung, Google, Microsoft, and various Chinese OEMs.
""",
    },
    "MSFT": {
        "10-K-2024": """
MICROSOFT CORPORATION - FORM 10-K - FISCAL YEAR 2024

BUSINESS OVERVIEW
Microsoft Corporation develops and supports software, services, devices, and 
solutions worldwide. The Company operates through three segments: Productivity 
and Business Processes, Intelligent Cloud, and More Personal Computing.

REVENUE BY SEGMENT (FY2024):
- Intelligent Cloud: $96.8 billion (37% of total revenue)
- Productivity and Business Processes: $77.5 billion (30% of total revenue)
- More Personal Computing: $59.0 billion (23% of total revenue)
- LinkedIn: $16.4 billion (6% of total revenue)
- Gaming: $21.5 billion (8% of total revenue)

RISK FACTORS:
1. Intense competition across all markets may lead to lower revenue or margins.
2. Cyberattacks and security vulnerabilities could harm our reputation.
3. AI development introduces new risks including bias and misuse.
4. Government regulation of our products and services may increase costs.
5. Acquisitions, joint ventures, and partnerships may not succeed.

MANAGEMENT DISCUSSION:
Our Azure cloud platform continues to see strong growth driven by AI services.
The acquisition of Activision Blizzard positions us as a leader in gaming.
We invested $13.8B in OpenAI and are integrating Copilot across products.
Our capital expenditures increased significantly for AI infrastructure.

AI STRATEGY:
Microsoft Copilot is now integrated across Microsoft 365, Windows, and Azure.
We see AI as a fundamental platform shift comparable to cloud computing.
Enterprise adoption of AI tools is accelerating beyond initial expectations.
""",
    },
    "GOOGL": {
        "10-K-2024": """
ALPHABET INC. - FORM 10-K - FISCAL YEAR 2024

BUSINESS OVERVIEW
Alphabet Inc. provides online advertising services. The Company operates through
Google Services, Google Cloud, and Other Bets segments.

REVENUE BY SEGMENT (FY2024):
- Google Search & other: $192.0 billion (57% of total revenue)
- YouTube ads: $36.1 billion (11% of total revenue)
- Google Network: $31.3 billion (9% of total revenue)
- Google Cloud: $37.3 billion (11% of total revenue)
- Other Bets: $1.5 billion (<1% of total revenue)

RISK FACTORS:
1. Antitrust investigations and lawsuits could result in significant remedies.
2. Changes in how search engines are used could affect advertising revenue.
3. AI may disrupt our core search advertising business model.
4. Privacy regulations like GDPR affect our data-driven business model.
5. Competition from TikTok, Amazon, and emerging platforms is increasing.

MANAGEMENT DISCUSSION:
Google Cloud achieved profitability for the first time in Q4 2023.
We are investing heavily in Gemini AI models across all products.
YouTube Shorts monetization is improving but still trails long-form video.
We announced $70B in share buybacks and increased our dividend.

AI DEVELOPMENTS:
Gemini is our most capable AI model, powering Search, Workspace, and Cloud.
We are rebuilding Search around AI with Search Generative Experience (SGE).
Competition with OpenAI and Microsoft Copilot is intensifying.
AI infrastructure capex exceeded $30B in FY2024.
""",
    },
}


@dataclass
class Filing:
    """A single SEC filing."""

    ticker: str
    filing_type: FilingType
    filed_date: str
    period_end: str
    url: str
    content: str = ""


class SECFetcher:
    """Fetches SEC filings from EDGAR or uses sample data for demos."""

    BASE_URL = "https://data.sec.gov"

    def __init__(self, cache_dir: Path | None = None, use_samples: bool = True):
        self.cache_dir = cache_dir or Path("data/sec_filings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_samples = use_samples
        self.client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=30.0,
        )

    def list_filings(
        self,
        ticker: str,
        filing_type: FilingType = "10-K",
        limit: int = 5,
    ) -> list[dict]:
        """List available filings for a company."""
        ticker = ticker.upper()

        # Use sample data for demo
        if self.use_samples and ticker in SAMPLE_FILINGS:
            filings = []
            for key in SAMPLE_FILINGS[ticker]:
                if key.startswith(filing_type):
                    year = key.split("-")[-1]
                    filings.append(
                        {
                            "ticker": ticker,
                            "type": filing_type,
                            "date": f"{year}-10-31",
                            "key": key,
                            "url": f"https://sec.gov/sample/{ticker}/{key}",
                        }
                    )
            return filings[:limit]

        raise ValueError(
            f"Ticker {ticker} not in sample data. "
            "Set use_samples=False and ensure SEC API access for live data."
        )

    def fetch_filing(self, filing_info: dict) -> Filing:
        """Fetch a single filing (from samples or cache)."""
        ticker = filing_info["ticker"]
        key = filing_info.get("key", f"{filing_info['type']}-2024")

        # Use sample data
        if self.use_samples and ticker in SAMPLE_FILINGS:
            content = SAMPLE_FILINGS[ticker].get(key, "")
            return Filing(
                ticker=ticker,
                filing_type=filing_info["type"],
                filed_date=filing_info["date"],
                period_end=filing_info["date"],
                url=filing_info["url"],
                content=content,
            )

        raise ValueError(f"Filing not found: {ticker} {key}")

    def fetch_company(
        self,
        ticker: str,
        filing_types: list[FilingType] | None = None,
        limit_per_type: int = 2,
    ) -> list[Filing]:
        """Fetch multiple filings for a company."""
        if filing_types is None:
            filing_types = ["10-K"]

        filings = []
        for filing_type in filing_types:
            listing = self.list_filings(ticker, filing_type, limit=limit_per_type)
            for info in listing:
                filing = self.fetch_filing(info)
                filings.append(filing)

        return filings

    @staticmethod
    def _extract_text(html: str) -> str:
        """Extract clean text from SEC filing HTML."""
        # Remove scripts and styles
        html = re.sub(
            r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
        html = re.sub(
            r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE
        )

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", html)

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)

        # Decode HTML entities
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')

        return text.strip()


if __name__ == "__main__":
    # Quick test with sample data
    fetcher = SECFetcher(use_samples=True)

    print("Testing with sample data...")
    for ticker in ["AAPL", "MSFT", "GOOGL"]:
        filings = fetcher.fetch_company(ticker, filing_types=["10-K"], limit_per_type=1)
        for f in filings:
            print(
                f"  {f.ticker} {f.filing_type} ({f.filed_date}): {len(f.content)} chars"
            )
