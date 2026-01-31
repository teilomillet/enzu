# Financial Research Assistant

Analyze SEC filings, earnings calls, and financial documents to extract insights — without building a RAG pipeline.

## What It Does

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Fetch     │ ──▶ │   Parse     │ ──▶ │   Load      │ ──▶ │   Analyze   │
│  SEC EDGAR  │     │  10-K/10-Q  │     │  into enzu  │     │   with RLM  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

1. **Fetch**: Download 10-K, 10-Q filings from SEC EDGAR (free, public)
2. **Parse**: Extract clean text from HTML/XML filings
3. **Load**: Combine multiple documents into enzu context
4. **Analyze**: Ask questions, find trends, compare periods

## Quick Start

```bash
export OPENAI_API_KEY=sk-...

# Analyze a single company
python examples/production/financial_research/analyze.py AAPL

# Compare multiple companies
python examples/production/financial_research/analyze.py AAPL MSFT GOOGL --compare
```

## Example Queries

```python
from financial_research import FinancialResearcher

researcher = FinancialResearcher()

# Load filings
researcher.load_company("AAPL", years=[2023, 2024])

# Ask questions
researcher.ask("What are the main risk factors?")
researcher.ask("How has revenue changed year-over-year?")
researcher.ask("What does management say about AI investments?")
```

## Sample Output

```
📊 ANALYSIS: Apple Inc (AAPL) - 10-K FY2024

RISK FACTORS:
1. Supply chain concentration in China (mentioned 12 times)
2. Foreign exchange exposure, particularly EUR and CNY
3. Regulatory scrutiny on App Store practices

REVENUE TRENDS:
- Services revenue grew 14% YoY ($85B → $97B)
- iPhone revenue flat (-2% YoY)
- Wearables declined 8%

MANAGEMENT OUTLOOK:
- "Investing significantly in generative AI capabilities"
- Expects Services to maintain double-digit growth
- Cautious on China consumer demand

Cost: $0.08 | Tokens: 4,200
```

## Why This Matters

| Traditional Approach | enzu Approach |
|---------------------|---------------|
| Bloomberg Terminal: $24,000/year | Pay per query: ~$0.05-0.20 |
| Manual reading: 2-4 hours per 10-K | Instant analysis |
| Build RAG pipeline: days of setup | Zero setup |

## Files

- `analyze.py` — Main CLI tool
- `fetcher.py` — SEC EDGAR downloader
- `parser.py` — Filing text extractor  
- `researcher.py` — enzu integration

## Data Sources

All data is publicly available from [SEC EDGAR](https://www.sec.gov/edgar/searchedgar/companysearch).

- 10-K: Annual reports (comprehensive)
- 10-Q: Quarterly reports
- 8-K: Material events
- DEF 14A: Proxy statements (executive compensation)

## Limitations

- SEC filings only (no earnings call transcripts in this demo)
- Text extraction may miss some tables/charts
- Very large filings (500+ pages) may need multiple RLM passes

## Extending

Add earnings call transcripts:
```python
researcher.load_earnings_call("AAPL", "Q4 2024")
researcher.ask("What questions did analysts ask about margins?")
```

Add news/sentiment:
```python
researcher.load_news("AAPL", days=30)
researcher.ask("What's the market sentiment on Apple's AI strategy?")
```
