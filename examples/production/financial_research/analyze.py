#!/usr/bin/env python3
"""
Financial Research CLI - Analyze SEC filings with enzu.

Usage:
    # Analyze a single company
    python analyze.py AAPL

    # Analyze with specific question
    python analyze.py AAPL --ask "What are the main risk factors?"

    # Compare multiple companies
    python analyze.py AAPL MSFT GOOGL --compare

    # Extract financial metrics
    python analyze.py AAPL --metrics

    # Identify risks
    python analyze.py AAPL --risks

Requirements:
    export OPENAI_API_KEY=sk-...
"""

import argparse
import sys
from pathlib import Path

# Add examples to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from examples.production.financial_research.researcher import FinancialResearcher


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SEC filings with enzu",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze.py AAPL
    python analyze.py AAPL --ask "What's their AI strategy?"
    python analyze.py AAPL MSFT --compare
    python analyze.py TSLA --risks
        """,
    )

    parser.add_argument(
        "tickers",
        nargs="+",
        help="Company ticker symbols (e.g., AAPL MSFT)",
    )
    parser.add_argument(
        "--ask",
        "-a",
        help="Ask a specific question about the filings",
    )
    parser.add_argument(
        "--compare",
        "-c",
        action="store_true",
        help="Compare the companies",
    )
    parser.add_argument(
        "--metrics",
        "-m",
        action="store_true",
        help="Extract key financial metrics",
    )
    parser.add_argument(
        "--risks",
        "-r",
        action="store_true",
        help="Identify and categorize risks",
    )
    parser.add_argument(
        "--summary",
        "-s",
        action="store_true",
        help="Generate a business summary",
    )
    parser.add_argument(
        "--filing-type",
        default="10-K",
        choices=["10-K", "10-Q"],
        help="Type of filing to analyze (default: 10-K)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=1,
        help="Number of years of filings to load (default: 1)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=1500,
        help="Max output tokens (default: 1500)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FINANCIAL RESEARCH ASSISTANT")
    print("=" * 70)

    # Initialize researcher
    researcher = FinancialResearcher(max_tokens=args.tokens)

    # Load filings
    for ticker in args.tickers:
        researcher.load_company(
            ticker.upper(),
            filing_types=[args.filing_type],
            years=args.years,
        )

    print("-" * 70)

    # Run analysis
    if args.ask:
        print(f"\n📊 QUESTION: {args.ask}\n")
        result = researcher.ask(args.ask)
    elif args.compare and len(args.tickers) > 1:
        print("\n📊 COMPARISON ANALYSIS\n")
        result = researcher.compare()
    elif args.metrics:
        print("\n📊 FINANCIAL METRICS\n")
        result = researcher.extract_metrics()
    elif args.risks:
        print("\n📊 RISK ANALYSIS\n")
        result = researcher.find_risks()
    elif args.summary:
        print("\n📊 BUSINESS SUMMARY\n")
        result = researcher.summarize()
    else:
        # Default: summary
        print("\n📊 BUSINESS SUMMARY\n")
        result = researcher.summarize()

    print(result.answer)

    print("\n" + "-" * 70)
    print("📈 ANALYSIS DETAILS:")
    print(f"   Filings: {', '.join(result.filings_analyzed)}")
    print(f"   Tokens:  {result.tokens_used:,}")
    if result.cost_usd:
        print(f"   Cost:    ${result.cost_usd:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
