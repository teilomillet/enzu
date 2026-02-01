#!/usr/bin/env python3
"""
Document Q&A Pipeline: PDF analysis with Docling integration.

This example demonstrates:
- Single query with document parsing
- Multi-turn session with cached documents
- Per-query additional documents
- Session statistics and persistence

Requirements:
    pip install enzu[docling]

Run:
    export OPENAI_API_KEY=sk-...
    python examples/production/document_qa/pipeline.py

    # Or with OpenRouter for cost tracking
    export OPENROUTER_API_KEY=sk-or-...
    python examples/production/document_qa/pipeline.py
"""

import os
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from enzu import Enzu


def check_docling_available() -> bool:
    """Check if docling is installed."""
    try:
        from enzu.tools.docling_parser import documents_available
        return documents_available()
    except ImportError:
        return False


def demo_single_query(client: Enzu, pdf_path: str) -> None:
    """Demonstrate single query with document."""
    print("\n" + "=" * 60)
    print("SINGLE QUERY WITH DOCUMENT")
    print("=" * 60)

    result = client.run(
        "What are the key findings in this document? Summarize in 3 bullet points.",
        documents=[pdf_path],
        tokens=500,
    )

    print(f"\nDocument: {pdf_path}")
    print(f"\nSummary:\n{result}")


def demo_multi_turn_session(client: Enzu, pdf_path: str) -> None:
    """Demonstrate multi-turn session with cached documents."""
    print("\n" + "=" * 60)
    print("MULTI-TURN SESSION WITH CACHED DOCUMENTS")
    print("=" * 60)

    # Create session with document - parsed once
    session = client.session(
        documents=[pdf_path],
        max_tokens=2000,
    )

    print(f"\nSession created with document: {pdf_path}")
    print("Documents are parsed once and cached for all queries.\n")

    # First query
    print("Query 1: 'What is the main topic of this document?'")
    answer1 = session.run(
        "What is the main topic of this document?",
        tokens=200,
    )
    print(f"Answer: {answer1}\n")

    # Second query - uses same cached document
    print("Query 2: 'What methodology or approach is described?'")
    answer2 = session.run(
        "What methodology or approach is described?",
        tokens=200,
    )
    print(f"Answer: {answer2}\n")

    # Third query - builds on conversation history
    print("Query 3: 'Based on what you told me, what are the limitations?'")
    answer3 = session.run(
        "Based on what you told me, what are the limitations?",
        tokens=200,
    )
    print(f"Answer: {answer3}\n")

    # Show session statistics
    print("-" * 40)
    print("Session Statistics:")
    print(f"  Total exchanges: {len(session)}")
    print(f"  Total tokens used: {session.total_tokens:,}")
    print(f"  Total cost: ${session.total_cost_usd:.4f}")

    # Optionally save session for later
    session_path = Path(__file__).parent / "session.json"
    session.save(str(session_path))
    print(f"\nSession saved to: {session_path}")


def demo_per_query_documents(client: Enzu, pdf_paths: list) -> None:
    """Demonstrate adding documents to individual queries."""
    print("\n" + "=" * 60)
    print("PER-QUERY ADDITIONAL DOCUMENTS")
    print("=" * 60)

    if len(pdf_paths) < 2:
        print("Need at least 2 documents for this demo.")
        return

    # Session with base document
    session = client.session(
        documents=[pdf_paths[0]],
        max_tokens=2000,
    )

    print(f"\nSession base document: {pdf_paths[0]}")

    # Query with additional document
    print(f"\nQuery with additional document: {pdf_paths[1]}")
    print("Question: 'Compare the two documents. What are the main differences?'")

    answer = session.run(
        "Compare the two documents. What are the main differences?",
        documents=[pdf_paths[1]],  # Additional document for this query
        tokens=400,
    )
    print(f"\nAnswer: {answer}")


def create_sample_pdf() -> str:
    """Create a sample PDF for testing if none provided."""
    sample_dir = Path(__file__).parent / "sample_docs"
    sample_dir.mkdir(exist_ok=True)
    sample_path = sample_dir / "sample_report.txt"

    # Create a simple text file (Docling can parse text files too)
    if not sample_path.exists():
        sample_path.write_text("""
# Annual Research Report 2024

## Executive Summary

This report presents the findings from our comprehensive study on
artificial intelligence adoption in enterprise environments.

## Key Findings

1. **Adoption Rate**: 73% of enterprises have integrated AI solutions
   into at least one business process.

2. **Primary Use Cases**: Customer service (45%), data analysis (38%),
   and process automation (32%) lead adoption.

3. **ROI Impact**: Companies report an average 23% improvement in
   operational efficiency after AI implementation.

## Methodology

Our research employed a mixed-methods approach:
- Quantitative surveys of 500 enterprise leaders
- Qualitative interviews with 50 CIOs
- Analysis of public financial disclosures

## Limitations

- Sample bias toward technology-forward companies
- Self-reported metrics may overstate benefits
- Limited longitudinal data

## Conclusions

AI adoption continues to accelerate, with clear benefits for
early adopters. However, implementation challenges remain,
particularly in legacy system integration.
""")
        print(f"Created sample document: {sample_path}")

    return str(sample_path)


def main():
    print("=" * 60)
    print("DOCUMENT Q&A PIPELINE - Docling Integration Demo")
    print("=" * 60)

    # Check Docling availability
    if not check_docling_available():
        print("\nDocling is not installed.")
        print("Install with: pip install enzu[docling]")
        print("\nRunning demo with mock data instead...")

        # Create a simple demonstration without actual parsing
        from enzu import Enzu
        client = Enzu()

        sample_text = """
        This is a sample research document about AI adoption.
        Key findings: 73% adoption rate, 23% efficiency improvement.
        """

        result = client.run(
            "Summarize these findings",
            data=sample_text,
            tokens=200,
        )
        print(f"\nDemo result (without Docling):\n{result}")
        return

    # Initialize client
    provider = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "openai"
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = Enzu(model=model, provider=provider)

    print(f"\nUsing: {provider}/{model}")

    # Get or create sample document
    sample_pdf = create_sample_pdf()

    # Run demos
    try:
        # Demo 1: Single query
        demo_single_query(client, sample_pdf)

        # Demo 2: Multi-turn session
        demo_multi_turn_session(client, sample_pdf)

        # Demo 3: Per-query documents (needs multiple docs)
        # Uncomment with real PDFs:
        # demo_per_query_documents(client, [sample_pdf, "other.pdf"])

    except ImportError as e:
        print(f"\nImportError: {e}")
        print("Install Docling: pip install enzu[docling]")
    except Exception as e:
        print(f"\nError: {e}")
        raise

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
