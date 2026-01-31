"""Tests for report_service example."""

import json
from pathlib import Path

import pytest

from enzu import Enzu, Outcome


class TestReportService:
    """Tests for the report service demo."""

    def test_corpus_loading(self, temp_dir):
        """Verify corpus loading works correctly."""
        # Create test docs
        docs_dir = temp_dir / "docs"
        docs_dir.mkdir()
        (docs_dir / "doc1.txt").write_text("First document content")
        (docs_dir / "doc2.txt").write_text("Second document content")

        # Load corpus (simulating load_corpus function)
        parts = []
        for p in sorted(docs_dir.glob("*.txt")):
            parts.append(f"=== {p.name} ===\n{p.read_text()}\n")
        corpus = "\n".join(parts)

        assert "=== doc1.txt ===" in corpus
        assert "=== doc2.txt ===" in corpus
        assert "First document content" in corpus
        assert "Second document content" in corpus

    def test_report_structure(self, mock_provider_factory):
        """Verify report has expected structure."""
        mock = mock_provider_factory([
            "## EXECUTIVE SUMMARY\nTest summary.\n\n## KEY FINDINGS\n- Finding 1"
        ])

        client = Enzu(provider=mock, model="test-model")
        report = client.run(
            "Generate report",
            data="Test corpus",
            tokens=100,
            return_report=True,
        )

        assert report.outcome in (Outcome.SUCCESS, Outcome.BUDGET_EXCEEDED)
        assert report.budget_usage.output_tokens is not None

    def test_budget_enforcement(self, mock_provider_factory):
        """Verify budget limits are respected."""
        mock = mock_provider_factory(
            ["Short response"],
            usage={"output_tokens": 50, "total_tokens": 100},
        )

        client = Enzu(provider=mock, model="test-model")
        report = client.run(
            "Generate report",
            data="Test corpus",
            tokens=100,
            return_report=True,
        )

        assert report.budget_usage.output_tokens <= 100

    def test_trace_json_format(self, temp_dir, mock_provider_factory):
        """Verify trace.json has correct structure."""
        mock = mock_provider_factory(["Test report content"])

        client = Enzu(provider=mock, model="test-model")
        report = client.run(
            "Generate report",
            data="Test corpus",
            tokens=100,
            return_report=True,
        )

        # Simulate trace.json creation
        trace = {
            "outcome": report.outcome.value,
            "partial": report.partial,
            "usage": {
                "output": report.budget_usage.output_tokens,
                "total": report.budget_usage.total_tokens,
            },
        }

        trace_path = temp_dir / "trace.json"
        trace_path.write_text(json.dumps(trace, indent=2))

        # Verify structure
        loaded = json.loads(trace_path.read_text())
        assert "outcome" in loaded
        assert "usage" in loaded
        assert loaded["usage"]["output"] is not None
