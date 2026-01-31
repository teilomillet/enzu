"""Tests for invoice processing pipeline."""

import json
import sys
from pathlib import Path

import pytest

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from enzu.models import ProviderResult, TaskSpec, BudgetUsage, Outcome
from enzu.providers.base import BaseProvider


class MockProvider(BaseProvider):
    """Mock provider for testing."""

    name = "mock"

    def __init__(self, responses=None, usage=None):
        self._responses = list(responses) if responses else []
        self._usage = usage or {"output_tokens": 50, "total_tokens": 100, "cost_usd": 0.001}
        self.calls = []

    def generate(self, task: TaskSpec) -> ProviderResult:
        self.calls.append(task.input_text)
        output = self._responses.pop(0) if self._responses else "{}"
        return ProviderResult(
            output_text=output,
            raw={"mock": True},
            usage=dict(self._usage),
            provider=self.name,
            model=task.model,
        )


# Import pipeline components after path setup
from examples.production.invoice_processing.pipeline import (
    InvoiceData,
    ProcessingResult,
    InvoicePipeline,
)


class TestInvoiceData:
    """Tests for InvoiceData dataclass."""

    def test_default_values(self):
        """InvoiceData has sensible defaults."""
        data = InvoiceData()
        assert data.invoice_number == ""
        assert data.total == 0.0
        assert data.line_items == []

    def test_with_values(self):
        """InvoiceData stores values correctly."""
        data = InvoiceData(
            invoice_number="INV-001",
            vendor_name="Acme Corp",
            total=1234.56,
            line_items=[{"description": "Widget", "amount": 100}],
        )
        assert data.invoice_number == "INV-001"
        assert data.vendor_name == "Acme Corp"
        assert data.total == 1234.56
        assert len(data.line_items) == 1


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""

    def test_success_result(self):
        """Success result has expected fields."""
        result = ProcessingResult(
            file_path="/path/to/invoice.txt",
            success=True,
            outcome="success",
            tokens_used=150,
            cost_usd=0.001,
        )
        assert result.success is True
        assert result.error is None

    def test_failure_result(self):
        """Failure result includes error."""
        result = ProcessingResult(
            file_path="/path/to/invoice.txt",
            success=False,
            outcome="budget_exceeded",
            error="Budget exceeded",
        )
        assert result.success is False
        assert result.error == "Budget exceeded"


class TestInvoicePipeline:
    """Tests for InvoicePipeline class."""

    def test_pipeline_initialization(self):
        """Pipeline initializes with config."""
        pipeline = InvoicePipeline(
            budget_per_item=0.05,
            tokens_per_item=500,
            max_retries=3,
            parallelism=5,
        )
        assert pipeline.budget_per_item == 0.05
        assert pipeline.tokens_per_item == 500
        assert pipeline.max_retries == 3
        assert pipeline.parallelism == 5

    def test_parse_valid_json(self):
        """Pipeline parses valid JSON response."""
        pipeline = InvoicePipeline()

        # Create a mock report object
        class MockReport:
            answer = json.dumps({
                "invoice_number": "INV-123",
                "vendor_name": "Test Vendor",
                "customer_name": "Test Customer",
                "date": "2024-01-15",
                "due_date": "2024-02-15",
                "subtotal": 100.0,
                "tax": 8.0,
                "total": 108.0,
                "line_items": [{"description": "Item 1", "amount": 100}],
            })

        result = pipeline._parse_response(MockReport())
        assert result is not None
        assert result.invoice_number == "INV-123"
        assert result.vendor_name == "Test Vendor"
        assert result.total == 108.0

    def test_parse_json_with_markdown(self):
        """Pipeline extracts JSON from markdown code blocks."""
        pipeline = InvoicePipeline()

        class MockReport:
            answer = '''Here's the extracted data:
```json
{"invoice_number": "INV-456", "total": 500.0}
```
'''

        result = pipeline._parse_response(MockReport())
        assert result is not None
        assert result.invoice_number == "INV-456"
        assert result.total == 500.0

    def test_parse_invalid_json(self):
        """Pipeline handles invalid JSON gracefully."""
        pipeline = InvoicePipeline()

        class MockReport:
            answer = "This is not valid JSON at all"

        result = pipeline._parse_response(MockReport())
        assert result is not None
        assert result.raw_response == "This is not valid JSON at all"
        assert result.invoice_number == ""  # Default value

    def test_parse_empty_response(self):
        """Pipeline handles empty response."""
        pipeline = InvoicePipeline()

        class MockReport:
            answer = None
            output_text = None

        result = pipeline._parse_response(MockReport())
        assert result is None

    def test_metrics_summary_empty(self):
        """Metrics summary works with no results."""
        pipeline = InvoicePipeline()
        metrics = pipeline.get_metrics_summary()

        assert metrics["total_invoices"] == 0
        assert metrics["successful"] == 0
        assert metrics["failed"] == 0
        assert metrics["total_cost_usd"] == 0

    def test_metrics_summary_with_results(self):
        """Metrics summary aggregates correctly."""
        pipeline = InvoicePipeline()

        # Add mock results
        pipeline.results = [
            ProcessingResult(
                file_path="inv1.txt",
                success=True,
                outcome="success",
                tokens_used=100,
                cost_usd=0.001,
            ),
            ProcessingResult(
                file_path="inv2.txt",
                success=True,
                outcome="success",
                tokens_used=150,
                cost_usd=0.0015,
            ),
            ProcessingResult(
                file_path="inv3.txt",
                success=False,
                outcome="budget_exceeded",
                tokens_used=50,
                cost_usd=0.0005,
            ),
        ]

        metrics = pipeline.get_metrics_summary()

        assert metrics["total_invoices"] == 3
        assert metrics["successful"] == 2
        assert metrics["failed"] == 1
        assert metrics["total_cost_usd"] == 0.003
        assert metrics["total_tokens"] == 300


class TestSampleInvoices:
    """Tests for sample invoice files."""

    def test_sample_invoices_exist(self):
        """Sample invoices are present."""
        sample_dir = Path(__file__).parent.parent / "sample_invoices"
        invoices = list(sample_dir.glob("*.txt"))
        assert len(invoices) >= 5

    def test_sample_invoices_readable(self):
        """Sample invoices are readable text files."""
        sample_dir = Path(__file__).parent.parent / "sample_invoices"
        for invoice in sample_dir.glob("*.txt"):
            content = invoice.read_text(encoding="utf-8")
            assert len(content) > 100  # Not empty
            assert "invoice" in content.lower() or "INVOICE" in content
