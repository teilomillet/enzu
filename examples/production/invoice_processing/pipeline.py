#!/usr/bin/env python3
"""
Invoice Processing Pipeline: Document extraction at scale with budget control.

This example demonstrates:
- Per-item budget allocation ($0.02 max per document)
- Batch processing with configurable parallelism
- Graceful degradation when budget is tight
- Smart retry with budget tracking
- p50/p95 cost metrics

Run:
    export OPENAI_API_KEY=sk-...
    python examples/production/invoice_processing/pipeline.py

    # Or with OpenRouter for cost tracking
    export OPENROUTER_API_KEY=sk-or-...
    python examples/production/invoice_processing/pipeline.py
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from enzu import Enzu, Outcome
from enzu.metrics import RunEvent, RunMetricsCollector

# Configuration
INVOICES_DIR = Path(__file__).parent / "sample_invoices"
OUTPUT_DIR = Path(__file__).parent / "output"
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PROVIDER = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "openai"

# Budget configuration
BUDGET_PER_INVOICE_USD = 0.02  # $0.02 max per invoice
BUDGET_PER_INVOICE_TOKENS = 300  # Token limit per invoice
MAX_RETRIES = 2
PARALLELISM = 3  # Concurrent processing threads


@dataclass
class InvoiceData:
    """Extracted invoice data."""

    invoice_number: str = ""
    vendor_name: str = ""
    customer_name: str = ""
    date: str = ""
    due_date: str = ""
    subtotal: float = 0.0
    tax: float = 0.0
    total: float = 0.0
    line_items: List[Dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""


@dataclass
class ProcessingResult:
    """Result of processing a single invoice."""

    file_path: str
    success: bool
    data: Optional[InvoiceData] = None
    outcome: str = ""
    tokens_used: int = 0
    cost_usd: float = 0.0
    elapsed_seconds: float = 0.0
    retries: int = 0
    error: Optional[str] = None


class InvoicePipeline:
    """
    Batch invoice processing with budget control.

    Key features:
    - Per-document budget caps
    - Parallel processing
    - Graceful degradation
    - Metrics collection
    """

    def __init__(
        self,
        budget_per_item: float = BUDGET_PER_INVOICE_USD,
        tokens_per_item: int = BUDGET_PER_INVOICE_TOKENS,
        max_retries: int = MAX_RETRIES,
        parallelism: int = PARALLELISM,
    ):
        self.budget_per_item = budget_per_item
        self.tokens_per_item = tokens_per_item
        self.max_retries = max_retries
        self.parallelism = parallelism
        self.client = Enzu(provider=PROVIDER, model=MODEL)
        self.metrics = RunMetricsCollector()
        self.results: List[ProcessingResult] = []

    def process_batch(self, invoice_paths: List[Path]) -> List[ProcessingResult]:
        """Process a batch of invoices in parallel."""
        print(f"\nProcessing {len(invoice_paths)} invoices...")
        print(f"  Budget per invoice: ${self.budget_per_item:.4f}")
        print(f"  Token limit: {self.tokens_per_item}")
        print(f"  Parallelism: {self.parallelism}")
        print("-" * 50)

        with ThreadPoolExecutor(max_workers=self.parallelism) as executor:
            futures = {
                executor.submit(self._process_single, path): path
                for path in invoice_paths
            }

            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    self._log_result(result)
                except Exception as e:
                    error_result = ProcessingResult(
                        file_path=str(path),
                        success=False,
                        error=str(e),
                        outcome="ERROR",
                    )
                    self.results.append(error_result)
                    self._log_result(error_result)

        return self.results

    def _process_single(self, invoice_path: Path) -> ProcessingResult:
        """Process a single invoice with retry logic."""
        content = invoice_path.read_text(encoding="utf-8")
        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                result = self._extract_invoice(invoice_path, content, retries)
                return result
            except Exception as e:
                last_error = str(e)
                retries += 1
                if retries <= self.max_retries:
                    time.sleep(0.5 * retries)  # Exponential backoff

        return ProcessingResult(
            file_path=str(invoice_path),
            success=False,
            error=f"Failed after {retries} retries: {last_error}",
            outcome="RETRY_EXHAUSTED",
            retries=retries,
        )

    def _extract_invoice(
        self, path: Path, content: str, retry_count: int
    ) -> ProcessingResult:
        """Extract data from a single invoice."""
        extraction_prompt = """Extract the following from this invoice and return as JSON:
{
  "invoice_number": "...",
  "vendor_name": "...",
  "customer_name": "...",
  "date": "YYYY-MM-DD",
  "due_date": "YYYY-MM-DD",
  "subtotal": 0.00,
  "tax": 0.00,
  "total": 0.00,
  "line_items": [
    {"description": "...", "quantity": 0, "unit_price": 0.00, "amount": 0.00}
  ]
}

Return ONLY valid JSON, no other text."""

        start_time = time.time()

        report = self.client.run(
            extraction_prompt,
            data=content,
            tokens=self.tokens_per_item,
            cost=self.budget_per_item,
            return_report=True,
        )

        elapsed = time.time() - start_time
        usage = report.budget_usage

        # Record metrics
        event = RunEvent(
            run_id=f"{path.stem}-{retry_count}",
            outcome=report.outcome.value,
            elapsed_seconds=usage.elapsed_seconds,
            cost_usd=usage.cost_usd,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        )
        self.metrics.observe(event)

        # Handle outcome
        if report.outcome == Outcome.SUCCESS:
            data = self._parse_response(report)
            return ProcessingResult(
                file_path=str(path),
                success=True,
                data=data,
                outcome=report.outcome.value,
                tokens_used=usage.output_tokens or 0,
                cost_usd=usage.cost_usd or 0.0,
                elapsed_seconds=elapsed,
                retries=retry_count,
            )
        elif report.outcome == Outcome.BUDGET_EXCEEDED:
            # Graceful degradation: try to parse partial result
            data = self._parse_response(report)
            return ProcessingResult(
                file_path=str(path),
                success=data is not None,  # Partial success if we got some data
                data=data,
                outcome=report.outcome.value,
                tokens_used=usage.output_tokens or 0,
                cost_usd=usage.cost_usd or 0.0,
                elapsed_seconds=elapsed,
                retries=retry_count,
                error="Budget exceeded - partial extraction" if data else "Budget exceeded",
            )
        else:
            return ProcessingResult(
                file_path=str(path),
                success=False,
                outcome=report.outcome.value,
                tokens_used=usage.output_tokens or 0,
                cost_usd=usage.cost_usd or 0.0,
                elapsed_seconds=elapsed,
                retries=retry_count,
                error=f"Outcome: {report.outcome.value}",
            )

    def _parse_response(self, report) -> Optional[InvoiceData]:
        """Parse LLM response into InvoiceData."""
        raw = getattr(report, "answer", None) or getattr(report, "output_text", None)
        if not raw:
            return None

        try:
            # Try to extract JSON from response
            text = raw.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            data = json.loads(text)
            return InvoiceData(
                invoice_number=data.get("invoice_number", ""),
                vendor_name=data.get("vendor_name", ""),
                customer_name=data.get("customer_name", ""),
                date=data.get("date", ""),
                due_date=data.get("due_date", ""),
                subtotal=float(data.get("subtotal", 0)),
                tax=float(data.get("tax", 0)),
                total=float(data.get("total", 0)),
                line_items=data.get("line_items", []),
                raw_response=raw,
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            # Return partial data with raw response
            return InvoiceData(raw_response=raw)

    def _log_result(self, result: ProcessingResult) -> None:
        """Log processing result."""
        status = "OK" if result.success else "FAIL"
        filename = Path(result.file_path).name
        cost = f"${result.cost_usd:.4f}" if result.cost_usd else "N/A"
        print(
            f"  [{status}] {filename:20} "
            f"tokens={result.tokens_used:4} cost={cost:8} "
            f"outcome={result.outcome}"
        )

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get processing metrics summary."""
        stats = self.metrics.snapshot()

        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        partial = [r for r in self.results if r.outcome == "budget_exceeded" and r.data]

        total_cost = sum(r.cost_usd for r in self.results)
        total_tokens = sum(r.tokens_used for r in self.results)

        return {
            "total_invoices": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "partial_extractions": len(partial),
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "avg_cost_per_invoice": total_cost / len(self.results) if self.results else 0,
            "avg_tokens_per_invoice": total_tokens / len(self.results) if self.results else 0,
            "percentiles": stats.get("percentiles", {}),
            "outcome_distribution": stats.get("outcome_distribution", {}),
        }

    def save_results(self, output_dir: Path) -> None:
        """Save extraction results to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save individual results
        results_data = []
        for r in self.results:
            result_dict = {
                "file": r.file_path,
                "success": r.success,
                "outcome": r.outcome,
                "tokens": r.tokens_used,
                "cost_usd": r.cost_usd,
                "elapsed_seconds": r.elapsed_seconds,
                "retries": r.retries,
            }
            if r.data:
                result_dict["extracted"] = {
                    "invoice_number": r.data.invoice_number,
                    "vendor_name": r.data.vendor_name,
                    "customer_name": r.data.customer_name,
                    "date": r.data.date,
                    "total": r.data.total,
                }
            if r.error:
                result_dict["error"] = r.error
            results_data.append(result_dict)

        # Write results
        (output_dir / "results.json").write_text(
            json.dumps(results_data, indent=2), encoding="utf-8"
        )

        # Write metrics
        metrics = self.get_metrics_summary()
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, default=str), encoding="utf-8"
        )

        print(f"\nResults saved to: {output_dir}")


def main():
    print("=" * 60)
    print("INVOICE PROCESSING PIPELINE")
    print("=" * 60)

    # Find invoices
    invoice_paths = list(INVOICES_DIR.glob("*.txt"))
    if not invoice_paths:
        print(f"No invoices found in {INVOICES_DIR}")
        return

    print(f"\nFound {len(invoice_paths)} invoices in {INVOICES_DIR}")

    # Initialize pipeline
    pipeline = InvoicePipeline(
        budget_per_item=BUDGET_PER_INVOICE_USD,
        tokens_per_item=BUDGET_PER_INVOICE_TOKENS,
        max_retries=MAX_RETRIES,
        parallelism=PARALLELISM,
    )

    # Process batch
    results = pipeline.process_batch(invoice_paths)

    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)

    metrics = pipeline.get_metrics_summary()
    print(f"\nTotal invoices:     {metrics['total_invoices']}")
    print(f"Successful:         {metrics['successful']}")
    print(f"Failed:             {metrics['failed']}")
    print(f"Partial extracts:   {metrics['partial_extractions']}")
    print(f"\nTotal cost:         ${metrics['total_cost_usd']:.4f}")
    print(f"Total tokens:       {metrics['total_tokens']}")
    print(f"Avg cost/invoice:   ${metrics['avg_cost_per_invoice']:.4f}")
    print(f"Avg tokens/invoice: {metrics['avg_tokens_per_invoice']:.0f}")

    # Percentiles
    if metrics.get("percentiles"):
        print("\nPercentiles:")
        for metric_name, values in metrics["percentiles"].items():
            if values.get("p50") is not None:
                print(
                    f"  {metric_name}: "
                    f"p50={values['p50']:.4f} "
                    f"p95={values['p95']:.4f}"
                )

    # Outcome distribution
    if metrics.get("outcome_distribution"):
        print("\nOutcome distribution:")
        for outcome, count in metrics["outcome_distribution"].items():
            print(f"  {outcome}: {count}")

    # Save results
    pipeline.save_results(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    # Show extracted data summary
    print("\nExtracted invoices:")
    for r in results:
        if r.success and r.data:
            print(
                f"  {r.data.invoice_number or 'N/A':20} "
                f"{r.data.vendor_name[:25]:25} "
                f"${r.data.total:>10,.2f}"
            )


if __name__ == "__main__":
    main()
