# Invoice Processing Pipeline

Batch document extraction at scale with strict budget control.

## Use Case

Process invoices with per-document budget caps. Example scenario:
- 1000 invoices to process
- $10 total budget
- $0.02 max per document

## Features

- **Per-item budget allocation** - Each invoice has its own cost cap
- **Batch processing** - Parallel execution with configurable concurrency
- **Graceful degradation** - Partial results when budget is tight
- **Retry logic** - Smart retry with exponential backoff
- **Cost reporting** - p50/p95 metrics, budget utilization

## Quick Start

```bash
# Set your API key
export OPENAI_API_KEY=sk-...

# Or use OpenRouter for real-time cost tracking
export OPENROUTER_API_KEY=sk-or-...

# Run the pipeline
python examples/production/invoice_processing/pipeline.py
```

## Output

```
INVOICE PROCESSING PIPELINE
============================================================

Found 5 invoices in sample_invoices

Processing 5 invoices...
  Budget per invoice: $0.0200
  Token limit: 300
  Parallelism: 3
--------------------------------------------------
  [OK] invoice_001.txt     tokens= 180 cost=$0.0012  outcome=success
  [OK] invoice_002.txt     tokens= 165 cost=$0.0011  outcome=success
  [OK] invoice_003.txt     tokens= 195 cost=$0.0013  outcome=success
  ...

PROCESSING SUMMARY
============================================================

Total invoices:     5
Successful:         5
Failed:             0
Partial extracts:   0

Total cost:         $0.0058
Total tokens:       890
Avg cost/invoice:   $0.0012
```

## Configuration

Edit the constants in `pipeline.py`:

```python
BUDGET_PER_INVOICE_USD = 0.02   # $0.02 max per invoice
BUDGET_PER_INVOICE_TOKENS = 300 # Token limit per invoice
MAX_RETRIES = 2                 # Retry attempts on failure
PARALLELISM = 3                 # Concurrent processing threads
```

## Architecture

See [architecture.md](architecture.md) for design details.

## Sample Data

The `sample_invoices/` directory contains 5 test invoices:

| File | Type | Total |
|------|------|-------|
| invoice_001.txt | SaaS Services | $912.55 |
| invoice_002.txt | Office Supplies | $647.29 |
| invoice_003.txt | Consulting | $20,940.90 |
| invoice_004.txt | Design Services | $11,395.50 |
| invoice_005.txt | Manufacturing Parts | $8,800.00 |

## Extracted Fields

The pipeline extracts:
- Invoice number
- Vendor name
- Customer name
- Date / Due date
- Subtotal, tax, total
- Line items (description, qty, price, amount)

## Budget Strategies

### Conservative (minimize failures)
```python
BUDGET_PER_INVOICE_USD = 0.03   # More headroom
BUDGET_PER_INVOICE_TOKENS = 400
```

### Aggressive (minimize cost)
```python
BUDGET_PER_INVOICE_USD = 0.01
BUDGET_PER_INVOICE_TOKENS = 200
```

### Balanced (recommended)
```python
BUDGET_PER_INVOICE_USD = 0.02
BUDGET_PER_INVOICE_TOKENS = 300
```

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Budget exceeded | Returns partial result if available |
| API error | Retries with exponential backoff |
| Parse failure | Logs error, continues batch |
| All retries fail | Records as failed, batch continues |

## Scaling

For production use with thousands of invoices:

1. **Increase parallelism**: `PARALLELISM = 10` or more
2. **Use connection pooling**: `use_pool=True` in Enzu client
3. **Add checkpointing**: Save progress periodically
4. **Use queue**: Consider `enzu.queue` for very large batches

## Output Files

Results are saved to `output/`:
- `results.json` - Per-invoice extraction results
- `metrics.json` - Aggregate metrics (costs, percentiles)
