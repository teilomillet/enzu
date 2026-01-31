# Report Service Cost Analysis

## Budget Configuration

| Budget Type | Default Value | Purpose |
|------------|---------------|---------|
| `BUDGET_TOKENS` | 1200 | Output length limit |
| `BUDGET_SECONDS` | 120 | Execution time limit |
| `BUDGET_COST` | $0.50 | Direct spend limit |

## Expected Costs

### With GPT-4o-mini (Default)
| Corpus Size | Input Tokens | Output Tokens | Est. Cost |
|-------------|--------------|---------------|-----------|
| 4 docs (~4KB) | ~1,500 | ~1,000 | ~$0.002 |
| 10 docs (~10KB) | ~4,000 | ~1,200 | ~$0.006 |
| 50 docs (~50KB) | ~15,000 | ~1,200 | ~$0.02 |

### With GPT-4o
| Corpus Size | Input Tokens | Output Tokens | Est. Cost |
|-------------|--------------|---------------|-----------|
| 4 docs (~4KB) | ~1,500 | ~1,000 | ~$0.03 |
| 10 docs (~10KB) | ~4,000 | ~1,200 | ~$0.08 |
| 50 docs (~50KB) | ~15,000 | ~1,200 | ~$0.25 |

*Costs estimated at Jan 2025 OpenAI pricing. Actual costs may vary.*

## Optimization Opportunities

### 1. Right-size the output budget
```python
# Current: generous output budget
tokens=1200

# Optimized: match actual need
tokens=800  # Typical report needs ~600-800 tokens
```
**Savings**: ~30% on output costs

### 2. Use cheaper models for simple corpora
```python
# For simple summarization
client = Enzu(model="gpt-4o-mini")  # 15x cheaper than GPT-4o
```

### 3. Pre-filter documents
```python
# Only include relevant docs
def load_corpus(keywords: list[str]) -> str:
    parts = []
    for p in DOCS_DIR.glob("*.txt"):
        content = p.read_text()
        if any(kw in content.lower() for kw in keywords):
            parts.append(f"=== {p.name} ===\n{content}\n")
    return "\n".join(parts)
```
**Savings**: Proportional to filtered docs

### 4. Cache expensive operations
```python
import hashlib

def get_cached_report(corpus_hash: str) -> str | None:
    cache_path = OUTPUT_DIR / f".cache/{corpus_hash}.md"
    if cache_path.exists():
        return cache_path.read_text()
    return None

corpus_hash = hashlib.sha256(corpus.encode()).hexdigest()[:16]
```
**Savings**: 100% on cache hits

## Budget Allocation Strategy

### Conservative (minimize risk of overrun)
```python
BUDGET_TOKENS = 800    # Leave headroom
BUDGET_SECONDS = 60    # Tight time limit
BUDGET_COST = 0.10     # Low spend cap
```

### Balanced (recommended)
```python
BUDGET_TOKENS = 1200   # Typical report needs
BUDGET_SECONDS = 120   # Comfortable time
BUDGET_COST = 0.50     # Room for retries
```

### Generous (maximize quality)
```python
BUDGET_TOKENS = 2000   # Allow detailed reports
BUDGET_SECONDS = 300   # Handle slow networks
BUDGET_COST = 2.00     # Allow GPT-4o
```

## Cost Monitoring

The `trace.json` output includes:
```json
{
  "usage": {
    "output": 847,
    "total": 2341,
    "elapsed": 3.2,
    "cost": 0.0034
  }
}
```

Track these metrics over time to:
1. Right-size budgets
2. Detect cost anomalies
3. Forecast monthly spend

## ROI Considerations

| Factor | Impact |
|--------|--------|
| Analyst time saved | ~2 hours per report |
| Cost per report | ~$0.01-0.10 |
| Break-even | ~1 report/month |

The value is in bounded execution: you know the maximum cost *before* running.
