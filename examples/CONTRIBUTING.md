# Contributing Examples

Guidelines for adding new examples to the enzu examples directory.

## Example Types

| Type | Location | When to Use |
|------|----------|-------------|
| **Minimal** | `basics/` | Ultra-short demos (< 20 lines), quick starts |
| **Concept** | `concepts/` | Single-feature demos, learning material |
| **Production** | `production/` | Real-world patterns, multi-file examples |
| **Advanced** | `advanced/` | RLM, metrics, stress testing |

## Quick Start

Use the template generator:

```bash
# Minimal example
python scripts/new_example.py --type minimal --name my_example

# Concept demo
python scripts/new_example.py --type concept --name budget_demo

# Production example
python scripts/new_example.py --type production --name my_service
```

## Structure Guidelines

### Minimal Examples

Single file, < 20 lines:

```python
#!/usr/bin/env python3
"""One-line description.

Run: python examples/basics/my_example.py
"""

from enzu import ask

print(ask("Your prompt"))
```

### Concept Examples

Single file with README, 20-80 lines:

```
examples/concepts/my_demo/
├── my_demo.py       # Main code
└── README.md        # What it demonstrates
```

### Production Examples

Full structure:

```
examples/production/my_service/
├── README.md           # Setup, quick start
├── architecture.md     # Design decisions
├── main.py             # Entry point
├── config.py           # Configuration (if needed)
├── tests/
│   ├── __init__.py
│   └── test_main.py
└── requirements.txt    # Extra dependencies (if needed)
```

## Testing Requirements

### All Examples Must

1. **Run without errors** with current API
2. **Have docstrings** explaining purpose and usage
3. **Use environment variables** for API keys (never hardcode)

### Production Examples Must Also

1. **Have tests** in `tests/` subdirectory
2. **Have architecture.md** explaining design
3. **Use MockProvider** for testing (no API calls in tests)

## Testing Approach

### Using MockProvider

Tests should not make real API calls. Use the mock provider:

```python
from enzu import Enzu
from conftest import MockProvider

def test_my_example(mock_provider_factory):
    mock = mock_provider_factory(["Expected response"])
    client = Enzu(provider=mock, model="test-model")

    result = client.run("Test task", tokens=100, return_report=True)

    assert result.outcome in (Outcome.SUCCESS, Outcome.BUDGET_EXCEEDED)
```

### Test Categories

| Category | Purpose | Example |
|----------|---------|---------|
| Smoke | Runs without error | `test_basic_run()` |
| Structure | Output has expected fields | `test_report_fields()` |
| Budget | Limits are respected | `test_token_limit()` |
| Logic | Business logic works | `test_history_truncation()` |

### Running Tests

```bash
# Run all example tests
uv run pytest examples/production/tests/ -v

# Run specific test file
uv run pytest examples/production/tests/test_my_example.py -v

# Run with coverage
uv run pytest examples/production/tests/ --cov=examples
```

## Code Style

### Imports

```python
# Standard library
import os
import sys
from pathlib import Path

# Third party (if needed)
from dotenv import load_dotenv

# Enzu
from enzu import Enzu, Outcome, Limits
```

### Provider Detection

Support both OpenAI and OpenRouter:

```python
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PROVIDER = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "openai"
```

### Error Messages

Be specific about requirements:

```python
if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
    raise SystemExit("Set OPENAI_API_KEY or OPENROUTER_API_KEY")
```

## Documentation

### README.md Template

```markdown
# Example Name

## What It Demonstrates

- Key concept 1
- Key concept 2

## Quick Start

\`\`\`bash
export OPENAI_API_KEY=sk-...
python examples/category/my_example.py
\`\`\`

## Expected Output

\`\`\`
Output here
\`\`\`

## Key Code

\`\`\`python
# The important part
\`\`\`
```

### architecture.md Template

```markdown
# Architecture

## Component Diagram

[ASCII or Mermaid diagram]

## Data Flow

1. Step 1
2. Step 2

## Key Decisions

### Decision 1

**Trade-off**: What was considered
**Chosen**: What was chosen and why
```

## Checklist

Before submitting a new example:

- [ ] Runs without errors
- [ ] Has docstring with run instructions
- [ ] Uses environment variables for keys
- [ ] Has README (for concept/production)
- [ ] Has tests (for production)
- [ ] Has architecture.md (for production)
- [ ] Added to parent README's table of contents
- [ ] CI passes

## Need Help?

- Check existing examples for patterns
- Look at `tests/providers/mock_provider.py` for mock usage
- Open an issue with questions
