# Contributing

Thanks for helping improve enzu.

## Setup

```bash
uv pip install -e ".[dev]"
```

## Checks

```bash
ruff format enzu/ tests/
ruff check enzu/ tests/
pytest tests/
```

## Pull Requests

- Keep changes focused and easy to review.
- Add or update tests when behavior changes.
- Update docs if public APIs or usage change.

## Code style

- Python 3.9+
- Typed APIs (Pydantic v2)
- Clear, explicit errors
