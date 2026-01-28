# AGENTS.md

## Commands
- **Package manager**: `uv` (preferred)
- **Install**: `uv pip install -e ".[dev]"` (fallback: `pip install -e ".[dev]"`)
- **Test all**: `pytest tests/`
- **Test fast (parallel, no containers)**: `uv run pytest -n auto -k "not container_pool_integration and not rlm_podman_security and not rlm_container_isolation" -m "not integration"`
- **Single test**: `pytest tests/test_file.py::test_function -v`
- **Type check**: `ty check enzu/`
- **Lint**: `ruff check enzu/ tests/`
- **Format**: `ruff format enzu/ tests/`

## Architecture
- `enzu/` - Core library: LLM orchestration with budget constraints
- `enzu/rlm/` - Reasoning Language Model engine (multi-step tasks)
- `enzu/providers/` - LLM provider adapters (OpenAI, Anthropic, etc.)
- `enzu/tools/` - Built-in tools for sandbox execution
- `enzu/server/` - FastAPI server for HTTP API
- `tests/` - pytest tests; prefix with `test_`

## Code Style
- Python 3.9+, strict typing with Pydantic v2
- Imports: stdlib → third-party → local (absolute imports)
- Types: Use type hints everywhere; `py.typed` marker present
- Errors: Raise specific exceptions, not generic Exception
- Naming: snake_case functions/vars, PascalCase classes
- No comments unless complex logic requires explanation
