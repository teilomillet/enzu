## RLM examples tests

These tests exercise the RLM loop with deterministic fixtures and a mock provider.

## Run locally

- `pytest tests/test_rlm_example1.py`
- `pytest tests/test_rlm_example2.py`
- `pytest tests/test_rlm_example3.py`
- `pytest tests/test_failures.py`

## Fixtures

- `tests/fixtures/example1/context.txt` is a small synthetic corpus.
- `tests/fixtures/example2/context.txt` is a tiny codebase snapshot.
- `tests/fixtures/example3/questions.txt` is a short batching dataset.

## Mock provider

`tests/providers/mock_provider.py` scripts model outputs and subcalls so the RLM
loop can run without network access. It uses the `SUBCALL:` prefix to route
`llm_query()` calls to deterministic responses.

## Failure mode coverage

`tests/test_failures.py` exercises:

- Engine preflight budget rejection
- Provider errors
- Verification failures
- RLM sandbox code errors
- RLM max-steps without FINAL
- RLM budget exhaustion during subcalls

## Optional integration runs

Set these env vars to run the integration tests:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `EXAMPLE1_CONTEXT_PATH`
- `EXAMPLE2_CONTEXT_PATH`
