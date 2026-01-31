# Enzu Examples

Learn to use enzu through practical examples, from simple quickstarts to production-ready patterns.

## Quick Start

```bash
# Set your API key
export OPENAI_API_KEY="your-key"
# or
export OPENROUTER_API_KEY="your-key"

# Run any example
python examples/basics/python_quickstart.py
```

## Structure

| Directory | Description | Complexity |
|-----------|-------------|------------|
| [`basics/`](basics/) | Quickstarts and minimal examples | 2-20 lines |
| [`concepts/`](concepts/) | Single-feature demos (budgets, outcomes) | 20-80 lines |
| [`production/`](production/) | Real-world patterns (sessions, jobs) | 50-200+ lines |
| [`advanced/`](advanced/) | RLM, metrics, stress testing | 100-300+ lines |

## Learning Path

### 1. Getting Started (5 min)
Start here if you're new to enzu:
- [`basics/python_quickstart.py`](basics/python_quickstart.py) - Your first enzu call
- [`basics/python_budget_guardrails.py`](basics/python_budget_guardrails.py) - Set spending limits

### 2. Core Concepts (15 min)
Understand budget enforcement:
- [`concepts/budget_hardstop_demo.py`](concepts/budget_hardstop_demo.py) - **The flagship demo**: budgets are enforced, not suggested
- [`concepts/typed_outcomes_demo.py`](concepts/typed_outcomes_demo.py) - Handle SUCCESS, BUDGET_EXCEEDED, TIMEOUT gracefully
- [`concepts/minimal_budget_exceeded.py`](concepts/minimal_budget_exceeded.py) - See what happens when budget runs out

### 3. Budget Strategies (15 min)
Different ways to cap costs:
- [`concepts/budget_cap_total_tokens.py`](concepts/budget_cap_total_tokens.py) - Cap by total tokens
- [`concepts/budget_cap_seconds.py`](concepts/budget_cap_seconds.py) - Cap by elapsed time
- [`concepts/budget_cap_cost_openrouter.py`](concepts/budget_cap_cost_openrouter.py) - Cap by USD cost

### 4. Production Patterns (30 min)
Build real applications:
- [`production/file_researcher.py`](production/file_researcher.py) - Session-based research with persistence
- [`production/file_chatbot.py`](production/file_chatbot.py) - Multi-turn chat with history management
- [`production/job_delegation_demo.py`](production/job_delegation_demo.py) - Async fire-and-forget tasks
- [`production/report_service/`](production/report_service/) - Document analysis with bounded output

### 5. Advanced Topics (30+ min)
Deep dives for power users:
- [`advanced/rlm_with_context.py`](advanced/rlm_with_context.py) - RLM with context documents
- [`advanced/run_metrics_demo.py`](advanced/run_metrics_demo.py) - Collect p50/p95/p99 metrics
- [`advanced/stress_testing_demo.py`](advanced/stress_testing_demo.py) - Failure injection testing

## API Keys

Most examples support multiple providers:

| Provider | Environment Variable | Notes |
|----------|---------------------|-------|
| OpenAI | `OPENAI_API_KEY` | Default for most examples |
| OpenRouter | `OPENROUTER_API_KEY` | Required for cost-based budgets |
| Exa | `EXA_API_KEY` | Only for `research_with_exa.py` |

Create a `.env` file to manage keys:
```bash
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...
```

## Running Examples

```bash
# From repo root
cd enzu

# Install dependencies
pip install -e .

# Run any example
python examples/basics/python_quickstart.py

# Run with specific provider
OPENROUTER_API_KEY="..." python examples/concepts/budget_cap_cost_openrouter.py
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding new examples.
