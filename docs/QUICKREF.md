# Quick Reference

## Providers

| Provider | Env Var | Model Format | Example |
|----------|---------|--------------|---------|
| openai | `OPENAI_API_KEY` | `model-name` | `gpt-4o`, `gpt-4o-mini` |
| openrouter | `OPENROUTER_API_KEY` | `org/model` | `openai/gpt-4o`, `anthropic/claude-3.5-sonnet` |
| groq | `GROQ_API_KEY` | `model-name` | `llama-3.3-70b-versatile`, `llama-3.1-8b-instant` |
| together | `TOGETHER_API_KEY` | `org/model` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| mistral | `MISTRAL_API_KEY` | `model-name` | `mistral-large-latest`, `mistral-small-latest` |
| deepseek | `DEEPSEEK_API_KEY` | `model-name` | `deepseek-chat`, `deepseek-reasoner` |
| gemini | `GEMINI_API_KEY` | `model-name` | `gemini-2.0-flash`, `gemini-1.5-pro` |
| xai | `XAI_API_KEY` | `model-name` | `grok-2`, `grok-2-mini` |
| fireworks | `FIREWORKS_API_KEY` | `accounts/org/models/name` | `accounts/fireworks/models/llama-v3p1-70b-instruct` |
| ollama | none (local) | `model-name` | `llama3`, `mistral`, `codellama` |
| lmstudio | none (local) | `model-name` | (whatever you loaded) |

## OpenAI extra env vars

```bash
OPENAI_API_KEY=sk-...        # Required
OPENAI_ORG=org-...           # Optional: organization ID
OPENAI_PROJECT=proj-...      # Optional: project ID
```

## OpenRouter extra env vars

```bash
OPENROUTER_API_KEY=sk-or-... # Required
OPENROUTER_REFERER=...       # Optional: your app's URL
OPENROUTER_APP_NAME=...      # Optional: your app's name
```

## Default behavior

```python
from enzu import run

# No provider specified → defaults to "openrouter"
# So this looks for OPENROUTER_API_KEY, not OPENAI_API_KEY
run("Hello", model="openai/gpt-4o")

# Explicit provider → uses that provider's env var
run("Hello", model="gpt-4o", provider="openai")  # uses OPENAI_API_KEY
```

## Common patterns

### Simple call (OpenAI)

```bash
export OPENAI_API_KEY=sk-...
```

```python
from enzu import run
answer = run("What is 2+2?", model="gpt-4o", provider="openai")
```

### Simple call (OpenRouter)

```bash
export OPENROUTER_API_KEY=sk-or-...
```

```python
from enzu import run
answer = run("What is 2+2?", model="openai/gpt-4o", provider="openrouter")
```

### With budget

```python
answer = run(
    "Summarize this",
    data=document,
    model="gpt-4o",
    provider="openai",
    cost=0.10,      # max $0.10
    tokens=500,     # max 500 output tokens
    seconds=30,     # max 30 seconds
)
```

### Multi-turn session

```python
from enzu import Session

session = Session(model="gpt-4o", provider="openai", max_cost_usd=1.00)
answer1 = session.run("Hello")
answer2 = session.run("What did I just say?")  # has context
```

### Local model (Ollama)

```bash
ollama run llama3  # start model first
```

```python
from enzu import run
answer = run("Hello", model="llama3", provider="ollama")
```

## run() parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `task` | str | The prompt |
| `model` | str | Model name (required) |
| `provider` | str | Provider name (default: "openrouter") |
| `data` | str | Context to include with prompt |
| `cost` | float | Max cost in USD |
| `tokens` | int | Max output tokens |
| `seconds` | float | Max wall-clock seconds |
| `return_report` | bool | Return full report instead of just text |

## Session parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | str | Model name (required) |
| `provider` | str | Provider name (default: "openrouter") |
| `max_cost_usd` | float | Session cost cap |
| `max_tokens` | int | Session output token cap (cumulative) |
| `history_max_chars` | int | Max history size (default: 10000) |

## Register custom provider

```python
from enzu import register_provider

register_provider(
    "mycompany",
    base_url="https://api.mycompany.com/v1",
    supports_responses=False,
)

# Now use it
run("Hello", model="my-model", provider="mycompany")
# Looks for MYCOMPANY_API_KEY
```

## Sources

- [OpenRouter models](https://openrouter.ai/models)
- [Groq models](https://console.groq.com/docs/models)
- [Together AI models](https://www.together.ai/models)
- [Mistral models](https://docs.mistral.ai/getting-started/models)
- [DeepSeek API](https://api-docs.deepseek.com/)
