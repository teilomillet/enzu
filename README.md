![enzu](github-32.jpeg)

# enzu

LLM tasks with hard budget limits.

![PyPI](https://img.shields.io/pypi/v/enzu)
![Python](https://img.shields.io/pypi/pyversions/enzu)
![License](https://img.shields.io/github/license/teilomillet/enzu)

enzu is a Python toolkit for running LLM tasks with **guaranteed spending limits**. Set a token, time, or cost cap—enzu enforces it. No runaway API bills.

## Quickstart

```bash
pip install enzu
export OPENAI_API_KEY=sk-...
```

```python
from enzu import ask

print(ask("Explain quantum computing in one sentence."))
```

## The problem enzu solves

LLM APIs charge per token. Without limits, a single bad prompt can cost $50+. enzu stops execution **before** you exceed your budget:

```python
from enzu import Enzu

client = Enzu()

# You ask for 500 words, but cap output at 50 tokens
result = client.run(
    "Write a 500-word essay on climate change.",
    tokens=50,  # Hard limit: stops here, guaranteed
)
# Result: partial output, no surprise bill
```

## What enzu is / isn't

| enzu is | enzu is not |
|---------|-------------|
| A budget enforcement layer | A prompt library |
| Hard stops when limits hit | "Best effort" throttling |
| Works with any OpenAI-compatible API | Tied to one provider |
| ~2k lines of code | A heavyweight framework |

## Core features

### 1. Budget limits

Cap by tokens, seconds, or dollars:

```python
client = Enzu()

# Cap by tokens
result = client.run("Summarize this", data=text, tokens=200)

# Cap by time
result = client.run("Research this topic", seconds=30)

# Cap by cost (requires OpenRouter)
result = client.run("Analyze this data", cost=0.50)  # Max $0.50
```

### 2. Predictable error handling

Every call returns a status you can check:

```python
from enzu import Enzu, Outcome

client = Enzu()
result = client.run("Analyze this", data=doc, tokens=100, return_report=True)

if result.outcome == Outcome.SUCCESS:
    print(result.answer)
elif result.outcome == Outcome.BUDGET_EXCEEDED:
    print("Hit the limit, partial result available")
elif result.outcome == Outcome.TIMEOUT:
    print("Took too long")
```

### 3. Document analysis

Parse PDFs, Word docs, and other files:

```bash
pip install enzu[docling]
```

```python
from enzu import Enzu

client = Enzu()

# Ask questions about a PDF
result = client.run(
    "What are the key findings?",
    documents=["quarterly-report.pdf"],
    tokens=500,
)

# Multi-turn conversation with document context
session = client.session(documents=["research-paper.pdf"])
answer1 = session.run("What's the main argument?")
answer2 = session.run("What evidence supports it?")
answer3 = session.run("What are the limitations?")
```

### 4. Long document handling

Documents too large for the model's context window? enzu automatically splits them into chunks and synthesizes the answer:

```python
client = Enzu()

# Works even with 100k+ token documents
answer = client.run(
    "Summarize the main conclusions",
    data=open("huge-report.txt").read(),
    tokens=500,
)
```

### 5. Background jobs

For long-running tasks, fire and poll:

```python
from enzu import Enzu, JobStatus

client = Enzu()
job = client.submit("Analyze this dataset", data=data, cost=5.0)

# Check later
job = client.status(job.job_id)
if job.status == JobStatus.COMPLETED:
    print(job.answer)
```

## HTTP API

Run enzu as a server:

```bash
pip install enzu[server]
uvicorn enzu.server:app --port 8000
```

```bash
curl http://localhost:8000/v1/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Say hello", "model": "gpt-4o"}'
```

## Examples

| Folder | What's inside |
|--------|---------------|
| [`examples/basics/`](examples/basics/) | First steps, minimal code |
| [`examples/concepts/`](examples/concepts/) | Budget caps, error handling |
| [`examples/production/`](examples/production/) | Document Q&A, async jobs, sessions |
| [`examples/advanced/`](examples/advanced/) | Metrics, stress testing |
| [`examples/usecases/`](examples/usecases/) | Code reviewer, summarizer, data extractor |

Start here:
- [`basics/python_quickstart.py`](examples/basics/python_quickstart.py) — First call
- [`concepts/budget_hardstop_demo.py`](examples/concepts/budget_hardstop_demo.py) — See budget enforcement in action
- [`production/document_qa/pipeline.py`](examples/production/document_qa/pipeline.py) — PDF analysis

## Documentation

- [Getting started](docs/README.md)
- [Provider setup](docs/QUICKREF.md) — OpenAI, OpenRouter, etc.
- [HTTP API reference](docs/SERVER.md)
- [Python API reference](docs/PYTHON_API_REFERENCE.md)
- [Recipes & patterns](docs/COOKBOOK.md)

## Requirements

Python 3.9+

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
