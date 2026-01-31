# Production

Real-world patterns for building applications with enzu.

## Session Management

### File Researcher
[`file_researcher.py`](file_researcher.py) - Session-based research with persistence

Features:
- Multi-turn research sessions
- Session save/load for continuity
- Budget tracking across steps
- Step limits for bounded exploration

### File Chatbot
[`file_chatbot.py`](file_chatbot.py) - Multi-turn chat with history management

Features:
- State persistence between runs
- History truncation (8k char limit) for context windows
- Conversation formatting

## Async Patterns

### Job Delegation
[`job_delegation_demo.py`](job_delegation_demo.py) - Fire-and-forget async tasks

Features:
- Non-blocking task submission
- Job status polling
- Background task handling
- `JobStatus` enum for state tracking

## External Integration

### Research with Exa
[`research_with_exa.py`](research_with_exa.py) - Web research integration

Features:
- Exa API for live search
- High-level `run()` API
- Provider selection

Requires: `EXA_API_KEY`

## Report Generation

### Report Service
[`report_service/`](report_service/) - Document analysis with bounded output

A complete production scenario:
- Loads corpus of documents
- Generates report with citations
- Enforces token, time, and cost budgets
- Outputs `report.md` + `trace.json`

**The core pattern:** Goal + Corpus + Hard Budgets = Bounded Output

## Architecture

Each production example should have:
- Clear purpose/use case
- Budget enforcement strategy
- Error handling for outcomes
- State management (if stateful)

See the `architecture.md` files in each subdirectory for detailed design docs.

## What's Next?

For advanced topics like RLM, metrics aggregation, and stress testing, see [`../advanced/`](../advanced/).
