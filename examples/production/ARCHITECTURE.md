# Production Examples Architecture

This document covers the architectural patterns used in production examples.

---

## File Researcher (`file_researcher.py`)

### Purpose
Multi-turn research sessions with state persistence.

### Data Flow
```
research_prompt.txt ──▶ Session.run() ──▶ research_output.txt
                              │
                              ▼
                       research_session.json (persisted state)
```

### Budget Enforcement
```python
Session(
    max_cost_usd=0.05,   # Total session budget
    max_tokens=1200,     # Total session tokens
)
session.run(
    tokens=400,          # Per-step token budget
    max_steps=4,         # Step limit
)
```

### Key Decisions

**Why session-based?**
- Maintains conversation context across runs
- Accumulates budget usage
- Enables iterative refinement

**Why file-based I/O?**
- Simple Unix-style interface
- Works with any editor
- Easy to script/automate

### Failure Modes
| Scenario | Behavior |
|----------|----------|
| Session budget exceeded | Raises `SessionBudgetExceeded` |
| Step limit hit | Returns best result so far |
| No prompt file | Exits with clear error |

---

## File Chatbot (`file_chatbot.py`)

### Purpose
Stateful chat with history management.

### Data Flow
```
chat_input.txt ──▶ run() ──▶ chat_output.txt
       │               │
       │               ▼
       └────── chat_state.json (history)
```

### Context Management
```
┌─────────────────────────────────────┐
│         Context Window              │
├─────────────────────────────────────┤
│ History (up to 8000 chars)          │
│ ┌─────────────────────────────────┐ │
│ │ User: Previous message 1        │ │
│ │ Assistant: Response 1           │ │
│ │ User: Previous message 2        │ │
│ │ Assistant: Response 2           │ │
│ │ ...                             │ │
│ └─────────────────────────────────┘ │
│                                     │
│ Current Input                       │
└─────────────────────────────────────┘
```

### Key Decisions

**Why truncate history at 8k chars?**
- Fits comfortably in most model context windows
- Leaves room for output
- Avoids cost explosion on long conversations

**Why reverse-order truncation?**
- Recent context is most relevant
- Gracefully degrades for long conversations

### Budget
```python
tokens=300  # Per-turn output limit
```

---

## Job Delegation (`job_delegation_demo.py`)

### Purpose
Async fire-and-forget task execution with polling.

### State Machine
```
┌──────────┐    submit()    ┌──────────┐    (auto)     ┌──────────┐
│ PENDING  │───────────────▶│ RUNNING  │──────────────▶│ COMPLETE │
└──────────┘                └──────────┘               └──────────┘
                                  │
                                  │ (error)
                                  ▼
                            ┌──────────┐
                            │ FAILED   │
                            └──────────┘
```

### API Pattern
```python
# Submit (non-blocking)
job = client.submit(task, tokens=200)

# Poll
while job.status in (JobStatus.PENDING, JobStatus.RUNNING):
    job = client.status(job.job_id)
    sleep(0.5)

# Result
if job.answer:
    print(job.answer)
```

### Key Decisions

**Why job-based?**
- Decouples submission from completion
- Enables cancellation
- Works with webhook patterns

**Why not async/await?**
- Simpler mental model
- Works across process boundaries
- File-based job store (no Redis needed)

---

## Research with Exa (`research_with_exa.py`)

### Purpose
Web research with external tool integration.

### Data Flow
```
TOPIC ──▶ run(data=TOPIC) ──▶ RLM Engine
                                  │
                                  ├─▶ research() tool ──▶ Exa API
                                  │                          │
                                  │◀──────────────── search results
                                  │
                                  └─▶ synthesize ──▶ output
```

### Tool Integration
```
┌─────────────────────────────────────────┐
│              RLM Sandbox                │
├─────────────────────────────────────────┤
│                                         │
│  Available Tools:                       │
│  ┌─────────────────────────────────┐   │
│  │ research(query)                 │   │
│  │   └─▶ Exa.search()             │   │
│  └─────────────────────────────────┘   │
│                                         │
│  Budget: tokens=1200                    │
│                                         │
└─────────────────────────────────────────┘
```

### Key Decisions

**Why Exa?**
- Clean API for web search
- Returns structured content
- Good for research tasks

**Why RLM mode?**
- Automatic tool discovery
- Goal-oriented execution
- Model decides when to search

---

## Common Patterns

### Provider Detection
```python
PROVIDER = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "openai"
```
Works with either provider, preferring OpenRouter for cost-based budgets.

### Environment Variables
| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI authentication |
| `OPENROUTER_API_KEY` | OpenRouter authentication |
| `EXA_API_KEY` | Exa search (research_with_exa only) |

### File Convention
| File | Purpose |
|------|---------|
| `*_input.txt` | User input |
| `*_output.txt` | Model output |
| `*_state.json` / `*_session.json` | Persistent state |
