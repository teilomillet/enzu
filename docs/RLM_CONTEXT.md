# RLM Context Management: Mitigating Wasted Context

Production RLM (Recursive Language Model) workloads face unique context challenges. This guide explains how to measure and mitigate wasted context in enzu's RLM mode.

## The Problem

Based on findings from [Recursive Language Models (arxiv.org/html/2512.24601v2)](https://arxiv.org/html/2512.24601v2), RLM systems exhibit:

1. **Cost variance**: Trajectory length unpredictability causes 2-10x cost variation
2. **Context waste**: Large prompts/retrieved context that rarely change the answer
3. **Scaling challenges**: Direct context feeding doesn't scale beyond model windows

Traditional approaches (passing all context inline) break down with:
- Large RAG document sets
- Multi-step reasoning requiring selective context access
- Budgets that can't absorb trajectory variance

## Symbolic vs Direct Context

### Direct (Inline) Context

**What it is**: Passing all data directly in the prompt.

```python
from enzu.rlm import RLMEngine

# All documents concatenated into prompt
context = "Doc1: ...\nDoc2: ...\nDoc3: ..."
inline_data = f"{query}\n\n{context}"

engine = RLMEngine()
report = engine.run(task, provider, data=None)
```

**Problems**:
- Every LLM call (main + subcalls) includes ALL context
- Tokens consumed even for irrelevant documents
- Doesn't scale beyond context window
- High cost variance due to retry amplification

### Symbolic (File-Based) Context

**What it is**: Treating context as external data accessible programmatically.

```python
from enzu.rlm import RLMEngine
from pathlib import Path

# Write context to file
context_file = Path("documents.txt")
context_file.write_text(large_context)

# Pass file reference, not inline data
engine = RLMEngine()
report = engine.run(task, provider, data=large_context)
```

**Benefits**:
- RLM reads only what it needs via file operations
- Selective access reduces token consumption
- Scales to arbitrarily large context
- Lower cost variance (fewer wasted tokens in subcalls)

## Quick Start

```python
from enzu import Budget, TaskSpec, SuccessCriteria
from enzu.rlm import RLMEngine, ContextBreakdown
from pathlib import Path

# Prepare symbolic context
context_file = Path("retrieved_docs.txt")
context_file.write_text(large_context_from_rag)

# Configure task with file reference
task = TaskSpec(
    task_id="rag-query",
    input_text="Summarize the key findings about X",
    model="gpt-4o-mini",
    budget=Budget(max_total_tokens=5000),
    success_criteria=SuccessCriteria(min_word_count=50),
    metadata={"context_file": str(context_file)},
)

# Run with symbolic context
engine = RLMEngine(max_steps=8)
report = engine.run(task, provider, data=large_context_from_rag)

# Check context breakdown
if report.context_breakdown:
    breakdown = report.context_breakdown
    print(f"Total context: {breakdown['total_context_chars']} chars")
    print(f"Symbolic ratio: {breakdown['symbolic_ratio']:.1%}")
    print(f"Context efficiency: {breakdown['context_efficiency']:.1%}")
```

## Context Breakdown Metrics

Enzu's RLM mode tracks context usage patterns:

| Metric | Description |
|--------|-------------|
| `system_prompt_chars` | System prompt + instructions overhead |
| `task_prompt_chars` | User task/query |
| `inline_data_chars` | Data passed directly in prompts |
| `file_data_chars` | Data accessible via file reads |
| `file_reads` | Number of file read operations |
| `file_bytes_read` | Bytes actually accessed |
| `symbolic_ratio` | Proportion of symbolic vs inline context |
| `context_efficiency` | Ratio of accessed vs available data |

### Interpreting Metrics

**High symbolic ratio (>0.8)**:
- Good: Most context is file-based
- RLM can selectively access only needed portions
- Lower token overhead

**Low context efficiency (<0.3)**:
- Good: RLM accessed only 30% of available data
- Selective reading saves tokens
- Consider: Is unused context necessary?

**High inline_data_chars**:
- Warning: Large inline data increases every LLM call cost
- Action: Move to file-based symbolic context

## Trajectory Variance

RLM trajectories vary in length due to:
- Recursion depth (subcall spawning)
- Step count (iteration until FINAL())
- Context access patterns

### Tracking Variance

```python
from enzu.rlm import get_rlm_context_tracker, RLMTrajectoryMetrics

tracker = get_rlm_context_tracker()

# Run multiple tasks
for task in tasks:
    report = engine.run(task, provider)

    # Record trajectory
    metrics = RLMTrajectoryMetrics(
        run_id=task.task_id,
        max_depth=len(report.steps),  # Simplified
        total_steps=len(report.steps),
        total_input_tokens=report.budget_usage.input_tokens or 0,
        total_output_tokens=report.budget_usage.output_tokens or 0,
        elapsed_seconds=report.budget_usage.elapsed_seconds,
        cost_usd=report.budget_usage.cost_usd,
        success=report.success,
        outcome=report.outcome.value,
    )
    tracker.record(metrics)

# Analyze variance
summary = tracker.summary()
print(f"Steps p50: {summary['steps_p50']}")
print(f"Steps p95: {summary['steps_p95']}")
print(f"Complexity p95: {summary['complexity_p95']}")
```

### Mitigating Variance

**Budget conservatively for p95**:
```python
# Don't budget for median case
budget = Budget(max_total_tokens=2000)  # ✗ Fails at p95

# Budget for p95 trajectory length
budget = Budget(max_total_tokens=8000)  # ✓ Handles variance
```

**Limit recursion depth**:
```python
engine = RLMEngine(
    max_recursion_depth=1,  # Limit subcall nesting
    subcall_max_steps=3,    # Shorter subcall trajectories
)
```

**Use concurrency limits**:
```python
from enzu.limits import configure_global_limiter

# Prevent retry storms during rate limit events
configure_global_limiter(max_concurrent=10)
```

## Best Practices

### 1. Prefer Symbolic Context for RAG

❌ **Don't** pass retrieved documents inline:
```python
# Wasteful: all docs in every prompt
docs = "\n".join(retrieved_documents)
report = engine.run(task, provider, data=None)
```

✅ **Do** use file-based symbolic context:
```python
# Efficient: RLM reads selectively
context_file.write_text("\n".join(retrieved_documents))
report = engine.run(task, provider, data=retrieved_documents)
```

### 2. Track Context Breakdown

Monitor context usage patterns:
```python
if report.context_breakdown:
    breakdown = report.context_breakdown

    # Alert on high inline usage
    if breakdown['inline_data_chars'] > 10000:
        logger.warning(f"Large inline context: {breakdown['inline_data_chars']} chars")

    # Check symbolic efficiency
    if breakdown['symbolic_ratio'] > 0.5 and breakdown['context_efficiency'] < 0.2:
        logger.info("Good: selective context access (20% efficiency)")
```

### 3. Budget for Trajectory Variance

Don't assume median trajectory:
```python
# Collect p95 metrics in staging
tracker = get_rlm_context_tracker()
for _ in range(100):
    report = engine.run(task, provider)
    # ... track metrics

summary = tracker.summary()
p95_steps = summary['steps_p95']

# Budget for p95, not average
budget = Budget(
    max_total_tokens=int(p95_steps * avg_tokens_per_step * 1.2)  # +20% buffer
)
```

### 4. Use Step Limits

Prevent runaway trajectories:
```python
engine = RLMEngine(
    max_steps=8,              # Main trajectory limit
    subcall_max_steps=3,      # Subcall limit
    max_recursion_depth=1,    # Nesting limit
)
```

### 5. Monitor Cost Variance

Track p95 cost/run:
```python
from enzu.metrics import RunEvent, get_run_metrics

collector = get_run_metrics()

for task in tasks:
    started = datetime.now(timezone.utc)
    report = engine.run(task, provider)
    finished = datetime.now(timezone.utc)

    event = RunEvent.from_execution_report(
        run_id=task.task_id,
        report=report,
        started_at=started,
        finished_at=finished,
    )
    collector.observe(event)

# Check p95 cost
stats = collector.snapshot()
p95_cost = stats['percentiles']['cost_usd']['p95']
print(f"p95 cost/run: ${p95_cost:.4f}")
```

## Context Optimization Workflow

1. **Baseline measurement**:
   - Run with inline context
   - Record token usage and cost
   - Note trajectory variance

2. **Switch to symbolic**:
   - Move context to files
   - Use file-based data parameter
   - RLM accesses programmatically

3. **Measure improvement**:
   - Compare token savings
   - Check context efficiency
   - Validate answer quality

4. **Tune budgets**:
   - Set max_total_tokens for p95
   - Adjust step limits
   - Configure recursion depth

## Advanced: Custom Context Instrumentation

Track file access patterns:

```python
from enzu.rlm import RLMEngine, ContextBreakdown

class InstrumentedRLMEngine(RLMEngine):
    def run(self, task, provider, data=None, **kwargs):
        # Track context access
        breakdown = ContextBreakdown(
            task_prompt_chars=len(task.input_text),
            file_data_chars=len(data) if data else 0,
            used_symbolic_context=data is not None,
        )

        report = super().run(task, provider, data, **kwargs)

        # Attach breakdown
        report.context_breakdown = breakdown.to_dict()
        return report
```

## Integration with Run Metrics

RLM context metrics integrate with enzu's run metrics:

```python
from enzu.metrics import RunEvent, get_run_metrics
from enzu.rlm import get_rlm_context_tracker

# Collect both run and trajectory metrics
run_collector = get_run_metrics()
rlm_tracker = get_rlm_context_tracker()

# Run tasks
for task in tasks:
    report = engine.run(task, provider)

    # Record run event
    event = RunEvent.from_execution_report(...)
    run_collector.observe(event)

    # Record RLM trajectory
    trajectory = RLMTrajectoryMetrics(
        run_id=task.task_id,
        # ... populate from report
    )
    rlm_tracker.record(trajectory)

# Analyze both dimensions
run_stats = run_collector.snapshot()
rlm_summary = rlm_tracker.summary()

print(f"p95 cost/run: ${run_stats['percentiles']['cost_usd']['p95']}")
print(f"p95 steps: {rlm_summary['steps_p95']}")
print(f"p95 complexity: {rlm_summary['complexity_p95']}")
```

## Examples

See:
- [examples/rlm_context_optimization.py](../examples/rlm_context_optimization.py) - Before/after comparison
- [examples/rlm_with_context.py](../examples/rlm_with_context.py) - Basic RLM usage

## References

- [Recursive Language Models (arxiv.org/html/2512.24601v2)](https://arxiv.org/html/2512.24601v2) - Original paper
- [docs/RUN_METRICS.md](./RUN_METRICS.md) - p95 cost tracking
- [docs/STRESS_TESTING.md](./STRESS_TESTING.md) - Testing degraded conditions

## Why This Matters

| Approach | Context Size | Token Cost | Scales? |
|----------|--------------|------------|---------|
| Inline (direct) | Fixed in prompt | High (every call) | No (context window limit) |
| Symbolic (file) | Accessible on-demand | Low (selective access) | Yes (unbounded) |

**Symbolic context makes RLM workloads production-viable by:**
- Reducing token overhead 30-70% for RAG workloads
- Enabling context sizes beyond model windows
- Lowering cost variance through selective access
- Matching the architecture from the RLM paper
