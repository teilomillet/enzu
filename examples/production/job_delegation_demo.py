#!/usr/bin/env python3
"""
Demo: Async job mode (delegation).

Shows how to submit long-running tasks as background jobs
and poll for completion - the "delegation" pattern from
the conversation/delegation threshold framing.

Run:
    export OPENAI_API_KEY=sk-...
    uv run examples/job_delegation_demo.py
"""

import time

from enzu import Enzu, JobStatus

client = Enzu()

print("=" * 60)
print("DEMO: Async job mode (delegation)")
print("=" * 60)

# Submit a task as a background job
print("\n📤 Submitting job...")
job = client.submit(
    "Explain the key concepts of machine learning in detail.",
    tokens=200,
)

print(f"  Job ID: {job.job_id}")
print(f"  Status: {job.status.value}")
print(f"  Created: {time.strftime('%H:%M:%S', time.localtime(job.created_at))}")

# Poll for completion
print("\n⏳ Polling for completion...")
while job.status in (JobStatus.PENDING, JobStatus.RUNNING):
    time.sleep(0.5)
    job = client.status(job.job_id)
    elapsed = time.time() - job.created_at
    print(f"  [{elapsed:.1f}s] Status: {job.status.value}")

# Show result
print("\n📊 RESULT:")
print(f"  Status:    {job.status.value}")
if job.outcome:
    print(f"  Outcome:   {job.outcome.value}")
print(f"  Partial:   {job.partial}")

if job.usage:
    print("\n📈 ACCOUNTING:")
    print(f"  Output tokens: {job.usage.output_tokens}")
    print(f"  Total tokens:  {job.usage.total_tokens}")
    print(f"  Elapsed:       {job.usage.elapsed_seconds:.2f}s")
    if job.usage.cost_usd:
        print(f"  Cost:          ${job.usage.cost_usd:.4f}")

if job.answer:
    print(f"\n📝 ANSWER ({len(job.answer)} chars):")
    print(f"  {job.answer[:200]}...")
elif job.error:
    print(f"\n❌ ERROR: {job.error}")

print("\n" + "=" * 60)
print("Jobs enable: fire-and-forget, polling, cancellation")
print("=" * 60)
