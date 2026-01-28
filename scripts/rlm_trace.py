#!/usr/bin/env python3
"""
RLM Trace: Save complete execution trace for debugging.

Captures all tokens, API responses, and RLM steps to diagnose issues.

Usage:
    python scripts/rlm_trace.py --model "z-ai/glm-4.7" --output artifacts/trace.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

from enzu.models import TaskSpec, Budget, SuccessCriteria, RLMExecutionReport
from enzu.api import resolve_provider
from enzu.providers.pool import close_all_providers
from enzu.rlm.engine import RLMEngine


@dataclass
class TraceEvent:
    """Single trace event."""
    timestamp: float
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionTrace:
    """Complete execution trace."""
    task_id: str
    model: str
    provider: str
    start_time: float
    end_time: float = 0.0
    events: List[TraceEvent] = field(default_factory=list)
    raw_api_responses: List[Dict[str, Any]] = field(default_factory=list)
    rlm_steps: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    success: bool = False
    error: Optional[str] = None
    
    def add_event(self, event_type: str, **data: Any) -> None:
        self.events.append(TraceEvent(
            timestamp=time.time(),
            event_type=event_type,
            data=data
        ))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model": self.model,
            "provider": self.provider,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": (self.end_time - self.start_time) * 1000,
            "success": self.success,
            "error": self.error,
            "final_answer": self.final_answer,
            "final_answer_len": len(self.final_answer),
            "events": [
                {
                    "timestamp": e.timestamp,
                    "relative_ms": (e.timestamp - self.start_time) * 1000,
                    "event_type": e.event_type,
                    "data": e.data
                }
                for e in self.events
            ],
            "raw_api_responses": self.raw_api_responses,
            "rlm_steps": self.rlm_steps,
            "stats": {
                "total_events": len(self.events),
                "total_api_calls": len(self.raw_api_responses),
                "total_rlm_steps": len(self.rlm_steps),
            }
        }


def build_test_context() -> str:
    """Build test context document."""
    marker = f"RLM-TRACE-{uuid4().hex[:8]}"
    return f"""=== TECHNICAL ANALYSIS CONTEXT ===
Report ID: {marker}
Dataset: trace-test-dataset
Domain: operational

METRICS:
- Latency: 45.2ms (stable)
- Throughput: 120 req/s (increasing)
- Error rate: 0.5% (decreasing)

ANALYSIS REQUIRED:
Generate an executive summary that includes the Report ID ({marker}).
The summary must be at least 50 words.
"""


def run_traced_execution(
    model: str,
    provider_name: str = "openrouter",
    with_research: bool = False,
) -> ExecutionTrace:
    """Run RLM execution with full tracing."""
    task_id = f"trace-{uuid4().hex[:8]}"
    trace = ExecutionTrace(
        task_id=task_id,
        model=model,
        provider=provider_name,
        start_time=time.time()
    )
    
    context = build_test_context()
    # Extract marker from context
    import re
    marker_match = re.search(r"RLM-TRACE-[a-f0-9]+", context)
    marker = marker_match.group(0) if marker_match else "UNKNOWN"
    
    trace.add_event("trace_started", marker=marker, context_len=len(context))
    
    try:
        provider = resolve_provider(provider_name, use_pool=True)
        trace.add_event("provider_initialized", provider=provider_name)
        
        # Build task
        input_text = f"""Analyze the following data and generate an executive summary report.

You have access to llm_query() to delegate analysis tasks to sub-agents.

DATA TO ANALYZE:
{context}

Generate a comprehensive executive summary that includes the Report ID."""
        
        task = TaskSpec(
            task_id=task_id,
            input_text=input_text,
            model=model,
            responses={},
            budget=Budget(max_tokens=8000, max_total_tokens=32000),
            success_criteria=SuccessCriteria(
                required_substrings=[marker],
                min_word_count=50,
            ),
            metadata={"trace_id": task_id}
        )
        
        trace.add_event("task_built", 
                       input_text_len=len(task.input_text),
                       budget=task.budget.model_dump() if task.budget else None)
        
        # Create RLM engine with tracing
        engine = RLMEngine(
            max_steps=4,
            recursive_subcalls=True,
            inject_search_tools=with_research,
            allowed_imports=["json", "re", "math"],
        )
        
        trace.add_event("engine_created", with_research=with_research)
        
        # Run and capture report
        report: RLMExecutionReport = engine.run(task, provider, data=context)
        
        trace.end_time = time.time()
        trace.add_event("engine_completed")
        
        # Extract step details
        for i, step in enumerate(report.steps):
            step_data = {
                "step_index": i,
                "model_output": step.model_output[:500] if step.model_output else None,
                "model_output_len": len(step.model_output) if step.model_output else 0,
                "code": step.code[:500] if step.code else None,
                "code_len": len(step.code) if step.code else 0,
                "stdout": step.stdout[:500] if step.stdout else None,
                "error": step.error[:500] if step.error else None,
            }
            trace.rlm_steps.append(step_data)
            trace.add_event("rlm_step", **step_data)
        
        trace.final_answer = report.answer or ""
        trace.success = report.success
        
        if report.budget_usage:
            trace.add_event("budget_usage", **report.budget_usage.model_dump())
        
        # Derive verification_passed from errors (RLMExecutionReport has no verification_passed field)
        verification_passed = not any("verification_failed" in e for e in report.errors)
        trace.add_event("execution_complete",
                       success=report.success,
                       answer_len=len(report.answer or ""),
                       verification_passed=verification_passed)
        
    except Exception as e:
        trace.end_time = time.time()
        trace.error = str(e)
        trace.add_event("execution_error", error=str(e), error_type=type(e).__name__)
        import traceback
        trace.add_event("traceback", tb=traceback.format_exc())
    
    finally:
        close_all_providers()
    
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description="RLM Trace - Full execution trace")
    parser.add_argument("--model", type=str, default="openai/gpt-4o-mini",
                       help="Model name")
    parser.add_argument("--provider", type=str, default="openrouter",
                       help="Provider name")
    parser.add_argument("--with-research", action="store_true",
                       help="Enable research tools")
    parser.add_argument("--output", type=str, default="artifacts/rlm-trace.json",
                       help="Output file path")
    args = parser.parse_args()
    
    print(f"Running traced execution with {args.model}...")
    print(f"Provider: {args.provider}")
    print(f"Research tools: {args.with_research}")
    print()
    
    trace = run_traced_execution(
        model=args.model,
        provider_name=args.provider,
        with_research=args.with_research,
    )
    
    # Save trace
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(trace.to_dict(), f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("TRACE COMPLETE")
    print(f"{'='*60}")
    print(f"Task ID: {trace.task_id}")
    print(f"Model: {trace.model}")
    print(f"Duration: {(trace.end_time - trace.start_time)*1000:.0f}ms")
    print(f"Success: {trace.success}")
    print(f"RLM Steps: {len(trace.rlm_steps)}")
    print(f"Events: {len(trace.events)}")
    print(f"Answer length: {len(trace.final_answer)} chars")
    if trace.error:
        print(f"Error: {trace.error}")
    print(f"\nTrace saved to: {output_path}")
    
    # Print first 200 chars of answer
    if trace.final_answer:
        print("\nAnswer preview:")
        print(trace.final_answer[:300])
        print("..." if len(trace.final_answer) > 300 else "")


if __name__ == "__main__":
    main()
