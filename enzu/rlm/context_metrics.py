"""
RLM context usage metrics and tracking.

Measures and analyzes context patterns in RLM workloads:
- Symbolic vs direct context usage
- File-based vs inline data access
- Trajectory depth and subcall patterns
- Context breakdown (system/task vs data)

Addresses findings from arxiv.org/html/2512.24601v2 about cost variance
in recursive language model workloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ContextBreakdown:
    """
    Breakdown of context usage for a single RLM run.

    Tracks how context is provided and accessed during execution.
    """

    # Total context sizes (in characters)
    system_prompt_chars: int = 0  # System prompt + instructions
    task_prompt_chars: int = 0  # User task/query
    inline_data_chars: int = 0  # Data passed directly in prompt
    file_data_chars: int = 0  # Data accessed via file reads

    # Access patterns
    file_reads: int = 0  # Number of file read operations
    file_bytes_read: int = 0  # Total bytes read from files

    # Trajectory metrics
    depth: int = 0  # Maximum recursion depth reached
    total_steps: int = 0  # Total steps across all levels
    subcalls: int = 0  # Number of recursive subcalls
    llm_invocations: int = 0  # Total LLM calls (main + subs)

    # Metadata
    used_symbolic_context: bool = False  # True if file-based context was used
    context_path: Optional[str] = None  # Path to context file if used

    def total_context_chars(self) -> int:
        """Total context size across all sources."""
        return (
            self.system_prompt_chars
            + self.task_prompt_chars
            + self.inline_data_chars
            + self.file_data_chars
        )

    def symbolic_ratio(self) -> float:
        """
        Ratio of symbolic (file-based) to direct (inline) context.

        Returns 1.0 if all context is symbolic, 0.0 if all inline.
        """
        total = self.inline_data_chars + self.file_data_chars
        if total == 0:
            return 0.0
        return self.file_data_chars / total

    def context_efficiency(self) -> float:
        """
        Ratio of bytes read to total available file context.

        High efficiency (close to 1.0) means RLM accessed most of the data.
        Low efficiency suggests selective access (good for symbolic context).
        """
        if self.file_data_chars == 0:
            return 1.0  # No symbolic context to optimize
        if self.file_bytes_read == 0:
            return 0.0  # Context provided but never accessed
        return min(1.0, self.file_bytes_read / self.file_data_chars)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/metrics."""
        return {
            "system_prompt_chars": self.system_prompt_chars,
            "task_prompt_chars": self.task_prompt_chars,
            "inline_data_chars": self.inline_data_chars,
            "file_data_chars": self.file_data_chars,
            "total_context_chars": self.total_context_chars(),
            "file_reads": self.file_reads,
            "file_bytes_read": self.file_bytes_read,
            "depth": self.depth,
            "total_steps": self.total_steps,
            "subcalls": self.subcalls,
            "llm_invocations": self.llm_invocations,
            "used_symbolic_context": self.used_symbolic_context,
            "symbolic_ratio": self.symbolic_ratio(),
            "context_efficiency": self.context_efficiency(),
        }


@dataclass
class RLMTrajectoryMetrics:
    """
    Trajectory-level metrics for RLM execution variance tracking.

    Captures the cost variance problem from the paper: RLM inference costs
    vary significantly due to unpredictable trajectory lengths.
    """

    run_id: str
    task_id: Optional[str] = None

    # Trajectory structure
    max_depth: int = 0  # Deepest recursion level
    total_steps: int = 0  # Sum of steps across all levels
    subcall_count: int = 0  # Number of recursive invocations

    # Token usage breakdown
    system_prompt_tokens: int = 0  # System prompt overhead
    task_tokens: int = 0  # User task/query tokens
    code_generation_tokens: int = 0  # Tokens for generated code
    total_input_tokens: int = 0  # Total input across all LLM calls
    total_output_tokens: int = 0  # Total output across all LLM calls

    # Context access
    context_breakdown: Optional[ContextBreakdown] = None

    # Cost and latency
    elapsed_seconds: float = 0.0
    cost_usd: Optional[float] = None

    # Outcome
    success: bool = False
    outcome: str = "unknown"

    def trajectory_complexity(self) -> float:
        """
        Complexity score based on depth, steps, and subcalls.

        Higher values indicate more complex execution paths.
        """
        return self.max_depth * 10 + self.total_steps + self.subcall_count * 5

    def token_efficiency(self) -> float:
        """
        Ratio of output tokens to total tokens.

        Higher values mean more generated output relative to input overhead.
        """
        total = self.total_input_tokens + self.total_output_tokens
        if total == 0:
            return 0.0
        return self.total_output_tokens / total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for metrics export."""
        result = {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "max_depth": self.max_depth,
            "total_steps": self.total_steps,
            "subcall_count": self.subcall_count,
            "system_prompt_tokens": self.system_prompt_tokens,
            "task_tokens": self.task_tokens,
            "code_generation_tokens": self.code_generation_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "elapsed_seconds": self.elapsed_seconds,
            "cost_usd": self.cost_usd,
            "success": self.success,
            "outcome": self.outcome,
            "trajectory_complexity": self.trajectory_complexity(),
            "token_efficiency": self.token_efficiency(),
        }

        if self.context_breakdown:
            result["context"] = self.context_breakdown.to_dict()

        return result


class RLMContextTracker:
    """
    Tracks context usage patterns across RLM executions.

    Use this to collect trajectory metrics for analyzing cost variance
    and context optimization opportunities.
    """

    def __init__(self) -> None:
        self._trajectories: List[RLMTrajectoryMetrics] = []

    def record(self, metrics: RLMTrajectoryMetrics) -> None:
        """Record trajectory metrics from an RLM execution."""
        self._trajectories.append(metrics)

    def get_trajectories(self) -> List[RLMTrajectoryMetrics]:
        """Get all recorded trajectory metrics."""
        return list(self._trajectories)

    def clear(self) -> None:
        """Clear recorded trajectories."""
        self._trajectories.clear()

    def summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics across all trajectories.

        Useful for analyzing cost variance and complexity distributions.
        """
        if not self._trajectories:
            return {
                "total_runs": 0,
                "avg_depth": 0.0,
                "avg_steps": 0.0,
                "avg_subcalls": 0.0,
                "avg_complexity": 0.0,
                "success_rate": 0.0,
            }

        total = len(self._trajectories)
        successes = sum(1 for t in self._trajectories if t.success)

        depths = [t.max_depth for t in self._trajectories]
        steps = [t.total_steps for t in self._trajectories]
        subcalls = [t.subcall_count for t in self._trajectories]
        complexities = [t.trajectory_complexity() for t in self._trajectories]

        # Calculate percentiles for variance analysis
        depths_sorted = sorted(depths)
        steps_sorted = sorted(steps)
        complexities_sorted = sorted(complexities)

        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            idx = int(len(data) * p)
            return data[min(idx, len(data) - 1)]

        return {
            "total_runs": total,
            "success_rate": successes / total if total > 0 else 0.0,
            "avg_depth": sum(depths) / total,
            "avg_steps": sum(steps) / total,
            "avg_subcalls": sum(subcalls) / total,
            "avg_complexity": sum(complexities) / total,
            "depth_p50": percentile(depths_sorted, 0.5),
            "depth_p95": percentile(depths_sorted, 0.95),
            "steps_p50": percentile(steps_sorted, 0.5),
            "steps_p95": percentile(steps_sorted, 0.95),
            "complexity_p50": percentile(complexities_sorted, 0.5),
            "complexity_p95": percentile(complexities_sorted, 0.95),
        }


# Global tracker instance
_global_tracker: Optional[RLMContextTracker] = None


def get_rlm_context_tracker() -> RLMContextTracker:
    """Get the global RLM context tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = RLMContextTracker()
    return _global_tracker


def reset_rlm_context_tracker() -> None:
    """Reset the global RLM context tracker."""
    global _global_tracker
    _global_tracker = RLMContextTracker()
