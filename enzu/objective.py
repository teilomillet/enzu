"""
Objective: goal decomposition via RLM.

Two APIs, same engine:

1. Fire-and-forget (one RLM loop, runs until done):
    result = enzu.objective("Run SEPA experiments", budget={"cost": 50, "hours": 8})

2. Stepped execution with idle/wake (for long-running objectives):
    obj = Objective(goal="Run SEPA experiments", budget={"cost": 50, "hours": 8})
    while not obj.done:
        obj.step()
        obj.save()
        time.sleep(300)
    print(obj.result)

Both use RLM under the hood. The difference:
- objective() = one long RLM loop (max_steps=25)
- Objective.step() = one short RLM loop per phase (max_steps=5),
  with Session-like state persistence between phases
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from enzu.models import (
    Budget,
    BudgetUsage,
    Outcome,
    RLMExecutionReport,
    SuccessCriteria,
    TaskSpec,
)


# ---------------------------------------------------------------------------
# Phase record
# ---------------------------------------------------------------------------

@dataclass
class Phase:
    """One step's record in an Objective execution."""

    number: int
    answer: Optional[str]
    budget_usage: Dict[str, Any]
    outcome: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "number": self.number,
            "answer": self.answer,
            "budget_usage": self.budget_usage,
            "outcome": self.outcome,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Phase":
        return cls(
            number=d["number"],
            answer=d.get("answer"),
            budget_usage=d.get("budget_usage", {}),
            outcome=d.get("outcome", "unknown"),
            timestamp=d.get("timestamp", datetime.now().isoformat()),
        )


# ---------------------------------------------------------------------------
# Budget parsing
# ---------------------------------------------------------------------------

def _parse_budget(budget: Optional[Union[Dict[str, Any], Budget]]) -> Budget:
    """Parse a user-friendly budget dict into a Budget model.

    Accepts:
        cost  (USD)
        hours / minutes / seconds  (time)
        tokens  (output tokens)

    Default if nothing: Budget(max_tokens=200_000).
    """
    if budget is None:
        return Budget(max_tokens=200_000)
    if isinstance(budget, Budget):
        return budget

    max_tokens = budget.get("tokens")
    max_cost_usd = budget.get("cost")

    # Time: accept hours, minutes, seconds — accumulate into max_seconds
    max_seconds: Optional[float] = None
    if "seconds" in budget:
        max_seconds = float(budget["seconds"])
    if "minutes" in budget:
        max_seconds = (max_seconds or 0) + float(budget["minutes"]) * 60
    if "hours" in budget:
        max_seconds = (max_seconds or 0) + float(budget["hours"]) * 3600

    # Need at least one limit
    if max_tokens is None and max_cost_usd is None and max_seconds is None:
        return Budget(max_tokens=200_000)

    return Budget(
        max_tokens=max_tokens,
        max_cost_usd=max_cost_usd,
        max_seconds=max_seconds,
    )


# ---------------------------------------------------------------------------
# Planner prompt
# ---------------------------------------------------------------------------

_HARNESS_PLANNER_TEMPLATE = """\

== HARNESS MODE ==
You are an autonomous execution harness. Your goal:

{goal}

{constraints_block}
== CAPABILITIES ==
- pip_install("package") is available — install any PyPI package at runtime.
- Expanded imports: subprocess, os, pathlib, sys, and many more.
- 1-hour timeout, 50 steps per phase.

== INSTRUCTIONS ==
1. PLAN: Decompose the goal into numbered sub-tasks. Print the plan.
2. INSTALL: Use pip_install() to install any missing tools or libraries.
3. EXECUTE: Work on ONE sub-task per iteration. Measure results quantitatively.
4. ADAPT: If a sub-task fails, inspect the error, adjust, and retry.
5. TRACK: Call budget_status() after each sub-task. Print remaining budget.
6. SYNTHESIZE: When all sub-tasks are done or budget < 20%, \
call FINAL() with the combined result including metrics.

{history_block}
{budget_block}
"""

_PLANNER_TEMPLATE = """\

== OBJECTIVE MODE ==
You are executing a multi-phase objective. Your goal:

{goal}

{constraints_block}
== INSTRUCTIONS ==
1. PLAN: Decompose the goal into numbered sub-tasks. Print the plan.
2. EXECUTE: Work on ONE sub-task per iteration via llm_query(). \
The RLM feedback loop brings you back.
3. TRACK: Call budget_status() after each sub-task. \
Print remaining budget and progress.
4. ADAPT: If a sub-task fails (error visible in next iteration's feedback), \
retry or adjust plan.
5. SYNTHESIZE: When all sub-tasks are done or budget < 20%, \
call FINAL() with the combined result.

{history_block}
{budget_block}
"""


def _build_planner_prompt(
    goal: str,
    constraints: Optional[List[str]] = None,
    phase_number: int = 1,
    history: str = "",
    budget_status: str = "",
    harness: bool = False,
) -> str:
    """Build the planner prompt injected via metadata.tools_guidance."""
    constraints_block = ""
    if constraints:
        items = "\n".join(f"- {c}" for c in constraints)
        constraints_block = f"Constraints:\n{items}\n"

    history_block = ""
    if history:
        history_block = (
            f"== Previous Phases ==\n{history}\n"
            f"Current phase: {phase_number}\n"
        )

    budget_block = ""
    if budget_status:
        budget_block = f"Budget status: {budget_status}\n"

    template = _HARNESS_PLANNER_TEMPLATE if harness else _PLANNER_TEMPLATE
    return template.format(
        goal=goal,
        constraints_block=constraints_block,
        history_block=history_block,
        budget_block=budget_block,
    )


# ---------------------------------------------------------------------------
# Objective class — stepped harness
# ---------------------------------------------------------------------------

@dataclass
class Objective:
    """Goal decomposition via RLM with idle/wake support.

    Example (stepped with checkpointing):
        obj = Objective(goal="Run experiments", budget={"cost": 50, "hours": 8})
        while not obj.done:
            obj.step()
            obj.save("experiments.json")
            time.sleep(300)
        print(obj.result)

    Example (fire-and-forget via run()):
        obj = Objective(goal="Compare Python and Rust", budget={"tokens": 50000})
        print(obj.run())
    """

    goal: str
    budget: Optional[Union[Dict[str, Any], Budget]] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    constraints: Optional[List[str]] = None
    context: Optional[str] = None
    max_steps_per_phase: int = 5
    api_key: Optional[str] = None
    isolation: Optional[str] = None
    harness: bool = False

    # Internal state
    _phases: List[Phase] = field(default_factory=list)
    _budget_used: Dict[str, float] = field(
        default_factory=lambda: {
            "cost_usd": 0.0,
            "elapsed_seconds": 0.0,
            "output_tokens": 0,
        }
    )
    _done: bool = False
    _result: Optional[str] = None
    _created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self) -> None:
        # Parse budget once for internal use
        self._parsed_budget = _parse_budget(self.budget)
        # Harness mode: bump default max_steps_per_phase to 50
        if self.harness and self.max_steps_per_phase == 5:
            self.max_steps_per_phase = 50

    def step(self) -> str:
        """Run one phase (one RLM loop). Returns phase result."""
        if self.done:
            return self._result or ""

        from enzu.api import run as enzu_run

        # Build prompt with history of prior phases
        history = self._format_history()
        budget_status = self._format_budget_status()
        planner_prompt = _build_planner_prompt(
            self.goal,
            self.constraints,
            phase_number=len(self._phases) + 1,
            history=history,
            budget_status=budget_status,
            harness=self.harness,
        )

        # Build remaining budget
        remaining = self._remaining_budget()

        # Resolve model: use provided or auto-detect
        model = self.model
        if model is None:
            from enzu.client import _detect_provider_and_model
            provider_name, model = _detect_provider_and_model()

        # Build TaskSpec
        spec = TaskSpec(
            task_id=f"obj-{uuid4().hex[:8]}",
            input_text=self.goal,
            model=model,
            budget=remaining,
            success_criteria=SuccessCriteria(goal=self.goal),
            metadata={"tools_guidance": planner_prompt},
            max_output_tokens=min(remaining.max_tokens or 4096, 4096),
        )

        # Run one short RLM phase
        harness_kwargs: Dict[str, Any] = {}
        if self.harness:
            from enzu.security import HARNESS_PROFILE

            harness_kwargs = dict(
                enable_pip=True,
                allowed_imports=list(HARNESS_PROFILE.allowed_imports),
                output_char_limit=HARNESS_PROFILE.output_char_limit,
                prompt_style="extended",
                inject_search_tools=True,
            )
        report = enzu_run(
            task=spec,
            model=model,
            provider=self.provider,
            api_key=self.api_key,
            mode="rlm",
            max_steps=self.max_steps_per_phase,
            data=self.context or "",
            return_report=True,
            isolation=self.isolation,
            **harness_kwargs,
        )

        # Record phase
        if isinstance(report, RLMExecutionReport):
            answer = report.answer
            usage = report.budget_usage.model_dump()
            outcome = report.outcome.value
        else:
            answer = getattr(report, "output_text", None) or str(report)
            usage = getattr(report, "budget_usage", BudgetUsage(
                elapsed_seconds=0, output_tokens=0, total_tokens=0, cost_usd=None,
            )).model_dump()
            outcome = getattr(report, "outcome", Outcome.SUCCESS)
            if isinstance(outcome, Outcome):
                outcome = outcome.value

        phase = Phase(
            number=len(self._phases) + 1,
            answer=answer,
            budget_usage=usage,
            outcome=outcome,
        )
        self._phases.append(phase)
        self._accumulate_budget(usage)

        # Check if done
        if isinstance(report, RLMExecutionReport):
            if report.success or report.outcome == Outcome.BUDGET_EXCEEDED:
                self._done = True
                self._result = report.answer
        elif outcome == Outcome.SUCCESS.value:
            self._done = True
            self._result = answer

        return answer or ""

    def run(self) -> str:
        """Run to completion (fire-and-forget). All phases in a loop."""
        while not self.done:
            self.step()
        return self.result or ""

    @property
    def done(self) -> bool:
        return self._done or self._budget_exhausted()

    @property
    def result(self) -> Optional[str]:
        return self._result or self._best_partial()

    @property
    def phases(self) -> List[Phase]:
        return list(self._phases)

    # ----- Persistence -----

    def save(self, path: Optional[str] = None) -> str:
        """Checkpoint to JSON. Returns path."""
        if path is None:
            path = f"objective-{uuid4().hex[:8]}.json"
        data = {
            "goal": self.goal,
            "budget": self.budget if not isinstance(self.budget, Budget) else self.budget.model_dump(),
            "model": self.model,
            "provider": self.provider,
            "constraints": self.constraints,
            "context": self.context,
            "max_steps_per_phase": self.max_steps_per_phase,
            "isolation": self.isolation,
            "harness": self.harness,
            "phases": [p.to_dict() for p in self._phases],
            "budget_used": self._budget_used,
            "done": self._done,
            "result": self._result,
            "created_at": self._created_at,
        }
        Path(path).write_text(json.dumps(data, indent=2))
        return path

    @classmethod
    def load(cls, path: str) -> "Objective":
        """Resume from checkpoint."""
        data = json.loads(Path(path).read_text())
        obj = cls(
            goal=data["goal"],
            budget=data.get("budget"),
            model=data.get("model"),
            provider=data.get("provider"),
            constraints=data.get("constraints"),
            context=data.get("context"),
            max_steps_per_phase=data.get("max_steps_per_phase", 5),
            isolation=data.get("isolation"),
            harness=data.get("harness", False),
        )
        obj._phases = [Phase.from_dict(p) for p in data.get("phases", [])]
        obj._budget_used = data.get("budget_used", {
            "cost_usd": 0.0,
            "elapsed_seconds": 0.0,
            "output_tokens": 0,
        })
        obj._done = data.get("done", False)
        obj._result = data.get("result")
        obj._created_at = data.get("created_at", datetime.now().isoformat())
        return obj

    # ----- Internal helpers -----

    def _format_history(self) -> str:
        if not self._phases:
            return ""
        lines = []
        for p in self._phases:
            cost = p.budget_usage.get("cost_usd")
            elapsed = p.budget_usage.get("elapsed_seconds", 0)
            cost_str = f"${cost:.2f}" if cost else "n/a"
            elapsed_min = elapsed / 60 if elapsed else 0
            snippet = (p.answer or "")[:200]
            lines.append(
                f"- Phase {p.number} ({p.outcome}, {cost_str}, "
                f"{elapsed_min:.0f}min): {snippet}"
            )
        return "\n".join(lines)

    def _format_budget_status(self) -> str:
        budget = self._parsed_budget
        parts = []
        if budget.max_cost_usd is not None:
            remaining_cost = budget.max_cost_usd - self._budget_used.get("cost_usd", 0)
            parts.append(f"${remaining_cost:.2f} remaining of ${budget.max_cost_usd:.2f}")
        if budget.max_seconds is not None:
            remaining_sec = budget.max_seconds - self._budget_used.get("elapsed_seconds", 0)
            hours = remaining_sec / 3600
            parts.append(f"{hours:.1f}h remaining")
        if budget.max_tokens is not None:
            remaining_tok = budget.max_tokens - int(self._budget_used.get("output_tokens", 0))
            parts.append(f"{remaining_tok:,} tokens remaining")
        return " / ".join(parts) if parts else ""

    def _remaining_budget(self) -> Budget:
        budget = self._parsed_budget
        max_tokens = budget.max_tokens
        if max_tokens is not None:
            max_tokens = max(1, max_tokens - int(self._budget_used.get("output_tokens", 0)))

        max_cost = budget.max_cost_usd
        if max_cost is not None:
            max_cost = max(0.01, max_cost - self._budget_used.get("cost_usd", 0))

        max_seconds = budget.max_seconds
        if max_seconds is not None:
            max_seconds = max(1.0, max_seconds - self._budget_used.get("elapsed_seconds", 0))

        # Ensure at least one limit exists
        if max_tokens is None and max_cost is None and max_seconds is None:
            max_tokens = 200_000

        return Budget(
            max_tokens=max_tokens,
            max_cost_usd=max_cost,
            max_seconds=max_seconds,
        )

    def _accumulate_budget(self, usage: Dict[str, Any]) -> None:
        if usage.get("cost_usd"):
            self._budget_used["cost_usd"] = (
                self._budget_used.get("cost_usd", 0) + usage["cost_usd"]
            )
        if usage.get("elapsed_seconds"):
            self._budget_used["elapsed_seconds"] = (
                self._budget_used.get("elapsed_seconds", 0) + usage["elapsed_seconds"]
            )
        if usage.get("output_tokens"):
            self._budget_used["output_tokens"] = (
                self._budget_used.get("output_tokens", 0) + usage["output_tokens"]
            )

    def _budget_exhausted(self) -> bool:
        budget = self._parsed_budget
        if budget.max_tokens is not None:
            if int(self._budget_used.get("output_tokens", 0)) >= budget.max_tokens:
                return True
        if budget.max_cost_usd is not None:
            if self._budget_used.get("cost_usd", 0) >= budget.max_cost_usd:
                return True
        if budget.max_seconds is not None:
            if self._budget_used.get("elapsed_seconds", 0) >= budget.max_seconds:
                return True
        return False

    def _best_partial(self) -> Optional[str]:
        """Return the most recent phase answer as a partial result."""
        if self._phases:
            return self._phases[-1].answer
        return None

    def __repr__(self) -> str:
        return (
            f"Objective(goal={self.goal!r}, phases={len(self._phases)}, "
            f"done={self.done})"
        )


# ---------------------------------------------------------------------------
# objective() convenience function — fire-and-forget
# ---------------------------------------------------------------------------

def objective(
    goal: str,
    *,
    budget: Optional[Union[Dict[str, Any], Budget]] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    constraints: Optional[List[str]] = None,
    context: Optional[str] = None,
    max_steps: int = 25,
    api_key: Optional[str] = None,
    isolation: Optional[str] = None,
    harness: bool = False,
) -> str:
    """One-shot objective. Runs a single long RLM loop.

    Example:
        result = enzu.objective(
            "List 3 pros and 3 cons of Python vs Rust for CLI tools",
            budget={"tokens": 50000},
            model="gpt-4o",
        )
    """
    from enzu.api import run as enzu_run

    parsed = _parse_budget(budget)

    # Harness mode: bump default max_steps to 50
    if harness and max_steps == 25:
        max_steps = 50

    # Resolve model
    effective_model = model
    if effective_model is None:
        from enzu.client import _detect_provider_and_model
        _, effective_model = _detect_provider_and_model()

    planner_prompt = _build_planner_prompt(goal, constraints, harness=harness)

    spec = TaskSpec(
        task_id=f"obj-{uuid4().hex[:8]}",
        input_text=goal,
        model=effective_model,
        budget=parsed,
        success_criteria=SuccessCriteria(goal=goal),
        metadata={"tools_guidance": planner_prompt},
        max_output_tokens=min(parsed.max_tokens or 4096, 4096),
    )

    harness_kwargs: Dict[str, Any] = {}
    if harness:
        from enzu.security import HARNESS_PROFILE

        harness_kwargs = dict(
            enable_pip=True,
            allowed_imports=list(HARNESS_PROFILE.allowed_imports),
            output_char_limit=HARNESS_PROFILE.output_char_limit,
            prompt_style="extended",
            inject_search_tools=True,
        )

    report = enzu_run(
        task=spec,
        model=effective_model,
        provider=provider,
        api_key=api_key,
        mode="rlm",
        max_steps=max_steps,
        data=context or "",
        return_report=True,
        isolation=isolation,
        **harness_kwargs,
    )

    if isinstance(report, RLMExecutionReport):
        return report.answer or ""
    return getattr(report, "output_text", "") or ""


# ---------------------------------------------------------------------------
# harness() convenience function — fire-and-forget with harness preset
# ---------------------------------------------------------------------------

def harness(
    goal: str,
    *,
    budget: Optional[Union[Dict[str, Any], Budget]] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    constraints: Optional[List[str]] = None,
    context: Optional[str] = None,
    max_steps: int = 50,
    api_key: Optional[str] = None,
    isolation: Optional[str] = None,
) -> str:
    """Run an objective in harness mode (expanded imports, pip, 1-hour timeout).

    Thin wrapper around objective() with harness=True and max_steps=50.

    Example:
        result = enzu.harness(
            "Retrain the classifier and report F1",
            budget={"cost": 10, "hours": 1},
        )
    """
    return objective(
        goal,
        budget=budget,
        model=model,
        provider=provider,
        constraints=constraints,
        context=context,
        max_steps=max_steps,
        api_key=api_key,
        isolation=isolation,
        harness=True,
    )
