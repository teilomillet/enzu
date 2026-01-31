"""
RLM Engine: Recursive Language Model execution.

Architecture from the paper (arxiv.org/html/2512.24601v1):
- Main RLM: orchestrates via code, delegates semantic work to sub-LLMs
- Sub-LLMs: have tools (search, etc.) to do actual work

This module provides the public RLMEngine class. Implementation is split across:
- budget.py: token budget pools and usage tracking
- feedback.py: error hints and code analysis
- llm_executor.py: LLM query execution with fallback
- prompts.py: system prompt constants
- runner.py: main step loop
- sandbox_factory.py: sandbox creation for isolation modes
- verification.py: output verification logic
"""

from __future__ import annotations

import time
import threading
from uuid import uuid4
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from enzu.contract import DEFAULT_MAX_OUTPUT_TOKENS
from enzu.models import (
    Budget,
    RLMExecutionReport,
    SuccessCriteria,
    TaskSpec,
)
from enzu.budget import count_tokens_exact, estimate_tokens_conservative
from enzu.providers.base import BaseProvider
from enzu import telemetry

# Extracted modules: see docstring for responsibilities.
from enzu.rlm.budget import (
    TokenBudgetPool,
    BudgetTracker,
    build_budget_usage,
    parse_budget_critical_threshold,
)
from enzu.rlm.llm_executor import LLMExecutor, LLMExecutorConfig
from enzu.rlm.prompts import (
    PIP_INSTALL_GUIDANCE,
    SEARCH_TOOLS_GUIDANCE,
    STRATEGY_HINTS,
    SYSTEM_PROMPT_GUARDRAILS,
    format_success_criteria as _format_success_criteria,
    has_strong_success_criteria as _has_strong_success_criteria,
)
from enzu.rlm.runner import StepRunner
from enzu.repl.protocol import SandboxFactory, SandboxProtocol
from enzu.rlm.sandbox_factory import create_sandbox
from enzu.rlm.verification import verify_output


def _trim_text(text: Optional[str], limit: int = 2000) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit]


class RLMEngine:
    """
    RLM (Recursive Language Model) engine following the paper's architecture.

    Orchestrates multi-step execution where each step:
    1. Generates code via LLM
    2. Executes in sandbox with llm_query/llm_batch available
    3. Provides feedback and loops until FINAL() called

    Configuration:
    - max_steps: iteration limit
    - recursive_subcalls: True = llm_query spawns sub-RLM (paper's approach)
    - inject_search_tools: False = main delegates tools to sub-RLMs
    - isolation: None/"subprocess"/"container" for sandbox security
    """

    def __init__(
        self,
        *,
        max_steps: int = 8,
        output_char_limit: int = 8192,
        allowed_imports: Optional[List[str]] = None,
        verify_on_final: bool = True,
        recursive_subcalls: bool = True,
        max_recursion_depth: int = 1,
        subcall_max_steps: int = 3,
        subcall_max_output_tokens: int = 1024,
        subcall_verify_on_final: bool = False,
        enable_pip: bool = False,
        prompt_style: str = "paper",
        inject_search_tools: bool = False,
        subcall_inject_search_tools: bool = True,
        isolation: Optional[str] = None,
        # Custom container image (SandboxImage, BuiltImage, or string tag)
        # Only used when isolation="container"
        sandbox_image: Optional[Any] = None,
    ) -> None:
        self._max_steps = max_steps
        self._output_char_limit = output_char_limit
        self._allowed_imports = allowed_imports or [
            "re",
            "math",
            "json",
            "datetime",
            "collections",
            "itertools",
            "functools",
        ]
        self._enable_pip = enable_pip
        if prompt_style not in {"paper", "extended"}:
            raise ValueError("prompt_style must be 'paper' or 'extended'.")
        self._prompt_style = prompt_style
        self._verify_on_final = verify_on_final
        self._recursive_subcalls = recursive_subcalls
        self._max_recursion_depth = max(0, max_recursion_depth)
        self._subcall_max_steps = subcall_max_steps
        self._subcall_max_output_tokens = subcall_max_output_tokens
        self._subcall_verify_on_final = subcall_verify_on_final
        self._inject_search_tools = inject_search_tools
        self._subcall_inject_search_tools = subcall_inject_search_tools
        if isolation is not None and isolation not in {"subprocess", "container"}:
            raise ValueError("isolation must be None, 'subprocess', or 'container'")
        self._isolation = isolation
        # sandbox_image: SandboxImage, BuiltImage, or string (see enzu.sandbox.image)
        # Only used when isolation="container"
        self._sandbox_image = sandbox_image

    @staticmethod
    def _build_context(query: str, data: Any) -> List[Any]:
        """Construct the RLM prompt chunks (query first, data after)."""
        if data is None:
            return [query]
        if isinstance(data, list):
            return [query, *data]
        return [query, data]

    @staticmethod
    def _context_stats(context: Any) -> Tuple[str, int, List[int]]:
        """Summarize context for prompt metadata."""
        if isinstance(context, (list, tuple)):
            lengths = [len(str(item)) for item in context]
            return type(context).__name__, sum(lengths), lengths
        text = str(context)
        return type(context).__name__, len(text), [len(text)]

    def _build_sub_engine(self) -> "RLMEngine":
        """Create sub-RLM for recursive calls. Per paper: sub-RLMs have tools."""
        return RLMEngine(
            max_steps=self._subcall_max_steps,
            output_char_limit=self._output_char_limit,
            allowed_imports=list(self._allowed_imports),
            verify_on_final=self._subcall_verify_on_final,
            recursive_subcalls=False,
            max_recursion_depth=0,
            subcall_max_steps=self._subcall_max_steps,
            subcall_max_output_tokens=self._subcall_max_output_tokens,
            subcall_verify_on_final=self._subcall_verify_on_final,
            enable_pip=self._enable_pip,
            prompt_style="extended",
            inject_search_tools=self._subcall_inject_search_tools,
            subcall_inject_search_tools=False,
            isolation=self._isolation,
            sandbox_image=self._sandbox_image,
        )

    def _run_subcall(
        self,
        *,
        parent_task: TaskSpec,
        provider: BaseProvider,
        prompt: str,
        depth: int,
        max_output_tokens: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
        sandbox_factory: Optional[SandboxFactory] = None,
    ) -> RLMExecutionReport:
        """Run a depth-limited recursive RLM subcall."""
        import json

        sub_metadata = {
            k: v
            for k, v in parent_task.metadata.items()
            if k not in {"prelude_code", "prelude_allow_final"}
        }
        sub_metadata["_subcall_prompt"] = prompt
        sub_metadata["allow_weak_success_criteria"] = True

        if sub_metadata.get("subcall_mode") == "research" and not str(
            prompt
        ).startswith("__DECIDE__:"):
            query = json.dumps(prompt)
            sub_metadata["prelude_code"] = (
                "import json\n"
                f"query = {query}\n"
                "sources = []\n"
                "try:\n"
                "    result = research(\n"
                "        query,\n"
                "        include_news=True,\n"
                "        days_back=180,\n"
                "        min_score=0.2,\n"
                "        num_results=6,\n"
                "        max_characters=4000,\n"
                "        max_chars_per_source=600,\n"
                "    )\n"
                "    raw = result.get('sources', [])\n"
                "    for s in raw[:3]:\n"
                "        sources.append({\n"
                "            'title': s.get('title', ''),\n"
                "            'url': s.get('url', ''),\n"
                "            'published_date': s.get('published_date', ''),\n"
                "            'text': (s.get('text', '') or '')[:300],\n"
                "        })\n"
                "except Exception:\n"
                "    sources = []\n"
                "FINAL(json.dumps(sources))\n"
            )
            sub_metadata["prelude_allow_final"] = True

        sub_max_output = self._subcall_max_output_tokens
        if max_output_tokens is not None:
            sub_max_output = min(sub_max_output, max_output_tokens)
        if sub_max_output <= 0:
            raise RuntimeError("budget_exhausted_preflight")

        sub_task = TaskSpec(
            task_id=f"{parent_task.task_id}:sub:{uuid4().hex[:8]}",
            input_text="Execute prelude code and return JSON only.",
            model=parent_task.model,
            # Pass max_total_tokens to subcall budget so TokenBudgetPool enforces limits during execution
            # Without this, subcall's TokenBudgetPool would be unlimited (max_total_tokens=None)
            budget=Budget(max_tokens=sub_max_output, max_total_tokens=max_total_tokens),
            success_criteria=SuccessCriteria(min_word_count=1),
            max_output_tokens=sub_max_output,
            metadata=sub_metadata,
        )
        sub_engine = self._build_sub_engine()
        return sub_engine.run(
            sub_task,
            provider,
            data="",
            depth=depth,
            prompt_env=prompt,
            sandbox_factory=sandbox_factory,
        )

    def run(
        self,
        task: TaskSpec,
        provider: BaseProvider,
        *,
        data: str,
        namespace: Optional[Dict[str, Any]] = None,
        sandbox: Optional[SandboxProtocol] = None,
        sandbox_factory: Optional[SandboxFactory] = None,
        on_progress: Optional[Callable[[str], None]] = None,
        depth: int = 0,
        prompt_env: Optional[Any] = None,
        fallback_providers: Optional[List[BaseProvider]] = None,
    ) -> RLMExecutionReport:
        """
        Execute the RLM loop.

        Args:
            task: Task specification with input, budget, success criteria
            provider: Primary LLM provider
            data: Context data for the sandbox
            namespace: Additional sandbox namespace bindings
            sandbox: Pre-constructed sandbox (root run only)
            sandbox_factory: Factory for creating sandboxes with llm callbacks
            on_progress: Progress callback
            depth: Recursion depth (internal use)
            prompt_env: Override prompt environment (internal use)
            fallback_providers: Backup providers on failure

        Returns:
            RLMExecutionReport with success, answer, steps, errors
        """
        errors: list[str] = []

        # Validate and adjust max_output_tokens
        max_output_tokens = task.max_output_tokens
        if task.budget.max_tokens is not None:
            if max_output_tokens is None:
                max_output_tokens = task.budget.max_tokens
            elif max_output_tokens > task.budget.max_tokens:
                errors.append("task.max_output_tokens exceeds budget.max_tokens")
                budget_usage = build_budget_usage(task.budget, {}, 0.0)
                return RLMExecutionReport(
                    success=False,
                    task_id=task.task_id,
                    provider=provider.name,
                    model=task.model,
                    answer=None,
                    steps=[],
                    budget_usage=budget_usage,
                    errors=errors,
                )
        if max_output_tokens != task.max_output_tokens:
            task = task.model_copy(update={"max_output_tokens": max_output_tokens})

        # Require explicit success criteria
        if not _has_strong_success_criteria(
            task.success_criteria
        ) and not task.metadata.get("allow_weak_success_criteria"):
            errors.append("success_criteria_missing_or_weak")
            budget_usage = build_budget_usage(task.budget, {}, 0.0)
            return RLMExecutionReport(
                success=False,
                task_id=task.task_id,
                provider=provider.name,
                model=task.model,
                answer=None,
                steps=[],
                budget_usage=budget_usage,
                errors=errors,
            )

        start_ts = time.time()
        usage_accumulator: Dict[str, Any] = {}
        token_pool = TokenBudgetPool(task.budget.max_total_tokens)
        budget_tracker = BudgetTracker(task.budget)
        critical_threshold = parse_budget_critical_threshold(task.metadata)
        context_path = task.metadata.get("context_path")
        context_before: Optional[Dict[str, Any]] = None

        # Budget notice system: collects notices during LLM calls, drained into prompts
        budget_notice_lock = threading.Lock()
        budget_notices: list[str] = []

        def add_budget_notice(message: str) -> None:
            with budget_notice_lock:
                budget_notices.append(message)

        def drain_budget_notices() -> Optional[str]:
            with budget_notice_lock:
                if not budget_notices:
                    return None
                message = " | ".join(budget_notices)
                budget_notices.clear()
                return message

        # Build context environment
        context_env = (
            prompt_env
            if prompt_env is not None
            else self._build_context(task.input_text, data)
        )
        context_type, context_len, context_lengths = self._context_stats(context_env)

        def emit(message: str) -> None:
            if on_progress:
                try:
                    on_progress(message)
                except Exception:
                    pass
            if message.startswith("llm_stream:") and not telemetry.stream_enabled():
                return
            telemetry.log("info", "rlm_progress", progress_message=message)

        with telemetry.span(
            "rlm.run", task_id=task.task_id, provider=provider.name, model=task.model
        ):
            # Load context if path specified
            if context_path:
                try:
                    from enzu.tools.context import ctx_load, ctx_stats

                    path = Path(context_path)
                    if path.exists():
                        ctx_load(str(path))
                    context_before = ctx_stats()
                    telemetry.log(
                        "info",
                        "context_loaded",
                        path=str(path),
                        stats=context_before,
                    )
                except Exception:
                    context_before = None

            # Create LLM executor
            executor_config = LLMExecutorConfig(
                task=task,
                provider=provider,
                fallback_providers=fallback_providers,
                usage_accumulator=usage_accumulator,
                token_pool=token_pool,
                budget_tracker=budget_tracker,
                emit=emit,
                errors=errors,
                add_budget_notice=add_budget_notice,
            )
            executor = LLMExecutor(executor_config)

            # Set up subcall runner if recursive mode enabled
            if self._recursive_subcalls and depth < self._max_recursion_depth:

                def subcall_runner(
                    prompt: str, max_output: Optional[int], max_total: Optional[int]
                ) -> RLMExecutionReport:
                    return self._run_subcall(
                        parent_task=task,
                        provider=provider,
                        prompt=prompt,
                        depth=depth + 1,
                        max_output_tokens=max_output,
                        max_total_tokens=max_total,
                        sandbox_factory=sandbox_factory,
                    )

                executor.set_subcall_runner(subcall_runner)

            # Create sandbox
            sandbox_namespace = dict(namespace or {})
            if sandbox is not None and sandbox_factory is not None:
                raise ValueError("sandbox and sandbox_factory cannot both be set.")

            if sandbox is None and sandbox_factory is None:
                # budget_status is only available in in-process sandbox (not picklable)
                # For isolated sandboxes, we skip it since it can't be serialized
                if self._isolation is None:
                    sandbox_namespace["budget_status"] = lambda: {
                        **token_pool.snapshot(),
                        "model": task.model,
                        "output_cap": task.max_output_tokens
                        or DEFAULT_MAX_OUTPUT_TOKENS,
                        "remaining_output_tokens": budget_tracker.remaining_output_tokens(),
                        "critical_threshold": critical_threshold,
                    }

                sandbox = create_sandbox(
                    isolation=self._isolation,
                    data=data,
                    context=context_env,
                    namespace=sandbox_namespace,
                    allowed_imports=self._allowed_imports,
                    output_char_limit=self._output_char_limit,
                    timeout_seconds=task.budget.max_seconds,
                    inject_search_tools=self._inject_search_tools,
                    enable_pip=self._enable_pip,
                    llm_query=executor.sandbox_query,
                    llm_batch=executor.batch_query,
                    # Custom container image (only used when isolation="container")
                    sandbox_image=self._sandbox_image,
                )
            elif sandbox_factory is not None:
                # Factory runs after llm callbacks exist; it owns serialization rules.
                sandbox = sandbox_factory(
                    isolation=self._isolation,
                    data=data,
                    context=context_env,
                    namespace=sandbox_namespace,
                    allowed_imports=self._allowed_imports,
                    output_char_limit=self._output_char_limit,
                    timeout_seconds=task.budget.max_seconds,
                    inject_search_tools=self._inject_search_tools,
                    enable_pip=self._enable_pip,
                    llm_query=executor.sandbox_query,
                    llm_batch=executor.batch_query,
                    sandbox_image=self._sandbox_image,
                )

            assert sandbox is not None

            # Run prelude code if specified
            prelude_code = task.metadata.get("prelude_code")
            if prelude_code:
                with telemetry.span("rlm.prelude", code_len=len(prelude_code)):
                    result = sandbox.exec(prelude_code)
                telemetry.log(
                    "info",
                    "rlm_prelude_result",
                    stdout=_trim_text(result.stdout),
                    error=_trim_text(result.error),
                )
                if sandbox.answer.get("ready"):
                    if task.metadata.get("prelude_allow_final"):
                        answer = sandbox.answer.get("content") or ""
                        verification = verify_output(
                            task.success_criteria, answer, goal_based_trust=True
                        )
                        if verification.passed:
                            telemetry.log("info", "rlm_prelude_final_accept")
                            elapsed_seconds = time.time() - start_ts
                            budget_usage = build_budget_usage(
                                task.budget, usage_accumulator, elapsed_seconds
                            )
                            return RLMExecutionReport(
                                success=True,
                                task_id=task.task_id,
                                provider=provider.name,
                                model=task.model,
                                answer=answer,
                                steps=[],
                                budget_usage=budget_usage,
                                errors=[],
                            )
                        telemetry.log(
                            "info",
                            "rlm_prelude_final_reject",
                            reasons=verification.reasons,
                        )
                    sandbox.answer["ready"] = False
                    sandbox.answer["content"] = ""

            # Build system prompt with upfront budget info
            has_search_tools = bool(sandbox.namespace.get("__search_tools_available__"))
            budget_snapshot = token_pool.snapshot()
            prompt = self._system_prompt(
                task,
                data_len=context_len,
                context_type=context_type,
                context_lengths=context_lengths,
                has_search_tools=has_search_tools,
                budget_snapshot=budget_snapshot,
                critical_threshold=critical_threshold,
            )

            # Estimate system prompt tokens - this is overhead users should be aware of
            system_prompt_tokens = count_tokens_exact(prompt, task.model)
            if system_prompt_tokens is None:
                system_prompt_tokens = estimate_tokens_conservative(prompt)
            telemetry.log(
                "info",
                "rlm_prompt_init",
                task_id=task.task_id,
                data_len=context_len,
                has_search_tools=has_search_tools,
                prompt=_trim_text(prompt),
                system_prompt_tokens=system_prompt_tokens,
            )

            # Run step loop
            runner = StepRunner(
                task=task,
                sandbox=sandbox,
                token_pool=token_pool,
                budget_tracker=budget_tracker,
                usage_accumulator=usage_accumulator,
                llm_query=lambda p, s: executor.direct_query(p, span_name=s),
                emit=emit,
                max_steps=self._max_steps,
                verify_on_final=self._verify_on_final,
                context_path=context_path,
                context_before=context_before,
                drain_budget_notices=drain_budget_notices,
                system_prompt_tokens=system_prompt_tokens,
            )
            return runner.run(prompt, start_ts, provider.name)

    def _system_prompt(
        self,
        task: TaskSpec,
        *,
        data_len: int,
        context_type: str = "str",
        context_lengths: Optional[List[int]] = None,
        has_search_tools: bool = False,
        budget_snapshot: Optional[Dict[str, Any]] = None,
        critical_threshold: Optional[int] = None,
    ) -> str:
        """Build the system prompt text with upfront budget information."""
        chunk_lengths = context_lengths or [data_len]
        base_prompt = (
            "You are tasked with answering a query with associated context. You can access, "
            "transform, and analyze this context interactively in a REPL environment that can "
            "recursively query sub-LLMs, which you are strongly encouraged to use as much as "
            "possible. You will be queried iteratively until you provide a final answer.\n\n"
            f"Your context is a {context_type} with {data_len} total characters, and is broken "
            f"up into chunks of char lengths: {chunk_lengths}.\n\n"
            "The REPL environment is initialized with:\n"
            "1. A 'context' variable containing all information related to the query.\n"
            "2. 'llm_query(prompt)' - query a sub-LLM (has tools: search, research, etc.).\n"
            "3. 'llm_batch(prompts)' - parallel sub-LLM queries (faster for multiple items).\n"
            "4. 'print()' to debug and observe outputs (truncated).\n\n"
            "Sub-LLMs have access to tools you don't. Delegate tasks requiring external "
            "information (search, research, web content) to them via llm_query/llm_batch.\n\n"
            "Make sure to explicitly look through the entire context. The context contains the "
            "answer to the query, so it is important that you parse it carefully and "
            "completely.\n\n"
            "You can use the REPL environment for as many iterations as you need to answer the "
            "query. In the REPL environment, you should primarily do computation. Write out the "
            "final answer when you are ready. You are not required to use the REPL environment "
            "if it is not helpful.\n\n"
            "When you want to execute Python code, you should wrap it in a block of triple "
            "backticks with the python language specifier (```repl```). For example:\n\n"
            "```repl\n"
            "# Example: delegate search to sub-LLM\n"
            "result = llm_query('Search for recent news about AI agents and summarize')\n"
            "print(result)\n"
            "```\n\n"
            "IMPORTANT: When you are done with the iterative process, you MUST provide a final "
            "answer inside a FINAL function when you have completed your task, NOT in code. "
            "Do not use these tags unless you have completed your task. You have two options:\n"
            "1. Use FINAL(your final answer here) to provide the answer directly\n"
            "2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL "
            "environment as your final output\n\n"
            "Note: If you are ready to provide a final answer, you cannot write anything other "
            "than the final answer in the FINAL or FINAL_VAR tags.\n\n"
            "Think step by step carefully, plan, and execute this plan immediately in your "
            "response -- do not just say 'I will do this' or 'I will do that'. Output to the "
            "REPL environment and recursive LLMs as much as possible. Remember to explicitly "
            "answer the original query in your final answer."
        )

        # Add budget constraints to system prompt so model knows limits upfront
        budget_text = self._format_budget_constraints(
            task.budget,
            budget_snapshot=budget_snapshot,
            critical_threshold=critical_threshold,
        )

        tools_guidance = task.metadata.get("tools_guidance") or ""
        criteria_text = _format_success_criteria(task.success_criteria)
        prompt = (
            self._model_prompt_extra(task.model)
            + base_prompt
            + budget_text
            + criteria_text
        )

        if self._prompt_style == "extended":
            suppress_probe = bool(task.metadata.get("suppress_probe_guidance"))

            def strip_probe(text: str) -> str:
                lines = [line for line in text.splitlines() if "Probe:" not in line]
                return "\n".join(lines)

            guardrails = SYSTEM_PROMPT_GUARDRAILS
            strategy = STRATEGY_HINTS if data_len >= 10000 else ""
            search_guidance = SEARCH_TOOLS_GUIDANCE if has_search_tools else ""
            if suppress_probe:
                guardrails = strip_probe(guardrails)
                strategy = strip_probe(strategy)
                search_guidance = strip_probe(search_guidance)
            pip_guidance = PIP_INSTALL_GUIDANCE if self._enable_pip else ""
            prompt = (
                prompt
                + guardrails
                + search_guidance
                + pip_guidance
                + strategy
                + tools_guidance
            )
        elif tools_guidance:
            prompt = prompt + tools_guidance

        return prompt

    @staticmethod
    def _format_budget_constraints(
        budget: Budget,
        budget_snapshot: Optional[Dict[str, Any]] = None,
        critical_threshold: Optional[int] = None,
    ) -> str:
        """Format budget limits for inclusion in system prompt.

        Communicates hard limits upfront so model can plan accordingly.
        These limits are enforced by the system during LLM calls.

        Args:
            budget: The Budget model with limits
            budget_snapshot: Optional snapshot from TokenBudgetPool with remaining_tokens
        """
        constraints = []

        if budget.max_total_tokens is not None:
            constraints.append(f"max_total_tokens: {budget.max_total_tokens}")
        if budget.max_tokens is not None:
            constraints.append(f"max_output_tokens: {budget.max_tokens}")
        if budget.max_cost_usd is not None:
            constraints.append(f"max_cost_usd: ${budget.max_cost_usd:.4f}")
        if budget.max_seconds is not None:
            constraints.append(f"max_seconds: {budget.max_seconds:.1f}")

        if not constraints:
            return ""

        # Add remaining tokens from snapshot if available
        remaining_info = ""
        if budget_snapshot and budget_snapshot.get("remaining_tokens") is not None:
            remaining = budget_snapshot["remaining_tokens"]
            max_total = budget_snapshot.get("max_total_tokens")
            if max_total:
                remaining_info = f"\nYou have {remaining:,} tokens remaining out of {max_total:,} total."
                if remaining < 5000:
                    remaining_info += (
                        "\n*** LOW BUDGET WARNING: You have very limited tokens. "
                        "Work efficiently and call FINAL() as soon as possible. ***"
                    )

        critical_text = ""
        if isinstance(critical_threshold, int) and critical_threshold > 0:
            if critical_threshold <= 100:
                critical_text = (
                    f"Critical budget threshold: {critical_threshold}% used. "
                    "At/over this threshold, sub-LLM calls are disabled.\n"
                )
            else:
                critical_text = (
                    f"Critical budget threshold: {critical_threshold} tokens remaining. "
                    "At/under this threshold, sub-LLM calls are disabled.\n"
                )

        return (
            "\n\n"
            "BUDGET CONSTRAINTS (HARD LIMITS - ENFORCED BY SYSTEM):\n"
            f"Your execution has the following budget limits: {', '.join(constraints)}."
            f"{remaining_info}\n"
            "These are hard limits enforced by the system. The system will stop generation "
            "when limits are reached. You CANNOT exceed these limits. Plan your approach accordingly:\n"
            "- Use sub-LLMs efficiently (each subcall consumes tokens from YOUR budget)\n"
            "- Avoid unnecessary iterations\n"
            "- Prioritize essential steps\n"
            "- Call FINAL() promptly when you have a sufficient answer\n"
            "You will receive budget warnings at 50%, 80%, and 90% usage. "
            "At 90%+, you must call FINAL() immediately.\n"
            "Call budget_status() to check remaining budget at any time.\n"
            f"{critical_text}\n"
        )

    @staticmethod
    def _model_prompt_extra(model_name: str) -> str:
        model_key = model_name.lower()
        model_prompt_extras = {
            "qwen": (
                "IMPORTANT: Be very careful about using llm_query too many times. "
                "There is a cost to each call. Aim for around ~200k characters in each "
                "llm_query call. Thus, you should batch/aggregate information at each "
                "step and only make a small number of llm_query calls.\n\n"
            ),
            "gpt": "",
        }
        for prefix, extra in model_prompt_extras.items():
            if model_key.startswith(prefix):
                return extra
        return ""
