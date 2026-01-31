"""
Enzu client - State-of-the-art LLM interface.

Simplest usage:
    from enzu import Enzu

    client = Enzu()  # Auto-detects from OPENAI_API_KEY or OPENROUTER_API_KEY
    answer = client.run("What is 2+2?")

Explicit configuration:
    client = Enzu("gpt-4o")  # Model as first arg
    client = Enzu("gpt-4o", provider="openai", cost=0.10)

Streaming:
    for chunk in client.stream("Write a story"):
        print(chunk, end="", flush=True)

Multi-turn:
    chat = client.session()
    chat.run("Find the bug", data=logs)
    chat.run("Fix it")

Context manager:
    with Enzu("gpt-4o") as client:
        answer = client.run("Hello")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Optional,
    Union,
    TYPE_CHECKING,
)

from enzu.models import ExecutionReport, RLMExecutionReport, ProgressEvent
from enzu.session import Session

if TYPE_CHECKING:
    from enzu.providers.base import BaseProvider


@dataclass
class StreamChunk:
    """A chunk of streamed output."""

    text: str
    done: bool = False

    def __str__(self) -> str:
        return self.text


def _detect_provider_and_model() -> tuple[str, str]:
    """
    Auto-detect provider and model from environment.

    Priority:
    1. ENZU_MODEL + ENZU_PROVIDER (explicit)
    2. OPENAI_API_KEY → openai + gpt-4o
    3. OPENROUTER_API_KEY → openrouter + openai/gpt-4o
    4. ANTHROPIC_API_KEY → anthropic + claude-3-5-sonnet-latest
    """
    if os.getenv("ENZU_MODEL"):
        provider = os.getenv("ENZU_PROVIDER", "openrouter")
        model = os.getenv("ENZU_MODEL")
        if provider is None or model is None:
            raise ValueError("ENZU_MODEL and ENZU_PROVIDER must both be set")
        return (provider, model)

    if os.getenv("OPENAI_API_KEY"):
        return ("openai", "gpt-4o")

    if os.getenv("OPENROUTER_API_KEY"):
        return ("openrouter", "openai/gpt-4o")

    if os.getenv("ANTHROPIC_API_KEY"):
        return ("anthropic", "claude-3-5-sonnet-latest")

    if os.getenv("MISTRAL_API_KEY"):
        return ("mistral", "mistral-large-latest")

    if os.getenv("GROQ_API_KEY"):
        return ("groq", "llama-3.3-70b-versatile")

    raise ValueError(
        "No API key found. Set one of: OPENAI_API_KEY, OPENROUTER_API_KEY, "
        "ANTHROPIC_API_KEY, or use Enzu(model='...', provider='...')"
    )


class Enzu:
    """
    Enzu client - single entry point for LLM tasks.

    Configure once, use everywhere. Supports streaming, sessions, and budgets.

    Examples:
        # Auto-detect from environment
        client = Enzu()

        # Explicit model
        client = Enzu("gpt-4o")

        # Full configuration
        client = Enzu("gpt-4o", provider="openai", cost=0.10, temperature=0.7)

        # Streaming
        for chunk in client.stream("Write a poem"):
            print(chunk, end="")

        # Multi-turn conversation
        chat = client.session(max_cost_usd=5.00)
        chat.run("Analyze this", data=document)
        chat.run("Explain more")
    """

    __slots__ = (
        "model",
        "provider",
        "api_key",
        "_cost",
        "_tokens",
        "_seconds",
        "_temperature",
        "_max_steps",
        "_provider_instance",
    )

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        cost: Optional[float] = None,
        tokens: Optional[int] = None,
        seconds: Optional[float] = None,
        temperature: Optional[float] = None,
        max_steps: Optional[int] = None,
    ):
        """
        Create an Enzu client.

        Args:
            model: Model identifier. If None, auto-detects from environment.
            provider: Provider name. If None, auto-detects from environment.
            api_key: API key. If None, uses environment variable.
            cost: Default max cost per call in USD. **OpenRouter only** - other
                providers don't report cost in responses. Use tokens= instead.
            tokens: Default max output tokens per call (works with all providers).
            seconds: Default max seconds per call.
            temperature: Default temperature (0.0-2.0).
            max_steps: Default max RLM reasoning steps.
        """
        if model is None and provider is None:
            provider, model = _detect_provider_and_model()
        elif model is None:
            raise ValueError("model is required when provider is specified")
        elif provider is None:
            provider = "openrouter"

        self.model: str = model
        self.provider: str = provider
        self.api_key: Optional[str] = api_key
        self._cost: Optional[float] = cost
        self._tokens: Optional[int] = tokens
        self._seconds: Optional[float] = seconds
        self._temperature: Optional[float] = temperature
        self._max_steps: Optional[int] = max_steps
        self._provider_instance: Optional[BaseProvider] = None

        if cost is not None and provider != "openrouter":
            import warnings

            warnings.warn(
                f"cost= budget is only enforced with OpenRouter (provider={provider!r}). "
                f"Other providers don't report cost in API responses. "
                f"Use tokens= for reliable budget enforcement.",
                UserWarning,
                stacklevel=2,
            )

    def __enter__(self) -> "Enzu":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def __repr__(self) -> str:
        return f"Enzu({self.model!r}, provider={self.provider!r})"

    # =========================================================================
    # Configuration methods (fluent API)
    # =========================================================================

    def with_model(self, model: str) -> "Enzu":
        """Return a new client with a different model."""
        return Enzu(
            model,
            provider=self.provider,
            api_key=self.api_key,
            cost=self._cost,
            tokens=self._tokens,
            seconds=self._seconds,
            temperature=self._temperature,
            max_steps=self._max_steps,
        )

    def with_budget(
        self,
        *,
        cost: Optional[float] = None,
        tokens: Optional[int] = None,
        seconds: Optional[float] = None,
    ) -> "Enzu":
        """Return a new client with different budget defaults."""
        return Enzu(
            self.model,
            provider=self.provider,
            api_key=self.api_key,
            cost=cost if cost is not None else self._cost,
            tokens=tokens if tokens is not None else self._tokens,
            seconds=seconds if seconds is not None else self._seconds,
            temperature=self._temperature,
            max_steps=self._max_steps,
        )

    # =========================================================================
    # Core methods
    # =========================================================================

    def run(
        self,
        task: str,
        *,
        data: Optional[str] = None,
        cost: Optional[float] = None,
        tokens: Optional[int] = None,
        seconds: Optional[float] = None,
        temperature: Optional[float] = None,
        max_steps: Optional[int] = None,
        contains: Optional[List[str]] = None,
        matches: Optional[List[str]] = None,
        min_words: Optional[int] = None,
        goal: Optional[str] = None,
        mode: Optional[str] = None,
        on_progress: Optional[Callable[[str], None]] = None,
        return_report: bool = False,
        sandbox: Optional[Any] = None,
        sandbox_factory: Optional[Any] = None,
        runtime: Optional[Any] = None,
    ) -> Union[str, ExecutionReport, RLMExecutionReport]:
        """
        Run a task and return the answer.

        Args:
            task: The prompt/task to execute.
            data: Context data. Triggers RLM (reasoning) mode.
            cost: Max cost in USD for this call.
            tokens: Max output tokens for this call.
            seconds: Max seconds for this call.
            temperature: Temperature for this call (0.0-2.0).
            max_steps: Max reasoning steps (RLM mode).
            contains: Output must contain these substrings.
            matches: Output must match these regex patterns.
            min_words: Minimum word count required.
            goal: Goal for model self-verification.
            mode: Force "chat" or "rlm". Default: auto-detect.
            on_progress: Callback for progress updates.
            return_report: Return full report instead of just answer.
            sandbox: Pre-constructed sandbox (root run only).
            sandbox_factory: Factory for creating sandboxes with llm callbacks.
            runtime: Custom RLM runtime backend.

        Returns:
            Answer string, or ExecutionReport/RLMExecutionReport if return_report=True.
        """
        from enzu.api import run as enzu_run

        return enzu_run(
            task,
            model=self.model,
            provider=self.provider,
            api_key=self.api_key,
            data=data,
            cost=cost if cost is not None else self._cost,
            tokens=tokens if tokens is not None else self._tokens,
            seconds=seconds if seconds is not None else self._seconds,
            temperature=temperature if temperature is not None else self._temperature,
            max_steps=max_steps if max_steps is not None else self._max_steps,
            contains=contains,
            matches=matches,
            min_words=min_words,
            goal=goal,
            mode=mode,
            on_progress=on_progress,
            return_report=return_report,
            sandbox=sandbox,
            sandbox_factory=sandbox_factory,
            runtime=runtime,
        )

    def stream(
        self,
        task: str,
        *,
        data: Optional[str] = None,
        tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """
        Stream a response, yielding text chunks as they arrive.

        For simple chat-style streaming. For complex tasks with data,
        use run() with on_progress callback.

        Args:
            task: The prompt to execute.
            data: Optional context (system message).
            tokens: Max output tokens.
            temperature: Temperature (0.0-2.0).

        Yields:
            Text chunks as they are generated.

        Example:
            for chunk in client.stream("Write a haiku"):
                print(chunk, end="", flush=True)
            print()  # newline at end
        """
        from enzu.api import _resolve_provider
        from enzu.models import Budget, SuccessCriteria, TaskSpec

        provider = _resolve_provider(
            self.provider,
            api_key=self.api_key,
        )

        max_tokens = tokens if tokens is not None else self._tokens or 1024
        temp = temperature if temperature is not None else self._temperature

        spec = TaskSpec(
            task_id="stream",
            input_text=task if not data else f"{data}\n\n{task}",
            model=self.model,
            budget=Budget(max_tokens=max_tokens),
            success_criteria=SuccessCriteria(min_word_count=1),
            max_output_tokens=max_tokens,
            temperature=temp,
        )

        chunks: List[str] = []

        def collect_chunk(event: ProgressEvent) -> None:
            if event.is_partial and event.message:
                chunks.append(event.message)

        provider.stream(spec, on_progress=collect_chunk)

        for chunk in chunks:
            yield chunk

    def session(
        self,
        *,
        max_cost_usd: Optional[float] = None,
        max_tokens: Optional[int] = None,
        history_max_chars: int = 10000,
    ) -> Session:
        """
        Create a conversation session.

        Sessions maintain history across multiple run() calls, enabling
        multi-turn conversations.

        Args:
            max_cost_usd: Session-wide cost cap.
            max_tokens: Session-wide output token cap (cumulative).
            history_max_chars: Max chars of history to retain.

        Returns:
            Session configured with this client's model/provider.

        Example:
            chat = client.session(max_cost_usd=5.00)
            chat.run("Find the bug", data=logs)
            chat.run("Explain what you found")
            chat.run("Now fix it")
            chat.save("debug_session.json")
        """
        return Session(
            model=self.model,
            provider=self.provider,
            api_key=self.api_key,
            max_cost_usd=max_cost_usd,
            max_tokens=max_tokens,
            history_max_chars=history_max_chars,
        )

    # =========================================================================
    # Convenience methods
    # =========================================================================

    def ask(self, question: str) -> str:
        """
        Simple Q&A - shortest path to an answer.

        Equivalent to run() but with a clearer intent for simple questions.

        Example:
            answer = client.ask("What is the capital of France?")
        """
        return self.run(question)  # type: ignore

    def analyze(
        self,
        task: str,
        data: str,
        *,
        cost: Optional[float] = None,
        goal: Optional[str] = None,
    ) -> str:
        """
        Analyze data with a task/question.

        Uses RLM (reasoning) mode for complex analysis.

        Args:
            task: What to do with the data.
            data: The data to analyze.
            cost: Max cost in USD.
            goal: Success criteria for the analysis.

        Example:
            answer = client.analyze(
                "Find the root cause of the error",
                data=log_content,
                cost=1.00,
            )
        """
        return self.run(task, data=data, cost=cost, goal=goal)  # type: ignore

    def batch(
        self,
        tasks: List[str],
        *,
        data: Optional[str] = None,
    ) -> List[str]:
        """
        Run multiple tasks and return all answers.

        Tasks are executed sequentially. For parallel execution,
        use the queue module.

        Args:
            tasks: List of tasks/prompts.
            data: Optional shared context for all tasks.

        Returns:
            List of answers in the same order as tasks.
        """
        return [self.run(task, data=data) for task in tasks]  # type: ignore


# =============================================================================
# Module-level API - The primary interface
# =============================================================================


def new(
    model: Optional[str] = None,
    *,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    cost: Optional[float] = None,
    tokens: Optional[int] = None,
    seconds: Optional[float] = None,
    temperature: Optional[float] = None,
    max_steps: Optional[int] = None,
) -> Enzu:
    """
    Create a new Enzu client.

    Example:
        from enzu import new

        client = new("gpt-4o")
        client.run("Hello")

        # Multi-model setup
        gpt = new("gpt-4o", provider="openai")
        claude = new("claude-3-5-sonnet", provider="anthropic")
    """
    return Enzu(
        model,
        provider=provider,
        api_key=api_key,
        cost=cost,
        tokens=tokens,
        seconds=seconds,
        temperature=temperature,
        max_steps=max_steps,
    )


def ask(question: str, **kwargs: Any) -> str:
    """
    Simple Q&A - shortest path to an answer.

    Example:
        import enzu
        enzu.ask("What is 2+2?")
    """
    return Enzu(**kwargs).ask(question)


def run(
    task: str,
    *,
    data: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    cost: Optional[float] = None,
    tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    sandbox: Optional[Any] = None,
    sandbox_factory: Optional[Any] = None,
    runtime: Optional[Any] = None,
    **kwargs: Any,
) -> str:
    """
    Run a task and return the answer.

    Example:
        import enzu
        enzu.run("Analyze this code", data=source_code)
    """
    client = Enzu(
        model, provider=provider, cost=cost, tokens=tokens, temperature=temperature
    )
    return client.run(
        task,
        data=data,
        sandbox=sandbox,
        sandbox_factory=sandbox_factory,
        runtime=runtime,
        **kwargs,
    )  # type: ignore


def stream(
    task: str,
    *,
    data: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Iterator[str]:
    """
    Stream a response, yielding text chunks.

    Example:
        import enzu
        for chunk in enzu.stream("Write a poem"):
            print(chunk, end="", flush=True)
    """
    client = Enzu(model, provider=provider, tokens=tokens, temperature=temperature)
    return client.stream(task, data=data)


def session(
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    max_cost_usd: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Session:
    """
    Create a conversation session.

    Example:
        import enzu
        chat = enzu.session(max_cost_usd=5.00)
        chat.run("Find the bug", data=logs)
        chat.run("Fix it")
    """
    client = Enzu(model, provider=provider)
    return client.session(max_cost_usd=max_cost_usd, max_tokens=max_tokens)


def analyze(
    task: str,
    data: str,
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    cost: Optional[float] = None,
    goal: Optional[str] = None,
) -> str:
    """
    Analyze data with reasoning.

    Example:
        import enzu
        enzu.analyze("Find the root cause", data=logs, cost=1.00)
    """
    client = Enzu(model, provider=provider, cost=cost)
    return client.analyze(task, data, goal=goal)
