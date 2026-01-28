from __future__ import annotations

from enzu.models import Budget, SuccessCriteria, TaskSpec
from enzu.providers.openai_compat import OpenAICompatProvider


class _FakeDelta:
    content = "hi"


class _FakeChoice:
    delta = _FakeDelta()


class _FakeChunk:
    def __init__(self, usage: dict | None = None) -> None:
        self.choices = [_FakeChoice()]
        self.usage = usage


class _FakeStream:
    def __iter__(self):
        yield _FakeChunk({"output_tokens": 1, "total_tokens": 1})


class _FakeChatCompletions:
    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _FakeStream()


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeChatCompletions()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()


class _EmptyDelta:
    content = ""


class _EmptyChoice:
    delta = _EmptyDelta()


class _EmptyChunk:
    def __init__(self) -> None:
        self.choices = [_EmptyChoice()]
        self.usage = None


class _EmptyStream:
    def __iter__(self):
        yield _EmptyChunk()


class _FakeMessage:
    content = "fallback"


class _FakeMessageChoice:
    message = _FakeMessage()


class _FallbackResponse:
    choices = [_FakeMessageChoice()]
    usage = {"output_tokens": 2, "total_tokens": 2}


class _FallbackChatCompletions:
    def __init__(self) -> None:
        self.kwargs_list: list[dict] = []

    def create(self, **kwargs):
        self.kwargs_list.append(kwargs)
        if kwargs.get("stream"):
            return _EmptyStream()
        return _FallbackResponse()


class _FallbackChat:
    def __init__(self) -> None:
        self.completions = _FallbackChatCompletions()


class _FallbackClient:
    def __init__(self) -> None:
        self.chat = _FallbackChat()


def test_stream_chat_completions_requests_usage() -> None:
    client = _FakeClient()
    provider = OpenAICompatProvider(name="openai", client=client)  # type: ignore[arg-type]
    task = TaskSpec(
        task_id="stream-usage",
        input_text="say hi",
        model="mock-model",
        budget=Budget(max_tokens=5),
        success_criteria=SuccessCriteria(min_word_count=1),
    )

    result = provider.stream(task)

    assert result.output_text == "hi"
    assert client.chat.completions.kwargs is not None
    assert client.chat.completions.kwargs["stream_options"] == {"include_usage": True}


def test_stream_chat_completions_falls_back_when_no_delta() -> None:
    # Covers chat stream fallback to non-streaming generate.
    client = _FallbackClient()
    provider = OpenAICompatProvider(name="openai", client=client)  # type: ignore[arg-type]
    task = TaskSpec(
        task_id="stream-fallback",
        input_text="say hi",
        model="mock-model",
        budget=Budget(max_tokens=5),
        success_criteria=SuccessCriteria(min_word_count=1),
    )

    result = provider.stream(task)

    assert result.output_text == "fallback"
    assert any(kwargs.get("stream") for kwargs in client.chat.completions.kwargs_list)
    assert any(not kwargs.get("stream") for kwargs in client.chat.completions.kwargs_list)
