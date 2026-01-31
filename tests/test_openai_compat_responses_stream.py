from __future__ import annotations

from enzu.models import Budget, SuccessCriteria, TaskSpec
from enzu.providers.openai_compat import OpenAICompatProvider


class _FakeEvent:
    def __init__(self, event_type: str, **fields: object) -> None:
        self.type = event_type
        for key, value in fields.items():
            setattr(self, key, value)


class _FakeStream:
    def __init__(self, events: list[object]) -> None:
        self._events = events

    def __iter__(self):
        yield from self._events


class _FakeResponses:
    def __init__(self, events: list[object]) -> None:
        self.kwargs: dict | None = None
        self._events = events

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _FakeStream(self._events)


class _FallbackResponses:
    def __init__(self, events: list[object], response: dict) -> None:
        self._events = events
        self._response = response
        self.kwargs_list: list[dict] = []

    def create(self, **kwargs):
        self.kwargs_list.append(kwargs)
        if kwargs.get("stream"):
            return _FakeStream(self._events)
        return self._response


class _FakeClient:
    def __init__(self, events: list[object]) -> None:
        self.responses = _FakeResponses(events)


class _FallbackClient:
    def __init__(self, events: list[object], response: dict) -> None:
        self.responses = _FallbackResponses(events, response)


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="stream-responses",
        input_text="say hi",
        model="mock-model",
        budget=Budget(max_tokens=5),
        success_criteria=SuccessCriteria(min_word_count=1),
        max_output_tokens=5,
    )


def test_stream_responses_handles_output_text_done() -> None:
    events = [
        _FakeEvent(
            "response.output_text.done",
            text="hi",
            output_index=0,
            content_index=0,
        )
    ]
    client = _FakeClient(events)
    provider = OpenAICompatProvider(
        name="openai", client=client, supports_responses=True
    )  # type: ignore[arg-type]

    result = provider.stream(_task())

    assert result.output_text == "hi"
    assert client.responses.kwargs is not None
    assert client.responses.kwargs.get("stream") is True


def test_stream_responses_handles_dict_output_item_done() -> None:
    events = [
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "content": [
                    {
                        "type": "output_text",
                        "text": "hello",
                    }
                ]
            },
        },
        {
            "type": "response.completed",
            "response": {"usage": {"output_tokens": 1}},
        },
    ]
    client = _FakeClient(events)
    provider = OpenAICompatProvider(
        name="openai", client=client, supports_responses=True
    )  # type: ignore[arg-type]

    result = provider.stream(_task())

    assert result.output_text == "hello"
    assert result.usage.get("output_tokens") == 1


def test_stream_responses_falls_back_on_failure_event() -> None:
    # Covers response.failed -> fallback to non-streaming responses.
    events = [
        _FakeEvent("response.failed", error="boom"),
    ]
    response = {"output_text": "fallback", "usage": {"output_tokens": 2}}
    client = _FallbackClient(events, response)
    provider = OpenAICompatProvider(
        name="openai", client=client, supports_responses=True
    )  # type: ignore[arg-type]
    progress: list[str] = []

    result = provider.stream(_task(), on_progress=lambda e: progress.append(e.message))

    assert result.output_text == "fallback"
    assert any(kwargs.get("stream") for kwargs in client.responses.kwargs_list)
    assert any(not kwargs.get("stream") for kwargs in client.responses.kwargs_list)
    assert any("responses_stream_failed_fallback" in msg for msg in progress)
