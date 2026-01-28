from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from openai import OpenAI

from enzu.models import ProgressEvent, ProviderResult, TaskSpec
# Shared helpers: _get_field, _as_list are used for response parsing.
# Defined in usage.py to avoid duplication across modules.
from enzu.usage import _as_list, _get_field, build_task_input_text
from enzu.providers.base import BaseProvider
from enzu.retry import with_retry


class OpenAICompatProvider(BaseProvider):
    """
    Unified provider for all OpenAI-compatible APIs.
    
    Supports both Open Responses API (per openresponses.org spec) and
    Chat Completions API. Tries Responses API first if supported, falls
    back to Chat Completions for compatibility.
    """

    def __init__(
        self,
        name: str,
        *,
        api_key: Optional[str] = None,
        client: Optional[OpenAI] = None,
        base_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        supports_responses: bool = False,
    ) -> None:
        self.name = name
        self._supports_responses = supports_responses
        if client is None:
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers=headers,
                organization=organization,
                project=project,
            )
        else:
            self._client = client

    def generate(self, task: TaskSpec) -> ProviderResult:
        """
        Generate response using Open Responses API if supported, else Chat Completions.
        
        Per openresponses.org spec, Responses API uses 'input' parameter and
        returns 'output_text' field. Chat Completions uses 'messages' and
        returns 'choices[0].message.content'.
        """
        if self._supports_responses:
            try:
                return self._generate_responses(task)
            except Exception:
                if self._requires_responses(task):
                    raise
                return self._generate_chat_completions(task)
        if self._requires_responses(task):
            raise ValueError("Provider does not support Open Responses API.")
        return self._generate_chat_completions(task)

    def _extra_body(self) -> Optional[Dict[str, Any]]:
        if self.name == "openrouter":
            return {"usage": {"include": True}}
        return None

    def _requires_responses(self, task: TaskSpec) -> bool:
        return bool(task.responses)

    def _build_responses_params(self, task: TaskSpec, *, stream: bool) -> Dict[str, Any]:
        params: Dict[str, Any] = dict(task.responses) if isinstance(task.responses, dict) else {}
        if stream:
            params["stream"] = True
        else:
            params.pop("stream", None)

        if params.get("input") is None:
            params["input"] = self._build_input(task)
        params["model"] = task.model
        if task.max_output_tokens is not None:
            params["max_output_tokens"] = task.max_output_tokens
        if task.temperature is not None:
            params["temperature"] = task.temperature

        extra_body = self._extra_body()
        if extra_body:
            params["extra_body"] = self._merge_extra_body(params.get("extra_body"), extra_body)
        return self._clean_params(params)

    @staticmethod
    def _merge_extra_body(base: Any, overlay: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(base, dict):
            return dict(overlay)
        merged = dict(overlay)
        for key, value in base.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return merged

    @with_retry()
    def _generate_responses(self, task: TaskSpec) -> ProviderResult:
        """Generate using Open Responses API (openresponses.org spec)."""
        params = self._build_responses_params(task, stream=False)
        response = self._client.responses.create(**params)
        output_text = self._extract_responses_output_text(response)
        usage = self._extract_usage(response)
        return ProviderResult(
            output_text=output_text,
            raw=response,
            usage=usage,
            provider=self.name,
            model=task.model,
        )

    @with_retry()
    def _generate_chat_completions(self, task: TaskSpec) -> ProviderResult:
        """Generate using Chat Completions API (fallback)."""
        messages = self._build_messages(task)
        extra_body = self._extra_body()
        params = self._clean_params(
            {
                "model": task.model,
                "messages": messages,
                "max_tokens": task.max_output_tokens,
                "temperature": task.temperature,
                "extra_body": extra_body,
            }
        )
        response = self._client.chat.completions.create(**params)
        output_text = self._extract_chat_output_text(response)
        usage = self._extract_usage(response)
        return ProviderResult(
            output_text=output_text,
            raw=response,
            usage=usage,
            provider=self.name,
            model=task.model,
        )

    def stream(
        self,
        task: TaskSpec,
        on_progress: Optional[Callable[[ProgressEvent], None]] = None,
    ) -> ProviderResult:
        """
        Stream response using Open Responses API if supported, else Chat Completions.
        
        Open Responses uses semantic events like 'response.output_text.delta'.
        Chat Completions uses 'choices[0].delta.content'.
        """
        if self._supports_responses:
            try:
                return self._stream_responses(task, on_progress)
            except Exception:
                if self._requires_responses(task):
                    raise
                return self._stream_chat_completions(task, on_progress)
        if self._requires_responses(task):
            raise ValueError("Provider does not support Open Responses API.")
        return self._stream_chat_completions(task, on_progress)

    @with_retry()
    def _stream_responses(self, task: TaskSpec, on_progress: Optional[Callable[[ProgressEvent], None]]) -> ProviderResult:
        """Stream using Open Responses API (openresponses.org spec)."""
        output_chunks: list[str] = []
        refusal_chunks: list[str] = []
        usage: Dict[str, Any] = {}
        done_items: Dict[int, str] = {}
        done_items_fallback: list[str] = []
        done_parts: Dict[tuple[int, int], str] = {}
        done_parts_fallback: list[str] = []
        refusal_parts: Dict[tuple[int, int], str] = {}
        refusal_parts_fallback: list[str] = []
        completed_response: Any = None
        try:
            params = self._build_responses_params(task, stream=True)
            stream = self._client.responses.create(**params)
            for event in stream:
                payload = self._event_payload(event)
                event_type = _get_field(payload, "type")
                if event_type in {"response.failed", "response.error"}:
                    error = _get_field(payload, "error") or _get_field(payload, "message")
                    raise RuntimeError(f"Responses stream failed: {error}")
                if event_type == "response.output_text.delta":
                    delta = _get_field(payload, "delta", "")
                    if isinstance(delta, str) and delta:
                        output_chunks.append(delta)
                        if on_progress:
                            on_progress(
                                ProgressEvent(
                                    phase="generation",
                                    message=delta,
                                    is_partial=True,
                                    data={"provider": self.name},
                                )
                            )
                elif event_type == "response.refusal.delta":
                    delta = _get_field(payload, "delta", "")
                    if isinstance(delta, str) and delta:
                        refusal_chunks.append(delta)
                        if on_progress:
                            on_progress(
                                ProgressEvent(
                                    phase="generation",
                                    message=delta,
                                    is_partial=True,
                                    data={"provider": self.name},
                                )
                            )
                elif event_type == "response.output_text.done":
                    text = _get_field(payload, "text", "")
                    if isinstance(text, str) and text:
                        output_index = self._get_int(payload, "output_index")
                        content_index = self._get_int(payload, "content_index")
                        if output_index is not None and content_index is not None:
                            done_parts[(output_index, content_index)] = text
                        else:
                            done_parts_fallback.append(text)
                elif event_type == "response.refusal.done":
                    text = _get_field(payload, "refusal")
                    if text is None:
                        text = _get_field(payload, "text", "")
                    if isinstance(text, str) and text:
                        output_index = self._get_int(payload, "output_index")
                        content_index = self._get_int(payload, "content_index")
                        if output_index is not None and content_index is not None:
                            refusal_parts[(output_index, content_index)] = text
                        else:
                            refusal_parts_fallback.append(text)
                elif event_type == "response.content_part.done":
                    part = _get_field(payload, "part")
                    if part:
                        part_type = _get_field(part, "type")
                        if part_type in {"output_text", "refusal"}:
                            text = _get_field(part, "text", "")
                            if part_type == "refusal":
                                text = _get_field(part, "refusal", text)
                            if isinstance(text, str) and text:
                                output_index = self._get_int(payload, "output_index")
                                content_index = self._get_int(payload, "content_index")
                                if output_index is not None and content_index is not None:
                                    if part_type == "output_text":
                                        done_parts[(output_index, content_index)] = text
                                    else:
                                        refusal_parts[(output_index, content_index)] = text
                                else:
                                    if part_type == "output_text":
                                        done_parts_fallback.append(text)
                                    else:
                                        refusal_parts_fallback.append(text)
                elif event_type == "response.output_item.done":
                    item = _get_field(payload, "item")
                    if item:
                        text = self._extract_output_text_from_item(item)
                        if text:
                            output_index = self._get_int(payload, "output_index")
                            if output_index is not None:
                                done_items[output_index] = text
                            else:
                                done_items_fallback.append(text)
                elif event_type == "response.completed":
                    response = _get_field(payload, "response")
                    if response is None and _get_field(payload, "object") == "response":
                        response = payload
                    if response is not None:
                        completed_response = response
                        usage = self._extract_usage(response)
            output_text = "".join(output_chunks)
            done_items_text = "".join(done_items[index] for index in sorted(done_items))
            if done_items_fallback:
                done_items_text += "".join(done_items_fallback)
            done_parts_text = "".join(done_parts[key] for key in sorted(done_parts))
            if done_parts_fallback:
                done_parts_text += "".join(done_parts_fallback)
            refusal_parts_text = "".join(refusal_parts[key] for key in sorted(refusal_parts))
            if refusal_parts_fallback:
                refusal_parts_text += "".join(refusal_parts_fallback)
            refusal_text = "".join(refusal_chunks)
            fallback_text = done_items_text or done_parts_text or refusal_parts_text or refusal_text
            if not fallback_text and completed_response is not None:
                fallback_text = self._extract_responses_output_text(completed_response)
            if fallback_text and (not output_text or len(fallback_text) > len(output_text)):
                output_text = fallback_text
            if (
                not output_text
                and completed_response is None
                and not done_items
                and not done_items_fallback
                and not done_parts
                and not done_parts_fallback
                and not refusal_chunks
                and not refusal_parts
                and not refusal_parts_fallback
            ):
                return self._generate_responses(task)
            return ProviderResult(
                output_text=output_text,
                raw=completed_response if completed_response is not None else {"streamed": True},
                usage=usage,
                provider=self.name,
                model=task.model,
            )
        except Exception:
            if on_progress:
                on_progress(
                    ProgressEvent(
                        phase="generation",
                        message=f"{self.name}_responses_stream_failed_fallback",
                        data={"provider": self.name},
                    )
                )
            return self._generate_responses(task)

    @with_retry()
    def _stream_chat_completions(
        self, task: TaskSpec, on_progress: Optional[Callable[[ProgressEvent], None]]
    ) -> ProviderResult:
        """Stream using Chat Completions API (fallback)."""
        messages = self._build_messages(task)
        output_chunks: list[str] = []
        usage: Dict[str, Any] = {}
        try:
            extra_body = self._extra_body()
            params = self._clean_params(
                {
                    "model": task.model,
                    "messages": messages,
                    "max_tokens": task.max_output_tokens,
                    "temperature": task.temperature,
                    # Request final usage chunk for budget accounting where supported.
                    "stream_options": {"include_usage": True},
                    "stream": True,
                    "extra_body": extra_body,
                }
            )
            stream = self._client.chat.completions.create(**params)
            for chunk in stream:
                delta = self._extract_chat_delta(chunk)
                if delta:
                    output_chunks.append(delta)
                    if on_progress:
                        on_progress(
                            ProgressEvent(
                                phase="generation",
                                message=delta,
                                is_partial=True,
                                data={"provider": self.name},
                            )
                        )
                if chunk.usage:
                    usage = self._extract_usage(chunk)
            output_text = "".join(output_chunks)
            if not output_text:
                return self._generate_chat_completions(task)
            return ProviderResult(
                output_text=output_text,
                raw={"streamed": True},
                usage=usage,
                provider=self.name,
                model=task.model,
            )
        except Exception:
            if on_progress:
                on_progress(
                    ProgressEvent(
                        phase="generation",
                        message=f"{self.name}_chat_stream_failed_fallback_generate",
                        data={"provider": self.name},
                    )
                )
            return self._generate_chat_completions(task)

    def _build_input(self, task: TaskSpec) -> Any:
        """
        Build input string for Open Responses API.
        
        Per openresponses.org spec, Responses API uses 'input' parameter
        (string or structured). Includes success criteria if present.
        """
        if isinstance(task.responses, dict):
            responses_input = task.responses.get("input")
            if responses_input is not None:
                return responses_input
        return build_task_input_text(task)

    def _build_messages(self, task: TaskSpec) -> list[Dict[str, str]]:
        """
        Build messages array for Chat Completions API.
        
        Includes success criteria in the user message if present.
        """
        content = build_task_input_text(task)
        return [{"role": "user", "content": content}]

    @staticmethod
    def _extract_responses_output_text(response: Any) -> str:
        """
        Extract text from Open Responses API response.
        
        Per openresponses.org spec, Responses API returns 'output_text' field
        directly, or nested in 'output' items.
        """
        output_text = _get_field(response, "output_text")
        if isinstance(output_text, str) and output_text:
            return output_text
        output = _get_field(response, "output")
        chunks: list[str] = []
        for item in _as_list(output):
            text = OpenAICompatProvider._extract_output_text_from_item(item)
            if text:
                chunks.append(text)
        return "".join(chunks)

    @staticmethod
    def _extract_chat_output_text(response: Any) -> str:
        """
        Extract text from Chat Completions response.
        
        Response shape: response.choices[0].message.content
        
        Reasoning models (e.g., z-ai/glm-4.7) return output in 'reasoning' field
        with empty 'content'. Falls back to 'reasoning' when 'content' is empty.
        See: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
        """
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if not choices:
                return ""
            choice = choices[0] if isinstance(choices, list) else choices
            message = choice.get("message") if isinstance(choice, dict) else None
            if not isinstance(message, dict):
                return ""
            content = message.get("content")
            if content:
                return content
            # Reasoning models: output in 'reasoning' field when 'content' is empty
            reasoning = message.get("reasoning")
            return reasoning or ""
        if not hasattr(response, "choices") or not response.choices:
            return ""
        choice = response.choices[0]
        if not hasattr(choice, "message"):
            return ""
        message = choice.message
        content = getattr(message, "content", None)
        if content:
            return content
        # Reasoning models: output in 'reasoning' field when 'content' is empty
        reasoning = getattr(message, "reasoning", None)
        return reasoning or ""

    @staticmethod
    def _extract_chat_delta(chunk: Any) -> str:
        """
        Extract delta text from Chat Completions streaming chunk.
        
        Chunk shape: chunk.choices[0].delta.content
        
        Reasoning models stream thinking in 'reasoning' field. Falls back to
        'reasoning' when 'content' is empty.
        """
        if isinstance(chunk, dict):
            choices = chunk.get("choices") or []
            if not choices:
                return ""
            choice = choices[0] if isinstance(choices, list) else choices
            delta = choice.get("delta") if isinstance(choice, dict) else None
            if not isinstance(delta, dict):
                return ""
            content = delta.get("content")
            if content:
                return content
            # Reasoning models: delta in 'reasoning' field
            reasoning = delta.get("reasoning")
            return reasoning or ""
        if not hasattr(chunk, "choices") or not chunk.choices:
            return ""
        choice = chunk.choices[0]
        if not hasattr(choice, "delta"):
            return ""
        delta = choice.delta
        content = getattr(delta, "content", None)
        if content:
            return content
        # Reasoning models: delta in 'reasoning' field
        reasoning = getattr(delta, "reasoning", None)
        return reasoning or ""

    @staticmethod
    def _extract_usage(response: Any) -> Dict[str, Any]:
        """
        Extract usage stats from response.
        
        Handles both CompletionUsage objects and dicts.
        """
        if isinstance(response, dict):
            usage = response.get("usage")
        else:
            usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if isinstance(usage, dict):
            return usage
        return dict(usage)

    @staticmethod
    def _clean_params(params: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in params.items() if value is not None}

    @staticmethod
    def _get_int(obj: Any, key: str) -> Optional[int]:
        # Uses shared _get_field from usage.py
        value = _get_field(obj, key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return None

    @staticmethod
    def _event_payload(event: Any) -> Any:
        if event is None:
            return None
        if isinstance(event, dict):
            data = event.get("data")
            if isinstance(data, dict) or hasattr(data, "type"):
                return data
            return event
        data = getattr(event, "data", None)
        if isinstance(data, dict) or hasattr(data, "type"):
            return data
        return event

    @staticmethod
    def _extract_output_text_from_item(item: Any) -> str:
        content = _get_field(item, "content")
        chunks: list[str] = []
        for part in _as_list(content):
            part_type = _get_field(part, "type")
            if part_type == "output_text":
                text = _get_field(part, "text", "")
                if isinstance(text, str) and text:
                    chunks.append(text)
            elif part_type == "refusal":
                refusal = _get_field(part, "refusal", "")
                if isinstance(refusal, str) and refusal:
                    chunks.append(refusal)
            else:
                text = _get_field(part, "text")
                if isinstance(text, str) and text:
                    chunks.append(text)
        return "".join(chunks)
