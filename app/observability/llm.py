"""Metadata-only measurement for the shared LLM interface."""

from collections.abc import Sequence
from time import perf_counter
from typing import Any

from app.llm.client import LLMClient
from app.llm.schemas import ChatMessage, LLMResponse
from app.observability.events import record


class TracedLLM:
    def __init__(self, backend: LLMClient) -> None:
        self._backend = backend

    async def chat(
        self, messages: Sequence[ChatMessage], *, json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        started = perf_counter()
        prompt_chars = sum(len(message.content) for message in messages)
        try:
            response = await self._backend.chat(
                messages, json_mode=json_mode, json_schema=json_schema
            )
        except Exception as exc:
            record(
                "model_error", success=False,
                duration_ms=round((perf_counter() - started) * 1000, 2),
                prompt_chars=prompt_chars, error_type=type(exc).__name__,
            )
            raise
        usage = response.usage
        record(
            "model_call", success=True,
            duration_ms=round((perf_counter() - started) * 1000, 2),
            prompt_chars=prompt_chars, response_chars=len(response.content),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )
        return response
