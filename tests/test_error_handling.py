"""Bounded failures become stable, content-safe outcomes."""

import time
from typing import Any

import pytest
from pydantic import BaseModel

from app.config.settings import Settings
from app.errors import describe_error, format_cli_error
from app.llm.client import LlamaCppClient, LLMContextOverflowError, LLMTimeoutError
from app.llm.schemas import ChatMessage
from app.llm.structured import StructuredOutputError
from app.tools.base import BaseTool, ToolResult


class RaisingModel:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def create_chat_completion(self, **_kwargs: Any) -> Any:
        raise self.exc


@pytest.mark.asyncio
async def test_context_overflow_has_specific_safe_error() -> None:
    client = LlamaCppClient(
        Settings(_env_file=None),
        RaisingModel(ValueError("Requested tokens (9999) exceed context window of 4096")),
    )
    with pytest.raises(LLMContextOverflowError) as captured:
        await client.chat([ChatMessage(role="user", content="PRIVATE_PROMPT")])
    assert "PRIVATE_PROMPT" not in str(captured.value)
    assert describe_error(captured.value).code == "CONTEXT_OVERFLOW"


@pytest.mark.asyncio
async def test_backend_timeout_has_specific_safe_error() -> None:
    client = LlamaCppClient(Settings(_env_file=None), RaisingModel(TimeoutError("PRIVATE")))
    with pytest.raises(LLMTimeoutError) as captured:
        await client.chat([ChatMessage(role="user", content="x")])
    assert "PRIVATE" not in str(captured.value)
    assert describe_error(captured.value).retryable


class SlowModel:
    def create_chat_completion(self, **_kwargs: Any) -> dict[str, Any]:
        time.sleep(0.02)
        return {"choices": [{"message": {"content": "late"}}]}


@pytest.mark.asyncio
async def test_local_model_deadline_discards_late_response() -> None:
    settings = Settings(_env_file=None, llm_timeout_seconds=0.001)
    client = LlamaCppClient(settings, SlowModel())
    with pytest.raises(LLMTimeoutError):
        await client.chat([ChatMessage(role="user", content="x")])


class SerialModel:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    def create_chat_completion(self, **_kwargs: Any) -> dict[str, Any]:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        time.sleep(0.03)
        self.active -= 1
        return {"choices": [{"message": {"content": "ok"}}]}


@pytest.mark.asyncio
async def test_timeout_keeps_shared_model_serialized() -> None:
    model = SerialModel()
    settings = Settings(_env_file=None, llm_timeout_seconds=0.001)
    async with LlamaCppClient(settings, model) as client:
        with pytest.raises(LLMTimeoutError):
            await client.chat([ChatMessage(role="user", content="first")])
        settings.llm_timeout_seconds = 0.2
        answer = await client.chat([ChatMessage(role="user", content="second")])
    assert answer.content == "ok"
    assert model.max_active == 1


class EmptyInput(BaseModel):
    pass


class TimedOutTool(BaseTool[EmptyInput]):
    name = "timed_out"
    description = "A tool that exceeds its own deadline"
    input_type = EmptyInput

    async def execute(self, arguments: EmptyInput) -> ToolResult:
        raise TimeoutError("PRIVATE_TOOL_DETAIL")


@pytest.mark.asyncio
async def test_tool_timeout_is_standard_result() -> None:
    result = await TimedOutTool().run({})
    assert result.error_type == "Timeout"
    assert result.success is False
    assert "PRIVATE_TOOL_DETAIL" not in result.model_dump_json()


def test_cli_error_message_avoids_raw_model_output() -> None:
    error = StructuredOutputError("PRIVATE_MODEL_OUTPUT")
    shown = format_cli_error(error)
    assert shown.startswith("Hata [INVALID_OUTPUT]:")
    assert "PRIVATE_MODEL_OUTPUT" not in shown
