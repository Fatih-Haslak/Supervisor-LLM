from collections.abc import Sequence
from typing import Any

import pytest

from app.llm.client import LLMError
from app.llm.schemas import ChatMessage, LLMResponse
from app.observability.events import TraceRecorder, trace_session
from app.orchestration.auto_mode import AutoModeRouter


class ScriptedLLM:
    def __init__(self, responses: list[str], *, error: bool = False) -> None:
        self.responses = iter(responses)
        self.error = error
        self.calls: list[list[ChatMessage]] = []
        self.schemas: list[dict[str, Any]] = []

    async def chat(
        self, messages: Sequence[ChatMessage], *, json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        assert json_schema is not None
        self.calls.append(list(messages))
        self.schemas.append(json_schema)
        if self.error:
            raise LLMError("router unavailable")
        return LLMResponse(content=next(self.responses), model="scripted")


@pytest.mark.asyncio
async def test_small_talk_and_arithmetic_use_fast_safe_paths() -> None:
    llm = ScriptedLLM([])
    router = AutoModeRouter(llm)
    assert await router.select("Merhaba!") == "chat"
    assert await router.select("Adım neydi?") == "chat"
    assert await router.select("3+44 kaç eder?") == "single"
    assert llm.calls == []


@pytest.mark.asyncio
async def test_contextual_request_can_be_sent_to_planner() -> None:
    llm = ScriptedLLM(['{"mode":"plan","confidence":0.92}'])
    history = [
        ChatMessage(role="user", content="sales.csv içindeki tutarları incele"),
        ChatMessage(role="assistant", content="CSV dosyasını analiz ettim."),
    ]
    recorder = TraceRecorder()
    with trace_session(recorder):
        selected = await AutoModeRouter(llm).select("Bunları bir rapora dönüştür", history)
    assert selected == "plan"
    assert "sales.csv" in llm.calls[0][0].content
    assert llm.schemas[0]["properties"]["mode"]["enum"] == [
        "chat", "single", "supervisor", "plan",
    ]
    assert any(event.event == "mode_selected" and event.mode == "plan"
               for event in recorder.events)


@pytest.mark.asyncio
async def test_tool_and_code_requests_cannot_fall_into_chat() -> None:
    llm = ScriptedLLM([
        '{"mode":"chat","confidence":0.99}',
        '{"mode":"single","confidence":0.99}',
        '{"mode":"chat","confidence":0.99}',
    ])
    router = AutoModeRouter(llm)
    assert await router.select("workspace/note.txt dosyasını oku") == "supervisor"
    assert await router.select("Fatih Tekke kimdir?") == "supervisor"
    assert await router.select("Bana Şenol Güneş hakkında bilgi getir") == "supervisor"
    assert await router.select("```python\ndef f(n): return n\n```\nAnaliz et") == "supervisor"


@pytest.mark.asyncio
async def test_multi_step_request_forces_plan_even_if_model_would_misroute() -> None:
    llm = ScriptedLLM([])
    assert await AutoModeRouter(llm).select(
        "Önce workspace/sales.csv dosyasını analiz et, sonra rapor yaz"
    ) == "plan"
    assert llm.calls == []


@pytest.mark.asyncio
async def test_invalid_or_unavailable_router_has_bounded_fallback() -> None:
    invalid = ScriptedLLM(["{}", '{"mode":"chat","confidence":2}'])
    router = AutoModeRouter(invalid)
    assert await router.select("workspace/data.csv dosyasını analiz et") == "supervisor"
    assert len(invalid.calls) == 2
    unavailable = ScriptedLLM([], error=True)
    assert await AutoModeRouter(unavailable).select("Fatih Tekke kimdir?") == "supervisor"


@pytest.mark.asyncio
async def test_low_confidence_uses_safe_existing_fallback() -> None:
    llm = ScriptedLLM(['{"mode":"chat","confidence":0.4}'])
    assert await AutoModeRouter(llm).select("workspace/report.md dosyasını oku") == "supervisor"
