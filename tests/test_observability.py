"""Trace metadata is useful without exposing task or tool contents."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from app.llm.schemas import ChatMessage, LLMResponse, TokenUsage
from app.observability.events import TraceRecorder, agent_span, trace_session
from app.observability.llm import TracedLLM
from app.orchestration.state import AgentState
from app.tools.filesystem import FileWriteTool, Workspace
from app.tools.registry import ToolRegistry


class FakeBackend:
    async def chat(
        self, messages: Sequence[ChatMessage], *, json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content="MODEL_PRIVATE_VALUE", model="fake",
            usage=TokenUsage(prompt_tokens=12, completion_tokens=3, total_tokens=15),
        )


@pytest.mark.asyncio
async def test_trace_has_ids_timing_tokens_but_no_content(tmp_path: Path) -> None:
    recorder = TraceRecorder()
    root = tmp_path / "workspace"
    root.mkdir()
    registry = ToolRegistry()
    registry.register(FileWriteTool(Workspace(root)))
    with trace_session(recorder):
        state = AgentState.for_request(
            "USER_PRIVATE_VALUE", ChatMessage(role="system", content="system")
        )
        with agent_span("coder"):
            await TracedLLM(FakeBackend()).chat(state.messages)
            result = await registry.execute(
                "file_write", {"path": "note.txt", "content": "TOOL_PRIVATE_VALUE"},
                {"file_write"},
            )
    assert result.success
    assert state.task_id == recorder.task_id
    assert [event.event for event in recorder.events] == [
        "agent_enter", "model_call", "tool_call", "agent_exit"
    ]
    assert all(event.task_id == recorder.task_id for event in recorder.events)
    assert all(event.trace_id == recorder.trace_id for event in recorder.events)
    assert [event.sequence for event in recorder.events] == [1, 2, 3, 4]
    model_event = recorder.events[1]
    assert model_event.prompt_tokens == 12
    assert model_event.completion_tokens == 3
    assert model_event.duration_ms is not None
    assert recorder.events[2].tool == "file_write"
    dumped = "\n".join(event.model_dump_json() for event in recorder.events)
    for secret in ("USER_PRIVATE_VALUE", "MODEL_PRIVATE_VALUE", "TOOL_PRIVATE_VALUE"):
        assert secret not in dumped


@pytest.mark.asyncio
async def test_trace_records_tool_failure_without_arguments(tmp_path: Path) -> None:
    recorder = TraceRecorder()
    registry = ToolRegistry()
    registry.register(FileWriteTool(Workspace(tmp_path)))
    with trace_session(recorder):
        result = await registry.execute(
            "file_write", {"path": "outside.txt", "content": "PRIVATE"}, set()
        )
    assert result.error_type == "PermissionDenied"
    assert recorder.events[0].success is False
    assert recorder.events[0].error_type == "PermissionDenied"
    assert "PRIVATE" not in recorder.events[0].model_dump_json()


def test_trace_event_count_is_bounded() -> None:
    recorder = TraceRecorder(max_events=2)
    for _ in range(4):
        recorder.record("task_started")
    assert len(recorder.events) == 2
    assert recorder.dropped == 2
