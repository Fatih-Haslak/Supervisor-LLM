"""Worker prompts receive relevant bounded context while task state stays complete."""

from typing import Any

import pytest

from app.llm.client import LLMContextOverflowError
from app.llm.schemas import ChatMessage, LLMResponse
from app.memory.context import MemoryAwareLLM
from app.memory.store import MemoryEntry
from app.orchestration.context import bounded_messages, worker_assignment
from app.orchestration.state import AgentOutput, AgentState, PlannedTask, TaskPlan


def test_long_history_keeps_request_and_recent_tool_result() -> None:
    messages = [
        ChatMessage(role="system", content="system"),
        ChatMessage(role="user", content="original task"),
        *[ChatMessage(role="user", content=f"old-{i}:" + "x" * 900) for i in range(15)],
        ChatMessage(role="user", content="latest tool result: 47"),
    ]
    reduced = bounded_messages(messages, max_chars=2000)
    assert reduced[0].content == "system"
    assert reduced[1].content == "original task"
    assert reduced[-1].content == "latest tool result: 47"
    assert sum(len(message.content) for message in reduced) <= 2000
    assert len(messages) == 18


def test_oversized_original_request_fails_instead_of_silent_truncation() -> None:
    messages = [
        ChatMessage(role="system", content="system"),
        ChatMessage(role="user", content="x" * 2000),
        ChatMessage(role="user", content="later"),
    ]
    with pytest.raises(LLMContextOverflowError):
        bounded_messages(messages, max_chars=1000)


def test_planned_worker_gets_only_dependency_outputs() -> None:
    state = AgentState(user_request="Read A, ignore B, then write C")
    state.plan = TaskPlan(tasks=[
        PlannedTask(id=1, agent="researcher", task="read A"),
        PlannedTask(id=2, agent="researcher", task="read B"),
        PlannedTask(id=3, agent="coder", task="write C", depends_on=[1]),
    ])
    state.agent_outputs = [
        AgentOutput(agent="researcher", task="read A", answer="A_EVIDENCE", planned_id=1),
        AgentOutput(agent="researcher", task="read B", answer="B_UNRELATED", planned_id=2),
    ]
    prompt = worker_assignment(state, "write C", "coder")
    assert "A_EVIDENCE" in prompt
    assert "B_UNRELATED" not in prompt
    assert "Read A, ignore B, then write C" in prompt


class CapturingLLM:
    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    async def chat(
        self, messages: list[ChatMessage], *, json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.messages = list(messages)
        return LLMResponse(content="ok", model="fake")


@pytest.mark.asyncio
async def test_long_term_memory_has_character_budget() -> None:
    entries = [
        MemoryEntry(
            key=f"note_{index}", value="x" * 500,
            category="fact", updated_at="2026-01-01 00:00:00",
        )
        for index in range(20)
    ]
    backend = CapturingLLM()
    await MemoryAwareLLM(backend, entries).chat([ChatMessage(role="user", content="Hi")])
    system = backend.messages[0].content
    assert "note_0" in system
    assert "note_19" not in system
    assert len(system) < 2300
