from collections.abc import Sequence
from typing import Any

import pytest

from app.agents.supervisor import Supervisor, SupervisorLimitError, WorkerResult
from app.llm.schemas import ChatMessage, LLMResponse
from app.llm.structured import StructuredOutputError
from app.orchestration.state import AgentState


class ScriptedLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.requests: list[list[ChatMessage]] = []
        self.schemas: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        assert json_schema is not None
        self.requests.append(list(messages))
        self.schemas.append(json_schema)
        return LLMResponse(content=next(self._responses), model="scripted")


class FakeWorker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, AgentState]] = []

    async def run(self, task: str, state: AgentState) -> WorkerResult:
        self.calls.append((task, state))
        assert state.current_agent == "general"
        assert task in state.pending_tasks
        return WorkerResult(answer="Sonuç 47")


@pytest.mark.asyncio
async def test_supervisor_delegates_then_synthesizes() -> None:
    llm = ScriptedLLM(
        [
            '{"action":"delegate","next_agent":"general","task":"3+44 hesapla",'
            '"reason":"Hesap gerekli"}',
            '{"action":"final_answer","answer":"Sonuç 47"}',
        ]
    )
    worker = FakeWorker()
    state = await Supervisor(llm, {"general": worker}).run("3+44 kaç?")

    assert len(worker.calls) == 1
    assert worker.calls[0][1] is state
    assert state.final_answer == "Sonuç 47"
    assert state.current_agent is None
    assert state.step_count == 2
    assert state.completed_tasks == ["3+44 hesapla"]
    assert state.pending_tasks == []
    assert state.agent_outputs[0].agent == "general"
    assert "Worker result" in llm.requests[1][-1].content
    assert llm.schemas[0]["properties"]["action"]["const"] == "delegate"
    assert "oneOf" in llm.schemas[1]


@pytest.mark.asyncio
async def test_unknown_worker_is_retried_with_allowlist() -> None:
    llm = ScriptedLLM(
        [
            '{"action":"delegate","next_agent":"admin","task":"x","reason":"x"}',
            '{"action":"delegate","next_agent":"general","task":"x","reason":"x"}',
            '{"action":"final_answer","answer":"Bitti"}',
        ]
    )
    worker = FakeWorker()
    state = await Supervisor(llm, {"general": worker}).run("x")
    assert state.final_answer == "Bitti"
    assert len(worker.calls) == 1
    assert "Choose a listed worker" in llm.requests[1][-1].content


@pytest.mark.asyncio
async def test_supervisor_rejects_final_before_any_worker() -> None:
    llm = ScriptedLLM(['{"action":"final_answer","answer":"Hazır"}'] * 2)
    with pytest.raises(StructuredOutputError, match="invalid route"):
        await Supervisor(llm, {"general": FakeWorker()}, max_json_retries=1).run("x")


@pytest.mark.asyncio
async def test_supervisor_round_limit() -> None:
    llm = ScriptedLLM(
        ['{"action":"delegate","next_agent":"general","task":"x","reason":"x"}'] * 2
    )
    with pytest.raises(SupervisorLimitError, match="rounds"):
        await Supervisor(llm, {"general": FakeWorker()}, max_rounds=2).run("x")
    assert len(llm.requests) == 2
