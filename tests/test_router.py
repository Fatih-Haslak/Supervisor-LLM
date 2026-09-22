from collections.abc import Sequence
from typing import Any

import pytest

from app.agents.supervisor import WorkerResult
from app.llm.schemas import ChatMessage, LLMResponse
from app.llm.structured import StructuredOutputError
from app.orchestration.router import Router
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
        self.calls: list[str] = []

    async def run(self, task: str, state: AgentState) -> WorkerResult:
        self.calls.append(task)
        assert state.current_agent == "general"
        return WorkerResult(answer="47")


class FakeSupervisor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run_planned(
        self, user_request: str, *, history: Sequence[ChatMessage] = ()
    ) -> AgentState:
        self.calls.append(user_request)
        state = AgentState(user_request=user_request, current_agent="supervisor")
        if history:
            state.messages = list(history)
        state.finish("Planlı sonuç")
        return state


@pytest.mark.asyncio
async def test_high_confidence_route_runs_worker_without_supervisor() -> None:
    llm = ScriptedLLM(['{"agent":"general","confidence":0.95}'])
    worker, supervisor = FakeWorker(), FakeSupervisor()
    state = await Router(llm, {"general": worker}, supervisor).run("3+44 kaç?")

    assert worker.calls == ["3+44 kaç?"]
    assert supervisor.calls == []
    assert len(llm.requests) == 1
    assert state.final_answer == "47"
    assert state.completed_tasks == ["3+44 kaç?"]
    assert state.route is not None and state.route.selected_agent == "general"
    assert llm.schemas[0]["properties"]["agent"]["enum"] == ["general", "supervisor"]


@pytest.mark.asyncio
async def test_low_confidence_route_falls_back_to_planned_supervisor() -> None:
    llm = ScriptedLLM(['{"agent":"general","confidence":0.50}'])
    worker, supervisor = FakeWorker(), FakeSupervisor()
    state = await Router(llm, {"general": worker}, supervisor).run("Karmaşık görev")

    assert worker.calls == []
    assert supervisor.calls == ["Karmaşık görev"]
    assert state.route is not None
    assert state.route.suggested_agent == "general"
    assert state.route.selected_agent == "supervisor"


@pytest.mark.asyncio
async def test_explicit_supervisor_route_even_with_high_confidence() -> None:
    llm = ScriptedLLM(['{"agent":"supervisor","confidence":0.99}'])
    supervisor = FakeSupervisor()
    state = await Router(llm, {"general": FakeWorker()}, supervisor).run("İki aşama")
    assert supervisor.calls == ["İki aşama"]
    assert state.route is not None and state.route.selected_agent == "supervisor"


@pytest.mark.asyncio
async def test_explicit_sequence_uses_supervisor_despite_high_worker_confidence() -> None:
    llm = ScriptedLLM(['{"agent":"general","confidence":1.0}'])
    worker, supervisor = FakeWorker(), FakeSupervisor()
    state = await Router(llm, {"general": worker}, supervisor).run(
        "Önce dosyaları listele, sonra rapor oluştur"
    )
    assert worker.calls == []
    assert supervisor.calls == ["Önce dosyaları listele, sonra rapor oluştur"]
    assert state.route is not None and state.route.selected_agent == "supervisor"


@pytest.mark.asyncio
async def test_coder_route_uses_supervisor_when_review_is_enabled() -> None:
    llm = ScriptedLLM(['{"agent":"coder","confidence":1.0}'])
    supervisor = FakeSupervisor()
    state = await Router(
        llm, {"coder": FakeWorker()}, supervisor,
        review_code_with_supervisor=True,
    ).run("Python fonksiyonu yaz")
    assert supervisor.calls == ["Python fonksiyonu yaz"]
    assert state.route is not None and state.route.selected_agent == "supervisor"


@pytest.mark.asyncio
async def test_router_retries_invalid_agent_and_confidence() -> None:
    llm = ScriptedLLM(
        [
            '{"agent":"unknown","confidence":0.9}',
            '{"agent":"general","confidence":1.4}',
            '{"agent":"general","confidence":0.9}',
        ]
    )
    decision = await Router(llm, {"general": FakeWorker()}, FakeSupervisor()).decide("x")
    assert decision.agent == "general"
    assert len(llm.requests) == 3
    assert "Choose one listed worker" in llm.requests[1][-1].content


@pytest.mark.asyncio
async def test_router_has_bounded_invalid_output() -> None:
    llm = ScriptedLLM(["{}", "{}"])
    with pytest.raises(StructuredOutputError, match="invalid route"):
        await Router(llm, {"general": FakeWorker()}, FakeSupervisor(), max_retries=1).decide(
            "x"
        )


@pytest.mark.asyncio
async def test_invalid_route_falls_back_to_supervisor() -> None:
    llm = ScriptedLLM(["{}", "{}"])
    supervisor = FakeSupervisor()
    state = await Router(
        llm, {"general": FakeWorker()}, supervisor, max_retries=1
    ).run("Bir görev")
    assert supervisor.calls == ["Bir görev"]
    assert state.route is not None
    assert state.route.selected_agent == "supervisor"
    assert state.route.confidence == 0


@pytest.mark.asyncio
async def test_router_uses_supervisor_for_conversation_followup() -> None:
    llm = ScriptedLLM(['{"agent":"general","confidence":0.99}'])
    worker, supervisor = FakeWorker(), FakeSupervisor()
    history = [ChatMessage(role="user", content="Kod Orion-17")]
    state = await Router(llm, {"general": worker}, supervisor).run(
        "Az önceki kod neydi?", history=history
    )
    assert worker.calls == []
    assert state.route is not None and state.route.selected_agent == "supervisor"
    assert state.messages[0].content == "Kod Orion-17"
