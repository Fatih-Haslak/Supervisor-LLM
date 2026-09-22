from collections.abc import Sequence
from typing import Any

import pytest

from app.agents.supervisor import Supervisor, SupervisorLimitError, WorkerResult
from app.llm.schemas import ChatMessage, LLMResponse
from app.llm.structured import StructuredOutputError
from app.observability.events import TraceRecorder, trace_session
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


class ResearchWorker:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run(self, task: str, state: AgentState) -> WorkerResult:
        assert state.current_agent == "researcher"
        self.calls.append(task)
        return WorkerResult(answer="Orion-17 brief.txt dosyasında")


class CoderWorker:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run(self, task: str, state: AgentState) -> WorkerResult:
        assert state.current_agent == "coder"
        self.calls.append(task)
        return WorkerResult(answer="Kod Fibonacci dizisini özyinelemeli hesaplıyor.")


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


@pytest.mark.asyncio
async def test_planned_mode_falls_back_to_supervisor_if_model_plan_is_invalid() -> None:
    planner = ScriptedLLM(['{"tasks":[]}'] * 3)
    llm = ScriptedLLM([
        '{"action":"delegate","next_agent":"general","task":"3+44 hesapla",'
        '"reason":"Hesap gerekli"}',
        '{"action":"final_answer","answer":"Sonuç 47"}',
    ])
    recorder = TraceRecorder()
    with trace_session(recorder):
        state = await Supervisor(
            llm, {"general": FakeWorker()}, planner_llm=planner
        ).run_planned("3+44 hesapla")
    assert state.final_answer == "Sonuç 47"
    assert state.plan is None
    assert any(event.event == "plan_fallback" for event in recorder.events)


@pytest.mark.asyncio
async def test_supervisor_retries_english_final_for_turkish_request() -> None:
    llm = ScriptedLLM([
        '{"action":"delegate","next_agent":"general","task":"3+44 hesapla",'
        '"reason":"Hesap gerekli"}',
        '{"action":"final_answer","answer":"The calculation has been completed. 47"}',
        '{"action":"final_answer","answer":"Hesaplama tamamlandı: 47."}',
    ])
    state = await Supervisor(llm, {"general": FakeWorker()}).run("3+44 işlemini hesapla")
    assert state.final_answer == "Hesaplama tamamlandı: 47."
    assert "tamamen doğal Türkçe" in llm.requests[2][-1].content


@pytest.mark.asyncio
async def test_local_document_search_is_delegated_to_researcher() -> None:
    llm = ScriptedLLM([
        '{"action":"delegate","next_agent":"file_agent",'
        '"task":"workspace belgelerinde Orion kodunu ara",'
        '"reason":"dosya işi"}',
        '{"action":"final_answer","answer":"Orion-17 brief.txt dosyasında"}',
    ])
    researcher = ResearchWorker()
    state = await Supervisor(llm, {
        "file_agent": FakeWorker(), "researcher": researcher
    }).run("workspace belgelerinde Orion kodunu ara")
    assert researcher.calls == ["workspace belgelerinde Orion kodunu ara"]
    assert state.agent_outputs[0].agent == "researcher"


@pytest.mark.asyncio
async def test_inline_python_analysis_is_delegated_to_coder() -> None:
    llm = ScriptedLLM([
        '{"action":"delegate","next_agent":"general",'
        '"task":"workspace/python/fibonacci.py dosyasını incele",'
        '"reason":"genel açıklama"}',
        '{"action":"final_answer","answer":"Kod Fibonacci dizisini özyinelemeli hesaplıyor."}',
    ])
    coder = CoderWorker()
    state = await Supervisor(llm, {
        "general": FakeWorker(), "coder": coder,
    }).run("```python\ndef fibonacci(n):\n    return n\n```\nbu kodu analiz et")

    assert coder.calls == [
        "Kullanıcının mesajında verdiği satır içi Python kodunu doğrudan analiz et. "
        "Çalışma alanında dosya arama, dosya oluşturma veya araç kullanma. Kodun "
        "davranışını, örnek çıktısını, performansını ve olası sorunlarını Türkçe açıkla."
    ]
    assert state.agent_outputs[0].agent == "coder"
