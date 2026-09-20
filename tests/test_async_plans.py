import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from app.agents.supervisor import Supervisor, WorkerResult
from app.llm.schemas import ChatMessage, LLMResponse
from app.orchestration.state import AgentState, TaskPlan


class FinalModel:
    async def chat(
        self, _messages: Sequence[ChatMessage], *, json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content='{"action":"final_answer","answer":"Bitti"}', model="test"
        )


@pytest.mark.asyncio
async def test_independent_research_runs_in_parallel_then_dependent_worker_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = TaskPlan.model_validate({"tasks": [
        {"id": 1, "agent": "researcher", "task": "A"},
        {"id": 2, "agent": "researcher", "task": "B"},
        {"id": 3, "agent": "researcher", "task": "D"},
        {"id": 4, "agent": "general", "task": "C", "depends_on": [1, 2, 3]},
    ]})

    async def fake_plan(*_args: Any) -> TaskPlan:
        return plan

    monkeypatch.setattr("app.agents.supervisor.Planner.plan", fake_plan)
    both_started = asyncio.Event()
    active = 0
    peak = 0
    seen: list[list[str]] = []

    class Researcher:
        async def run(self, task: str, state: AgentState) -> WorkerResult:
            nonlocal active, peak
            assert not state.agent_outputs
            active += 1
            peak = max(peak, active)
            if active == 2:
                both_started.set()
            await both_started.wait()
            active -= 1
            return WorkerResult(answer=task)

    class General:
        async def run(self, _task: str, state: AgentState) -> WorkerResult:
            seen.append([output.answer for output in state.agent_outputs])
            return WorkerResult(answer="C")

    supervisor = Supervisor(
        FinalModel(), {"researcher": Researcher(), "general": General()}
    )
    state = await asyncio.wait_for(supervisor.run_planned("A ve B araştır, sonra C"), 2)
    assert peak == 2
    assert [output.answer for output in state.agent_outputs] == ["A", "B", "D", "C"]
    assert seen == [["A", "B", "D"]]
    assert state.final_answer == "Bitti"
