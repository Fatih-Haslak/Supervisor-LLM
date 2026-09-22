from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.agents.supervisor import Supervisor
from app.agents.workers import build_workers, worker_descriptions
from app.llm.schemas import ChatMessage, LLMResponse
from app.llm.structured import StructuredOutputError
from app.orchestration.planner import Planner
from app.orchestration.state import TaskPlan
from app.tools.filesystem import FileReadTool, FileWriteTool, Workspace
from app.tools.registry import ToolRegistry
from app.tools.search import SearchTool
from tests.support import ApproveWrites


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


def test_plan_rejects_duplicate_ids_and_forward_dependencies() -> None:
    with pytest.raises(ValidationError, match="unique"):
        TaskPlan.model_validate(
            {"tasks": [
                {"id": 1, "agent": "general", "task": "a"},
                {"id": 1, "agent": "general", "task": "b"},
            ]}
        )
    with pytest.raises(ValidationError, match="earlier"):
        TaskPlan.model_validate(
            {"tasks": [
                {"id": 1, "agent": "general", "task": "a", "depends_on": [2]},
                {"id": 2, "agent": "general", "task": "b"},
            ]}
        )


@pytest.mark.asyncio
async def test_planner_retries_unavailable_agent() -> None:
    llm = ScriptedLLM(
        [
            '{"tasks":[{"id":1,"agent":"admin","task":"x","depends_on":[]}]}',
            '{"tasks":[{"id":1,"agent":"general","task":"x","depends_on":[]}]}',
        ]
    )
    plan = await Planner(llm, {"general"}, max_retries=1).plan("x")
    assert plan.tasks[0].agent == "general"
    assert llm.schemas[0]["$defs"]["PlannedTask"]["properties"]["agent"]["enum"] == [
        "general"
    ]
    assert "Invalid plan" in llm.requests[1][-1].content


@pytest.mark.asyncio
async def test_planner_rejects_repeated_invalid_plan() -> None:
    llm = ScriptedLLM(["{}", "{}"])
    with pytest.raises(StructuredOutputError, match="invalid plan"):
        await Planner(llm, {"general"}, max_retries=1).plan("x")


@pytest.mark.asyncio
async def test_auto_review_drops_redundant_tail_tasks() -> None:
    llm = ScriptedLLM([
        '{"tasks":['
        '{"id":1,"agent":"coder","task":"Fix code and run tests","depends_on":[]},'
        '{"id":2,"agent":"file_agent","task":"Run function_test",'
        '"depends_on":[1]},'
        '{"id":3,"agent":"researcher","task":"Review code changes",'
        '"depends_on":[2]}]}'
    ])
    plan = await Planner(
        llm, {"coder", "file_agent", "researcher"}, auto_review=True
    ).plan("Fix code, test, and review")
    assert [task.agent for task in plan.tasks] == ["coder"]


@pytest.mark.asyncio
async def test_planned_supervisor_passes_first_result_to_second_worker(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "source.txt").write_text("Proje kodu: MAVİ", encoding="utf-8")
    workspace = Workspace(root)
    registry = ToolRegistry(approver=ApproveWrites())
    registry.register(SearchTool(workspace))
    registry.register(FileReadTool(workspace))
    registry.register(FileWriteTool(workspace))
    llm = ScriptedLLM(
        [
            '{"tasks":['
            '{"id":1,"agent":"researcher","task":"source.txt içinde proje kodunu bul",'
            '"depends_on":[]},'
            '{"id":2,"agent":"file_agent","task":"Bulunan kodu report.txt dosyasına yaz",'
            '"depends_on":[1]}]}',
            '{"action":"use_tool","tool":"search",'
            '"arguments":{"query":"Proje kodu"}}',
            '{"action":"final_answer","answer":"source.txt: Proje kodu MAVİ"}',
            '{"action":"use_tool","tool":"file_read",'
            '"arguments":{"path":"source.txt"}}',
            '{"action":"final_answer","answer":"source.txt: Proje kodu MAVİ"}',
            '{"action":"use_tool","tool":"file_write",'
            '"arguments":{"path":"report.txt","content":"MAVİ"}}',
            '{"action":"final_answer","answer":"report.txt yazıldı"}',
            '{"action":"final_answer","answer":"Kod MAVİ; rapor yazıldı."}',
        ]
    )
    state = await Supervisor(
        llm,
        build_workers(llm, registry),
        worker_descriptions=worker_descriptions(),
    ).run_planned("source.txt dosyasındaki proje kodunu report.txt içine yaz")

    assert (root / "report.txt").read_text(encoding="utf-8") == "MAVİ"
    assert state.plan is not None and len(state.plan.tasks) == 2
    assert state.completed_tasks == [
        "source.txt içinde proje kodunu bul", "Bulunan kodu report.txt dosyasına yaz"
    ]
    assert state.pending_tasks == []
    assert [output.agent for output in state.agent_outputs] == ["researcher", "file_agent"]
    assert [call.tool for call in state.tool_results] == ["search", "file_read", "file_write"]
    assert "MAVİ" in llm.requests[5][-1].content
    assert state.final_answer == "Kod MAVİ; rapor yazıldı."
