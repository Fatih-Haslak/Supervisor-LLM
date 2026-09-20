"""A multi-step local task crosses planner, workers, tools, reviewer, and final answer."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from app.agents.reviewer import ReviewerAgent
from app.agents.supervisor import Supervisor
from app.agents.workers import build_workers
from app.llm.schemas import ChatMessage, LLMResponse
from app.tools.calculator import CalculatorTool
from app.tools.filesystem import FileReadTool, FileWriteTool, Workspace
from app.tools.registry import ToolRegistry
from tests.support import ApproveWrites


class ScriptedLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)

    async def chat(
        self, messages: Sequence[ChatMessage], *, json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        assert json_schema is not None
        return LLMResponse(content=next(self.responses), model="fake")


@pytest.mark.asyncio
async def test_planned_report_is_written_and_reviewed(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "sales.csv").write_text("amount\n100\n200\n", encoding="utf-8")
    registry = ToolRegistry(approver=ApproveWrites())
    workspace = Workspace(root)
    registry.register(FileReadTool(workspace))
    registry.register(FileWriteTool(workspace))
    registry.register(CalculatorTool())
    llm = ScriptedLLM([
        '{"tasks":['
        '{"id":1,"agent":"researcher","task":"sales.csv oku","depends_on":[]},'
        '{"id":2,"agent":"general","task":"100+200 hesapla","depends_on":[1]},'
        '{"id":3,"agent":"coder","task":"summary.md raporunu yaz",'
        '"depends_on":[2]}]}',
        '{"action":"use_tool","tool":"file_read",'
        '"arguments":{"path":"sales.csv"}}',
        '{"action":"final_answer","answer":"sales.csv tutarları 100 ve 200"}',
        '{"action":"use_tool","tool":"calculator",'
        '"arguments":{"expression":"100+200"}}',
        '{"action":"final_answer","answer":"Toplam 300"}',
        '{"action":"use_tool","tool":"file_write",'
        '"arguments":{"path":"summary.md","content":"Toplam: 300"}}',
        '{"action":"final_answer","answer":"summary.md yazıldı"}',
        '{"status":"pass","issues":[]}',
        '{"action":"final_answer","answer":"Rapor yazıldı ve kontrol edildi"}',
    ])
    supervisor = Supervisor(
        llm, build_workers(llm, registry), reviewer=ReviewerAgent(llm, registry)
    )
    state = await supervisor.run_planned("sales.csv tutarlarını topla, rapor yaz ve kontrol et")
    assert state.final_answer == "Rapor yazıldı ve kontrol edildi"
    assert (root / "summary.md").read_text(encoding="utf-8") == "Toplam: 300"
    tool_names = [call.tool for call in state.tool_results]
    assert tool_names[:3] == ["file_read", "calculator", "file_write"]
    assert all(name == "file_read" for name in tool_names[3:])
    assert [review.status for review in state.reviews] == ["pass"]
    assert state.pending_tasks == []
    assert len(state.completed_tasks) == 3
