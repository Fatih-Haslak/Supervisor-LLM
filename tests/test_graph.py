"""The LangGraph path delegates and reviews without replacing the custom loop."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from app.agents.reviewer import ReviewerAgent
from app.agents.supervisor import SupervisorLimitError, WorkerResult
from app.llm.schemas import ChatMessage, LLMResponse
from app.orchestration.graph import GraphOrchestrator
from app.orchestration.state import AgentState, ToolCallRecord
from app.tools.filesystem import FileReadTool, FileWriteTool, Workspace
from app.tools.registry import ToolRegistry


class ScriptedLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)

    async def chat(
        self, messages: Sequence[ChatMessage], *, json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        assert json_schema is not None
        return LLMResponse(content=next(self.responses), model="scripted")


class FakeWorker:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run(self, task: str, state: AgentState) -> WorkerResult:
        self.calls.append(task)
        assert task in state.pending_tasks
        return WorkerResult(answer="47")


@pytest.mark.asyncio
async def test_graph_supervisor_worker_final() -> None:
    llm = ScriptedLLM([
        '{"action":"delegate","next_agent":"general","task":"3+44",'
        '"reason":"hesap"}',
        '{"action":"final_answer","answer":"47"}',
    ])
    worker = FakeWorker()
    state = await GraphOrchestrator(llm, {"general": worker}).run("3+44 kaç?")
    assert state.final_answer == "47"
    assert state.step_count == 2
    assert state.completed_tasks == ["3+44"]
    assert state.pending_tasks == []
    assert [item.agent for item in state.agent_outputs] == ["general"]
    assert worker.calls == ["3+44"]


@pytest.mark.asyncio
async def test_graph_honors_supervisor_round_limit() -> None:
    llm = ScriptedLLM([
        '{"action":"delegate","next_agent":"general","task":"x",'
        '"reason":"needed"}',
    ])
    with pytest.raises(SupervisorLimitError):
        await GraphOrchestrator(llm, {"general": FakeWorker()}, max_rounds=1).run("x")


class WritingWorker:
    def __init__(self, registry: ToolRegistry, *, repair: bool) -> None:
        self.registry = registry
        self.repair = repair
        self.calls: list[str] = []

    async def run(self, task: str, state: AgentState) -> WorkerResult:
        self.calls.append(task)
        valid = self.repair and len(self.calls) > 1
        arguments: dict[str, object] = {
            "path": "module.py",
            "content": "def add(a, b): return a + b\n" if valid else "def add(:\n",
            "overwrite": len(self.calls) > 1,
        }
        tool_result = await self.registry.execute("file_write", arguments, {"file_write"})
        return WorkerResult(
            answer="module.py yazıldı",
            tool_results=[
                ToolCallRecord(tool="file_write", arguments=arguments, result=tool_result)
            ],
        )


def make_registry(root: Path) -> ToolRegistry:
    root.mkdir()
    workspace = Workspace(root)
    registry = ToolRegistry()
    registry.register(FileReadTool(workspace))
    registry.register(FileWriteTool(workspace))
    return registry


@pytest.mark.asyncio
async def test_graph_reviewer_retries_coder_then_passes(tmp_path: Path) -> None:
    registry = make_registry(tmp_path / "workspace")
    llm = ScriptedLLM([
        '{"action":"delegate","next_agent":"coder","task":"module.py yaz",'
        '"reason":"kod"}',
        '{"status":"pass","issues":[]}',
        '{"action":"final_answer","answer":"Düzeltildi"}',
    ])
    coder = WritingWorker(registry, repair=True)
    state = await GraphOrchestrator(
        llm, {"coder": coder}, reviewer=ReviewerAgent(llm, registry)
    ).run("module.py dosyasına add yaz")
    assert state.final_answer == "Düzeltildi"
    assert [review.status for review in state.reviews] == ["fail", "pass"]
    assert len(coder.calls) == 2
    assert "Reviewer feedback" in coder.calls[1]
    assert state.completed_tasks == ["module.py yaz"]
    assert state.pending_tasks == []


@pytest.mark.asyncio
async def test_graph_stops_after_failed_review_limit(tmp_path: Path) -> None:
    registry = make_registry(tmp_path / "workspace")
    llm = ScriptedLLM([
        '{"action":"delegate","next_agent":"coder","task":"module.py yaz",'
        '"reason":"kod"}',
    ])
    coder = WritingWorker(registry, repair=False)
    state = await GraphOrchestrator(
        llm, {"coder": coder}, reviewer=ReviewerAgent(llm, registry),
        max_review_retries=1,
    ).run("module.py dosyasına add yaz")
    assert [review.status for review in state.reviews] == ["fail", "fail"]
    assert len(coder.calls) == 2
    assert state.pending_tasks == ["module.py yaz"]
    assert state.completed_tasks == []
    assert state.final_answer is not None and "tamamlanmadı" in state.final_answer
