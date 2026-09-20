from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from app.agents.reviewer import ReviewerAgent
from app.agents.supervisor import Supervisor, WorkerResult
from app.llm.schemas import ChatMessage, LLMResponse
from app.orchestration.state import AgentState, ToolCallRecord
from app.tools.base import ToolResult
from app.tools.filesystem import FileReadTool, FileWriteTool, Workspace
from app.tools.registry import ToolRegistry


class ScriptedLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.calls = 0

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        assert json_schema is not None
        self.calls += 1
        return LLMResponse(content=next(self._responses), model="scripted")


def make_registry(root: Path) -> ToolRegistry:
    workspace = Workspace(root)
    registry = ToolRegistry()
    registry.register(FileReadTool(workspace))
    registry.register(FileWriteTool(workspace))
    return registry


class WritingWorker:
    def __init__(self, registry: ToolRegistry, *, repair: bool) -> None:
        self.registry = registry
        self.repair = repair
        self.calls = 0

    async def run(self, task: str, state: AgentState) -> WorkerResult:
        self.calls += 1
        assert state.current_agent == "coder"
        code = (
            "def add(a, b):\n    return a + b\n"
            if self.repair and self.calls > 1
            else "def add(:\n    pass\n"
        )
        arguments: dict[str, object] = {
            "path": "module.py", "content": code, "overwrite": self.calls > 1,
        }
        tool_result = await self.registry.execute("file_write", arguments, {"file_write"})
        return WorkerResult(
            answer="module.py güncellendi",
            tool_results=[
                ToolCallRecord(tool="file_write", arguments=arguments, result=tool_result)
            ],
        )


class NeverCalledWorker:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, task: str, state: AgentState) -> WorkerResult:
        self.calls += 1
        return WorkerResult(answer="Should not run")


@pytest.mark.asyncio
async def test_reviewer_detects_syntax_error_without_running_code(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "module.py").write_text("def add(:\n    pass\n", encoding="utf-8")
    llm = ScriptedLLM([])
    result = await ReviewerAgent(llm, make_registry(root)).review(
        "Toplama fonksiyonu yaz", "module.py oluştur",
        WorkerResult(
            answer="Bitti",
            tool_results=[
                ToolCallRecord(
                    tool="file_write", arguments={"path": "module.py"},
                    result=ToolResult.ok("Wrote module.py"),
                )
            ],
        ),
    )
    assert result.verdict.status == "fail"
    assert "syntax error" in result.verdict.issues[0]
    assert [call.tool for call in result.tool_calls] == ["file_read"]
    assert llm.calls == 0
    assert (root / "module.py").read_text(encoding="utf-8").startswith("def add(:")


@pytest.mark.asyncio
async def test_reviewer_requires_file_evidence(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    llm = ScriptedLLM([])
    result = await ReviewerAgent(llm, make_registry(root)).review(
        "Kod yaz", "module.py oluştur", WorkerResult(answer="Yazıldı")
    )
    assert result.verdict.status == "fail"
    assert "No workspace file evidence" in result.verdict.issues[0]
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_reviewer_can_read_file_written_by_earlier_worker(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "module.py").write_text("def add(a, b): return a + b\n", encoding="utf-8")
    llm = ScriptedLLM(['{"status":"pass","issues":[]}'])
    earlier_write = ToolCallRecord(
        tool="file_write", arguments={"path": "module.py"},
        result=ToolResult.ok("Wrote module.py"),
    )
    result = await ReviewerAgent(llm, make_registry(root)).review(
        "Kod yaz", "module.py içeriğini kontrol et", WorkerResult(answer="Kontrol edildi"),
        evidence_tools=[earlier_write],
    )
    assert result.verdict.status == "pass"
    assert [call.tool for call in result.tool_calls] == ["file_read"]
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_supervisor_retries_coder_after_review_failure(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    registry = make_registry(root)
    llm = ScriptedLLM(
        [
            '{"tasks":[{"id":1,"agent":"coder","task":"module.py oluştur",'
            '"depends_on":[]}]}',
            '{"status":"pass","issues":[]}',
            '{"action":"final_answer","answer":"Kod düzeltildi ve incelemeden geçti"}',
        ]
    )
    worker = WritingWorker(registry, repair=True)
    state = await Supervisor(
        llm, {"coder": worker}, reviewer=ReviewerAgent(llm, registry)
    ).run_planned("module.py dosyasına add fonksiyonu yaz")

    assert worker.calls == 2
    assert (root / "module.py").read_text(encoding="utf-8").startswith("def add(a, b):")
    assert [review.status for review in state.reviews] == ["fail", "pass"]
    assert state.completed_tasks == ["module.py oluştur"]
    assert state.pending_tasks == []
    assert state.final_answer == "Kod düzeltildi ve incelemeden geçti"


@pytest.mark.asyncio
async def test_review_retries_stop_after_two_failed_revisions(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    registry = make_registry(root)
    llm = ScriptedLLM(
        ['{"tasks":[{"id":1,"agent":"coder","task":"module.py oluştur",'
         '"depends_on":[]},{"id":2,"agent":"file_agent",'
         '"task":"rapor yaz","depends_on":[1]}]}']
    )
    worker = WritingWorker(registry, repair=False)
    dependent_worker = NeverCalledWorker()
    state = await Supervisor(
        llm, {"coder": worker, "file_agent": dependent_worker},
        reviewer=ReviewerAgent(llm, registry),
        max_review_retries=2,
    ).run_planned("module.py dosyasına add fonksiyonu yaz")

    assert worker.calls == 3
    assert dependent_worker.calls == 0
    assert [review.attempt for review in state.reviews] == [1, 2, 3]
    assert all(review.status == "fail" for review in state.reviews)
    assert state.completed_tasks == []
    assert state.pending_tasks == ["module.py oluştur", "rapor yaz"]
    assert state.final_answer is not None and "tamamlanmadı" in state.final_answer
