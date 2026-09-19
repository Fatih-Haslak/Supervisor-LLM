from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from app.agents.supervisor import SingleAgentWorker, Supervisor
from app.agents.workers import build_workers, worker_descriptions
from app.llm.schemas import ChatMessage, LLMResponse
from app.orchestration.state import AgentState
from app.tools.calculator import CalculatorTool
from app.tools.filesystem import DirectoryListTool, FileReadTool, FileWriteTool, Workspace
from app.tools.registry import ToolRegistry
from app.tools.search import SearchTool


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


def make_registry(root: Path) -> ToolRegistry:
    workspace = Workspace(root)
    registry = ToolRegistry()
    for tool in (
        CalculatorTool(), FileReadTool(workspace), FileWriteTool(workspace),
        DirectoryListTool(workspace), SearchTool(workspace),
    ):
        registry.register(tool)
    return registry


@pytest.mark.asyncio
async def test_researcher_cannot_write_even_if_model_requests_it(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    llm = ScriptedLLM(
        [
            '{"action":"use_tool","tool":"file_write",'
            '"arguments":{"path":"forbidden.txt","content":"bad"}}',
            '{"action":"final_answer","answer":"Yazma iznim yok"}',
        ]
    )
    worker = build_workers(llm, make_registry(root))["researcher"]
    result = await worker.run("Dosya yaz", AgentState(user_request="Dosya yaz"))

    assert not (root / "forbidden.txt").exists()
    assert result.tool_results[0].result.error_type == "PermissionDenied"
    assert '"name": "search"' in llm.requests[0][1].content
    assert '"name": "file_write"' not in llm.requests[0][1].content


@pytest.mark.asyncio
async def test_supervisor_routes_to_file_agent_and_collects_file_result(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    llm = ScriptedLLM(
        [
            '{"action":"delegate","next_agent":"file_agent",'
            '"task":"note.txt dosyasına Merhaba yaz","reason":"Dosya görevi"}',
            '{"action":"use_tool","tool":"file_write",'
            '"arguments":{"path":"note.txt","content":"Merhaba"}}',
            '{"action":"final_answer","answer":"note.txt yazıldı"}',
            '{"action":"final_answer","answer":"note.txt dosyası oluşturuldu"}',
        ]
    )
    workers = build_workers(llm, make_registry(root))
    state = await Supervisor(
        llm, workers, worker_descriptions=worker_descriptions()
    ).run("note.txt dosyası oluştur ve Merhaba yaz")

    assert (root / "note.txt").read_text(encoding="utf-8") == "Merhaba"
    assert state.final_answer == "note.txt dosyası oluşturuldu"
    assert state.agent_outputs[0].agent == "file_agent"
    assert state.tool_results[0].tool == "file_write"
    assert state.tool_results[0].result.success
    assert llm.schemas[0]["$defs"]["DelegateDecision"]["properties"]["next_agent"]["enum"] == [
        "coder", "file_agent", "general", "researcher"
    ]


@pytest.mark.asyncio
async def test_worker_keeps_original_request(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    llm = ScriptedLLM(['{"action":"final_answer","answer":"Bulunamadı"}'])
    worker = build_workers(llm, make_registry(root))["researcher"]
    assert isinstance(worker, SingleAgentWorker)
    state = AgentState(user_request="Fatih Tekke adını belgelerde ara")
    await worker.run("Yerel belgelerde araştır", state)
    assert "Fatih Tekke" in llm.requests[0][-1].content
    assert "Yerel belgelerde araştır" in llm.requests[0][-1].content
    assert "Perform only the assigned subtask" in llm.requests[0][-1].content


@pytest.mark.asyncio
async def test_coder_can_write_but_cannot_run_python_by_default(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    llm = ScriptedLLM(
        [
            '{"action":"use_tool","tool":"file_write",'
            '"arguments":{"path":"module.py","content":"def add(a, b): return a + b"}}',
            '{"action":"use_tool","tool":"python_exec",'
            '"arguments":{"code":"print(1)"}}',
            '{"action":"final_answer","answer":"Kod yazıldı"}',
        ]
    )
    worker = build_workers(llm, make_registry(root))["coder"]
    result = await worker.run("Toplama fonksiyonu yaz", AgentState(user_request="Kod yaz"))
    assert (root / "module.py").is_file()
    assert result.tool_results[0].result.success
    assert result.tool_results[1].result.error_type == "PermissionDenied"
    assert '"name": "python_exec"' not in llm.requests[0][1].content
