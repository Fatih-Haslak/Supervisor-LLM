from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from app.agents.single import RepeatedToolError, SingleAgent
from app.llm.schemas import ChatMessage, LLMResponse
from app.orchestration.state import AgentState
from app.security.approvals import Approver
from app.service.tasks import TaskManager, TaskMode
from app.tools.filesystem import FileReadTool, Workspace
from app.tools.registry import ToolRegistry


class ScriptedLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            content='{"action":"use_tool","tool":"file_read",'
            '"arguments":{"path":"C:/outside.txt"}}',
            model="scripted",
        )


@pytest.mark.asyncio
async def test_repeated_permission_denied_stops_and_preserves_calls(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    registry = ToolRegistry()
    registry.register(FileReadTool(Workspace(root)))
    llm = ScriptedLLM()
    agent = SingleAgent(llm, registry, {"file_read"})

    with pytest.raises(RepeatedToolError) as caught:
        await agent.run("Read a file")

    assert caught.value.error_type == "PermissionDenied"
    assert llm.calls == 2
    assert [call.result.error_type for call in caught.value.state.tool_results] == [
        "PermissionDenied", "PermissionDenied"
    ]


@pytest.mark.asyncio
async def test_failed_task_snapshot_keeps_partial_tool_calls(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    registry = ToolRegistry()
    registry.register(FileReadTool(Workspace(root)))
    agent = SingleAgent(ScriptedLLM(), registry, {"file_read"})

    async def run(
        _message: str, _mode: TaskMode, _approver: Approver,
        _history: list[ChatMessage],
    ) -> AgentState:
        return (await agent.run("Read a file")).state

    manager = TaskManager(run)
    await manager.start()
    try:
        task = manager.submit("Read a file", "single")
        await manager._queue.join()
        view = manager.get(task.task_id)
        assert view is not None
        assert view.status == "failed"
        assert view.error is not None and view.error.code == "TOOL_FAILURE"
        assert len(view.tool_calls) == 2
        assert view.tool_calls[0].arguments["path"] == "C:/outside.txt"
    finally:
        await manager.close()
