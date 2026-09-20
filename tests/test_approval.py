"""Approval is explicit, scoped to a validated tool call, and fail closed."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from app.agents.single import SingleAgent
from app.llm.schemas import ChatMessage, LLMResponse
from app.security.approvals import ApprovalRequest, TerminalApprover, ToolApprovalError
from app.tools.filesystem import FileWriteTool, Workspace
from app.tools.registry import ToolRegistry


class SpyApprover:
    def __init__(self, approved: bool) -> None:
        self.approved = approved
        self.requests: list[ApprovalRequest] = []

    async def request_approval(self, request: ApprovalRequest) -> bool:
        self.requests.append(request)
        return self.approved


@pytest.mark.asyncio
async def test_write_requires_review_and_denial_preserves_file(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    path = root / "note.txt"
    registry = ToolRegistry()
    registry.register(FileWriteTool(Workspace(root)))
    assert registry.specs({"file_write"})[0].requires_approval
    required = await registry.execute(
        "file_write", {"path": "note.txt", "content": "draft"}, {"file_write"}
    )
    assert required.error_type == "ApprovalRequired"
    assert not path.exists()

    approver = SpyApprover(False)
    registry = ToolRegistry(approver=approver)
    registry.register(FileWriteTool(Workspace(root)))
    denied = await registry.execute(
        "file_write", {"path": "note.txt", "content": "draft"}, {"file_write"}
    )
    assert denied.error_type == "ApprovalDenied"
    assert approver.requests[0].arguments == {
        "path": "note.txt", "content": "draft", "overwrite": False
    }
    assert not path.exists()


@pytest.mark.asyncio
async def test_invalid_write_arguments_never_request_approval(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    approver = SpyApprover(True)
    registry = ToolRegistry(approver=approver)
    registry.register(FileWriteTool(Workspace(root)))
    result = await registry.execute(
        "file_write", {"path": "note.txt", "content": "draft", "extra": "x"},
        {"file_write"},
    )
    assert result.error_type == "InvalidArguments"
    assert approver.requests == []
    assert not (root / "note.txt").exists()
    outside = await registry.execute(
        "file_write", {"path": "../outside.txt", "content": "draft"}, {"file_write"}
    )
    assert outside.error_type == "PermissionDenied"
    assert approver.requests == []


class MutatingApprover:
    async def request_approval(self, request: ApprovalRequest) -> bool:
        request.arguments["content"] = "changed after approval"
        return True


@pytest.mark.asyncio
async def test_approver_cannot_change_executed_arguments(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    registry = ToolRegistry(approver=MutatingApprover())
    registry.register(FileWriteTool(Workspace(root)))
    result = await registry.execute(
        "file_write", {"path": "note.txt", "content": "original"}, {"file_write"}
    )
    assert result.success
    assert (root / "note.txt").read_text(encoding="utf-8") == "original"


class WriteDecisionLLM:
    async def chat(
        self, messages: Sequence[ChatMessage], *, json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            model="fake",
            content='{"action":"use_tool","tool":"file_write",'
            '"arguments":{"path":"note.txt","content":"draft"}}',
        )


@pytest.mark.asyncio
async def test_agent_stops_when_approval_is_missing(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    registry = ToolRegistry()
    registry.register(FileWriteTool(Workspace(root)))
    agent = SingleAgent(WriteDecisionLLM(), registry, {"file_write"})
    with pytest.raises(ToolApprovalError) as caught:
        await agent.run("note.txt yaz")
    assert caught.value.kind == "ApprovalRequired"
    assert not (root / "note.txt").exists()


@pytest.mark.asyncio
async def test_terminal_approver_requires_explicit_yes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FakeTTY:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("app.security.approvals.sys.stdin", FakeTTY())
    answers = iter(["hayır", "evet"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    request = ApprovalRequest(
        tool="file_write", arguments={"path": "note.txt", "content": "draft"}
    )
    approver = TerminalApprover()
    assert await approver.request_approval(request) is False
    assert await approver.request_approval(request) is True
    assert '"path": "note.txt"' in capsys.readouterr().out
