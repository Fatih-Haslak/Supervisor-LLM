from pathlib import Path

import pytest

from app.tools.calculator import CalculatorTool
from app.tools.filesystem import (
    DirectoryListTool,
    FileReadTool,
    FileWriteTool,
    Workspace,
)
from app.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_calculator_supports_four_operations_without_eval() -> None:
    tool = CalculatorTool()
    assert (await tool.run({"expression": "25*17"})).output == "425"
    assert (await tool.run({"expression": "(12+8)/4-1"})).output == "4"
    assert (await tool.run({"expression": "0.1+0.2"})).output == "0.3"
    for expression in ("__import__('os').system('whoami')", "2**8", "1/0"):
        result = await tool.run({"expression": expression})
        assert not result.success


@pytest.mark.asyncio
async def test_file_tools_stay_in_workspace_and_check_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    workspace = Workspace(root)
    writer, reader, listing = FileWriteTool(workspace), FileReadTool(workspace), DirectoryListTool(
        workspace
    )

    assert (await writer.run({"path": "note.txt", "content": "Merhaba"})).success
    assert (await reader.run({"path": "workspace/note.txt"})).output == "Merhaba"
    assert (await reader.run({"path": "note.txt"})).output == "Merhaba"
    assert "note.txt" in (await listing.run({})).output
    denied = await writer.run({"path": "note.txt", "content": "Sil"})
    assert denied.error_type == "AlreadyExists"
    assert (await reader.run({"path": "note.txt"})).output == "Merhaba"
    assert (await writer.run({"path": "note.txt", "content": "Yeni", "overwrite": True})).success
    assert (await reader.run({"path": "note.txt"})).output == "Yeni"

    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    for path in (
        "../outside.txt", "workspace/../outside.txt", str(outside),
        ".env", "workspace/.env", "C:\\Windows\\win.ini",
    ):
        result = await reader.run({"path": path})
        assert result.error_type == "PermissionDenied"
    assert not (await writer.run({"path": "../escape.txt", "content": "bad"})).success
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.asyncio
async def test_symlink_escape_is_blocked(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    try:
        (root / "link.txt").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable on this machine")
    result = await FileReadTool(Workspace(root)).run({"path": "link.txt"})
    assert result.error_type == "PermissionDenied"
    write_result = await FileWriteTool(Workspace(root)).run(
        {"path": "link.txt", "content": "changed", "overwrite": True}
    )
    assert write_result.error_type == "PermissionDenied"
    assert outside.read_text(encoding="utf-8") == "private"


@pytest.mark.asyncio
async def test_registry_exposes_only_allowed_tools(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(FileWriteTool(Workspace(root)))
    assert [spec.name for spec in registry.specs({"calculator"})] == ["calculator"]
    denied = await registry.execute("file_write", {"path": "x", "content": "x"}, {"calculator"})
    result = await registry.execute("calculator", {"expression": "2+2"}, {"calculator"})
    invalid = await registry.execute(
        "calculator", {"expression": "2+2", "extra": 1}, {"calculator"}
    )
    assert denied.error_type == "PermissionDenied"
    assert result.output == "4"
    assert invalid.error_type == "InvalidArguments"
    with pytest.raises(ValueError, match="already registered"):
        registry.register(CalculatorTool())
