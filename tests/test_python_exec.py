from pathlib import Path

import pytest

from app.tools.filesystem import Workspace
from app.tools.python_exec import PythonExecTool


@pytest.mark.asyncio
async def test_restricted_python_runs_in_subprocess(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = PythonExecTool(Workspace(workspace))
    result = await tool.run({"code": "numbers = [1, 2, 3]\nprint(sum(numbers))"})
    assert result.success
    assert result.output == "6"


@pytest.mark.asyncio
async def test_restricted_python_cannot_import_or_read_files(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    tool = PythonExecTool(Workspace(workspace))
    for code in ("import os", "print(open('../secret.txt'))", "print((1).__class__)"):
        result = await tool.run({"code": code})
        assert result.error_type == "PythonError"
        assert "secret" not in (result.output or "")


@pytest.mark.asyncio
async def test_restricted_python_timeout(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = PythonExecTool(Workspace(workspace))
    result = await tool.run(
        {"code": "for i in range(1000000000):\n    pass", "timeout_seconds": 0.2}
    )
    assert result.error_type == "Timeout"


@pytest.mark.asyncio
async def test_restricted_python_output_is_capped(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = PythonExecTool(Workspace(workspace))
    result = await tool.run({"code": "for i in range(1000):\n    print('x' * 100)"})
    assert result.success
    assert result.output is not None
    assert "[output limit reached]" in result.output
    assert len(result.output) < 17000
