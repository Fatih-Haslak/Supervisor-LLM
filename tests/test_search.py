import json
from pathlib import Path

import pytest

from app.tools.filesystem import Workspace
from app.tools.search import SearchTool


@pytest.mark.asyncio
async def test_search_finds_local_lines_and_skips_protected_paths(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "notes.txt").write_text("Ankara\nFatih Tekke bir futbolcudur\n", encoding="utf-8")
    (root / ".env").write_text("SECRET=Fatih Tekke", encoding="utf-8")
    tool = SearchTool(Workspace(root))

    result = await tool.run({"query": "fatih tekke"})
    assert result.success
    matches = json.loads(result.output or "[]")
    assert matches == [{"path": "notes.txt", "line": 2, "text": "Fatih Tekke bir futbolcudur"}]

    denied = await tool.run({"query": "secret", "path": "../"})
    assert denied.error_type == "PermissionDenied"


@pytest.mark.asyncio
async def test_search_does_not_follow_external_symlink(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "private.txt"
    outside.write_text("do-not-find-me", encoding="utf-8")
    try:
        (root / "link.txt").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable on this machine")

    result = await SearchTool(Workspace(root)).run({"query": "do-not-find-me"})
    assert result.success and result.output == "[]"
