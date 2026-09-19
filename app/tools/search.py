"""Bounded local text search restricted to workspace files."""

import asyncio
import json
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.tools.base import BaseTool, ToolResult
from app.tools.filesystem import Workspace, WorkspaceAccessError

_MAX_FILES = 200
_MAX_MATCHES = 20
_MAX_FILE_BYTES = 100_000


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=200)
    path: str = Field(default=".", min_length=1, max_length=260)


class SearchTool(BaseTool[SearchInput]):
    name = "search"
    description = "Search UTF-8 text inside workspace files; this does not search the internet."
    input_type = SearchInput

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    def _search(self, directory: Path, query: str) -> list[dict[str, object]]:
        matches: list[dict[str, object]] = []
        examined = 0
        needle = query.casefold()
        for root, dirs, files in os.walk(directory, followlinks=False):
            dirs[:] = [
                name for name in sorted(dirs)
                if not (Path(root) / name).is_symlink()
                and self._is_allowed(Path(root) / name)
            ]
            for name in sorted(files):
                target = Path(root) / name
                if not self._is_allowed(target) or not target.is_file():
                    continue
                if target.stat().st_size > _MAX_FILE_BYTES:
                    continue
                examined += 1
                if examined > _MAX_FILES:
                    return matches
                try:
                    lines = target.read_text(encoding="utf-8").splitlines()
                except (OSError, UnicodeError):
                    continue
                for number, line in enumerate(lines, start=1):
                    if needle in line.casefold():
                        matches.append(
                            {"path": str(target.relative_to(self._workspace.root)),
                             "line": number, "text": line[:300]}
                        )
                        if len(matches) >= _MAX_MATCHES:
                            return matches
        return matches

    def _is_allowed(self, path: Path) -> bool:
        try:
            self._workspace.resolve(str(path.relative_to(self._workspace.root)))
            return True
        except (ValueError, WorkspaceAccessError):
            return False

    async def execute(self, arguments: SearchInput) -> ToolResult:
        try:
            directory = self._workspace.resolve(arguments.path)
            if not directory.is_dir():
                return ToolResult.fail("DirectoryNotFound", "Workspace directory not found")
            matches = await asyncio.to_thread(self._search, directory, arguments.query)
            return ToolResult.ok(json.dumps(matches, ensure_ascii=False))
        except WorkspaceAccessError as exc:
            return ToolResult.fail("PermissionDenied", str(exc))
        except OSError:
            return ToolResult.fail("FileError", "Could not search workspace files")
