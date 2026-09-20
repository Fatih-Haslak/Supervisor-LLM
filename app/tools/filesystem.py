"""Workspace-only file tools with path and overwrite checks."""

import asyncio
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.tools.base import BaseTool, ToolResult

_MAX_FILE_BYTES = 1_000_000
_BLOCKED_PARTS = {".git", ".venv", ".env"}
_WINDOWS_DEVICES = {"con", "prn", "aux", "nul", "com1", "com2", "com3", "com4",
                    "com5", "com6", "com7", "com8", "com9", "lpt1", "lpt2",
                    "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9"}


class WorkspaceAccessError(ValueError):
    """A path is outside the allowed workspace or otherwise disallowed."""


class Workspace:
    def __init__(self, root: Path) -> None:
        if not root.is_dir() or root.is_symlink():
            raise ValueError("Workspace root must be an existing directory")
        self.root = root.resolve()

    def resolve(self, path: str) -> Path:
        relative = Path(path)
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            raise WorkspaceAccessError("Path must stay inside workspace")
        # Accept user-facing paths such as workspace/report.md as root-relative.
        if relative.parts and relative.parts[0].casefold() == self.root.name.casefold():
            relative = Path(*relative.parts[1:]) if len(relative.parts) > 1 else Path(".")
        for part in relative.parts:
            normalized = part.rstrip(" .").casefold()
            if ":" in part or any(ord(character) < 32 for character in part):
                raise WorkspaceAccessError("Unsupported workspace path")
            if normalized.split(".", 1)[0] in _WINDOWS_DEVICES:
                raise WorkspaceAccessError("Windows device paths are not allowed")
            if normalized in _BLOCKED_PARTS or normalized.startswith(".env."):
                raise WorkspaceAccessError("Protected workspace path")
        target = (self.root / relative).resolve()
        if not target.is_relative_to(self.root):
            raise WorkspaceAccessError("Path must stay inside workspace")
        return target


class FileReadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=260)


class FileWriteInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=260)
    content: str = Field(max_length=_MAX_FILE_BYTES)
    overwrite: bool = False


class DirectoryListInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(default=".", min_length=1, max_length=260)


class FileReadTool(BaseTool[FileReadInput]):
    name = "file_read"
    description = "Read a UTF-8 text file inside workspace."
    input_type = FileReadInput

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    async def execute(self, arguments: FileReadInput) -> ToolResult:
        try:
            target = self._workspace.resolve(arguments.path)
            if not target.is_file():
                return ToolResult.fail("FileNotFound", "Workspace file not found")
            if target.stat().st_size > _MAX_FILE_BYTES:
                return ToolResult.fail("FileTooLarge", "Workspace file exceeds 1 MB")
            content = await asyncio.to_thread(target.read_text, encoding="utf-8")
            return ToolResult.ok(content)
        except WorkspaceAccessError as exc:
            return ToolResult.fail("PermissionDenied", str(exc))
        except UnicodeError:
            return ToolResult.fail("InvalidEncoding", "Workspace file is not UTF-8 text")
        except OSError:
            return ToolResult.fail("FileError", "Could not read workspace file")


class FileWriteTool(BaseTool[FileWriteInput]):
    name = "file_write"
    requires_approval = True
    description = "Write a UTF-8 file inside workspace; overwrite must be explicit."
    input_type = FileWriteInput

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    def preflight(self, arguments: FileWriteInput) -> ToolResult | None:
        try:
            target = self._workspace.resolve(arguments.path)
            if not target.parent.is_dir():
                return ToolResult.fail("DirectoryNotFound", "Parent directory does not exist")
            if target.is_dir():
                return ToolResult.fail("InvalidPath", "Target is a directory")
            if target.exists() and not arguments.overwrite:
                return ToolResult.fail("AlreadyExists", "Overwrite was not allowed")
        except WorkspaceAccessError as exc:
            return ToolResult.fail("PermissionDenied", str(exc))
        return None

    @staticmethod
    def _write(target: Path, content: str, overwrite: bool) -> None:
        mode = "w" if overwrite else "x"
        with target.open(mode, encoding="utf-8") as file:
            file.write(content)

    async def execute(self, arguments: FileWriteInput) -> ToolResult:
        try:
            target = self._workspace.resolve(arguments.path)
            if not target.parent.is_dir():
                return ToolResult.fail("DirectoryNotFound", "Parent directory does not exist")
            if target.is_dir():
                return ToolResult.fail("InvalidPath", "Target is a directory")
            if target.exists() and not arguments.overwrite:
                return ToolResult.fail("AlreadyExists", "Overwrite was not allowed")
            await asyncio.to_thread(self._write, target, arguments.content, arguments.overwrite)
            return ToolResult.ok(f"Wrote {target.relative_to(self._workspace.root)}")
        except WorkspaceAccessError as exc:
            return ToolResult.fail("PermissionDenied", str(exc))
        except FileExistsError:
            return ToolResult.fail("AlreadyExists", "Overwrite was not allowed")
        except OSError:
            return ToolResult.fail("FileError", "Could not write workspace file")


class DirectoryListTool(BaseTool[DirectoryListInput]):
    name = "directory_list"
    description = "List one directory level inside workspace."
    input_type = DirectoryListInput

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    def _visible(self, entry: Path) -> bool:
        if entry.is_symlink():
            return False
        try:
            self._workspace.resolve(str(entry.relative_to(self._workspace.root)))
            return True
        except (ValueError, WorkspaceAccessError):
            return False

    async def execute(self, arguments: DirectoryListInput) -> ToolResult:
        try:
            target = self._workspace.resolve(arguments.path)
            if not target.is_dir():
                return ToolResult.fail("DirectoryNotFound", "Workspace directory not found")
            entries = await asyncio.to_thread(
                lambda: sorted(target.iterdir(), key=lambda p: p.name)
            )
            visible = [entry for entry in entries if self._visible(entry)]
            lines = [f"{entry.name}/" if entry.is_dir() else entry.name for entry in visible[:200]]
            if len(visible) > 200:
                lines.append("... (listing limited to 200 entries)")
            return ToolResult.ok("\n".join(lines))
        except WorkspaceAccessError as exc:
            return ToolResult.fail("PermissionDenied", str(exc))
        except OSError:
            return ToolResult.fail("FileError", "Could not list workspace directory")
