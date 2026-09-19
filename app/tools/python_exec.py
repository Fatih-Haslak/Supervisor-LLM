"""Bounded subprocess wrapper for restricted Python snippets."""

import asyncio
import os
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.tools.base import BaseTool, ToolResult
from app.tools.filesystem import Workspace

_RUNNER = Path(__file__).with_name("_python_runner.py")
_MAX_OUTPUT_BYTES = 20000


class PythonExecInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1, max_length=4000)
    timeout_seconds: float = Field(default=2, gt=0, le=5)


class PythonExecTool(BaseTool[PythonExecInput]):
    name = "python_exec"
    description = (
        "Run a restricted Python snippet for small calculations. "
        "No imports, attributes, file access, shell calls, or arbitrary functions."
    )
    input_type = PythonExecInput

    def __init__(self, workspace: Workspace, python_executable: Path | None = None) -> None:
        self._workspace = workspace
        self._python_executable = python_executable or Path(sys.executable)

    async def execute(self, arguments: PythonExecInput) -> ToolResult:
        environment = {
            key: os.environ[key]
            for key in ("SystemRoot", "WINDIR", "PATH", "TEMP", "TMP")
            if key in os.environ
        }
        environment["PYTHONIOENCODING"] = "utf-8"
        try:
            process = await asyncio.create_subprocess_exec(
                str(self._python_executable),
                "-I",
                "-S",
                str(_RUNNER),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._workspace.root,
                env=environment,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(arguments.code.encode("utf-8")),
                    timeout=arguments.timeout_seconds,
                )
            except TimeoutError:
                process.kill()
                await process.communicate()
                return ToolResult.fail("Timeout", "Python execution exceeded its time limit")
        except OSError:
            return ToolResult.fail("ExecutionError", "Could not start Python subprocess")

        output = stdout[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace").rstrip()
        error = stderr[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace").rstrip()
        if process.returncode != 0:
            return ToolResult(
                success=False,
                output=output,
                error_type="PythonError",
                message=error or "Restricted Python code failed",
            )
        return ToolResult.ok(output)
