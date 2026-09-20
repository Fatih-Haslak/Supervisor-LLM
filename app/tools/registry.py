"""Registry that exposes and executes only explicitly allowed tools."""

from collections.abc import Collection, Mapping
from time import perf_counter
from typing import Any

from app.observability.events import record
from app.tools.base import BaseTool, ToolResult, ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool[Any]] = {}

    def register(self, tool: BaseTool[Any]) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def specs(self, allowed_tools: Collection[str]) -> list[ToolSpec]:
        return [
            self._tools[name].spec()
            for name in sorted(set(allowed_tools))
            if name in self._tools
        ]

    async def execute(
        self, name: str, arguments: Mapping[str, object], allowed_tools: Collection[str]
    ) -> ToolResult:
        started = perf_counter()
        try:
            if name not in allowed_tools:
                result = ToolResult.fail("PermissionDenied", "Tool is not allowed for this caller")
            else:
                tool = self._tools.get(name)
                result = (
                    ToolResult.fail("UnknownTool", "Tool is not registered")
                    if tool is None else await tool.run(arguments)
                )
        except Exception as exc:
            record(
                "tool_call", tool=name, success=False,
                duration_ms=round((perf_counter() - started) * 1000, 2),
                error_type=type(exc).__name__,
            )
            raise
        record(
            "tool_call", tool=name, success=result.success,
            duration_ms=round((perf_counter() - started) * 1000, 2),
            error_type=result.error_type,
        )
        return result
