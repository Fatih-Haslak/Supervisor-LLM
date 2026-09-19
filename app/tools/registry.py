"""Registry that exposes and executes only explicitly allowed tools."""

from collections.abc import Collection, Mapping
from typing import Any

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
        if name not in allowed_tools:
            return ToolResult.fail("PermissionDenied", "Tool is not allowed for this caller")
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.fail("UnknownTool", "Tool is not registered")
        return await tool.run(arguments)
