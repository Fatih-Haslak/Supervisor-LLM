"""Registry that exposes and executes only explicitly allowed tools."""

from collections.abc import Collection, Mapping
from copy import deepcopy
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from app.observability.events import record
from app.security.approvals import ApprovalRequest, Approver
from app.tools.base import BaseTool, ToolResult, ToolSpec


class ToolRegistry:
    def __init__(self, approver: Approver | None = None) -> None:
        self._tools: dict[str, BaseTool[Any]] = {}
        self._approver = approver

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
                if tool is None:
                    result = ToolResult.fail("UnknownTool", "Tool is not registered")
                else:
                    try:
                        validated = tool.input_type.model_validate(dict(arguments))
                    except ValidationError:
                        result = ToolResult.fail(
                            "InvalidArguments", "Tool arguments failed validation"
                        )
                    else:
                        normalized = validated.model_dump()
                        preflight_error = tool.preflight(validated)
                        if preflight_error is not None:
                            result = preflight_error
                        elif tool.requires_approval:
                            if self._approver is None:
                                result = ToolResult.fail(
                                    "ApprovalRequired", "Human approval is required for this tool"
                                )
                            else:
                                request = ApprovalRequest(
                                    tool=name, arguments=deepcopy(normalized)
                                )
                                approved = await self._approver.request_approval(request)
                                result = (
                                    await tool.run(normalized) if approved is True
                                    else ToolResult.fail(
                                        "ApprovalDenied", "Human approval was denied"
                                    )
                                )
                        else:
                            result = await tool.run(normalized)
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
