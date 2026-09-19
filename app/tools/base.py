"""Tool contracts and input validation."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    output: str | None = None
    error_type: str | None = None
    message: str | None = None

    @model_validator(mode="after")
    def check_result(self) -> "ToolResult":
        if self.success and (self.output is None or self.error_type is not None):
            raise ValueError("Successful tool results need output and no error")
        if not self.success and (self.error_type is None or self.message is None):
            raise ValueError("Failed tool results need error_type and message")
        return self

    @classmethod
    def ok(cls, output: str) -> "ToolResult":
        return cls(success=True, output=output)

    @classmethod
    def fail(cls, error_type: str, message: str) -> "ToolResult":
        return cls(success=False, error_type=error_type, message=message)


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


ArgumentsT = TypeVar("ArgumentsT", bound=BaseModel)


class BaseTool(ABC, Generic[ArgumentsT]):
    name: str
    description: str
    input_type: type[ArgumentsT]

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.input_type.model_json_schema(),
        )

    async def run(self, arguments: Mapping[str, object]) -> ToolResult:
        try:
            parsed = self.input_type.model_validate(dict(arguments))
        except ValidationError:
            return ToolResult.fail("InvalidArguments", "Tool arguments failed validation")
        try:
            return await self.execute(parsed)
        except Exception:
            return ToolResult.fail("ToolExecutionError", "Tool execution failed")

    @abstractmethod
    async def execute(self, arguments: ArgumentsT) -> ToolResult:
        """Run deterministic work with validated arguments."""
