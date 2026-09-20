"""Validated JSON decisions with a bounded correction loop."""

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from app.llm.client import LLMClient
from app.llm.schemas import ChatMessage
from app.observability.events import record
from app.tools.base import ToolSpec


class UseToolDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["use_tool"]
    tool: str = Field(min_length=1)
    arguments: dict[str, object]


class FinalAnswerDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["final_answer"]
    answer: str = Field(min_length=1)


AgentDecision = Annotated[
    UseToolDecision | FinalAnswerDecision,
    Field(discriminator="action"),
]
_decision_adapter: TypeAdapter[AgentDecision] = TypeAdapter(AgentDecision)


def decision_schema(tools: Sequence[ToolSpec]) -> dict[str, object]:
    """Constrain each tool call to its actual arguments in local-model decoding."""
    def compact(value: object) -> object:
        if isinstance(value, dict):
            # llama.cpp grammar expands maxLength into thousands of rules for
            # file content and can overflow its sampler stack. Pydantic still
            # enforces these bounds when the tool executes.
            return {
                key: compact(item) for key, item in value.items()
                if key not in {"maxLength", "title", "description", "default"}
            }
        if isinstance(value, list):
            return [compact(item) for item in value]
        return value

    choices: list[dict[str, object]] = [{
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {"const": "final_answer"},
            "answer": {"type": "string", "minLength": 1},
        },
        "required": ["action", "answer"],
    }]
    for tool in tools:
        choices.append({
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {"const": "use_tool"},
                "tool": {"const": tool.name},
                "arguments": compact(tool.parameters),
            },
            "required": ["action", "tool", "arguments"],
        })
    return {"oneOf": choices}

DECISION_INSTRUCTIONS = (
    "Return exactly one JSON object and no markdown. "
    'Choose {"action":"final_answer","answer":"..."} or '
    '{"action":"use_tool","tool":"...","arguments":{...}}. '
    "Use only tools explicitly provided in the task. Do not invent tool names."
)


class StructuredOutputError(Exception):
    """The model did not produce a valid decision within the retry limit."""


def parse_decision(content: str) -> UseToolDecision | FinalAnswerDecision:
    return _decision_adapter.validate_json(content)


class StructuredDecisionClient:
    def __init__(self, llm: LLMClient, max_retries: int = 2) -> None:
        if not 0 <= max_retries <= 5:
            raise ValueError("max_retries must be between 0 and 5")
        self._llm = llm
        self._max_retries = max_retries

    async def decide(
        self, messages: Sequence[ChatMessage], *, tools: Sequence[ToolSpec] | None = None
    ) -> UseToolDecision | FinalAnswerDecision:
        request = [ChatMessage(role="system", content=DECISION_INSTRUCTIONS), *messages]
        schema = decision_schema(tools) if tools is not None else _decision_adapter.json_schema()
        for attempt in range(self._max_retries + 1):
            response = await self._llm.chat(
                request, json_schema=schema
            )
            try:
                return parse_decision(response.content)
            except ValidationError as exc:
                record("model_retry", retry_count=attempt + 1)
                if attempt == self._max_retries:
                    raise StructuredOutputError(
                        f"Model returned invalid structured output after {attempt + 1} attempts"
                    ) from exc
                if response.content:
                    request.append(ChatMessage(role="assistant", content=response.content))
                request.append(
                    ChatMessage(
                        role="user",
                        content=(
                            "Invalid decision JSON. Return only one object in the required schema."
                        ),
                    )
                )
        raise AssertionError("Unreachable retry state")
