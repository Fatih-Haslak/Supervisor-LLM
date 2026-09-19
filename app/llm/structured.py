"""Validated JSON decisions with a bounded correction loop."""

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from app.llm.client import LLMClient
from app.llm.schemas import ChatMessage


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
        self, messages: Sequence[ChatMessage]
    ) -> UseToolDecision | FinalAnswerDecision:
        request = [ChatMessage(role="system", content=DECISION_INSTRUCTIONS), *messages]
        for attempt in range(self._max_retries + 1):
            response = await self._llm.chat(
                request, json_schema=_decision_adapter.json_schema()
            )
            try:
                return parse_decision(response.content)
            except ValidationError as exc:
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
