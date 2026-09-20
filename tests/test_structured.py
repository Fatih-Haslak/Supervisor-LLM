from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import ValidationError

from app.llm.schemas import ChatMessage, LLMResponse
from app.llm.structured import (
    FinalAnswerDecision,
    StructuredDecisionClient,
    StructuredOutputError,
    UseToolDecision,
    decision_schema,
    parse_decision,
)
from app.tools.base import ToolSpec
from app.tools.csv_analysis import CsvSummaryInput


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls: list[list[ChatMessage]] = []

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        assert json_schema is not None
        self.calls.append(list(messages))
        return LLMResponse(content=next(self.responses), model="fake")


def test_parse_both_decision_variants() -> None:
    answer = parse_decision('{"action":"final_answer","answer":"425"}')
    call = parse_decision(
        '{"action":"use_tool","tool":"calculator","arguments":{"expression":"25*17"}}'
    )
    assert isinstance(answer, FinalAnswerDecision) and answer.answer == "425"
    assert isinstance(call, UseToolDecision) and call.arguments == {"expression": "25*17"}


def test_extra_or_missing_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_decision('{"action":"final_answer","answer":"ok","extra":1}')
    with pytest.raises(ValidationError):
        parse_decision('{"action":"use_tool","tool":"calculator"}')


def test_tool_schema_limits_arguments_to_registered_input() -> None:
    schema = decision_schema([ToolSpec(
        name="csv_summary", description="Summarize CSV",
        parameters=CsvSummaryInput.model_json_schema(),
    )])
    choices = schema["oneOf"]
    assert isinstance(choices, list)
    call = choices[1]
    assert call["properties"]["tool"] == {"const": "csv_summary"}
    assert call["properties"]["arguments"]["additionalProperties"] is False
    assert set(call["properties"]["arguments"]["properties"]) == {"path", "column"}


@pytest.mark.asyncio
async def test_invalid_json_is_retried_with_limit() -> None:
    llm = FakeLLM(["not json", '{"action":"final_answer","answer":"Tamam"}'])
    decision = await StructuredDecisionClient(llm, max_retries=1).decide(
        [ChatMessage(role="user", content="Merhaba")]
    )
    assert isinstance(decision, FinalAnswerDecision)
    assert len(llm.calls) == 2
    assert llm.calls[1][-1].role == "user"


@pytest.mark.asyncio
async def test_retry_limit_is_enforced() -> None:
    llm = FakeLLM(["invalid", "still invalid"])
    with pytest.raises(StructuredOutputError, match="2 attempts"):
        await StructuredDecisionClient(llm, max_retries=1).decide(
            [ChatMessage(role="user", content="Merhaba")]
        )
    assert len(llm.calls) == 2
