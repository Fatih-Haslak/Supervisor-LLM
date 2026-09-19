from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from app.agents.single import AgentLimitError, SingleAgent
from app.llm.schemas import ChatMessage, LLMResponse
from app.tools.calculator import CalculatorTool
from app.tools.filesystem import FileReadTool, Workspace
from app.tools.registry import ToolRegistry


class ScriptedLLM:
    def __init__(self, decisions: list[str]) -> None:
        self._decisions = iter(decisions)
        self.requests: list[list[ChatMessage]] = []

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        assert json_schema is not None
        self.requests.append(list(messages))
        return LLMResponse(content=next(self._decisions), model="scripted")


@pytest.mark.asyncio
async def test_agent_reads_calculates_and_answers(tmp_path: Path) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    (workspace_path / "numbers.txt").write_text("12\n8\n5", encoding="utf-8")
    workspace = Workspace(workspace_path)
    registry = ToolRegistry()
    registry.register(FileReadTool(workspace))
    registry.register(CalculatorTool())
    llm = ScriptedLLM(
        [
            '{"action":"use_tool","tool":"file_read","arguments":{"path":"numbers.txt"}}',
            '{"action":"use_tool","tool":"calculator","arguments":{"expression":"12+8+5"}}',
            '{"action":"final_answer","answer":"Toplam 25"}',
        ]
    )
    result = await SingleAgent(llm, registry, {"file_read", "calculator"}).run(
        "numbers.txt içindeki sayıları topla"
    )
    assert result.answer == "Toplam 25"
    assert result.step_count == 3
    assert result.state.task_id
    assert result.state.current_agent is None
    assert result.state.pending_tasks == []
    assert result.state.completed_tasks == ["numbers.txt içindeki sayıları topla"]
    assert result.state.final_answer == "Toplam 25"
    assert [call.tool for call in result.tool_calls] == ["file_read", "calculator"]
    assert result.tool_calls[1].result.output == "25"
    assert "12\\n8\\n5" in llm.requests[1][-1].content
    assert "calculator" in llm.requests[2][-1].content


@pytest.mark.asyncio
async def test_agent_limits_stop_repeated_tool_calls() -> None:
    llm = ScriptedLLM(
        ['{"action":"use_tool","tool":"calculator","arguments":{"expression":"1+1"}}'] * 3
    )
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    agent = SingleAgent(llm, registry, {"calculator"}, max_steps=2, max_tool_calls=2)
    with pytest.raises(AgentLimitError, match="steps"):
        await agent.run("Keep calculating")
    assert len(llm.requests) == 2


@pytest.mark.asyncio
async def test_agent_does_not_expose_disallowed_tool() -> None:
    llm = ScriptedLLM(['{"action":"final_answer","answer":"Bitti"}'])
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    result = await SingleAgent(llm, registry, set()).run("Sadece cevap ver")
    assert result.answer == "Bitti"
    assert '"calculator"' not in llm.requests[0][1].content
