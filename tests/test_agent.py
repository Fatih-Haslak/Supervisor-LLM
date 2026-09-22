from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from app.agents.single import AgentLimitError, SingleAgent, _public_title_from_request
from app.llm.schemas import ChatMessage, LLMResponse
from app.security.approvals import ApprovalRequest
from app.tools.calculator import CalculatorTool
from app.tools.filesystem import FileReadTool, FileWriteTool, Workspace
from app.tools.function_test import FunctionTestTool
from app.tools.registry import ToolRegistry
from app.tools.search import SearchTool
from app.tools.wikipedia import WikipediaLookupTool


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


class ApproveWrites:
    async def request_approval(self, request: ApprovalRequest) -> bool:
        return request.tool == "file_write"


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


@pytest.mark.asyncio
async def test_researcher_reads_search_hit_before_answer(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "brief.txt").write_text("Proje kodu Orion-17.\n", encoding="utf-8")
    workspace = Workspace(root)
    registry = ToolRegistry()
    registry.register(SearchTool(workspace))
    registry.register(FileReadTool(workspace))
    llm = ScriptedLLM([
        '{"action":"use_tool","tool":"search",'
        '"arguments":{"query":"Orion","path":"."}}',
        '{"action":"final_answer","answer":"brief.txt"}',
        '{"action":"use_tool","tool":"file_read",'
        '"arguments":{"path":"brief.txt"}}',
        '{"action":"final_answer","answer":"brief.txt: Orion-17"}',
    ])
    result = await SingleAgent(
        llm, registry, {"search", "file_read"}, role_name="researcher"
    ).run("Orion proje kodunu ara")
    assert result.answer == "brief.txt: Orion-17"
    assert [call.tool for call in result.tool_calls] == ["search", "file_read"]
    assert "Before answering, use file_read" in llm.requests[2][-1].content


@pytest.mark.asyncio
async def test_researcher_retries_empty_phrase_with_distinctive_term(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "brief.txt").write_text("Proje kodu Orion-17.\n", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(SearchTool(Workspace(root)))
    registry.register(FileReadTool(Workspace(root)))
    llm = ScriptedLLM([
        '{"action":"use_tool","tool":"search",'
        '"arguments":{"query":"Orion proje kodu","path":"."}}',
        '{"action":"use_tool","tool":"file_read",'
        '"arguments":{"path":"brief.txt"}}',
        '{"action":"final_answer","answer":"brief.txt: Orion-17"}',
    ])
    result = await SingleAgent(
        llm, registry, {"search", "file_read"}, role_name="researcher"
    ).run("workspace belgelerinde Orion proje kodunu ara")
    assert [call.tool for call in result.tool_calls] == ["search", "search", "file_read"]
    assert result.tool_calls[0].result.output == "[]"
    assert result.tool_calls[1].arguments["query"] == "Orion"
    assert "Orion-17" in result.answer


@pytest.mark.asyncio
async def test_coder_runs_json_tests_after_edit_before_answer(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "math.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (root / "cases.json").write_text(
        '{"function":"add","cases":[{"args":[2,3],"expected":5}]}',
        encoding="utf-8",
    )
    workspace = Workspace(root)
    registry = ToolRegistry(approver=ApproveWrites())
    registry.register(FileWriteTool(workspace))
    registry.register(FunctionTestTool(workspace))
    llm = ScriptedLLM([
        '{"action":"use_tool","tool":"file_write","arguments":'
        '{"path":"math.py","content":"def add(a, b):\\n    return a + b\\n",'
        '"overwrite":true}}',
        '{"action":"final_answer","answer":"Düzeltildi"}',
        '{"action":"use_tool","tool":"function_test","arguments":'
        '{"path":"math.py","tests_path":"cases.json"}}',
        '{"action":"final_answer","answer":"1 test geçti"}',
    ])
    result = await SingleAgent(
        llm, registry, {"file_write", "function_test"}, role_name="coder"
    ).run("math.py düzelt ve cases.json testlerini çalıştır")
    assert result.answer == "1 test geçti"
    assert [call.tool for call in result.tool_calls] == ["file_write", "function_test"]
    assert "Call function_test" in llm.requests[2][-1].content


@pytest.mark.asyncio
async def test_researcher_looks_up_public_fact_before_answer() -> None:
    registry = ToolRegistry()
    registry.register(WikipediaLookupTool(lambda _title: {
        "query": {"pages": [{"title": "Fatih Tekke", "extract": "Türk futbolcu."}]}
    }))
    llm = ScriptedLLM([
        '{"action":"final_answer","answer":"Tanınmıyor"}',
        '{"action":"use_tool","tool":"wikipedia_lookup",'
        '"arguments":{"title":"Fatih Tekke"}}',
        '{"action":"final_answer","answer":"Fatih Tekke Türk futbolcudur."}',
    ])
    result = await SingleAgent(
        llm, registry, {"wikipedia_lookup"}, role_name="researcher"
    ).run("Fatih Tekke kimdir?")
    assert "futbolcudur" in result.answer
    assert [call.tool for call in result.tool_calls] == ["wikipedia_lookup"]
    assert "Public fact lookup" in llm.requests[1][-1].content


@pytest.mark.asyncio
async def test_public_lookup_repairs_missing_title_from_original_question() -> None:
    seen: list[str] = []

    def fetch(title: str) -> dict[str, object]:
        seen.append(title)
        return {"query": {"pages": [{"title": title, "extract": "Futbolcudur."}]}}

    registry = ToolRegistry()
    registry.register(WikipediaLookupTool(fetch))
    llm = ScriptedLLM([
        '{"action":"use_tool","tool":"wikipedia_lookup","arguments":{}}',
        '{"action":"final_answer","answer":"Fatih Tekke futbolcudur."}',
    ])
    result = await SingleAgent(
        llm, registry, {"wikipedia_lookup"}, role_name="researcher"
    ).run(
        "Original user request (context only): FATİH TEKKE KİMDİR?\n"
        "Assigned subtask: Fatih Tekke kimdir?"
    )
    assert result.tool_calls[0].arguments == {"title": "FATİH TEKKE"}
    assert seen == ["Fatih Tekke"]


@pytest.mark.asyncio
async def test_public_lookup_repairs_natural_question_and_query_argument() -> None:
    question = "Bilmiyorumda benim için Fatih Terim kimdir araştırır mısın"
    assert _public_title_from_request(question) == "Fatih Terim"
    assert _public_title_from_request("fatih tekke kimdir") == "fatih tekke"
    assert _public_title_from_request("fatih tekke (futbolcu) olan kimdir") == "fatih tekke"
    seen: list[str] = []

    def fetch(title: str) -> dict[str, object]:
        seen.append(title)
        return {"query": {"pages": [{"title": title, "extract": "Türk teknik direktör."}]}}

    registry = ToolRegistry()
    registry.register(WikipediaLookupTool(fetch))
    llm = ScriptedLLM([
        '{"action":"use_tool","tool":"wikipedia_lookup","arguments":{}}',
        '{"action":"final_answer","answer":"Fatih Terim teknik direktördür."}',
    ])
    result = await SingleAgent(
        llm, registry, {"wikipedia_lookup"}, role_name="researcher"
    ).run("Original user request (context only): " + question)
    assert result.tool_calls[0].arguments == {"title": "Fatih Terim"}
    assert seen == ["Fatih Terim"]

    repaired = SingleAgent(llm, registry, {"wikipedia_lookup"})._tool_arguments(
        "wikipedia_lookup", {"query": question}, question
    )
    assert repaired == {"title": "Fatih Terim"}


@pytest.mark.asyncio
async def test_unavailable_public_source_returns_uncertainty_without_retry() -> None:
    def fail(_title: str) -> dict[str, object]:
        raise OSError("offline")

    registry = ToolRegistry()
    registry.register(WikipediaLookupTool(fail))
    llm = ScriptedLLM([
        '{"action":"use_tool","tool":"wikipedia_lookup",'
        '"arguments":{"title":"Fatih Tekke"}}',
    ])
    result = await SingleAgent(
        llm, registry, {"wikipedia_lookup"}, role_name="researcher"
    ).run("Fatih Tekke kimdir?")
    assert "doğrulayamadım" in result.answer
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].result.error_type == "LookupUnavailable"
