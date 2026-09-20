import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from app.agents.reviewer import ReviewerAgent
from app.agents.supervisor import Supervisor
from app.agents.workers import build_workers
from app.llm.schemas import ChatMessage, LLMResponse
from app.tools.filesystem import DirectoryListTool, FileReadTool, FileWriteTool, Workspace
from app.tools.function_test import FunctionTestTool
from app.tools.registry import ToolRegistry
from tests.support import ApproveWrites


class ScriptedLLM:
    def __init__(self, answers: list[str]) -> None:
        self.answers = iter(answers)

    async def chat(
        self, _messages: Sequence[ChatMessage], *, json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        return LLMResponse(content=next(self.answers), model="scripted")


@pytest.mark.asyncio
async def test_coder_fixes_bug_and_reviewer_reruns_cases(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "buggy_math.py").write_text(
        "def add(a, b):\n    return a - b\n", encoding="utf-8"
    )
    (root / "buggy_math_tests.json").write_text(json.dumps({
        "function": "add", "cases": [
            {"args": [2, 3], "expected": 5},
            {"args": [-4, 7], "expected": 3},
        ],
    }), encoding="utf-8")
    registry = ToolRegistry(approver=ApproveWrites())
    workspace = Workspace(root)
    for tool in (DirectoryListTool(workspace), FileReadTool(workspace),
                 FileWriteTool(workspace), FunctionTestTool(workspace)):
        registry.register(tool)
    fixed = "def add(a, b):\n    return a + b\n"
    llm = ScriptedLLM([
        '{"tasks":[{"id":1,"agent":"coder","task":'
        '"buggy_math.py hatasını düzelt ve buggy_math_tests.json testlerini çalıştır",'
        '"depends_on":[]}]}',
        '{"action":"use_tool","tool":"directory_list","arguments":{"path":"."}}',
        '{"action":"use_tool","tool":"file_read",'
        '"arguments":{"path":"buggy_math.py"}}',
        '{"action":"use_tool","tool":"file_read",'
        '"arguments":{"path":"buggy_math_tests.json"}}',
        json.dumps({"action": "use_tool", "tool": "file_write", "arguments": {
            "path": "buggy_math.py", "content": fixed, "overwrite": True
        }}),
        '{"action":"use_tool","tool":"function_test","arguments":'
        '{"path":"buggy_math.py","tests_path":"buggy_math_tests.json"}}',
        '{"action":"final_answer","answer":"Hata düzeltildi; 2/2 test geçti."}',
        '{"status":"pass","issues":[]}',
        '{"action":"final_answer","answer":"Kod düzeltildi ve testler doğrulandı."}',
    ])
    state = await Supervisor(
        llm, build_workers(llm, registry), reviewer=ReviewerAgent(llm, registry)
    ).run_planned("buggy_math.py hatasını düzelt, testleri çalıştır ve incele")
    assert state.final_answer == "Kod düzeltildi ve testler doğrulandı."
    assert (root / "buggy_math.py").read_text(encoding="utf-8") == fixed
    assert [review.status for review in state.reviews] == ["pass"]
    assert [call.tool for call in state.tool_results].count("function_test") == 2
