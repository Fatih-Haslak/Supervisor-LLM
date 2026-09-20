import json
from pathlib import Path

import pytest

from app.tools.filesystem import Workspace
from app.tools.function_test import FunctionTestTool


@pytest.mark.asyncio
async def test_function_cases_fail_then_pass_after_fix(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    code = root / "math.py"
    code.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (root / "tests.json").write_text(json.dumps({
        "function": "add", "cases": [
            {"args": [2, 3], "expected": 5},
            {"args": [-4, 7], "expected": 3},
        ],
    }), encoding="utf-8")
    tool = FunctionTestTool(Workspace(root))
    before = await tool.run({"path": "math.py", "tests_path": "tests.json"})
    assert before.success and before.output is not None
    assert json.loads(before.output)["passed"] == 0

    code.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    after = await tool.run({"path": "math.py", "tests_path": "tests.json"})
    assert after.success and after.output is not None
    assert json.loads(after.output)["passed"] == 2


@pytest.mark.asyncio
async def test_function_tool_rejects_imports_and_outside_paths(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "math.py").write_text(
        "import os\ndef add(a, b):\n    return a + b\n", encoding="utf-8"
    )
    (root / "tests.json").write_text(json.dumps({
        "function": "add", "cases": [{"args": [2, 3], "expected": 5}],
    }), encoding="utf-8")
    tool = FunctionTestTool(Workspace(root))
    denied = await tool.run({"path": "../math.py", "tests_path": "tests.json"})
    assert denied.error_type == "PermissionDenied"
    unsafe = await tool.run({"path": "math.py", "tests_path": "tests.json"})
    assert unsafe.error_type == "InvalidTest"
