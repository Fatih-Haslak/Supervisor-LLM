"""Read-only tests for small pure arithmetic Python functions in workspace."""

import ast
import asyncio
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.tools.base import BaseTool, ToolResult
from app.tools.filesystem import Workspace, WorkspaceAccessError


class FunctionTestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=260)
    tests_path: str = Field(min_length=1, max_length=260)


class FunctionCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    args: list[int | float] = Field(max_length=8)
    expected: int | float


class FunctionSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function: str = Field(min_length=1, max_length=60)
    cases: list[FunctionCase] = Field(min_length=1, max_length=20)


def _number(value: int | float) -> Decimal:
    number = Decimal(str(value))
    if not number.is_finite() or abs(number) > 1_000_000_000:
        raise ValueError("Test number is outside allowed bounds")
    return number


def _expression(node: ast.expr, variables: dict[str, Decimal]) -> Decimal:
    if (isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)):
        result = _number(node.value)
    elif isinstance(node, ast.Name) and node.id in variables:
        result = variables[node.id]
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _expression(node.operand, variables)
        result = value if isinstance(node.op, ast.UAdd) else -value
    elif isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)
    ):
        left = _expression(node.left, variables)
        right = _expression(node.right, variables)
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            result = left / right
        elif isinstance(node.op, ast.FloorDiv):
            result = left // right
        else:
            result = left % right
    else:
        raise ValueError("Only pure arithmetic return expressions can be tested")
    if not result.is_finite() or abs(result) > 1_000_000_000_000:
        raise ValueError("Computed result is outside allowed bounds")
    return result


def _test_module(source: str, suite: FunctionSuite) -> dict[str, object]:
    tree = ast.parse(source)
    if len(list(ast.walk(tree))) > 100:
        raise ValueError("Python function is too complex for the safe test runner")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or len(tree.body) != 1 or functions[0].name != suite.function:
        raise ValueError("Expected exactly one matching Python function")
    function = functions[0]
    if (function.decorator_list or function.args.defaults or function.args.kw_defaults
            or function.args.vararg or function.args.kwarg or function.args.kwonlyargs
            or function.args.posonlyargs or len(function.body) != 1
            or not isinstance(function.body[0], ast.Return)
            or function.body[0].value is None):
        raise ValueError("Only a simple function with one return expression is supported")
    names = [argument.arg for argument in function.args.args]
    if len(names) > 8 or len(names) != len(set(names)):
        raise ValueError("Function parameters are outside allowed bounds")
    failures: list[dict[str, object]] = []
    for index, case in enumerate(suite.cases, start=1):
        if len(case.args) != len(names):
            raise ValueError("Test argument count does not match function parameters")
        variables = dict(zip(names, (_number(arg) for arg in case.args), strict=True))
        try:
            actual = _expression(function.body[0].value, variables)
            expected = _number(case.expected)
            if actual != expected:
                failures.append({"case": index, "expected": str(expected), "actual": str(actual)})
        except (ArithmeticError, InvalidOperation, ValueError) as exc:
            failures.append({"case": index, "error": type(exc).__name__})
    return {
        "function": suite.function,
        "passed": len(suite.cases) - len(failures),
        "total": len(suite.cases),
        "failures": failures,
    }


class FunctionTestTool(BaseTool[FunctionTestInput]):
    name = "function_test"
    description = (
        "Run JSON test cases for one pure arithmetic Python function. "
        "Reads workspace code and test JSON; never executes workspace Python code. "
        "Use path and tests_path relative to workspace."
    )
    input_type = FunctionTestInput

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    async def execute(self, arguments: FunctionTestInput) -> ToolResult:
        try:
            code = self._workspace.resolve(arguments.path)
            tests = self._workspace.resolve(arguments.tests_path)
            if not code.is_file() or not tests.is_file():
                return ToolResult.fail("FileNotFound", "Code or test file was not found")
            if code.suffix.casefold() != ".py" or tests.suffix.casefold() != ".json":
                return ToolResult.fail("InvalidPath", "Expected .py code and .json tests")
            if code.stat().st_size > 4000 or tests.stat().st_size > 10000:
                return ToolResult.fail("FileTooLarge", "Code or test file exceeds the safe limit")
            source, test_source = await asyncio.gather(
                asyncio.to_thread(Path.read_text, code, encoding="utf-8"),
                asyncio.to_thread(Path.read_text, tests, encoding="utf-8"),
            )
            suite = FunctionSuite.model_validate_json(test_source)
            result = _test_module(source, suite)
            return ToolResult.ok(json.dumps(result, ensure_ascii=False))
        except WorkspaceAccessError as exc:
            return ToolResult.fail("PermissionDenied", str(exc))
        except (OSError, UnicodeError):
            return ToolResult.fail("FileError", "Could not read code or tests")
        except (SyntaxError, ValueError, RecursionError, InvalidOperation) as exc:
            return ToolResult.fail("InvalidTest", str(exc))
