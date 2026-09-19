"""Small arithmetic evaluator with an explicit AST allowlist."""

import ast
from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext

from pydantic import BaseModel, ConfigDict, Field

from app.tools.base import BaseTool, ToolResult

_LIMIT = Decimal("1e12")


class CalculatorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expression: str = Field(min_length=1, max_length=200)


def _bounded(value: Decimal) -> Decimal:
    if not value.is_finite() or abs(value) > _LIMIT:
        raise ValueError("Arithmetic result is outside the allowed range")
    return value


def _evaluate(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        return _bounded(Decimal(str(node.value)))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate(node.operand)
        return _bounded(value if isinstance(node.op, ast.UAdd) else -value)
    if isinstance(node, ast.BinOp):
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Add):
            return _bounded(left + right)
        if isinstance(node.op, ast.Sub):
            return _bounded(left - right)
        if isinstance(node.op, ast.Mult):
            return _bounded(left * right)
        if isinstance(node.op, ast.Div):
            return _bounded(left / right)
    raise ValueError("Only +, -, *, / and numeric literals are allowed")


def calculate(expression: str) -> str:
    if len(expression) > 200:
        raise ValueError("Expression is too long")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("Expression is too complex")
    with localcontext() as context:
        context.prec = 28
        try:
            value = _evaluate(tree.body)
        except (DivisionByZero, InvalidOperation, ZeroDivisionError) as exc:
            raise ValueError("Invalid arithmetic operation") from exc
    return format(value.normalize(), "f")


class CalculatorTool(BaseTool[CalculatorInput]):
    name = "calculator"
    description = "Evaluate simple arithmetic using +, -, *, and /."
    input_type = CalculatorInput

    async def execute(self, arguments: CalculatorInput) -> ToolResult:
        try:
            return ToolResult.ok(calculate(arguments.expression))
        except (SyntaxError, ValueError) as exc:
            return ToolResult.fail("InvalidExpression", str(exc))
