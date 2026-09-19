"""Restricted child process for the optional Python tool.

This is deliberately a small Python subset, not a general Python interpreter.
"""

import ast
import sys

_MAX_CODE_LENGTH = 4000
_MAX_OUTPUT_CHARS = 16000
_ALLOWED_NODES = (
    ast.Module,
    ast.Expr,
    ast.Assign,
    ast.AugAssign,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.UAdd,
    ast.USub,
    ast.Call,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Subscript,
    ast.Slice,
    ast.For,
    ast.If,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Pass,
    ast.Break,
    ast.Continue,
)


class OutputLimitReached(Exception):
    pass


_written = 0


def safe_print(*values: object) -> None:
    global _written
    rendered = " ".join(str(value) for value in values) + "\n"
    remaining = _MAX_OUTPUT_CHARS - _written
    if len(rendered) > remaining:
        sys.stdout.write(rendered[:remaining] + "\n[output limit reached]\n")
        raise OutputLimitReached
    sys.stdout.write(rendered)
    _written += len(rendered)


_SAFE_CALLS = {
    "print": safe_print,
    "abs": abs,
    "int": int,
    "float": float,
    "str": str,
    "len": len,
    "sum": sum,
    "min": min,
    "max": max,
    "round": round,
    "range": range,
    "sorted": sorted,
}


def validate_code(code: str) -> ast.Module:
    if len(code) > _MAX_CODE_LENGTH:
        raise ValueError("Code exceeds 4000 characters")
    tree = ast.parse(code, mode="exec")
    nodes = list(ast.walk(tree))
    if len(nodes) > 200:
        raise ValueError("Code is too complex")
    for node in nodes:
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Unsupported Python syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError("Private names are not allowed")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_CALLS:
                raise ValueError("Only approved functions can be called")
            if node.keywords:
                raise ValueError("Keyword arguments are not allowed")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str) and len(node.value) > 1000:
                raise ValueError("String literal is too long")
            if type(node.value) is int and abs(node.value) > 1_000_000_000:
                raise ValueError("Integer literal is too large")
    return tree


def main() -> int:
    code = sys.stdin.read(_MAX_CODE_LENGTH + 1)
    try:
        tree = validate_code(code)
        exec(compile(tree, "<workspace-python>", "exec"), {"__builtins__": _SAFE_CALLS})
        return 0
    except OutputLimitReached:
        return 0
    except Exception as exc:
        sys.stderr.write(f"{type(exc).__name__}: {exc}"[:2000])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
