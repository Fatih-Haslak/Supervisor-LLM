import json
from pathlib import Path

import pytest

from app.tools.csv_analysis import CsvSummaryTool
from app.tools.filesystem import Workspace
from app.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_csv_summary_is_decimal_exact_and_workspace_scoped(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "sales.csv").write_text(
        "month,amount\nJan,100.10\nFeb,200.20\n", encoding="utf-8"
    )
    registry = ToolRegistry()
    registry.register(CsvSummaryTool(Workspace(root)))
    result = await registry.execute(
        "csv_summary", {"path": "sales.csv", "column": "amount"}, {"csv_summary"}
    )
    assert result.success and result.output is not None
    assert json.loads(result.output) == {
        "path": "sales.csv", "column": "amount", "count": 2,
        "total": "300.3", "average": "150.15",
        "minimum": "100.1", "maximum": "200.2",
    }
    denied = await registry.execute(
        "csv_summary", {"path": "../secret.csv"}, {"csv_summary"}
    )
    assert denied.error_type == "PermissionDenied"


@pytest.mark.asyncio
async def test_csv_summary_rejects_invalid_number(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "sales.csv").write_text("amount\nNaN\n", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(CsvSummaryTool(Workspace(root)))
    result = await registry.execute(
        "csv_summary", {"path": "sales.csv"}, {"csv_summary"}
    )
    assert result.error_type == "InvalidCSV"
