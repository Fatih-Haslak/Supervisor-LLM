"""Deterministic, workspace-only summary for numeric CSV columns."""

import asyncio
import csv
import json
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.tools.base import BaseTool, ToolResult
from app.tools.filesystem import Workspace, WorkspaceAccessError

_MAX_BYTES = 1_000_000
_MAX_ROWS = 10_000


class CsvSummaryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=260)
    column: str = Field(default="amount", min_length=1, max_length=100)


class CsvSummary(BaseModel):
    path: str
    column: str
    count: int
    total: str
    average: str
    minimum: str
    maximum: str


def _format_number(value: Decimal) -> str:
    return format(value.normalize(), "f")


def summarize_csv(path: Path, column: str, display_path: str) -> CsvSummary:
    values: list[Decimal] = []
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or column not in reader.fieldnames:
            raise ValueError("Requested numeric column was not found")
        for row_number, row in enumerate(reader, start=2):
            if len(values) >= _MAX_ROWS:
                raise ValueError("CSV exceeds 10000 data rows")
            raw = row.get(column)
            if raw is None:
                raise ValueError(f"Missing numeric value at row {row_number}")
            try:
                value = Decimal(raw.strip())
            except InvalidOperation as exc:
                raise ValueError(f"Invalid numeric value at row {row_number}") from exc
            if not value.is_finite():
                raise ValueError(f"Non-finite numeric value at row {row_number}")
            if abs(value.adjusted()) > 12 or len(value.as_tuple().digits) > 30:
                raise ValueError(f"Numeric value is too large at row {row_number}")
            values.append(value)
    if not values:
        raise ValueError("CSV has no data rows")
    total = sum(values, Decimal(0))
    average = (total / len(values)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return CsvSummary(
        path=display_path, column=column, count=len(values),
        total=_format_number(total), average=_format_number(average),
        minimum=_format_number(min(values)), maximum=_format_number(max(values)),
    )


class CsvSummaryTool(BaseTool[CsvSummaryInput]):
    name = "csv_summary"
    description = (
        "Summarize one numeric CSV column inside workspace: count, total, average, "
        "minimum, maximum. Use exact path and column name."
    )
    input_type = CsvSummaryInput

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    async def execute(self, arguments: CsvSummaryInput) -> ToolResult:
        try:
            target = self._workspace.resolve(arguments.path)
            if not target.is_file():
                return ToolResult.fail("FileNotFound", "Workspace CSV file not found")
            if target.stat().st_size > _MAX_BYTES:
                return ToolResult.fail("FileTooLarge", "Workspace CSV exceeds 1 MB")
            summary = await asyncio.to_thread(
                summarize_csv, target, arguments.column, arguments.path
            )
            return ToolResult.ok(json.dumps(summary.model_dump(), ensure_ascii=False))
        except WorkspaceAccessError as exc:
            return ToolResult.fail("PermissionDenied", str(exc))
        except (OSError, UnicodeError):
            return ToolResult.fail("FileError", "Could not read workspace CSV")
        except (csv.Error, InvalidOperation, ValueError) as exc:
            return ToolResult.fail("InvalidCSV", str(exc))
