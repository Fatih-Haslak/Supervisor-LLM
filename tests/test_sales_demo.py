import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from app.agents.reviewer import ReviewerAgent, _report_metric_issues
from app.agents.supervisor import Supervisor, WorkerResult
from app.agents.workers import build_workers
from app.llm.schemas import ChatMessage, LLMResponse
from app.orchestration.state import ToolCallRecord
from app.tools.base import ToolResult
from app.tools.csv_analysis import CsvSummary, CsvSummaryTool
from app.tools.filesystem import FileReadTool, FileWriteTool, Workspace
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


def registry_for(root: Path) -> ToolRegistry:
    registry = ToolRegistry(approver=ApproveWrites())
    workspace = Workspace(root)
    registry.register(CsvSummaryTool(workspace))
    registry.register(FileReadTool(workspace))
    registry.register(FileWriteTool(workspace))
    return registry


REPORT = (
    "# Satış raporu\n\nİki satışın toplamı 300.\n\n"
    "| Ölçüt | Değer |\n| --- | --- |\n"
    "| Satır sayısı | 2 |\n| Toplam | 300 |\n| Ortalama | 150 |\n"
    "| En düşük | 100 |\n| En yüksek | 200 |\n"
)


def test_horizontal_markdown_table_is_checked_against_csv() -> None:
    summary = CsvSummary(
        path="sales.csv", column="amount", count=2, total="300",
        average="150", minimum="100", maximum="200",
    )
    report = (
        "|Satır sayısı|Toplam|Ortalama|En düşük|En yüksek|\n"
        "|---|---|---|---|---|\n"
        "|2|300|150|100|200|"
    )
    assert _report_metric_issues(report, summary) == []
    assert any("Toplam" in issue for issue in _report_metric_issues(
        report.replace("|2|300|", "|2|999|"), summary
    ))


@pytest.mark.asyncio
async def test_sales_demo_data_writer_and_deterministic_review(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "sales.csv").write_text("amount\n100\n200\n", encoding="utf-8")
    registry = registry_for(root)
    llm = ScriptedLLM([
        '{"tasks":['
        '{"id":1,"agent":"data_agent","task":"sales.csv amount sütununu analiz et",'
        '"depends_on":[]},'
        '{"id":2,"agent":"writer","task":"sales-report.md raporunu yaz",'
        '"depends_on":[1]}]}',
        '{"action":"use_tool","tool":"csv_summary",'
        '"arguments":{"path":"sales.csv","column":"amount"}}',
        '{"action":"final_answer","answer":'
        '"sales.csv amount: count 2 total 300 average 150 minimum 100 maximum 200"}',
        '{"action":"use_tool","tool":"file_write","arguments":'
        '{"path":"sales-report.md","content":' + json.dumps(REPORT) + '}}',
        '{"action":"final_answer","answer":"sales-report.md yazıldı"}',
        '{"status":"pass","issues":[]}',
        '{"action":"final_answer","answer":"Rapor doğrulandı"}',
    ])
    state = await Supervisor(
        llm, build_workers(llm, registry), reviewer=ReviewerAgent(llm, registry)
    ).run_planned("sales.csv analiz et, sales-report.md yaz ve doğrula")
    assert state.final_answer == "Rapor doğrulandı"
    assert [output.agent for output in state.agent_outputs] == ["data_agent", "writer"]
    assert [review.status for review in state.reviews] == ["pass"]
    assert state.reviews[0].agent == "writer"
    assert (root / "sales-report.md").read_text(encoding="utf-8") == REPORT
    assert [call.tool for call in state.tool_results].count("csv_summary") == 2


@pytest.mark.asyncio
async def test_reviewer_rejects_wrong_report_total_before_model_call(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "sales.csv").write_text("amount\n100\n200\n", encoding="utf-8")
    (root / "sales-report.md").write_text(
        REPORT.replace("| Toplam | 300 |", "| Toplam | 999 |"), encoding="utf-8"
    )
    registry = registry_for(root)
    summary = await registry.execute(
        "csv_summary", {"path": "sales.csv", "column": "amount"}, {"csv_summary"}
    )
    evidence = [
        ToolCallRecord(tool="csv_summary", arguments={"path": "sales.csv", "column": "amount"},
                       result=summary),
        ToolCallRecord(tool="file_write", arguments={"path": "sales-report.md"},
                       result=ToolResult.ok("written")),
    ]
    review = await ReviewerAgent(ScriptedLLM([]), registry).review(
        "Rapor yaz", "sales-report.md yaz", WorkerResult(answer="Bitti", tool_results=evidence),
        evidence_tools=evidence,
    )
    assert review.verdict.status == "fail"
    assert any("Toplam" in issue for issue in review.verdict.issues)
