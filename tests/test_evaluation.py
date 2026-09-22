"""Evaluation distinguishes answers, routing, tools, failures, and measured cost."""

from pathlib import Path

import pytest

from app.evaluation.metrics import EvaluationCase, EvaluationObservation, evaluate
from app.evaluation.runner import load_cases, write_fixtures
from app.observability.events import TraceRecorder
from app.orchestration.state import AgentOutput, AgentState, ReviewRecord, ToolCallRecord
from app.tools.base import ToolResult


def sample_case(case_id: str, *, agent: str | None = None) -> EvaluationCase:
    return EvaluationCase(
        id=case_id, task="3+44", expected_tools=["calculator"],
        expected_agent=agent, expected_answer_contains="47",
    )


def test_evaluation_report_computes_success_and_cost() -> None:
    success = AgentState(user_request="3+44", step_count=2)
    success.agent_outputs.append(AgentOutput(agent="general", task="hesapla", answer="47"))
    success.tool_results.append(
        ToolCallRecord(
            tool="calculator", arguments={"expression": "3+44"}, result=ToolResult.ok("47")
        )
    )
    success.reviews.append(
        ReviewRecord(agent="coder", task="x", attempt=1, status="pass")
    )
    success.finish("Sonuç 47")
    recorder = TraceRecorder()
    recorder.record("model_call", prompt_tokens=10, completion_tokens=2)
    recorder.record("model_retry", retry_count=1)
    failed = AgentState(user_request="3+44", step_count=1)
    failed.tool_results.append(
        ToolCallRecord(
            tool="file_read", arguments={"path": "missing.txt"},
            result=ToolResult.fail("FileNotFound", "Missing"),
        )
    )
    failed.fail("Dosya yok")
    report = evaluate([
        EvaluationObservation(
            case=sample_case("good", agent="general"), state=success,
            events=recorder.events, latency_ms=100,
        ),
        EvaluationObservation(
            case=sample_case("bad", agent="general"), state=failed, latency_ms=300,
        ),
    ])
    assert report.case_count == 2
    assert report.task_success_rate == 0.5
    assert report.tool_selection_accuracy == 0.5
    assert report.agent_routing_accuracy == 0.5
    assert report.average_steps == 1.5
    assert report.tool_error_rate == 0.5
    assert report.average_latency_ms == 200
    assert report.prompt_tokens == 10 and report.completion_tokens == 2
    assert report.reviewer_pass_rate == 1
    assert report.retry_rate == 0.5


def test_evaluation_case_set_is_valid() -> None:
    cases = load_cases(Path("eval/cases.json"))
    assert len(cases) >= 18
    assert all(case.expected_answer_contains for case in cases)
    assert sum(case.mode == "auto" for case in cases) >= 12
    assert {"single", "plan", "router", "graph"}.issubset({case.mode for case in cases})
    assert any(case.prior_turns for case in cases)
    assert any(case.expected_file_contains for case in cases)


def test_evaluation_allows_review_reads_but_requires_written_artifact() -> None:
    case = EvaluationCase(
        id="write-report", task="Rapor yaz", expected_tools=["file_write"],
        expected_answer_contains="rapor", expected_file_contains={"report.md": "500"},
    )
    state = AgentState(user_request=case.task)
    state.tool_results.extend([
        ToolCallRecord(tool="file_write", arguments={}, result=ToolResult.ok("ok")),
        ToolCallRecord(tool="file_read", arguments={}, result=ToolResult.ok("500")),
    ])
    state.finish("rapor yazıldı")
    good = evaluate([EvaluationObservation(case=case, state=state, latency_ms=1)])
    missing = evaluate([
        EvaluationObservation(case=case, state=state, latency_ms=1, files_match=False)
    ])
    assert good.cases[0].success
    assert not missing.cases[0].success


def test_evaluation_rejects_english_answer_to_turkish_case() -> None:
    case = EvaluationCase(
        id="language-check", task="Hatayı düzelt", expected_answer_contains="47"
    )
    state = AgentState(user_request=case.task)
    state.finish("The bug has been fixed; result 47.")
    score = evaluate([EvaluationObservation(case=case, state=state, latency_ms=1)]).cases[0]
    assert not score.success
    assert not score.language_match


def test_duplicate_case_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        '[{"id":"same","task":"a","expected_answer_contains":"x"},'
        '{"id":"same","task":"b","expected_answer_contains":"y"}]',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique"):
        load_cases(path)


def test_fixture_paths_stay_in_temporary_workspace(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    valid = EvaluationCase(
        id="fixture-valid", task="read", expected_answer_contains="ok",
        fixtures={"data/numbers.txt": "10\n20\n"},
    )
    write_fixtures(root, valid)
    assert (root / "data/numbers.txt").read_text(encoding="utf-8") == "10\n20\n"
    invalid = valid.model_copy(update={"fixtures": {"../outside.txt": "bad"}})
    with pytest.raises(ValueError):
        write_fixtures(root, invalid)
    assert not (tmp_path / "outside.txt").exists()
