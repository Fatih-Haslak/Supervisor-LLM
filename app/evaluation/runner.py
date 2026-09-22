"""Exercise the same runtime and routing path as the browser, in isolated workspaces."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from pydantic import TypeAdapter

from app.config.settings import Settings
from app.errors import describe_error, format_cli_error
from app.evaluation.metrics import (
    EvaluationCase,
    EvaluationObservation,
    EvaluationReport,
    evaluate,
)
from app.llm.client import LlamaCppClient
from app.llm.schemas import ChatMessage
from app.observability.events import TraceRecorder, record, trace_session
from app.orchestration.state import AgentState
from app.security.approvals import ApprovalRequest
from app.service.runtime import AgentRuntime
from app.tools.filesystem import Workspace


def load_cases(path: Path) -> list[EvaluationCase]:
    cases = TypeAdapter(list[EvaluationCase]).validate_python(
        json.loads(path.read_text(encoding="utf-8"))
    )
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("Evaluation case IDs must be unique and nonempty")
    return cases


def write_fixtures(root: Path, case: EvaluationCase) -> None:
    workspace = Workspace(root)
    for name, content in case.fixtures.items():
        if len(content.encode("utf-8")) > 100_000:
            raise ValueError("Evaluation fixture exceeds 100 KB")
        target = workspace.resolve(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


class EvaluationApprover:
    """Approve writes only inside a throwaway evaluation workspace."""

    async def request_approval(self, request: ApprovalRequest) -> bool:
        return request.tool == "file_write"


async def run_evaluation(cases: list[EvaluationCase], settings: Settings) -> EvaluationReport:
    if not cases:
        raise ValueError("At least one evaluation case is required")
    observations: list[EvaluationObservation] = []
    async with LlamaCppClient(settings) as client:
        for index, case in enumerate(cases, start=1):
            recorder = TraceRecorder()
            state: AgentState | None = None
            error_code: str | None = None
            files_match = True
            started = perf_counter()
            with TemporaryDirectory(prefix="agent-eval-") as temporary:
                root = Path(temporary) / "workspace"
                root.mkdir()
                isolated = settings.model_copy(update={
                    "memory_db_path": Path(temporary) / "memory.sqlite3"
                })
                runtime = AgentRuntime(isolated, root, client=client)
                history: list[ChatMessage] = []
                with trace_session(recorder):
                    record("task_started")
                    try:
                        write_fixtures(root, case)
                        for turn in [*case.prior_turns, case.task]:
                            state = await runtime.run(
                                turn, case.mode, EvaluationApprover(), history
                            )
                            if not state.final_answer or state.pending_tasks:
                                raise RuntimeError("Evaluation turn did not complete")
                            history.extend([
                                ChatMessage(role="user", content=turn),
                                ChatMessage(role="assistant", content=state.final_answer),
                            ])
                        workspace = Workspace(root)
                        for path, expected in case.expected_file_contains.items():
                            target = workspace.resolve(path)
                            files_match = (files_match and target.is_file()
                                           and expected in target.read_text(encoding="utf-8"))
                        record("task_completed", success=True)
                    except Exception as exc:
                        error_code = describe_error(exc).code
                        partial = getattr(exc, "state", None)
                        if isinstance(partial, AgentState):
                            state = partial
                        record("task_failed", error_type=type(exc).__name__)
            observation = EvaluationObservation(
                case=case, state=state, events=recorder.events,
                latency_ms=round((perf_counter() - started) * 1000, 2),
                error_code=error_code, files_match=files_match,
            )
            observations.append(observation)
            passed = evaluate([observation]).cases[0].success
            print(
                f"[{index}/{len(cases)}] {case.id}: {'PASS' if passed else 'FAIL'}",
                file=sys.stderr, flush=True,
            )
    return evaluate(observations)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate browser-equivalent agent paths")
    parser.add_argument("--cases", type=Path, default=Path("eval/cases.json"))
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    try:
        cases = load_cases(args.cases)
        if args.case_ids:
            cases = [case for case in cases if case.id in args.case_ids]
            if len(cases) != len(set(args.case_ids)):
                raise ValueError("Unknown or duplicate evaluation case ID")
        if not 1 <= args.repeat <= 5:
            raise ValueError("Repeat count must be between 1 and 5")
        if args.repeat > 1:
            cases = [
                case.model_copy(update={"id": f"{case.id}-run{index}"})
                for index in range(1, args.repeat + 1)
                for case in cases
            ]
        report = asyncio.run(run_evaluation(cases, Settings()))
    except Exception as exc:
        print(format_cli_error(exc), file=sys.stderr)
        return 1
    print(report.model_dump_json(indent=2))
    return 0 if all(case.success for case in report.cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
