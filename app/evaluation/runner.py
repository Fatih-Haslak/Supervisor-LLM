"""Run a small local benchmark through the same agents and shared GGUF backend."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from pydantic import TypeAdapter

from app.agents.reviewer import ReviewerAgent
from app.agents.single import SingleAgent
from app.agents.supervisor import Supervisor
from app.agents.workers import build_workers, worker_descriptions
from app.config.settings import Settings
from app.errors import describe_error, format_cli_error
from app.evaluation.metrics import (
    EvaluationCase,
    EvaluationObservation,
    EvaluationReport,
    evaluate,
)
from app.llm.client import LlamaCppClient
from app.llm.strategy import SharedModelStrategy
from app.memory.context import MemoryAwareLLM
from app.observability.events import TraceRecorder, record, trace_session
from app.observability.llm import TracedLLM
from app.orchestration.state import AgentState
from app.tools.calculator import CalculatorTool
from app.tools.csv_analysis import CsvSummaryTool
from app.tools.filesystem import DirectoryListTool, FileReadTool, FileWriteTool, Workspace
from app.tools.function_test import FunctionTestTool
from app.tools.registry import ToolRegistry
from app.tools.search import SearchTool


def load_cases(path: Path) -> list[EvaluationCase]:
    cases = TypeAdapter(list[EvaluationCase]).validate_python(
        json.loads(path.read_text(encoding="utf-8"))
    )
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("Evaluation case IDs must be unique and nonempty")
    return cases


def make_registry(root: Path) -> ToolRegistry:
    workspace = Workspace(root)
    registry = ToolRegistry()
    for tool in (
        CalculatorTool(), CsvSummaryTool(workspace), FunctionTestTool(workspace),
        FileReadTool(workspace), FileWriteTool(workspace),
        DirectoryListTool(workspace), SearchTool(workspace),
    ):
        registry.register(tool)
    return registry


def write_fixtures(root: Path, case: EvaluationCase) -> None:
    workspace = Workspace(root)
    for name, content in case.fixtures.items():
        if len(content.encode("utf-8")) > 100_000:
            raise ValueError("Evaluation fixture exceeds 100 KB")
        target = workspace.resolve(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


async def run_evaluation(cases: list[EvaluationCase], settings: Settings) -> EvaluationReport:
    if not cases:
        raise ValueError("At least one evaluation case is required")
    observations: list[EvaluationObservation] = []
    async with LlamaCppClient(settings) as client:
        for case in cases:
            recorder = TraceRecorder()
            state: AgentState | None = None
            error_code: str | None = None
            started = perf_counter()
            with trace_session(recorder):
                record("task_started")
                llm = MemoryAwareLLM(TracedLLM(client), [])
                strategy = SharedModelStrategy(llm)
                with TemporaryDirectory(prefix="agent-eval-") as temporary:
                    root = Path(temporary) / "workspace"
                    root.mkdir()
                    try:
                        write_fixtures(root, case)
                        registry = make_registry(root)
                        if case.mode == "single":
                            state = (await SingleAgent(
                                strategy.for_role("single"), registry,
                                {"calculator", "file_read", "file_write", "directory_list"},
                            ).run(case.task)).state
                        else:
                            workers = build_workers(
                                strategy.for_role("general"), registry,
                                model_for_role=strategy.for_role,
                            )
                            reviewer = ReviewerAgent(strategy.for_role("reviewer"), registry)
                            if case.mode == "graph":
                                from app.orchestration.graph import GraphOrchestrator

                                state = await GraphOrchestrator(
                                    strategy.for_role("supervisor"), workers, reviewer=reviewer,
                                    worker_descriptions=worker_descriptions(),
                                    planner_llm=strategy.for_role("planner"),
                                ).run(case.task)
                            else:
                                state = await Supervisor(
                                    strategy.for_role("supervisor"), workers, reviewer=reviewer,
                                    worker_descriptions=worker_descriptions(),
                                    planner_llm=strategy.for_role("planner"),
                                ).run(case.task)
                        record("task_completed", success=True)
                    except Exception as exc:
                        error_code = describe_error(exc).code
                        record("task_failed", error_type=type(exc).__name__)
            observations.append(
                EvaluationObservation(
                    case=case, state=state, events=recorder.events,
                    latency_ms=round((perf_counter() - started) * 1000, 2),
                    error_code=error_code,
                )
            )
    return evaluate(observations)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the local agent on a case set")
    parser.add_argument("--cases", type=Path, default=Path("eval/cases.json"))
    args = parser.parse_args()
    try:
        report = asyncio.run(run_evaluation(load_cases(args.cases), Settings()))
    except Exception as exc:
        print(format_cli_error(exc), file=sys.stderr)
        return 1
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
