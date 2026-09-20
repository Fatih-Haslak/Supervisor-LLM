"""Interactive local chat, single-agent, and supervisor CLI."""

import argparse
import asyncio
import sqlite3
import sys
from contextlib import nullcontext
from pathlib import Path

from pydantic import ValidationError

from app.agents.reviewer import ReviewerAgent
from app.agents.single import AgentLimitError, AgentRunResult, SingleAgent
from app.agents.supervisor import Supervisor, SupervisorLimitError
from app.agents.workers import build_workers, worker_descriptions
from app.config.logging import configure_logging
from app.config.settings import Settings
from app.errors import format_cli_error
from app.llm.client import LlamaCppClient, LLMError
from app.llm.schemas import ChatMessage
from app.llm.strategy import SharedModelStrategy
from app.llm.structured import StructuredOutputError
from app.memory.context import MemoryAwareLLM
from app.memory.store import MemoryInput, SQLiteMemoryStore
from app.observability.events import TraceRecorder, agent_span, record, trace_session
from app.observability.llm import TracedLLM
from app.orchestration.router import Router
from app.security.approvals import TerminalApprover, ToolApprovalError
from app.tools.calculator import CalculatorTool
from app.tools.filesystem import DirectoryListTool, FileReadTool, FileWriteTool, Workspace
from app.tools.python_exec import PythonExecTool
from app.tools.registry import ToolRegistry
from app.tools.search import SearchTool


async def run_chat(prompt: str | None, settings: Settings) -> int:
    memories = await SQLiteMemoryStore(settings.memory_db_path).list()
    messages = [
        ChatMessage(
            role="system",
            content="You are a helpful assistant. Reply in the user's language. /no_think",
        )
    ]
    async with LlamaCppClient(settings) as client:
        strategy = SharedModelStrategy(MemoryAwareLLM(client, memories))
        llm = strategy.for_role("chat")
        while True:
            if prompt is None:
                try:
                    user_input = input("Siz> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return 0
                if user_input.lower() in {"exit", "quit", "çık"}:
                    return 0
                if not user_input:
                    continue
            else:
                user_input = prompt
            messages.append(ChatMessage(role="user", content=user_input))
            try:
                result = await llm.chat(messages)
            except LLMError as exc:
                print(format_cli_error(exc), file=sys.stderr)
                return 1
            print(f"Asistan> {result.content}")
            messages.append(ChatMessage(role="assistant", content=result.content))
            if prompt is not None:
                return 0


async def run_agent(
    prompt: str | None,
    settings: Settings,
    allow_python: bool,
    show_tools: bool,
    supervisor: bool = False,
    plan: bool = False,
    router: bool = False,
    graph: bool = False,
    trace: bool = False,
) -> int:
    memories = await SQLiteMemoryStore(settings.memory_db_path).list()
    workspace = Workspace(Path("workspace"))
    registry = ToolRegistry(approver=TerminalApprover() if sys.stdin.isatty() else None)
    registry.register(CalculatorTool())
    registry.register(FileReadTool(workspace))
    registry.register(FileWriteTool(workspace))
    registry.register(DirectoryListTool(workspace))
    registry.register(SearchTool(workspace))
    allowed = {"calculator", "file_read", "file_write", "directory_list"}
    if allow_python:
        registry.register(PythonExecTool(workspace))
        allowed.add("python_exec")

    async with LlamaCppClient(settings) as client:
        strategy = SharedModelStrategy(MemoryAwareLLM(TracedLLM(client), memories))
        agent = SingleAgent(strategy.for_role("single"), registry, allowed)
        workers = build_workers(
            strategy.for_role("general"), registry, allow_python=allow_python,
            model_for_role=strategy.for_role,
        )
        manager = (
            Supervisor(
                strategy.for_role("supervisor"),
                workers,
                worker_descriptions=worker_descriptions(),
                reviewer=ReviewerAgent(strategy.for_role("reviewer"), registry),
                planner_llm=strategy.for_role("planner"),
            )
            if supervisor or router
            else None
        )
        graph_engine = None
        if graph:
            from app.orchestration.graph import GraphOrchestrator

            graph_engine = GraphOrchestrator(
                strategy.for_role("supervisor"), workers,
                worker_descriptions=worker_descriptions(),
                reviewer=ReviewerAgent(strategy.for_role("reviewer"), registry),
                planner_llm=strategy.for_role("planner"),
            )
        route_engine = (
            Router(
                strategy.for_role("router"), workers, manager,
                worker_descriptions=worker_descriptions(),
                review_code_with_supervisor=True,
            )
            if router and manager is not None
            else None
        )
        while True:
            if prompt is None:
                try:
                    user_input = input("Görev> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return 0
                if user_input.lower() in {"exit", "quit", "çık"}:
                    return 0
                if not user_input:
                    continue
            else:
                user_input = prompt
            recorder = TraceRecorder() if trace else None
            context = trace_session(recorder) if recorder is not None else nullcontext()
            with context:
                record("task_started")
                try:
                    if graph_engine is not None:
                        result = AgentRunResult(state=await graph_engine.run(user_input))
                    elif route_engine is not None:
                        result = AgentRunResult(state=await route_engine.run(user_input))
                    elif manager is not None:
                        state = (
                            await manager.run_planned(user_input)
                            if plan
                            else await manager.run(user_input)
                        )
                        result = AgentRunResult(state=state)
                    else:
                        with agent_span("single"):
                            result = await agent.run(user_input)
                    record("task_completed", success=True)
                except (AgentLimitError, SupervisorLimitError, ToolApprovalError, LLMError,
                        StructuredOutputError) as exc:
                    record("task_failed", error_type=type(exc).__name__)
                    print(format_cli_error(exc), file=sys.stderr)
                    if prompt is not None:
                        return 1
                    continue
                finally:
                    if recorder is not None:
                        for event in recorder.events:
                            print("Trace> " + event.model_dump_json(exclude_none=True),
                                  file=sys.stderr)
            if show_tools:
                if result.state.route is not None:
                    route = result.state.route
                    print(
                        f"Rota> {route.suggested_agent} "
                        f"(güven {route.confidence:.2f}) -> {route.selected_agent}"
                    )
                if manager is not None or graph_engine is not None:
                    if result.state.plan is not None:
                        planned = " -> ".join(
                            f"{task.id}:{task.agent}" for task in result.state.plan.tasks
                        )
                        print(f"Plan> {planned}")
                    worker_names = " -> ".join(
                        output.agent for output in result.state.agent_outputs
                    )
                    if graph_engine is not None:
                        print(f"Agent akışı> graph: supervisor -> {worker_names} -> supervisor")
                    elif (
                        result.state.route is None
                        or result.state.route.selected_agent == "supervisor"
                    ):
                        print(f"Agent akışı> supervisor -> {worker_names} -> supervisor")
                    else:
                        print(f"Agent akışı> router -> {worker_names}")
                names = " -> ".join(call.tool for call in result.tool_calls) or "yok"
                print(f"Araç çağrıları> {names}")
                for review in result.state.reviews:
                    issues = "; ".join(review.issues)
                    print(
                        f"İnceleme> {review.agent} #{review.attempt}: "
                        f"{review.status.upper()}" + (f" — {issues}" if issues else "")
                    )
            print(f"Asistan> {result.answer}")
            if prompt is not None:
                return 0


async def run_memory_command(args: argparse.Namespace, settings: Settings) -> int:
    try:
        store = SQLiteMemoryStore(settings.memory_db_path)
        if args.memory_save is not None:
            key, value = args.memory_save
            item = MemoryInput(key=key, value=value, category=args.memory_category)
            saved = await store.save(item)
            print(f"Bellek kaydedildi> {saved.key} ({saved.category})")
        elif args.memory_list:
            entries = await store.list(limit=100)
            for entry in entries:
                print(f"{entry.key} [{entry.category}]: {entry.value}")
            if not entries:
                print("Bellek boş")
        elif args.memory_delete is not None:
            deleted = await store.delete(args.memory_delete)
            print("Bellek silindi" if deleted else "Bellek kaydı bulunamadı")
        return 0
    except ValidationError:
        print("Hata [INVALID_INPUT]: Geçersiz bellek kaydı veya gizli bilgi.", file=sys.stderr)
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(format_cli_error(exc), file=sys.stderr)
    return 1


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Local LLM agent CLI")
    parser.add_argument("--prompt", help="Send one prompt and exit")
    parser.add_argument("--agent", action="store_true", help="Use the single tool-calling agent")
    parser.add_argument("--supervisor", action="store_true", help="Route through the supervisor")
    parser.add_argument("--plan", action="store_true", help="Plan subtasks before delegation")
    parser.add_argument("--router", action="store_true", help="Route simple tasks directly")
    parser.add_argument("--graph", action="store_true", help="Use LangGraph orchestration")
    parser.add_argument("--allow-python", action="store_true", help="Enable restricted Python tool")
    parser.add_argument("--show-tools", action="store_true", help="Show tool names after each task")
    parser.add_argument("--trace", action="store_true", help="Print safe JSON trace events")
    memory_group = parser.add_mutually_exclusive_group()
    memory_group.add_argument("--memory-save", nargs=2, metavar=("KEY", "VALUE"))
    memory_group.add_argument("--memory-list", action="store_true")
    memory_group.add_argument("--memory-delete", metavar="KEY")
    parser.add_argument(
        "--memory-category", choices=("preference", "decision", "fact"),
        default="preference", help="Category for --memory-save",
    )
    args = parser.parse_args()
    memory_command = (
        args.memory_save is not None or args.memory_list or args.memory_delete is not None
    )
    if memory_command and (
        args.prompt is not None or args.agent or args.supervisor or args.router or args.graph
        or args.plan or args.allow_python or args.show_tools or args.trace
    ):
        parser.error("Memory commands cannot be combined with chat or agent options")
    if args.memory_category != "preference" and args.memory_save is None:
        parser.error("--memory-category requires --memory-save")
    if (
        args.allow_python or args.show_tools or args.supervisor or args.router or args.graph
        or args.trace
    ) and not args.agent:
        parser.error("Agent options require --agent")
    if args.plan and not args.supervisor:
        parser.error("--plan requires --supervisor")
    if args.router and args.supervisor:
        parser.error("--router and --supervisor are separate modes")
    if args.graph and (args.router or args.supervisor or args.plan):
        parser.error("--graph cannot be combined with --router, --supervisor, or --plan")
    try:
        settings = Settings()
        configure_logging(settings.log_level)
        if memory_command:
            return asyncio.run(run_memory_command(args, settings))
        if args.agent:
            return asyncio.run(
                run_agent(
                    args.prompt, settings, args.allow_python, args.show_tools,
                    args.supervisor, args.plan, args.router, args.graph, args.trace,
                )
            )
        return asyncio.run(run_chat(args.prompt, settings))
    except Exception as exc:
        print(format_cli_error(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
