"""Interactive local chat, single-agent, and supervisor CLI."""

import argparse
import asyncio
import sys
from pathlib import Path

from app.agents.single import AgentLimitError, AgentRunResult, SingleAgent
from app.agents.supervisor import Supervisor, SupervisorLimitError
from app.agents.workers import build_workers, worker_descriptions
from app.config.logging import configure_logging
from app.config.settings import Settings
from app.llm.client import LlamaCppClient, LLMError
from app.llm.schemas import ChatMessage
from app.llm.structured import StructuredOutputError
from app.tools.calculator import CalculatorTool
from app.tools.filesystem import DirectoryListTool, FileReadTool, FileWriteTool, Workspace
from app.tools.python_exec import PythonExecTool
from app.tools.registry import ToolRegistry
from app.tools.search import SearchTool


async def run_chat(prompt: str | None, settings: Settings) -> int:
    messages = [
        ChatMessage(
            role="system",
            content="You are a helpful assistant. Reply in the user's language. /no_think",
        )
    ]
    async with LlamaCppClient(settings) as client:
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
                result = await client.chat(messages)
            except LLMError as exc:
                print(f"Hata: {exc}", file=sys.stderr)
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
) -> int:
    workspace = Workspace(Path("workspace"))
    registry = ToolRegistry()
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
        agent = SingleAgent(client, registry, allowed)
        manager = (
            Supervisor(
                client,
                build_workers(client, registry, allow_python=allow_python),
                worker_descriptions=worker_descriptions(),
            )
            if supervisor
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
            try:
                if manager is not None:
                    state = (
                        await manager.run_planned(user_input)
                        if plan
                        else await manager.run(user_input)
                    )
                    result = AgentRunResult(state=state)
                else:
                    result = await agent.run(user_input)
            except (AgentLimitError, SupervisorLimitError, LLMError, StructuredOutputError) as exc:
                print(f"Hata: {exc}", file=sys.stderr)
                if prompt is not None:
                    return 1
                continue
            if show_tools:
                if manager is not None:
                    if result.state.plan is not None:
                        planned = " -> ".join(
                            f"{task.id}:{task.agent}" for task in result.state.plan.tasks
                        )
                        print(f"Plan> {planned}")
                    workers = " -> ".join(output.agent for output in result.state.agent_outputs)
                    print(f"Agent akışı> supervisor -> {workers} -> supervisor")
                names = " -> ".join(call.tool for call in result.tool_calls) or "yok"
                print(f"Araç çağrıları> {names}")
            print(f"Asistan> {result.answer}")
            if prompt is not None:
                return 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Local LLM agent CLI")
    parser.add_argument("--prompt", help="Send one prompt and exit")
    parser.add_argument("--agent", action="store_true", help="Use the single tool-calling agent")
    parser.add_argument("--supervisor", action="store_true", help="Route through the supervisor")
    parser.add_argument("--plan", action="store_true", help="Plan subtasks before delegation")
    parser.add_argument("--allow-python", action="store_true", help="Enable restricted Python tool")
    parser.add_argument("--show-tools", action="store_true", help="Show tool names after each task")
    args = parser.parse_args()
    if (args.allow_python or args.show_tools or args.supervisor) and not args.agent:
        parser.error("--allow-python, --show-tools and --supervisor require --agent")
    if args.plan and not args.supervisor:
        parser.error("--plan requires --supervisor")
    settings = Settings()
    configure_logging(settings.log_level)
    if args.agent:
        return asyncio.run(
            run_agent(
                args.prompt, settings, args.allow_python, args.show_tools,
                args.supervisor, args.plan,
            )
        )
    return asyncio.run(run_chat(args.prompt, settings))


if __name__ == "__main__":
    raise SystemExit(main())
