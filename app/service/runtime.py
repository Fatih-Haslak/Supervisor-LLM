"""One local model instance shared by all queued agent tasks."""

from pathlib import Path

from app.agents.chat import run_chat
from app.agents.reviewer import ReviewerAgent
from app.agents.single import SingleAgent
from app.agents.supervisor import Supervisor
from app.agents.workers import build_workers, worker_descriptions
from app.config.settings import Settings
from app.llm.client import LlamaCppClient
from app.llm.schemas import ChatMessage
from app.llm.strategy import SharedModelStrategy
from app.memory.context import MemoryAwareLLM
from app.memory.store import MemoryEntry, SQLiteMemoryStore
from app.observability.events import agent_span
from app.observability.llm import TracedLLM
from app.orchestration.auto_mode import AutoModeRouter
from app.orchestration.router import Router
from app.orchestration.state import AgentState
from app.security.approvals import Approver
from app.service.tasks import TaskMode
from app.tools.calculator import CalculatorTool
from app.tools.csv_analysis import CsvSummaryTool
from app.tools.filesystem import DirectoryListTool, FileReadTool, FileWriteTool, Workspace
from app.tools.function_test import FunctionTestTool
from app.tools.registry import ToolRegistry
from app.tools.search import SearchTool
from app.tools.wikipedia import WikipediaLookupTool


class AgentRuntime:
    def __init__(
        self, settings: Settings, workspace_root: Path,
        *, client: LlamaCppClient | None = None,
    ) -> None:
        self._settings = settings
        self._workspace = Workspace(workspace_root)
        self._client = client or LlamaCppClient(settings)
        self._memories: list[MemoryEntry] = []

    async def __aenter__(self) -> "AgentRuntime":
        self._memories = await SQLiteMemoryStore(self._settings.memory_db_path).list()
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self._client.__aexit__(*_exc)

    async def run(
        self, message: str, mode: TaskMode, approver: Approver,
        history: list[ChatMessage],
    ) -> AgentState:
        registry = ToolRegistry(approver=approver)
        for tool in (
            CalculatorTool(), CsvSummaryTool(self._workspace),
            FunctionTestTool(self._workspace),
            FileReadTool(self._workspace), FileWriteTool(self._workspace),
            DirectoryListTool(self._workspace), SearchTool(self._workspace),
        ):
            registry.register(tool)
        if self._settings.web_lookup_enabled:
            registry.register(WikipediaLookupTool())
        self._memories = await SQLiteMemoryStore(self._settings.memory_db_path).list()
        strategy = SharedModelStrategy(MemoryAwareLLM(
            TracedLLM(self._client), self._memories
        ))
        if mode == "auto":
            chosen = await AutoModeRouter(strategy.for_role("router")).select(
                message, history
            )
            if chosen == "chat":
                with agent_span("chat"):
                    return await run_chat(strategy.for_role("chat"), message, history)
            mode = chosen
        if mode == "single":
            with agent_span("single"):
                return (await SingleAgent(
                    strategy.for_role("single"), registry,
                    {"calculator", "file_read", "file_write", "directory_list"},
                    role_instructions=(
                        "For every arithmetic expression, call calculator first. "
                        "Use its result in the final answer; do not calculate mentally."
                    ),
                ).run(message, history=history)).state
        workers = build_workers(
            strategy.for_role("general"), registry,
            allow_web=self._settings.web_lookup_enabled,
            model_for_role=strategy.for_role,
        )
        reviewer = ReviewerAgent(strategy.for_role("reviewer"), registry)
        supervisor = Supervisor(
            strategy.for_role("supervisor"), workers, reviewer=reviewer,
            worker_descriptions=worker_descriptions(
                allow_web=self._settings.web_lookup_enabled
            ),
            planner_llm=strategy.for_role("planner"),
        )
        if mode == "supervisor":
            return await supervisor.run(message, history=history)
        if mode == "plan":
            return await supervisor.run_planned(message, history=history)
        if mode == "router":
            return await Router(
                strategy.for_role("router"), workers, supervisor,
                worker_descriptions=worker_descriptions(
                    allow_web=self._settings.web_lookup_enabled
                ),
                review_code_with_supervisor=True,
            ).run(message, history=history)
        from app.orchestration.graph import GraphOrchestrator

        return await GraphOrchestrator(
            strategy.for_role("supervisor"), workers, reviewer=reviewer,
            worker_descriptions=worker_descriptions(
                allow_web=self._settings.web_lookup_enabled
            ),
            planner_llm=strategy.for_role("planner"),
        ).run(message, history=history)
