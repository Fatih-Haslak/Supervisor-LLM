"""Specialized worker policies sharing one LLM and one tool registry."""

from collections.abc import Callable
from dataclasses import dataclass

from app.agents.single import SingleAgent
from app.agents.supervisor import SingleAgentWorker, Worker
from app.llm.client import LLMClient
from app.llm.strategy import ModelRole
from app.tools.registry import ToolRegistry


@dataclass(frozen=True)
class WorkerPolicy:
    description: str
    instructions: str
    allowed_tools: frozenset[str]


WORKER_POLICIES: dict[ModelRole, WorkerPolicy] = {
    "general": WorkerPolicy(
        description="Simple questions, calculations, and mixed workspace tasks.",
        instructions="Use calculator for arithmetic. Work only inside workspace.",
        allowed_tools=frozenset({"calculator", "file_read", "file_write", "directory_list"}),
    ),
    "researcher": WorkerPolicy(
        description="Find and summarize facts in local workspace documents; no internet access.",
        instructions=(
            "Search local workspace documents, then read relevant files. Give file paths "
            "for claims. You have no web search; do not present unverified external facts "
            "as researched findings. If search returns [], report that no local match was "
            "found. Do not keep searching unrelated words."
        ),
        allowed_tools=frozenset({"search", "file_read"}),
    ),
    "coder": WorkerPolicy(
        description="Inspect, write, and revise Python code inside workspace.",
        instructions=(
            "Inspect the target files before editing. Explain the actual changes. "
            "Do not claim tests were run unless python_exec returned a result."
        ),
        allowed_tools=frozenset({"file_read", "file_write", "directory_list"}),
    ),
    "file_agent": WorkerPolicy(
        description="Find, read, and write ordinary workspace files; not code implementation.",
        instructions=(
            "Manage ordinary workspace files. Preserve requested content and paths exactly. "
            "For code implementation, tell supervisor that coder is the appropriate role."
        ),
        allowed_tools=frozenset({"file_read", "file_write", "directory_list"}),
    ),
}


def build_workers(
    llm: LLMClient, registry: ToolRegistry, *, allow_python: bool = False,
    model_for_role: Callable[[ModelRole], LLMClient] | None = None,
) -> dict[str, Worker]:
    workers: dict[str, Worker] = {}
    for name, policy in WORKER_POLICIES.items():
        allowed = set(policy.allowed_tools)
        if allow_python and name in {"general", "coder"}:
            allowed.add("python_exec")
        workers[name] = SingleAgentWorker(
            SingleAgent(
                model_for_role(name) if model_for_role is not None else llm,
                registry,
                allowed,
                role_name=name,
                role_instructions=policy.instructions,
            )
        )
    return workers


def worker_descriptions() -> dict[str, str]:
    return {name: policy.description for name, policy in WORKER_POLICIES.items()}
