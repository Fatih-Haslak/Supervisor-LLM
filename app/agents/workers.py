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
        description="Search local documents or look up public facts on Turkish Wikipedia.",
        instructions=(
            "For a public person or encyclopedic question, call wikipedia_lookup with "
            "only the public topic title. Answer in 2-4 short Turkish sentences; "
            "cite its returned URL and keep facts within its extract. "
            "For workspace questions, search local documents and then read a "
            "matching file with file_read. Preserve exact codes and names. If no source "
            "is available, say that you cannot verify the answer; never claim a person "
            "does not exist. Do not send private workspace content to Wikipedia."
        ),
        allowed_tools=frozenset({"search", "file_read", "wikipedia_lookup"}),
    ),
    "coder": WorkerPolicy(
        description="Inspect, revise, and test small Python functions inside workspace.",
        instructions=(
            "When the original request contains inline Python code and the assigned task "
            "says to analyze it directly, analyze that code without file or test tools. "
            "Inspect the target code and JSON test cases before editing. "
            "After the change, call function_test with the code and test paths. "
            "Report exact pass/fail counts. The function_test tool supports only "
            "one pure arithmetic function; state its limitation honestly."
        ),
        allowed_tools=frozenset({
            "file_read", "file_write", "directory_list", "function_test"
        }),
    ),
    "file_agent": WorkerPolicy(
        description="Find, read, and write ordinary workspace files; not code implementation.",
        instructions=(
            "Manage ordinary workspace files. Preserve requested content and paths exactly. "
            "For code implementation, tell supervisor that coder is the appropriate role."
        ),
        allowed_tools=frozenset({"file_read", "file_write", "directory_list"}),
    ),
    "data_agent": WorkerPolicy(
        description="Analyze numeric data in local CSV files with deterministic tools.",
        instructions=(
            "For CSV analysis, call csv_summary with the exact workspace path and numeric "
            "column. Preserve every returned count, total, average, minimum and maximum "
            "exactly in your answer, including the source path. Never invent metrics."
        ),
        allowed_tools=frozenset({"file_read", "csv_summary"}),
    ),
    "writer": WorkerPolicy(
        description="Write Markdown reports from completed analysis inside workspace.",
        instructions=(
            "file_write ile istenen Markdown raporunu yaz. Kısa bir bulgular paragrafı "
            "ve sayısal tablo ekle. Tablo örneği: | Ölçüt | Değer |, "
            "| Satır sayısı | 4 |, | Toplam | 500 |, | Ortalama | 125 |, "
            "| En düşük | 80 |, | En yüksek | 200 |. Örnekteki rakamları kopyalama; "
            "data_agent sonuçlarını birebir kullan. Yeni rakam uydurma."
        ),
        allowed_tools=frozenset({"file_read", "file_write"}),
    ),
}


def build_workers(
    llm: LLMClient, registry: ToolRegistry, *, allow_python: bool = False,
    allow_web: bool = False,
    model_for_role: Callable[[ModelRole], LLMClient] | None = None,
) -> dict[str, Worker]:
    workers: dict[str, Worker] = {}
    for name, policy in WORKER_POLICIES.items():
        allowed = set(policy.allowed_tools)
        instructions = policy.instructions
        if name == "researcher" and not allow_web:
            allowed.discard("wikipedia_lookup")
            instructions += " Wikipedia lookup is unavailable; do not invent public facts."
        if allow_python and name in {"general", "coder"}:
            allowed.add("python_exec")
        workers[name] = SingleAgentWorker(
            SingleAgent(
                model_for_role(name) if model_for_role is not None else llm,
                registry,
                allowed,
                role_name=name,
                role_instructions=instructions,
            )
        )
    return workers


def worker_descriptions(*, allow_web: bool = False) -> dict[str, str]:
    descriptions: dict[str, str] = {
        name: policy.description for name, policy in WORKER_POLICIES.items()
    }
    if not allow_web:
        descriptions["researcher"] = "Find facts in local workspace documents only."
    return descriptions
