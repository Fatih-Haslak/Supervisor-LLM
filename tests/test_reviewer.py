import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from app.agents.reviewer import (
    NO_SOURCE_RESEARCH_ANSWER,
    RESEARCH_UNVERIFIED_ANSWER,
    ReviewerAgent,
)
from app.agents.supervisor import Supervisor, WorkerResult
from app.llm.client import LLMContextOverflowError
from app.llm.schemas import ChatMessage, LLMResponse
from app.orchestration.state import AgentState, ToolCallRecord
from app.tools.base import ToolResult
from app.tools.filesystem import FileReadTool, FileWriteTool, Workspace
from app.tools.registry import ToolRegistry
from tests.support import ApproveWrites


class ScriptedLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.calls = 0
        self.requests: list[list[ChatMessage]] = []

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        assert json_schema is not None
        self.calls += 1
        self.requests.append(list(messages))
        return LLMResponse(content=next(self._responses), model="scripted")


def make_registry(root: Path) -> ToolRegistry:
    workspace = Workspace(root)
    registry = ToolRegistry(approver=ApproveWrites())
    registry.register(FileReadTool(workspace))
    registry.register(FileWriteTool(workspace))
    return registry


class WritingWorker:
    def __init__(self, registry: ToolRegistry, *, repair: bool) -> None:
        self.registry = registry
        self.repair = repair
        self.calls = 0

    async def run(self, task: str, state: AgentState) -> WorkerResult:
        self.calls += 1
        assert state.current_agent == "coder"
        code = (
            "def add(a, b):\n    return a + b\n"
            if self.repair and self.calls > 1
            else "def add(:\n    pass\n"
        )
        arguments: dict[str, object] = {
            "path": "module.py", "content": code, "overwrite": self.calls > 1,
        }
        tool_result = await self.registry.execute("file_write", arguments, {"file_write"})
        return WorkerResult(
            answer="module.py güncellendi",
            tool_results=[
                ToolCallRecord(tool="file_write", arguments=arguments, result=tool_result)
            ],
        )


class NeverCalledWorker:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, task: str, state: AgentState) -> WorkerResult:
        self.calls += 1
        return WorkerResult(answer="Should not run")


@pytest.mark.asyncio
async def test_reviewer_detects_syntax_error_without_running_code(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "module.py").write_text("def add(:\n    pass\n", encoding="utf-8")
    llm = ScriptedLLM([])
    result = await ReviewerAgent(llm, make_registry(root)).review(
        "Toplama fonksiyonu yaz", "module.py oluştur",
        WorkerResult(
            answer="Bitti",
            tool_results=[
                ToolCallRecord(
                    tool="file_write", arguments={"path": "module.py"},
                    result=ToolResult.ok("Wrote module.py"),
                )
            ],
        ),
    )
    assert result.verdict.status == "fail"
    assert "syntax error" in result.verdict.issues[0]
    assert [call.tool for call in result.tool_calls] == ["file_read"]
    assert llm.calls == 0
    assert (root / "module.py").read_text(encoding="utf-8").startswith("def add(:")


@pytest.mark.asyncio
async def test_reviewer_requires_file_evidence(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    llm = ScriptedLLM([])
    result = await ReviewerAgent(llm, make_registry(root)).review(
        "Kod yaz", "module.py oluştur", WorkerResult(answer="Yazıldı")
    )
    assert result.verdict.status == "fail"
    assert "dosya" in result.verdict.issues[0]
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_reviewer_accepts_explicit_no_source_fallback(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    llm = ScriptedLLM([])
    result = await ReviewerAgent(llm, make_registry(root)).review(
        "Cihan Top kimdir?", "Kişiyi araştır",
        WorkerResult(answer=NO_SOURCE_RESEARCH_ANSWER),
    )
    assert result.verdict.status == "pass"
    assert result.tool_calls == []
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_reviewer_accepts_successful_wikipedia_source(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    llm = ScriptedLLM(['{"status":"pass","issues":[]}'])
    source = ToolCallRecord(
        tool="wikipedia_lookup", arguments={"title": "Şenol Güneş"},
        result=ToolResult.ok(
            '{"title":"Şenol Güneş","extract":"Türk futbol teknik direktörü.",'
            '"language":"tr","url":"https://tr.wikipedia.org/wiki/%C5%9Eenol_G%C3%BCne%C5%9F"}'
        ),
    )
    result = await ReviewerAgent(llm, make_registry(root)).review(
        "Şenol Güneş hakkında bilgi getir, reviewer'a sok",
        "Wikipedia'da araştır ve kaynak URL'siyle özetle",
        WorkerResult(answer="Türk teknik direktörüdür. Kaynak: Wikipedia", tool_results=[source]),
    )
    assert result.verdict.status == "pass"
    assert llm.calls == 1
    evidence = json.loads(llm.requests[0][-1].content)
    assert evidence["public_sources"][0]["title"] == "Şenol Güneş"
    assert evidence["public_sources"][0]["url"].startswith("https://tr.wikipedia.org/")


@pytest.mark.asyncio
async def test_reviewer_accepts_web_search_snippet_as_public_evidence(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    llm = ScriptedLLM(['{"status":"pass","issues":[]}'])
    source = ToolCallRecord(
        tool="web_search", arguments={"query": "Triton Server"},
        result=ToolResult.ok(json.dumps({"provider": "Bing RSS", "results": [{
            "title": "Triton Inference Server", "url": "https://developer.nvidia.com/triton",
            "snippet": "A platform for deploying machine learning models."
        }]})),
    )
    result = await ReviewerAgent(llm, make_registry(root)).review(
        "Triton Server'ın işlevini araştır", "Triton Server hakkında kaynaklı bilgi ver",
        WorkerResult(answer="Model dağıtım platformudur. Kaynak: https://developer.nvidia.com/triton",
                     tool_results=[source]),
    )
    assert result.verdict.status == "pass"
    evidence = json.loads(llm.requests[0][-1].content)
    assert evidence["public_sources"][0]["extract"].startswith("A platform")
    assert evidence["public_sources"][0]["url"] == "https://developer.nvidia.com/triton"


@pytest.mark.asyncio
async def test_no_public_research_results_complete_with_safe_fallback(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    registry = make_registry(root)

    class NoSourceResearcher:
        async def run(self, _task: str, _state: AgentState) -> WorkerResult:
            return WorkerResult(
                answer="Cihan Top tanınmış bir Türk oyuncudur.",
                tool_results=[
                    ToolCallRecord(
                        tool="wikipedia_lookup", arguments={"title": "Cihan Top"},
                        result=ToolResult.fail("NoArticle", "Madde bulunamadı"),
                    ),
                    ToolCallRecord(
                        tool="web_search", arguments={"query": "Cihan Top"},
                        result=ToolResult.fail("NoResults", "Sonuç bulunamadı"),
                    ),
                ],
            )

    llm = ScriptedLLM([
        '{"action":"delegate","next_agent":"researcher","task":"Araştır",'
        '"reason":"Kişi araştırması"}',
        '{"action":"final_answer","answer":"'
        + NO_SOURCE_RESEARCH_ANSWER + '"}',
    ])
    state = await Supervisor(
        llm, {"researcher": NoSourceResearcher()},
        reviewer=ReviewerAgent(llm, registry),
    ).run("Cihan Top kimdir?")

    assert state.final_answer == NO_SOURCE_RESEARCH_ANSWER
    assert state.completed_tasks
    assert state.pending_tasks == []
    assert [review.status for review in state.reviews] == ["pass"]
    assert state.agent_outputs[-1].answer == NO_SOURCE_RESEARCH_ANSWER


@pytest.mark.asyncio
async def test_reviewer_context_overflow_returns_safe_research_answer(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    registry = make_registry(root)
    source = ToolCallRecord(
        tool="wikipedia_lookup", arguments={"title": "Cihan Top"},
        result=ToolResult.ok(
            '{"title":"Cihan Top","extract":"Kişi hakkında kısa bilgi.",'
            '"url":"https://tr.wikipedia.org/wiki/Cihan_Top"}'
        ),
    )

    class ResearchWorker:
        async def run(self, _task: str, _state: AgentState) -> WorkerResult:
            return WorkerResult(
                answer="Cihan Top hakkında doğrulanamayan ayrıntılar.",
                tool_results=[source],
            )

    class OverflowDuringReviewLLM(ScriptedLLM):
        async def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> LLMResponse:
            if len(self.requests) == 1:
                self.requests.append(list(messages))
                raise LLMContextOverflowError("prompt exceeds context window")
            return await super().chat(messages, **kwargs)

    llm = OverflowDuringReviewLLM([
        '{"action":"delegate","next_agent":"researcher","task":"Araştır",'
        '"reason":"Kişi araştırması"}',
        '{"action":"final_answer","answer":"' + RESEARCH_UNVERIFIED_ANSWER + '"}',
    ])
    state = await Supervisor(
        llm, {"researcher": ResearchWorker()},
        reviewer=ReviewerAgent(llm, registry),
    ).run("Cihan Top kimdir?")

    assert state.final_answer == RESEARCH_UNVERIFIED_ANSWER
    assert state.completed_tasks and state.pending_tasks == []
    assert [review.status for review in state.reviews] == ["pass"]


@pytest.mark.asyncio
async def test_explicit_research_review_reviews_researcher_and_skips_report_writer(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    registry = make_registry(root)
    source = ToolCallRecord(
        tool="wikipedia_lookup", arguments={"title": "Şenol Güneş"},
        result=ToolResult.ok(
            '{"title":"Şenol Güneş","extract":"Türk futbol teknik direktörü.",'
            '"language":"tr","url":"https://tr.wikipedia.org/wiki/%C5%9Eenol_G%C3%BCne%C5%9F"}'
        ),
    )

    class ResearchWorker:
        async def run(self, task: str, state: AgentState) -> WorkerResult:
            assert state.current_agent == "researcher"
            return WorkerResult(
                answer="Türk futbol teknik direktörüdür. Kaynak: https://tr.wikipedia.org/wiki/",
                tool_results=[source],
            )

    llm = ScriptedLLM([
        '{"tasks":[{"id":1,"agent":"writer","task":"Rapor yaz",'
        '"depends_on":[]}]}',
        '{"status":"pass","issues":[]}',
        '{"action":"final_answer","answer":"Şenol Güneş Türk futbol teknik direktörüdür."}',
    ])
    state = await Supervisor(
        llm, {"researcher": ResearchWorker(), "writer": NeverCalledWorker()},
        reviewer=ReviewerAgent(llm, registry),
    ).run_planned("Şenol Güneş hakkında bilgi getir ve bunu reviewer'a sok")

    assert state.plan is not None
    assert [task.agent for task in state.plan.tasks] == ["researcher"]
    assert [review.status for review in state.reviews] == ["pass"]
    assert state.completed_tasks == [state.plan.tasks[0].task]
    assert state.final_answer is not None
    assert state.final_answer.startswith("Şenol Güneş Türk futbol teknik direktörüdür.")
    assert "wikipedia.org/wiki/" in state.final_answer


@pytest.mark.asyncio
async def test_read_only_coder_step_does_not_require_later_function_tests(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "module.py").write_text(
        "def add(a, b):\n    return a - b\n", encoding="utf-8"
    )
    read = ToolCallRecord(
        tool="file_read", arguments={"path": "module.py"},
        result=ToolResult.ok("def add(a, b):\n    return a - b\n"),
    )
    llm = ScriptedLLM(['{"status":"pass","issues":[]}'])
    result = await ReviewerAgent(llm, make_registry(root)).review(
        "module.py hatasını düzelt, function_test ile testleri çalıştır",
        "module.py dosyasını oku ve hatayı belirle",
        WorkerResult(answer="Toplama yerine çıkarma kullanılmış.", tool_results=[read]),
    )
    assert result.verdict.status == "pass"
    assert [call.tool for call in result.tool_calls] == ["file_read"]
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_reviewer_accepts_inline_code_as_analysis_evidence(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    llm = ScriptedLLM(['{"status":"pass","issues":[]}'])
    result = await ReviewerAgent(llm, make_registry(root)).review(
        "```python\ndef fibonacci(n):\n    return n\n```\nbu kodu analiz et",
        "Verilen Python kodunu analiz et",
        WorkerResult(answer="Fonksiyon n değerini doğrudan döndürüyor."),
    )
    assert result.verdict.status == "pass"
    assert result.tool_calls == []
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_reviewer_can_read_file_written_by_earlier_worker(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "module.py").write_text("def add(a, b): return a + b\n", encoding="utf-8")
    llm = ScriptedLLM(['{"status":"pass","issues":[]}'])
    earlier_write = ToolCallRecord(
        tool="file_write", arguments={"path": "module.py"},
        result=ToolResult.ok("Wrote module.py"),
    )
    result = await ReviewerAgent(llm, make_registry(root)).review(
        "Kod yaz", "module.py içeriğini kontrol et", WorkerResult(answer="Kontrol edildi"),
        evidence_tools=[earlier_write],
    )
    assert result.verdict.status == "pass"
    assert [call.tool for call in result.tool_calls] == ["file_read"]
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_supervisor_retries_coder_after_review_failure(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    registry = make_registry(root)
    llm = ScriptedLLM(
        [
            '{"tasks":[{"id":1,"agent":"coder","task":"module.py oluştur",'
            '"depends_on":[]}]}',
            '{"status":"pass","issues":[]}',
            '{"action":"final_answer","answer":"Kod düzeltildi ve incelemeden geçti"}',
        ]
    )
    worker = WritingWorker(registry, repair=True)
    state = await Supervisor(
        llm, {"coder": worker}, reviewer=ReviewerAgent(llm, registry)
    ).run_planned("module.py dosyasına add fonksiyonu yaz")

    assert worker.calls == 2
    assert (root / "module.py").read_text(encoding="utf-8").startswith("def add(a, b):")
    assert [review.status for review in state.reviews] == ["fail", "pass"]
    assert state.completed_tasks == ["module.py oluştur"]
    assert state.pending_tasks == []
    assert state.final_answer == "Kod düzeltildi ve incelemeden geçti"


@pytest.mark.asyncio
async def test_review_retries_stop_after_two_failed_revisions(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    registry = make_registry(root)
    llm = ScriptedLLM(
        ['{"tasks":[{"id":1,"agent":"coder","task":"module.py oluştur",'
         '"depends_on":[]},{"id":2,"agent":"file_agent",'
         '"task":"rapor yaz","depends_on":[1]}]}']
    )
    worker = WritingWorker(registry, repair=False)
    dependent_worker = NeverCalledWorker()
    state = await Supervisor(
        llm, {"coder": worker, "file_agent": dependent_worker},
        reviewer=ReviewerAgent(llm, registry),
        max_review_retries=2,
    ).run_planned("module.py dosyasına add fonksiyonu yaz")

    assert worker.calls == 3
    assert dependent_worker.calls == 0
    assert [review.attempt for review in state.reviews] == [1, 2, 3]
    assert all(review.status == "fail" for review in state.reviews)
    assert state.completed_tasks == []
    assert state.pending_tasks == ["module.py oluştur", "rapor yaz"]
    assert state.final_answer is not None and "tamamlanmadı" in state.final_answer
