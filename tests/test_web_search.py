"""Public web search is bounded, source-linked, and handles model argument mistakes."""

import json

import pytest

from app.agents.single import SingleAgent
from app.tools.base import ToolResult
from app.tools.web_search import WebSearchInput, WebSearchTool, _clean_result, _safe_result_url


def test_web_search_input_and_result_url_validation() -> None:
    assert WebSearchInput(query="Fatih Terim").query == "Fatih Terim"
    assert _safe_result_url("https://example.org/page") == "https://example.org/page"
    for url in ("http://example.org", "https://localhost/a", "https://127.0.0.1/a",
                "https://user:pass@example.org", "https://example.local/a"):
        assert _safe_result_url(url) is None
    clean = _clean_result("A title", "https://example.org", "<a>Readable &amp; safe</a>")
    assert clean is not None and clean["snippet"] == "Readable & safe"


@pytest.mark.asyncio
async def test_web_search_returns_only_clean_source_results() -> None:
    def search(query: str) -> dict[str, object]:
        assert query == "Şenol Güneş"
        return {"provider": "test", "query": query, "results": [
            {"title": "  Şenol Güneş  ", "url": "https://example.org/gunes",
             "snippet": "Turkish football coach."},
            {"title": "private", "url": "http://127.0.0.1/private", "snippet": "secret"},
        ]}

    result = await WebSearchTool(search=search).run({"query": "Şenol Güneş"})
    assert result.success and result.output
    data = json.loads(result.output)
    assert len(data["results"]) == 1
    assert data["results"][0]["url"] == "https://example.org/gunes"


@pytest.mark.asyncio
async def test_web_search_handles_provider_failure_without_leaking_details() -> None:
    def failing(_query: str) -> dict[str, object]:
        raise OSError("internal endpoint details")

    result = await WebSearchTool(search=failing).run({"query": "Triton Server"})
    assert not result.success
    assert result.error_type == "SearchUnavailable"
    assert "internal endpoint" not in (result.message or "")


@pytest.mark.asyncio
async def test_web_search_rejects_whitespace_query_before_provider_call() -> None:
    called = False

    def search(_query: str) -> dict[str, object]:
        nonlocal called
        called = True
        return {"results": []}

    result = await WebSearchTool(search=search).run({"query": "  "})
    assert not result.success and result.error_type == "InvalidArguments"
    assert not called


def test_web_search_missing_query_is_repaired_from_request() -> None:
    agent = SingleAgent.__new__(SingleAgent)
    agent._role_name = "researcher"
    arguments = agent._tool_arguments("web_search", {}, "Triton Server ne işe yarar?")
    assert arguments == {"query": "Triton Server ne işe yarar?"}


def test_grounding_adds_verified_source_links_without_rewriting_claims() -> None:
    from app.orchestration.evidence import ground_web_answer

    source = ToolResult.ok(json.dumps({"results": [
        {"url": "https://example.org/source", "title": "Source", "snippet": "Evidence"}
    ]}))
    assert ground_web_answer("Triton bir sunucudur.", [source]).endswith(
        "Kaynaklar: https://example.org/source"
    )


def test_grounding_removes_links_not_returned_by_search() -> None:
    from app.orchestration.evidence import ground_web_answer

    source = ToolResult.ok(json.dumps({"results": [
        {"url": "https://example.org/source", "title": "Source", "snippet": "Evidence"}
    ]}))
    answer = ground_web_answer(
        "Salah için [kaynak](https://made-up.example/facts) ve https://bad.example/page",
        [source],
    )
    assert "made-up.example" not in answer and "bad.example" not in answer
    assert "https://example.org/source" in answer
