import json

import pytest

from app.tools.wikipedia import WikipediaLookupTool


@pytest.mark.asyncio
async def test_wikipedia_lookup_returns_bounded_sourced_intro() -> None:
    seen: list[str] = []

    def fetch(title: str) -> dict[str, object]:
        seen.append(title)
        return {"query": {"pages": [{
            "title": "Fatih Tekke", "extract": "Türk eski futbolcu ve teknik direktör."
        }]}}

    result = await WikipediaLookupTool(fetch).run({"title": "Fatih Tekke"})
    assert result.success
    assert seen == ["Fatih Tekke"]
    data = json.loads(result.output)
    assert "futbolcu" in data["extract"]
    assert data["url"] == "https://tr.wikipedia.org/wiki/Fatih_Tekke"


@pytest.mark.asyncio
async def test_wikipedia_lookup_handles_missing_or_failed_source() -> None:
    missing = WikipediaLookupTool(lambda _title: {
        "query": {"pages": [{"missing": True, "title": "Unknown"}]}
    })
    absent = await missing.run({"title": "Unknown"})
    assert absent.error_type == "NoArticle"

    def fail(_title: str) -> dict[str, object]:
        raise OSError("network unavailable")

    unavailable = await WikipediaLookupTool(fail).run({"title": "Fatih Tekke"})
    assert unavailable.error_type == "LookupUnavailable"


@pytest.mark.asyncio
async def test_wikipedia_lookup_normalizes_uppercase_turkish_question() -> None:
    seen: list[str] = []

    def fetch(title: str) -> dict[str, object]:
        seen.append(title)
        return {"query": {"pages": [{"title": title, "extract": "Futbolcudur."}]}}

    result = await WikipediaLookupTool(fetch).run({"title": "FATİH TEKKE KİMDİR?"})
    assert result.success
    assert seen == ["Fatih Tekke"]

    lower = await WikipediaLookupTool(fetch).run({"title": "fatih tekke"})
    assert lower.success
    assert seen[-1] == "Fatih Tekke"


@pytest.mark.asyncio
async def test_wikipedia_lookup_bounds_long_article_intro() -> None:
    tool = WikipediaLookupTool(lambda _title: {
        "query": {"pages": [{"title": "Fatih Terim", "extract": "A" * 5000}]}
    })
    result = await tool.run({"title": "Fatih Terim"})
    assert result.success
    assert len(json.loads(result.output)["extract"]) == 1200
