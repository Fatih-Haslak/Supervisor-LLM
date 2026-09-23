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


@pytest.mark.asyncio
async def test_wikipedia_search_finds_article_when_qualified_title_is_missing() -> None:
    seen: list[str] = []

    def fetch(title: str) -> dict[str, object]:
        seen.append(title)
        if title == "Muhammed Salah":
            return {"query": {"pages": [{
                "title": title, "extract": "Mısırlı futbolcudur."
            }]}}
        return {"query": {"pages": [{"title": title, "missing": True}]}}

    def search(_title: str, _language: str) -> dict[str, object]:
        return {"query": {"search": [
            {"title": "Muhammed Salah Cedidi"},
            {"title": "Muhammed Salah"},
        ]}}

    result = await WikipediaLookupTool(fetch, search=search).run({
        "title": "Muhammed Salah (futbolcu)"
    })
    assert result.success
    assert seen == ["Muhammed Salah (futbolcu)", "Muhammed Salah"]
    assert json.loads(result.output)["url"].endswith("/Muhammed_Salah")


@pytest.mark.asyncio
async def test_wikipedia_english_fallback_accepts_name_transliteration() -> None:
    def missing(title: str) -> dict[str, object]:
        return {"query": {"pages": [{"title": title, "missing": True}]}}

    def fetch_english(title: str) -> dict[str, object]:
        if title == "Mohamed Salah":
            return {"query": {"pages": [{
                "title": title, "extract": "Egyptian professional footballer."
            }]}}
        return missing(title)

    tool = WikipediaLookupTool(
        missing, search=lambda _title, _language: {"query": {"search": []}},
        fetch_english=fetch_english,
    )
    result = await tool.run({"title": "Muhammed Salah"})
    assert result.success
    data = json.loads(result.output)
    assert data["language"] == "en"
    assert data["url"] == "https://en.wikipedia.org/wiki/Mohamed_Salah"


@pytest.mark.asyncio
async def test_wikipedia_search_rejects_other_people_and_wrong_qualifier() -> None:
    def fetch(title: str) -> dict[str, object]:
        if title == "Muhammed Salah Cedidi":
            return {"query": {"pages": [{"title": title, "extract": "An author."}]}}
        if title == "Muhammed Salah":
            return {"query": {"pages": [{"title": title, "extract": "An author."}]}}
        return {"query": {"pages": [{"title": title, "missing": True}]}}

    tool = WikipediaLookupTool(
        fetch, search=lambda _title, _language: {"query": {"search": [
            {"title": "Muhammed Salah Cedidi"}, {"title": "Muhammed Salah"},
        ]}},
    )
    result = await tool.run({"title": "Muhammed Salah (futbolcu)"})
    assert result.error_type == "NoArticle"


@pytest.mark.asyncio
async def test_wikipedia_fallback_keeps_birth_year_disambiguation() -> None:
    def missing(title: str) -> dict[str, object]:
        return {"query": {"pages": [{"title": title, "missing": True}]}}

    def english(title: str) -> dict[str, object]:
        return {"query": {"pages": [{
            "title": title,
            "extract": "Mohamed Salah El Boukammiri (born 27 May 2004) "
                       "is a Belgian professional footballer.",
        }]}}

    result = await WikipediaLookupTool(
        missing, search=lambda _title, _language: {"query": {"search": []}},
        fetch_english=english,
    ).run({"title": "Mo Salah (footballer, born 2004)"})
    assert result.success
    assert json.loads(result.output)["language"] == "en"


@pytest.mark.asyncio
async def test_wikipedia_short_given_name_accepts_canonical_redirect() -> None:
    tool = WikipediaLookupTool(lambda _title: {"query": {"pages": [{
        "title": "Muhammed Salah", "extract": "Mısırlı futbolcudur."
    }]}})
    result = await tool.run({"title": "Mo Salah"})
    assert result.success
    assert json.loads(result.output)["title"] == "Muhammed Salah"


@pytest.mark.asyncio
async def test_wikipedia_api_error_is_not_reported_as_missing_article() -> None:
    result = await WikipediaLookupTool(lambda _title: {
        "error": {"code": "ratelimited", "info": "Too many requests"}
    }).run({"title": "Muhammed Salah"})
    assert result.error_type == "LookupUnavailable"
