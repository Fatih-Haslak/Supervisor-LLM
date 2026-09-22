"""Bounded public-article lookup from Turkish Wikipedia."""

import asyncio
import json
import re
from collections.abc import Callable
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field

from app.tools.base import BaseTool, ToolResult

_API = "https://tr.wikipedia.org/w/api.php"


def _turkish_title_case(value: str) -> str:
    lowered = value.replace("İ", "i").replace("I", "ı").lower()
    words: list[str] = []
    for word in lowered.split():
        first = word[:1].replace("i", "İ").replace("ı", "I").upper()
        words.append(first + word[1:])
    return " ".join(words)


def _normalized_title(raw: str) -> str:
    title = re.sub(r"\s+(?:kimdir|kimdi|nedir)\s*[?.!]*$", "", raw.strip(),
                   flags=re.IGNORECASE).strip(" \"'?.!")
    if title.isupper() or title.islower():
        title = _turkish_title_case(title)
    return title


class WikipediaInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        min_length=2, max_length=120,
        description="Only the public article title, for example 'Triton Server'.",
    )


def _fetch_article(title: str) -> dict[str, object]:
    url = _API + "?" + urlencode({
        "action": "query", "format": "json", "formatversion": "2",
        "prop": "extracts", "exintro": "1", "explaintext": "1",
        "redirects": "1", "titles": title,
    })
    request = Request(url, headers={"User-Agent": "LocalAgent/0.1 (public article lookup)"})
    with urlopen(request, timeout=10) as response:
        payload = response.read(32_001)
    if len(payload) > 32_000:
        raise ValueError("Wikipedia response is too large")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("Invalid Wikipedia response")
    return data


class WikipediaLookupTool(BaseTool[WikipediaInput]):
    name = "wikipedia_lookup"
    description = (
        "Look up a public topic or person's introductory article on Turkish Wikipedia. "
        "Sends the title to tr.wikipedia.org; use for public facts, not private data."
    )
    input_type = WikipediaInput

    def __init__(
        self, fetch: Callable[[str], dict[str, object]] | None = None
    ) -> None:
        self._fetch = fetch or _fetch_article

    async def execute(self, arguments: WikipediaInput) -> ToolResult:
        try:
            data = await asyncio.to_thread(self._fetch, _normalized_title(arguments.title))
            query = data.get("query")
            pages = query.get("pages") if isinstance(query, dict) else None
            if not isinstance(pages, list) or not pages:
                return ToolResult.fail("NoArticle", "Wikipedia article was not found")
            page = pages[0]
            if not isinstance(page, dict) or page.get("missing") is True:
                return ToolResult.fail("NoArticle", "Wikipedia article was not found")
            title = page.get("title")
            extract = page.get("extract")
            if not isinstance(title, str) or not isinstance(extract, str) or not extract:
                return ToolResult.fail("NoArticle", "Wikipedia article has no introduction")
            return ToolResult.ok(json.dumps({
                "title": title,
                "extract": extract[:1200],
                "url": "https://tr.wikipedia.org/wiki/" + quote(title.replace(" ", "_")),
            }, ensure_ascii=False))
        except (OSError, ValueError, TypeError):
            return ToolResult.fail("LookupUnavailable", "Wikipedia lookup failed")
