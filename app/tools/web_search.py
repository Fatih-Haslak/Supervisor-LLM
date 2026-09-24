"""Public web search results with bounded, source-linked snippets."""

import asyncio
import ipaddress
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from html import unescape
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field

from app.tools.base import BaseTool, ToolResult

_MAX_RESPONSE_BYTES = 512_000


class WebSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=300)


def _safe_result_url(raw: object) -> str | None:
    if not isinstance(raw, str) or len(raw) > 2000:
        return None
    try:
        parsed = urlsplit(raw.strip())
        if parsed.scheme != "https" or not parsed.hostname or parsed.username:
            return None
        host = parsed.hostname.casefold()
        if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
            return None
        try:
            address = ipaddress.ip_address(host)
            if not address.is_global:
                return None
        except ValueError:
            pass
        return raw.strip()
    except ValueError:
        return None


def _clean_result(title: object, url: object, snippet: object) -> dict[str, str] | None:
    safe_url = _safe_result_url(url)
    if safe_url is None or not isinstance(title, str):
        return None
    summary = " ".join(unescape(re.sub(r"<[^>]+>", " ", snippet)).split()) \
        if isinstance(snippet, str) else ""
    cleaned_title = " ".join(title.split())[:240]
    if not cleaned_title:
        return None
    return {
        "title": cleaned_title,
        "url": safe_url,
        "snippet": summary[:900],
    }


def _bing_search(query: str) -> dict[str, object]:
    url = "https://www.bing.com/search?" + urlencode({
        "q": query, "format": "rss", "mkt": "tr-TR", "count": "5",
    })
    request = Request(url, headers={
        "User-Agent": "LocalAgent/0.1 (public web search)",
        "Accept": "application/rss+xml, application/xml, text/xml",
    })
    with urlopen(request, timeout=10) as response:
        if response.status != 200:
            raise OSError("Web search returned an unexpected status")
        payload = response.read(_MAX_RESPONSE_BYTES + 1)
    if len(payload) > _MAX_RESPONSE_BYTES:
        raise ValueError("Web search response is too large")
    root = ET.fromstring(payload)
    results: list[dict[str, str]] = []
    for item in root.findall("./channel/item")[:5]:
        result = _clean_result(
            item.findtext("title"), item.findtext("link"), item.findtext("description")
        )
        if result is not None:
            results.append(result)
    return {"provider": "Bing RSS", "query": query, "results": results}


def _google_news_search(query: str) -> dict[str, object]:
    """Search recent public reporting using Google's public News RSS feed."""
    url = "https://news.google.com/rss/search?" + urlencode({
        "q": query, "hl": "tr", "gl": "TR", "ceid": "TR:tr",
    })
    request = Request(url, headers={
        "User-Agent": "Mozilla/5.0 LocalAgent/0.1 (public web search)",
        "Accept": "application/rss+xml, application/xml, text/xml",
    })
    with urlopen(request, timeout=10) as response:
        if response.status != 200:
            raise OSError("Web search returned an unexpected status")
        payload = response.read(_MAX_RESPONSE_BYTES + 1)
    if len(payload) > _MAX_RESPONSE_BYTES:
        raise ValueError("Web search response is too large")
    root = ET.fromstring(payload)
    results: list[dict[str, str]] = []
    for item in root.findall("./channel/item")[:5]:
        title = item.findtext("title")
        publisher = item.findtext("source")
        excerpt = item.findtext("description") or title
        if publisher and excerpt:
            excerpt = f"Yayın: {publisher}. {excerpt}"
        result = _clean_result(title, item.findtext("link"), excerpt)
        if result is not None:
            results.append(result)
    return {"provider": "Google News RSS", "query": query, "results": results}


def _brave_search(query: str, api_key: str) -> dict[str, object]:
    url = "https://api.search.brave.com/res/v1/web/search?" + urlencode({
        "q": query, "country": "TR", "search_lang": "tr", "count": "5",
    })
    request = Request(url, headers={
        "Accept": "application/json",
        "X-Subscription-Token": api_key,
        "User-Agent": "LocalAgent/0.1",
    })
    with urlopen(request, timeout=10) as response:
        if response.status != 200:
            raise OSError("Brave Search returned an unexpected status")
        payload = response.read(_MAX_RESPONSE_BYTES + 1)
    if len(payload) > _MAX_RESPONSE_BYTES:
        raise ValueError("Web search response is too large")
    data: Any = json.loads(payload)
    web = data.get("web") if isinstance(data, dict) else None
    rows = web.get("results") if isinstance(web, dict) else None
    results: list[dict[str, str]] = []
    if isinstance(rows, list):
        for row in rows[:5]:
            if not isinstance(row, dict):
                continue
            result = _clean_result(row.get("title"), row.get("url"), row.get("description"))
            if result is not None:
                results.append(result)
    return {"provider": "Brave Search", "query": query, "results": results}


class WebSearchTool(BaseTool[WebSearchInput]):
    name = "web_search"
    description = (
        "Search the public web for current facts, technical topics, and sources. "
        "Returns up to five titles, short snippets, and HTTPS source URLs. Do not send "
        "private workspace text or secrets as queries."
    )
    input_type = WebSearchInput

    def __init__(
        self,
        *,
        api_key: str | None = None,
        search: Callable[[str], dict[str, object]] | None = None,
    ) -> None:
        self._api_key = api_key
        self._search = search

    async def execute(self, arguments: WebSearchInput) -> ToolResult:
        query = " ".join(arguments.query.split())
        if len(query) < 2:
            return ToolResult.fail("InvalidArguments", "Search query is empty")
        try:
            if self._search is not None:
                data = await asyncio.to_thread(self._search, query)
            elif self._api_key:
                data = await asyncio.to_thread(_brave_search, query, self._api_key)
            else:
                try:
                    data = await asyncio.to_thread(_google_news_search, query)
                except (OSError, URLError, TimeoutError, ValueError, ET.ParseError):
                    data = {"results": []}
                if not data.get("results"):
                    data = await asyncio.to_thread(_bing_search, query)
            raw_results = data.get("results")
            if not isinstance(raw_results, list):
                raw_results = []
            results = [clean for row in raw_results[:5]
                       if isinstance(row, dict)
                       if (clean := _clean_result(
                           row.get("title"), row.get("url"),
                           row.get("snippet", row.get("description")),
                       )) is not None]
            if not results:
                return ToolResult.fail("NoResults", "No public web search results were found")
            safe_data = {
                "provider": str(data.get("provider", "web search"))[:40],
                "query": query, "results": results,
            }
            return ToolResult.ok(json.dumps(safe_data, ensure_ascii=False))
        except HTTPError as exc:
            if exc.code == 429:
                return ToolResult.fail("RateLimited", "Web search rate limit was reached")
            return ToolResult.fail("SearchUnavailable", "Public web search is unavailable")
        except (OSError, URLError, TimeoutError, ValueError, ET.ParseError):
            return ToolResult.fail("SearchUnavailable", "Public web search is unavailable")
