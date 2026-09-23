"""Bounded public-article lookup with title search and language fallback."""

import asyncio
import json
import re
from collections.abc import Callable
from difflib import SequenceMatcher
from functools import lru_cache
from time import monotonic, sleep
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field

from app.tools.base import BaseTool, ToolResult

_LANGUAGES = ("tr", "en")


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


def _request(language: str, params: dict[str, str]) -> dict[str, object]:
    url = f"https://{language}.wikipedia.org/w/api.php?" + urlencode({
        "action": "query", "format": "json", "formatversion": "2",
        **params,
    })
    request = Request(url, headers={"User-Agent": "LocalAgent/0.1 (public article lookup)"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=10) as response:
                payload = response.read(32_001)
            break
        except HTTPError as exc:
            if exc.code not in {429, 503} or attempt == 2:
                raise
            sleep(0.5 * (attempt + 1))
    if len(payload) > 32_000:
        raise ValueError("Wikipedia response is too large")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("Invalid Wikipedia response")
    return data


@lru_cache(maxsize=256)
def _cached_request(language: str, title: str, search: bool, bucket: int) -> dict[str, object]:
    del bucket  # The five-minute key prevents stale biographies from persisting indefinitely.
    if search:
        return _request(language, {
            "list": "search", "srsearch": title, "srwhat": "title", "srlimit": "5",
        })
    return _request(language, {
        "prop": "extracts", "exintro": "1", "explaintext": "1",
        "redirects": "1", "titles": title,
    })


def _fetch_article(title: str, language: str = "tr") -> dict[str, object]:
    return _cached_request(language, title, False, int(monotonic() // 300))


def _search_articles(title: str, language: str) -> dict[str, object]:
    return _cached_request(language, title, True, int(monotonic() // 300))


def _article(data: dict[str, object], language: str) -> dict[str, str] | None:
    if "error" in data:
        raise ValueError("Wikipedia API returned an error")
    query = data.get("query")
    pages = query.get("pages") if isinstance(query, dict) else None
    if not isinstance(pages, list) or not pages:
        return None
    page = pages[0]
    if not isinstance(page, dict) or page.get("missing") is True:
        return None
    title = page.get("title")
    extract = page.get("extract")
    if not isinstance(title, str) or not isinstance(extract, str) or not extract.strip():
        return None
    return {
        "title": title, "extract": extract[:1200], "language": language,
        "url": f"https://{language}.wikipedia.org/wiki/" + quote(title.replace(" ", "_")),
    }


def _subject(title: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", title).casefold().strip()


def _related_title(requested: str, candidate: str) -> bool:
    requested_subject = _subject(requested)
    candidate_subject = _subject(candidate)
    if requested_subject == candidate_subject:
        return True
    requested_words = requested_subject.split()
    candidate_words = candidate_subject.split()
    # Accept spelling variants, not a different person with an added surname.
    if len(requested_words) < 2 or len(requested_words) != len(candidate_words):
        return False
    if requested_words[-1] != candidate_words[-1]:
        return False
    arabic_given_names = {"mo", "mohamed", "mohammed", "muhammad", "muhammed"}
    if (len(requested_words) == 2 and requested_words[0] in arabic_given_names
            and candidate_words[0] in arabic_given_names):
        return True
    return SequenceMatcher(None, requested_subject, candidate_subject).ratio() >= 0.78


def _title_variants(title: str) -> list[str]:
    first, separator, remainder = title.partition(" ")
    variants = [title]
    if separator and first.casefold() in {"muhammed", "muhammad", "mohammed", "mohamed"}:
        for spelling in ("Muhammed", "Mohamed", "Muhammad", "Mohammed"):
            variant = spelling + separator + remainder
            if variant not in variants:
                variants.append(variant)
    return variants


def _matches_qualifier(requested: str, extract: str) -> bool:
    qualifier = re.search(r"\(([^)]{2,40})\)\s*$", requested)
    if qualifier is None:
        return True
    word = qualifier.group(1).casefold()
    year = re.search(r"\b(?:18|19|20)\d{2}\b", word)
    if year and year.group() not in extract:
        return False
    if "futbolcu" in word or "footballer" in word:
        return "futbolcu" in extract.casefold() or "footballer" in extract.casefold()
    if year and re.fullmatch(r"(?:born|d\.|doğumlu)?[\s,]*(?:18|19|20)\d{2}"
                             r"(?:[\s,]*doğumlu)?", word):
        return True
    return word in extract.casefold()


class WikipediaLookupTool(BaseTool[WikipediaInput]):
    name = "wikipedia_lookup"
    description = (
        "Look up a public topic or person on Turkish Wikipedia, searching alternate "
        "titles and English Wikipedia if needed. Use for public facts, not private data."
    )
    input_type = WikipediaInput

    def __init__(
        self, fetch: Callable[[str], dict[str, object]] | None = None, *,
        search: Callable[[str, str], dict[str, object]] | None = None,
        fetch_english: Callable[[str], dict[str, object]] | None = None,
    ) -> None:
        self._fetch = fetch or _fetch_article
        self._search = search if search is not None else (
            _search_articles if fetch is None else None
        )
        self._fetch_english = fetch_english if fetch_english is not None else (
            (lambda title: _fetch_article(title, "en")) if fetch is None else None
        )

    def _lookup(self, requested: str) -> dict[str, str] | None:
        for language in _LANGUAGES:
            fetch = self._fetch if language == "tr" else self._fetch_english
            if fetch is None:
                continue
            article = _article(fetch(requested), language)
            if (article and _related_title(requested, article["title"])
                    and _matches_qualifier(requested, article["extract"])):
                return article
            base_title = re.sub(r"\s*\([^)]*\)\s*$", "", requested).strip()
            if base_title != requested:
                article = _article(fetch(base_title), language)
                if (article and _related_title(requested, article["title"])
                        and _matches_qualifier(requested, article["extract"])):
                    return article
            if self._search is not None:
                search_data = self._search(requested, language)
                if "error" in search_data:
                    raise ValueError("Wikipedia search returned an error")
                query = search_data.get("query")
                matches = query.get("search") if isinstance(query, dict) else None
                if isinstance(matches, list):
                    for match in matches:
                        candidate = match.get("title") if isinstance(match, dict) else None
                        if (not isinstance(candidate, str)
                                or not _related_title(requested, candidate)):
                            continue
                        if "(disambiguation)" in candidate.casefold():
                            continue
                        article = _article(fetch(candidate), language)
                        if article and _matches_qualifier(requested, article["extract"]):
                            return article
            for variant in _title_variants(requested)[1:]:
                article = _article(fetch(variant), language)
                if (article and _related_title(requested, article["title"])
                        and _matches_qualifier(requested, article["extract"])):
                    return article
        return None

    async def execute(self, arguments: WikipediaInput) -> ToolResult:
        try:
            article = await asyncio.to_thread(self._lookup, _normalized_title(arguments.title))
            if article is None:
                return ToolResult.fail("NoArticle", "Wikipedia article was not found")
            return ToolResult.ok(json.dumps(article, ensure_ascii=False))
        except (OSError, ValueError, TypeError, KeyError):
            return ToolResult.fail("LookupUnavailable", "Wikipedia lookup failed")
