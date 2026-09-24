"""Keep final Wikipedia answers consistent with successful tool evidence."""

import json
import re
from collections.abc import Sequence

from app.tools.base import ToolResult

_FALSE_ABSENCE = re.compile(
    r"(?:makale|madde|kaynak).{0,65}(?:bulunamad|yok|erişilemedi)|"
    r"(?:bilgi veremem|bilgi sağlayamıyorum|doğrulayamadım)",
    flags=re.IGNORECASE,
)


def ground_wikipedia_answer(answer: str, results: Sequence[ToolResult]) -> str:
    """Recover from a false no-article answer and require a visible source URL."""
    for result in reversed(results):
        if not result.success or result.output is None:
            continue
        try:
            article = json.loads(result.output)
        except (TypeError, ValueError):
            continue
        if not isinstance(article, dict):
            continue
        title, extract, url = (
            article.get("title"), article.get("extract"), article.get("url")
        )
        if not all(isinstance(value, str) for value in (title, extract, url)):
            continue
        assert isinstance(title, str) and isinstance(extract, str) and isinstance(url, str)
        if not re.fullmatch(r"https://(?:tr|en)\.wikipedia\.org/wiki/[^\s]+", url):
            continue
        if "//en." in url:
            # A small local model can change names or nationalities while translating.
            # Keep the source's wording visible when only an English article exists.
            intro = extract.split("\n", 1)[0][:500].strip()
            if len(extract.split("\n", 1)[0]) > 500:
                intro = intro.rsplit(" ", 1)[0] + "…"
            return (
                f"İngilizce Wikipedia'da **{title}** maddesi bulundu. "
                f"Kaynak giriş metni (İngilizce): “{intro}”\nKaynak: {url}"
            )
        if _FALSE_ABSENCE.search(answer):
            intro = extract[:380].strip()
            if len(extract) > 380:
                intro = intro.rsplit(" ", 1)[0] + "…"
            return f"Türkçe Wikipedia'da **{title}** maddesi bulundu. {intro}\nKaynak: {url}"
        if url not in answer:
            return answer.rstrip() + f"\nKaynak: {url}"
        return answer
    return answer


def ground_web_answer(answer: str, results: Sequence[ToolResult]) -> str:
    """Expose verified search URLs and remove links the model invented."""
    urls: list[str] = []
    for result in results:
        if not result.success or not result.output:
            continue
        try:
            data = json.loads(result.output)
        except (TypeError, ValueError):
            continue
        rows = data.get("results") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            continue
        for row in rows[:5]:
            if (isinstance(row, dict) and isinstance(row.get("url"), str)
                    and re.fullmatch(r"https://[^\s]+", row["url"])):
                urls.append(row["url"][:500])
    if not urls:
        return answer
    trusted = set(urls)
    link_pattern = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
    answer = link_pattern.sub(
        lambda match: match.group(0) if match.group(2) in trusted else match.group(1),
        answer,
    )
    url_pattern = re.compile(r"https?://[^\s)>\]]+")
    answer = url_pattern.sub(
        lambda match: match.group(0) if match.group(0).rstrip(".,;:!?\"'") in trusted else "",
        answer,
    )
    cited = list(dict.fromkeys(url for url in urls if url in answer))
    if cited:
        return answer
    # Keep the answer's claims untouched; show evidence links for user verification.
    return answer.rstrip() + "\nKaynaklar: " + " · ".join(dict.fromkeys(urls[:3]))
