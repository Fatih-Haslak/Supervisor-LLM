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
