"""Conversational path for short turns that do not need task planning."""

import re
from collections.abc import Sequence

from app.llm.client import LLMClient, LLMError
from app.llm.schemas import ChatMessage
from app.orchestration.state import AgentState

_TASK_CUES = re.compile(
    r"workspace[/\\]|\.(?:csv|py|md|txt|json)\b|"
    r"\b(?:dosyay[ıai]|dosyalar[ıi]?|rapor|analiz|düzelt|testleri|"
    r"projeyi|listele|oluştur|araştır|kaydet)\b",
    flags=re.IGNORECASE,
)
_ARITHMETIC = re.compile(r"\d\s*[-+*/]\s*\d")
_USER_NAME = re.compile(
    r"(?:^|[,.!?]\s*)(?:benim\s+)?(?:adım|ismim)\s+"
    r"(?!ne(?:ydi|dir)?\b|kim\b)"
    r"([A-Za-zÇĞİÖŞÜçğıöşü][A-Za-zÇĞİÖŞÜçğıöşü'-]{1,39})\b",
    flags=re.IGNORECASE,
)
_ASK_USER_NAME = re.compile(
    r"\b(?:(?:benim\s+)?(?:adım|ismim)(?:\s+ne(?:ydi|dir)?)?|adımı|ismimi)\b",
    flags=re.IGNORECASE,
)
_ASK_ASSISTANT_NAME = re.compile(
    r"\b(?:senin\s+adın|senin\s+ismin|adın\s+ne|ismin\s+ne)\b",
    flags=re.IGNORECASE,
)


def automatic_mode(message: str) -> str:
    """Keep ordinary conversation out of the planner."""
    if _TASK_CUES.search(message):
        return "supervisor"
    if _ARITHMETIC.search(message):
        return "single"
    return "chat"


async def run_chat(
    llm: LLMClient, message: str, history: Sequence[ChatMessage]
) -> AgentState:
    if _ASK_USER_NAME.search(message) and not _USER_NAME.search(message):
        for earlier in reversed(history):
            if earlier.role != "user":
                continue
            match = _USER_NAME.search(earlier.content)
            if match:
                answer = f"Adın {match.group(1)}."
                state = AgentState.for_request(message, ChatMessage(
                    role="system", content="Conversation name recall"
                ))
                state.finish(answer)
                return state
    if _ASK_ASSISTANT_NAME.search(message) and not _ASK_USER_NAME.search(message):
        state = AgentState.for_request(message, ChatMessage(
            role="system", content="Assistant identity"
        ))
        state.finish("Ben Yerel Agent adlı asistanım.")
        return state
    system = ChatMessage(
        role="system",
        content=(
            "Sen yerel Türkçe sohbet asistanısın. /no_think\n"
            "Adın Yerel Agent; başka bir ad uydurma. Doğal ve dilbilgisi düzgün "
            "Türkçe kullan. Önceki konuşma mesajlarını takip et. Kullanıcı adını söylediyse "
            "sonraki sorularda aynen hatırla. Bilmediğin bilgiyi uydurma. "
            "Kısa sohbet sorularına doğrudan, doğal Türkçe yanıt ver."
        ),
    )
    selected = list(history[-16:])
    while selected and sum(len(item.content) for item in selected) > 6000:
        selected = selected[2:]
    messages = [system, *selected, ChatMessage(role="user", content=message)]
    response = await llm.chat(messages)
    answer = response.content.strip()
    if not answer:
        raise LLMError("Local model returned an empty chat answer")
    state = AgentState.for_request(message, system)
    state.messages = [*messages, ChatMessage(role="assistant", content=answer)]
    state.finish(answer)
    return state
