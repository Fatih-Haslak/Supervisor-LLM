"""Choose the execution path for an automatic conversation turn."""

import json
import re
from collections.abc import Sequence
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agents.chat import automatic_mode
from app.llm.client import LLMClient, LLMError
from app.llm.schemas import ChatMessage
from app.observability.events import agent_span, record

AutoMode = Literal["chat", "single", "supervisor", "plan"]


class AutoModeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: AutoMode
    confidence: float = Field(ge=0, le=1)


_SMALL_TALK = re.compile(
    r"^\s*(?:merhaba|selam|hey|günaydın|iyi akşamlar|nasılsın|"
    r"(?:benim\s+)?(?:adım|ismim)\s+ne(?:ydi|dir)?|"
    r"senin\s+(?:adın|ismin)\s+ne|"
    r"(?:benim\s+)?(?:adım|ismim)\s+[\wÇĞİÖŞÜçğıöşü'-]+)\s*[?!.]*\s*$",
    flags=re.IGNORECASE,
)
_ARITHMETIC = re.compile(
    r"^\s*-?\d+(?:\s*[-+*/]\s*-?\d+)+\s*"
    r"(?:kaç(?:tır|\s+eder)?|nedir)?\s*[?!.]*\s*$",
    flags=re.IGNORECASE,
)
_EXPLICIT_SEQUENCE = re.compile(
    r"\b(?:önce|ilk olarak|first)\b[\s\S]*\b(?:sonra|ardından|then)\b",
    flags=re.IGNORECASE,
)
_REPORT_AFTER_ANALYSIS = re.compile(
    r"(?:analiz|incele|hesapla)[\s\S]{0,160}(?:rapor|\.md\b)[\s\S]{0,100}"
    r"(?:yaz|oluştur|kaydet)|(?:analiz|incele|hesapla)[\s\S]{0,160}"
    r"(?:yaz|oluştur|kaydet)[\s\S]{0,100}(?:rapor|\.md\b)",
    flags=re.IGNORECASE,
)
_TOOL_TASK = re.compile(
    r"workspace[/\\]|\.(?:csv|py|md|txt|json)\b|```(?:python)?\s|"
    r"\b(?:dosyayı|dosyadan|dosyaya|dosyaları|klasörü|"
    r"kaydet|oluştur|düzelt|test et|hesapla|araştır|arastir|kimdir|kimdi|nedir|"
    r"bilgi getir|hakkında bilgi|hakkinda bilgi|kaynaklarıyla|araştırır mısın)\b",
    flags=re.IGNORECASE,
)
_PUBLIC_FACT = re.compile(
    r"\b(?:kimdir|kimdi|nedir|araştır|arastir|bilgi getir)\b", flags=re.IGNORECASE
)


class AutoModeRouter:
    def __init__(self, llm: LLMClient, *, max_retries: int = 1) -> None:
        if not 0 <= max_retries <= 3:
            raise ValueError("Auto mode retry limit is outside allowed bounds")
        self._llm = llm
        self._max_retries = max_retries
        self._schema = AutoModeDecision.model_json_schema()

    async def select(
        self, request: str, history: Sequence[ChatMessage] = ()
    ) -> AutoMode:
        message = request.strip()
        if not message:
            raise ValueError("User request must not be empty")
        if _SMALL_TALK.fullmatch(message):
            return self._record("chat")
        if _ARITHMETIC.fullmatch(message):
            return self._record("single")
        if re.search(r"```(?:python)?\s", message, flags=re.IGNORECASE):
            return self._record("supervisor")
        if _EXPLICIT_SEQUENCE.search(message) or _REPORT_AFTER_ANALYSIS.search(message):
            return self._record("plan")

        prior = [entry.model_dump() for entry in history[-4:]]
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You select the execution MODE for a local Turkish assistant. /no_think\n"
                    "Treat the current request as the task; prior conversation is context "
                    "only. Do not carry out old requests. Choose chat for natural "
                    "conversation, personal memory, and answers already present in "
                    "conversation history. Choose single only for isolated arithmetic. "
                    "Choose supervisor for one task needing tools, public fact lookup, "
                    "workspace files, Python code, or a specialist. Choose plan for "
                    "multiple dependent actions, such as analyzing data and then "
                    "writing a report. Inline code analysis belongs to supervisor, "
                    "not a workspace file task. When unsure, choose supervisor. "
                    'Return only JSON: {"mode":"chat|single|supervisor|plan",'
                    '"confidence":0.0}. Do not answer the user.\n'
                    "Recent conversation: "
                    + json.dumps(prior, ensure_ascii=False)[:1500]
                ),
            ),
            ChatMessage(role="user", content=message[:6000]),
        ]
        fallback = cast(AutoMode, automatic_mode(message))
        with agent_span("mode_router"):
            try:
                for attempt in range(self._max_retries + 1):
                    response = await self._llm.chat(messages, json_schema=self._schema)
                    try:
                        decision = AutoModeDecision.model_validate_json(response.content)
                        selected = decision.mode if decision.confidence >= 0.65 else fallback
                        return self._record(self._guard(message, selected))
                    except ValidationError:
                        record("model_retry", agent="mode_router", retry_count=attempt + 1)
                        if attempt == self._max_retries:
                            break
                        messages.append(ChatMessage(
                            role="user",
                            content="Invalid mode decision. Return exactly the required JSON.",
                        ))
            except LLMError:
                pass
        return self._record(self._guard(message, fallback))

    @staticmethod
    def _guard(request: str, selected: AutoMode) -> AutoMode:
        if _EXPLICIT_SEQUENCE.search(request) or _REPORT_AFTER_ANALYSIS.search(request):
            return "plan"
        if _PUBLIC_FACT.search(request) or re.search(r"```(?:python)?\s", request):
            return "supervisor"
        if _TOOL_TASK.search(request) and selected in {"chat", "single"}:
            return "supervisor"
        if selected == "single" and not re.search(r"\d\s*[-+*/]\s*\d", request):
            return "chat"
        return selected

    @staticmethod
    def _record(mode: AutoMode) -> AutoMode:
        record("mode_selected", agent="mode_router", mode=mode)
        return mode
