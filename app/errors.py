"""Stable user-facing error codes without raw exception or traceback text."""

import sqlite3
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from app.agents.single import AgentLimitError
from app.agents.supervisor import SupervisorLimitError
from app.llm.client import LLMContextOverflowError, LLMError, LLMTimeoutError
from app.llm.structured import StructuredOutputError
from app.security.approvals import ToolApprovalError

ErrorCode = Literal[
    "LLM_TIMEOUT", "CONTEXT_OVERFLOW", "INVALID_OUTPUT", "AGENT_LIMIT",
    "SUPERVISOR_LIMIT", "LLM_FAILURE", "INTERNAL_ERROR",
    "INVALID_INPUT", "STORAGE_FAILURE",
    "APPROVAL_REQUIRED", "APPROVAL_DENIED",
]


class ErrorInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: ErrorCode
    message: str
    retryable: bool


def describe_error(exc: Exception) -> ErrorInfo:
    if isinstance(exc, ToolApprovalError):
        if exc.kind == "ApprovalRequired":
            return ErrorInfo(
                code="APPROVAL_REQUIRED", message="Araç işlemi için insan onayı gerekiyor.",
                retryable=True,
            )
        return ErrorInfo(
            code="APPROVAL_DENIED", message="Araç işlemi onaylanmadı.", retryable=False
        )
    if isinstance(exc, LLMTimeoutError):
        return ErrorInfo(
            code="LLM_TIMEOUT", message="Yerel modelin süre sınırı aşıldı.", retryable=True
        )
    if isinstance(exc, LLMContextOverflowError):
        return ErrorInfo(
            code="CONTEXT_OVERFLOW",
            message="İstek modelin bağlam penceresine sığmıyor; daha kısa bir istek deneyin.",
            retryable=False,
        )
    if isinstance(exc, StructuredOutputError):
        return ErrorInfo(
            code="INVALID_OUTPUT",
            message="Model tekrar denemelerden sonra geçerli yapılandırılmış karar üretemedi.",
            retryable=True,
        )
    if isinstance(exc, AgentLimitError):
        return ErrorInfo(
            code="AGENT_LIMIT", message="Agent adım veya araç çağrısı sınırına ulaştı.",
            retryable=False,
        )
    if isinstance(exc, SupervisorLimitError):
        return ErrorInfo(
            code="SUPERVISOR_LIMIT", message="Supervisor tur veya plan sınırına ulaştı.",
            retryable=False,
        )
    if isinstance(exc, LLMError):
        return ErrorInfo(
            code="LLM_FAILURE",
            message="Yerel model çalıştırılamadı; model yolunu ve Python ortamını kontrol edin.",
            retryable=False,
        )
    if isinstance(exc, (ValidationError, ValueError)):
        return ErrorInfo(
            code="INVALID_INPUT", message="Girdi veya ayar doğrulanamadı.", retryable=False
        )
    if isinstance(exc, (sqlite3.Error, OSError)):
        return ErrorInfo(
            code="STORAGE_FAILURE", message="Yerel dosya veya bellek erişimi başarısız.",
            retryable=False,
        )
    return ErrorInfo(
        code="INTERNAL_ERROR", message="Beklenmeyen bir hata oluştu.", retryable=False
    )


def format_cli_error(exc: Exception) -> str:
    info = describe_error(exc)
    return f"Hata [{info.code}]: {info.message}"
