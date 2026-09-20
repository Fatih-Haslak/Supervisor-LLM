"""Explicit human approval for mutating tools."""

import asyncio
import json
import sys
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str = Field(min_length=1)
    arguments: dict[str, object]


class Approver(Protocol):
    async def request_approval(self, request: ApprovalRequest) -> bool: ...


class ToolApprovalError(Exception):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        super().__init__(kind)


class TerminalApprover:
    """Show the exact validated operation; only terminal input can approve it."""

    async def request_approval(self, request: ApprovalRequest) -> bool:
        if not sys.stdin.isatty():
            return False
        preview = json.dumps(
            {"tool": request.tool, "arguments": request.arguments}, ensure_ascii=True
        )
        if len(preview) > 8000:
            print("Onay önizlemesi terminal sınırını aşıyor; işlem reddedildi.")
            return False
        print(f"Onay isteniyor> {preview}")
        try:
            answer = await asyncio.to_thread(input, "Bu işlemi onaylıyor musunuz? [e/H] ")
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.strip().casefold() in {"e", "evet"}
