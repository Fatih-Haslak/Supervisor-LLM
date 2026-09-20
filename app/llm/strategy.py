"""Explicit Phase 20 strategy: all roles share one local model backend."""

from dataclasses import dataclass
from typing import Literal

from app.llm.client import LLMClient

ModelRole = Literal[
    "chat", "single", "supervisor", "planner", "router", "reviewer",
    "general", "researcher", "coder", "file_agent", "data_agent", "writer",
]


@dataclass(frozen=True)
class SharedModelStrategy:
    """Role selection seam with exactly one live LLM client and model instance."""

    backend: LLMClient

    def for_role(self, role: ModelRole) -> LLMClient:
        # Keep role selection explicit without loading another GGUF into VRAM.
        return self.backend
