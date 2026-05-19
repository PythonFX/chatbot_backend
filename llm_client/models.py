from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    DOUBAO = "doubao"
    KIMI = "kimi"
    ZHIPU = "zhipu"
    MLX = "mlx"


class ToolUse(BaseModel):
    id: str
    name: str
    input: Dict[str, Any] = Field(default_factory=dict)


class ThinkingBlock(BaseModel):
    thinking: str = ""


class LLMResponse(BaseModel):
    content: str = ""
    thinking: str = ""
    tool_uses: List[ToolUse] = Field(default_factory=list)
    stop_reason: Optional[str] = None
    usage: Dict[str, int] = Field(default_factory=dict)
    raw: Any = Field(default=None, exclude=True)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_uses) > 0


class StreamEvent(str, Enum):
    TEXT = "text"
    THINKING = "thinking"
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_DELTA = "tool_use_delta"
    TOOL_USE_END = "tool_use_end"
    DONE = "done"


@dataclass
class StreamChunk:
    event: StreamEvent
    data: Any = None


@dataclass
class ToolDef:
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    role: str
    content: str = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


Messages = Union[List[Message], List[Dict[str, Any]]]


AzureTokenProvider = Union[str, Callable[[], str], Callable[[], Awaitable[str]]]


def detect_provider(model: str) -> Provider:
    if model.startswith("claude"):
        return Provider.ANTHROPIC
    if "minimax" in model.lower():
        return Provider.ANTHROPIC
    if "doubao" in model.lower():
        return Provider.DOUBAO
    if "kimi" in model.lower():
        return Provider.KIMI
    model_lower = model.strip().lower()
    if model_lower.startswith(("gpt-", "o1", "o3", "o4")):
        return Provider.OPENAI
    return Provider.ANTHROPIC
