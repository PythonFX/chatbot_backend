from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from .models import LLMResponse, Message, Messages, StreamChunk, ToolDef


class BaseLLMClient(ABC):
    def __init__(self, model: Optional[str] = None) -> None:
        self._default_model = model

    @staticmethod
    def _normalize_messages(messages: Messages) -> List[Message]:
        if not messages:
            return messages
        if isinstance(messages[0], Message):
            return messages
        return [
            Message(
                role=m["role"],
                content=m.get("content") or "",
                tool_calls=m.get("tool_calls"),
                tool_call_id=m.get("tool_call_id"),
            )
            for m in messages
        ]

    @abstractmethod
    def completion(
        self,
        messages: Messages,
        model: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[ToolDef]] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> LLMResponse: ...

    @abstractmethod
    async def async_completion(
        self,
        messages: Messages,
        model: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[ToolDef]] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> LLMResponse: ...

    @abstractmethod
    def stream(
        self,
        messages: Messages,
        model: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[ToolDef]] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]: ...

    @abstractmethod
    async def async_stream(
        self,
        messages: Messages,
        model: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[ToolDef]] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]: ...

    def collect_stream(self, chunks: Iterator[StreamChunk]) -> LLMResponse:
        from .models import StreamEvent, ToolUse

        content = ""
        thinking = ""
        tool_uses: List[ToolUse] = []
        stop_reason = "end_turn"

        for chunk in chunks:
            if chunk.event == StreamEvent.TEXT:
                content += chunk.data
            elif chunk.event == StreamEvent.THINKING:
                thinking += chunk.data
            elif chunk.event == StreamEvent.TOOL_USE_END:
                tool_uses.append(ToolUse(
                    id=chunk.data["id"],
                    name=chunk.data["name"],
                    input=chunk.data["input"],
                ))
                stop_reason = "tool_use"
            elif chunk.event == StreamEvent.DONE:
                break

        if tool_uses:
            stop_reason = "tool_use"

        return LLMResponse(
            content=content,
            thinking=thinking,
            tool_uses=tool_uses,
            stop_reason=stop_reason,
        )

    async def async_collect_stream(self, chunks: AsyncIterator[StreamChunk]) -> LLMResponse:
        from .models import StreamEvent, ToolUse

        content = ""
        thinking = ""
        tool_uses: List[ToolUse] = []
        stop_reason = "end_turn"

        try:
            async for chunk in chunks:
                if chunk.event == StreamEvent.TEXT:
                    content += chunk.data
                elif chunk.event == StreamEvent.THINKING:
                    thinking += chunk.data
                elif chunk.event == StreamEvent.TOOL_USE_END:
                    tool_uses.append(ToolUse(
                        id=chunk.data["id"],
                        name=chunk.data["name"],
                        input=chunk.data["input"],
                    ))
                    stop_reason = "tool_use"
                elif chunk.event == StreamEvent.DONE:
                    break
        finally:
            if hasattr(chunks, "aclose"):
                await chunks.aclose()

        if tool_uses:
            stop_reason = "tool_use"

        return LLMResponse(
            content=content,
            thinking=thinking,
            tool_uses=tool_uses,
            stop_reason=stop_reason,
        )

    def close(self) -> None:
        if hasattr(self, "_http"):
            self._http.close()

    async def aclose(self) -> None:
        if hasattr(self, "_ahttp"):
            await self._ahttp.aclose()

    def __enter__(self) -> BaseLLMClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    async def __aenter__(self) -> BaseLLMClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()
