from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import httpx

from .base import BaseLLMClient
from .models import LLMResponse, Message, Messages, StreamChunk, StreamEvent, ToolDef, ToolUse


class AnthropicClient(BaseLLMClient):
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        auth_mode: str = "x-api-key",
        model: Optional[str] = None,
        thinking: Optional[Dict[str, Any]] = None,
        timeout: float = 300.0,
    ) -> None:
        super().__init__(model=model)
        self._api_key = api_key
        self._base_url = base_url or "https://api.anthropic.com"
        self._auth_mode = auth_mode
        self._thinking = thinking
        self._http = httpx.Client(timeout=timeout)
        self._ahttp = httpx.AsyncClient(timeout=timeout)

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._auth_mode == "bearer":
            headers["Authorization"] = f"Bearer {self._api_key}"
        else:
            headers["x-api-key"] = self._api_key
            headers["anthropic-version"] = "2023-06-01"
        return headers

    @staticmethod
    def _build_tools(tools: Optional[List[ToolDef]]) -> Optional[List[Dict]]:
        if not tools:
            return None
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]

    @staticmethod
    def _messages_to_anthropic(messages: List[Message]) -> List[Dict]:
        result: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.tool_call_id:
                result.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content,
                        }
                    ],
                })
            elif msg.tool_calls:
                content_blocks: List[Dict] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["input"] if isinstance(tc["input"], dict) else json.loads(tc["input"]),
                    })
                result.append({"role": "assistant", "content": content_blocks})
            else:
                result.append({"role": msg.role, "content": msg.content})
        return result

    @staticmethod
    def _parse_response(resp: Dict) -> LLMResponse:
        content_text = ""
        thinking_text = ""
        tool_uses: List[ToolUse] = []

        for block in resp.get("content", []):
            block_type = block.get("type", "")
            if block_type == "text":
                content_text += block.get("text", "")
            elif block_type == "thinking":
                thinking_text += block.get("thinking", "")
            elif block_type == "tool_use":
                tool_uses.append(ToolUse(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    input=block.get("input", {}),
                ))

        usage = resp.get("usage", {})
        return LLMResponse(
            content=content_text,
            thinking=thinking_text,
            tool_uses=tool_uses,
            stop_reason=resp.get("stop_reason"),
            usage={
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            },
            raw=resp,
        )

    def _build_body(
        self,
        model: str,
        messages: List[Message],
        system: Optional[str],
        tools: Optional[List[ToolDef]],
        max_tokens: int,
        temperature: float,
        stream: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": model,
            "messages": self._messages_to_anthropic(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if system:
            body["system"] = system
        ant_tools = self._build_tools(tools)
        if ant_tools:
            body["tools"] = ant_tools
        thinking_config = kwargs.pop("thinking", self._thinking)
        if thinking_config:
            body["thinking"] = thinking_config
        body.update(kwargs)
        return body

    @staticmethod
    def _parse_stream_event(
        event: Dict,
        current_tool: Dict[str, Any],
        current_tool_index: Optional[int],
    ) -> tuple[list[StreamChunk], Dict[str, Any], Optional[int]]:
        chunks: list[StreamChunk] = []
        event_type = event.get("type", "")

        if event_type == "content_block_start":
            block = event.get("content_block", {})
            block_type = block.get("type", "")
            idx = event.get("index", 0)
            if block_type == "tool_use":
                current_tool = {"id": block.get("id", ""), "name": block.get("name", ""), "input": ""}
                current_tool_index = idx
                chunks.append(StreamChunk(
                    event=StreamEvent.TOOL_USE_START,
                    data={"id": current_tool["id"], "name": current_tool["name"], "index": idx},
                ))

        elif event_type == "content_block_delta":
            delta = event.get("delta", {})
            delta_type = delta.get("type", "")
            if delta_type == "text_delta":
                chunks.append(StreamChunk(event=StreamEvent.TEXT, data=delta.get("text", "")))
            elif delta_type == "thinking_delta":
                chunks.append(StreamChunk(event=StreamEvent.THINKING, data=delta.get("thinking", "")))
            elif delta_type == "input_json_delta":
                partial = delta.get("partial_json", "")
                if current_tool is not None:
                    current_tool["input"] += partial
                chunks.append(StreamChunk(
                    event=StreamEvent.TOOL_USE_DELTA,
                    data={"index": current_tool_index, "arguments": partial},
                ))

        elif event_type == "content_block_stop":
            if current_tool and current_tool_index is not None:
                try:
                    parsed = json.loads(current_tool["input"]) if current_tool["input"] else {}
                except json.JSONDecodeError:
                    parsed = {"raw": current_tool["input"]}
                chunks.append(StreamChunk(
                    event=StreamEvent.TOOL_USE_END,
                    data={
                        "index": current_tool_index,
                        "id": current_tool["id"],
                        "name": current_tool["name"],
                        "input": parsed,
                    },
                ))
                current_tool = {}
                current_tool_index = None

        elif event_type == "message_stop":
            chunks.append(StreamChunk(event=StreamEvent.DONE))

        return chunks, current_tool, current_tool_index

    def completion(
        self,
        messages: Messages,
        model: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[ToolDef]] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> LLMResponse:
        messages = self._normalize_messages(messages)
        model = model or self._default_model
        if not model:
            raise ValueError("model must be provided either at init or per-call")

        url = f"{self._base_url}/v1/messages"
        body = self._build_body(model, messages, system, tools, max_tokens, temperature, False, **kwargs)

        resp = self._http.post(url, headers=self._headers(), json=body)
        resp.raise_for_status()
        return self._parse_response(resp.json())

    async def async_completion(
        self,
        messages: Messages,
        model: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[ToolDef]] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> LLMResponse:
        messages = self._normalize_messages(messages)
        model = model or self._default_model
        if not model:
            raise ValueError("model must be provided either at init or per-call")

        url = f"{self._base_url}/v1/messages"
        body = self._build_body(model, messages, system, tools, max_tokens, temperature, False, **kwargs)

        resp = await self._ahttp.post(url, headers=self._headers(), json=body)
        resp.raise_for_status()
        return self._parse_response(resp.json())

    def stream(
        self,
        messages: Messages,
        model: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[ToolDef]] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        messages = self._normalize_messages(messages)
        model = model or self._default_model
        if not model:
            raise ValueError("model must be provided either at init or per-call")

        url = f"{self._base_url}/v1/messages"
        body = self._build_body(model, messages, system, tools, max_tokens, temperature, True, **kwargs)
        current_tool: Dict[str, Any] = {}
        current_tool_index: Optional[int] = None

        with self._http.stream("POST", url, headers=self._headers(), json=body) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:] if line.startswith("data: ") else line[5:].lstrip()
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                chunks, current_tool, current_tool_index = self._parse_stream_event(
                    event, current_tool, current_tool_index
                )
                for chunk in chunks:
                    yield chunk
                    if chunk.event == StreamEvent.DONE:
                        return

    async def async_stream(
        self,
        messages: Messages,
        model: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[ToolDef]] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        messages = self._normalize_messages(messages)
        model = model or self._default_model
        if not model:
            raise ValueError("model must be provided either at init or per-call")

        url = f"{self._base_url}/v1/messages"
        body = self._build_body(model, messages, system, tools, max_tokens, temperature, True, **kwargs)
        current_tool: Dict[str, Any] = {}
        current_tool_index: Optional[int] = None

        async with self._ahttp.stream("POST", url, headers=self._headers(), json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:] if line.startswith("data: ") else line[5:].lstrip()
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                chunks, current_tool, current_tool_index = self._parse_stream_event(
                    event, current_tool, current_tool_index
                )
                for chunk in chunks:
                    yield chunk
                    if chunk.event == StreamEvent.DONE:
                        return
