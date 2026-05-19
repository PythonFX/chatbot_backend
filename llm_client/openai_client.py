from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import httpx

from .base import BaseLLMClient
from .models import LLMResponse, Message, Messages, StreamChunk, StreamEvent, ToolDef, ToolUse


class OpenAIClient(BaseLLMClient):
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 300.0,
    ) -> None:
        super().__init__(model=model)
        self._api_key = api_key
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._http = httpx.Client(timeout=timeout)
        self._ahttp = httpx.AsyncClient(timeout=timeout)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _build_tools(tools: Optional[List[ToolDef]]) -> Optional[List[Dict]]:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    @staticmethod
    def _messages_to_openai(messages: List[Message], system: Optional[str] = None) -> List[Dict]:
        result: List[Dict[str, Any]] = []
        if system:
            result.append({"role": "system", "content": system})

        for msg in messages:
            if msg.tool_call_id:
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
            elif msg.tool_calls:
                result.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["input"]) if isinstance(tc["input"], dict) else tc["input"],
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })
            else:
                result.append({"role": msg.role, "content": msg.content})
        return result

    @staticmethod
    def _parse_response(resp: Dict) -> LLMResponse:
        choice = resp.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        finish_reason = choice.get("finish_reason", "stop")

        stop_reason = "end_turn"
        if finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"

        tool_uses: List[ToolUse] = []
        for tc in message.get("tool_calls", []):
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            tool_uses.append(ToolUse(id=tc.get("id", ""), name=fn.get("name", ""), input=args))

        usage = resp.get("usage", {})
        return LLMResponse(
            content=content,
            tool_uses=tool_uses,
            stop_reason=stop_reason,
            usage={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
            raw=resp,
        )

    @staticmethod
    def _parse_stream_chunks(line: str, current_tools: Dict[int, Dict[str, Any]]) -> List[StreamChunk]:
        chunks: List[StreamChunk] = []
        if not line or not line.startswith("data: "):
            return chunks
        data = line[6:]
        if data.strip() == "[DONE]":
            chunks.append(StreamChunk(event=StreamEvent.DONE))
            return chunks

        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            return chunks

        choices = chunk.get("choices", [])
        if not choices:
            return chunks

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        if delta.get("content"):
            chunks.append(StreamChunk(event=StreamEvent.TEXT, data=delta["content"]))

        for tc in delta.get("tool_calls", []):
            idx = tc.get("index", 0)
            if idx not in current_tools:
                current_tools[idx] = {
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": "",
                }
                chunks.append(StreamChunk(
                    event=StreamEvent.TOOL_USE_START,
                    data={"id": current_tools[idx]["id"], "name": current_tools[idx]["name"], "index": idx},
                ))

            fn_delta = tc.get("function", {})
            if fn_delta.get("name"):
                current_tools[idx]["name"] = fn_delta["name"]
            if fn_delta.get("arguments"):
                current_tools[idx]["arguments"] += fn_delta["arguments"]
                chunks.append(StreamChunk(
                    event=StreamEvent.TOOL_USE_DELTA,
                    data={"index": idx, "arguments": fn_delta["arguments"]},
                ))

        if finish_reason:
            for idx in sorted(current_tools):
                args_str = current_tools[idx]["arguments"]
                try:
                    args = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    args = {"raw": args_str}
                chunks.append(StreamChunk(
                    event=StreamEvent.TOOL_USE_END,
                    data={
                        "index": idx,
                        "id": current_tools[idx]["id"],
                        "name": current_tools[idx]["name"],
                        "input": args,
                    },
                ))
            chunks.append(StreamChunk(event=StreamEvent.DONE))

        return chunks

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
            "messages": self._messages_to_openai(messages, system),
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        oai_tools = self._build_tools(tools)
        if oai_tools:
            body["tools"] = oai_tools
        body.update(kwargs)
        return body

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

        url = f"{self._base_url}/chat/completions"
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

        url = f"{self._base_url}/chat/completions"
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

        url = f"{self._base_url}/chat/completions"
        body = self._build_body(model, messages, system, tools, max_tokens, temperature, True, **kwargs)
        current_tools: Dict[int, Dict[str, Any]] = {}

        with self._http.stream("POST", url, headers=self._headers(), json=body) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                for chunk in self._parse_stream_chunks(line, current_tools):
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

        url = f"{self._base_url}/chat/completions"
        body = self._build_body(model, messages, system, tools, max_tokens, temperature, True, **kwargs)
        current_tools: Dict[int, Dict[str, Any]] = {}

        async with self._ahttp.stream("POST", url, headers=self._headers(), json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                for chunk in self._parse_stream_chunks(line, current_tools):
                    yield chunk
                    if chunk.event == StreamEvent.DONE:
                        return
