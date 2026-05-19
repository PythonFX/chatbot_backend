from __future__ import annotations

import threading
import time
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Iterator, List, Optional, Union

import httpx

from .base import BaseLLMClient
from .models import AzureTokenProvider, LLMResponse, Message, Messages, StreamChunk, StreamEvent, ToolDef, ToolUse
from .openai_client import OpenAIClient


class AzureClient(BaseLLMClient):
    def __init__(
        self,
        deployment: str,
        endpoint: str,
        api_version: str = "2024-06-01",
        api_key: Optional[str] = None,
        ad_token_provider: Optional[AzureTokenProvider] = None,
        ad_token_ttl: float = 1800.0,
        timeout: float = 300.0,
    ) -> None:
        super().__init__(model=deployment or f"azure/{deployment}")
        self._deployment = deployment
        self._endpoint = endpoint.rstrip("/")
        self._api_version = api_version
        self._api_key = api_key
        self._ad_token_provider = ad_token_provider
        self._ad_token_ttl = ad_token_ttl
        self._cached_ad_token: str = ""
        self._cached_ad_token_at: float = 0.0
        self._token_lock = threading.Lock()
        self._http = httpx.Client(timeout=timeout)
        self._ahttp = httpx.AsyncClient(timeout=timeout)

    def _is_ad_token_expired(self) -> bool:
        if not self._cached_ad_token:
            return True
        return (time.monotonic() - self._cached_ad_token_at) >= self._ad_token_ttl

    def _resolve_azure_token(self) -> str:
        if self._api_key:
            return ""
        with self._token_lock:
            if not self._is_ad_token_expired():
                return self._cached_ad_token
        provider = self._ad_token_provider
        if provider is None:
            return ""
        if isinstance(provider, str):
            token = provider
        else:
            token = provider()
        with self._token_lock:
            self._cached_ad_token = token
            self._cached_ad_token_at = time.monotonic()
        return token

    async def _async_resolve_azure_token(self) -> str:
        if self._api_key:
            return ""
        with self._token_lock:
            if not self._is_ad_token_expired():
                return self._cached_ad_token
        provider = self._ad_token_provider
        if provider is None:
            return ""
        if isinstance(provider, str):
            token = provider
        else:
            result = provider()
            if isinstance(result, Awaitable):
                token = await result
            else:
                token = result
        with self._token_lock:
            self._cached_ad_token = token
            self._cached_ad_token_at = time.monotonic()
        return token

    def _headers(self, token: Optional[str] = None) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["api-key"] = self._api_key
        elif token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _url(self, deployment: Optional[str] = None) -> str:
        dep = deployment or self._deployment
        if not dep:
            raise ValueError("azure_deployment must be provided either at init or per-call")
        return (
            f"{self._endpoint}/openai/deployments/{dep}"
            f"/chat/completions?api-version={self._api_version}"
        )

    def _build_body(
        self,
        messages: List[Message],
        system: Optional[str],
        tools: Optional[List[ToolDef]],
        max_tokens: int,
        temperature: float,
        stream: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "messages": OpenAIClient._messages_to_openai(messages, system),
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        oai_tools = OpenAIClient._build_tools(tools)
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

        token = self._resolve_azure_token()
        deployment = model if model != f"azure/{self._deployment}" else None
        url = self._url(deployment)
        body = self._build_body(messages, system, tools, max_tokens, temperature, False, **kwargs)

        resp = self._http.post(url, headers=self._headers(token), json=body)
        resp.raise_for_status()
        return OpenAIClient._parse_response(resp.json())

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

        token = await self._async_resolve_azure_token()
        deployment = model if model != f"azure/{self._deployment}" else None
        url = self._url(deployment)
        body = self._build_body(messages, system, tools, max_tokens, temperature, False, **kwargs)

        resp = await self._ahttp.post(url, headers=self._headers(token), json=body)
        resp.raise_for_status()
        return OpenAIClient._parse_response(resp.json())

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

        token = self._resolve_azure_token()
        deployment = model if model != f"azure/{self._deployment}" else None
        url = self._url(deployment)
        body = self._build_body(messages, system, tools, max_tokens, temperature, True, **kwargs)
        current_tools: Dict[int, Dict[str, Any]] = {}

        with self._http.stream("POST", url, headers=self._headers(token), json=body) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                for chunk in OpenAIClient._parse_stream_chunks(line, current_tools):
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

        token = await self._async_resolve_azure_token()
        deployment = model if model != f"azure/{self._deployment}" else None
        url = self._url(deployment)
        body = self._build_body(messages, system, tools, max_tokens, temperature, True, **kwargs)
        current_tools: Dict[int, Dict[str, Any]] = {}

        async with self._ahttp.stream("POST", url, headers=self._headers(token), json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                for chunk in OpenAIClient._parse_stream_chunks(line, current_tools):
                    yield chunk
                    if chunk.event == StreamEvent.DONE:
                        return
