from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from .base import BaseLLMClient
from .models import LLMResponse, Message, Messages, Provider, StreamChunk, ToolDef


def _resolve_key(value: Union[Provider, str]) -> str:
    if isinstance(value, Provider):
        return value.value
    return str(value)


class LLMClient(BaseLLMClient):
    def __init__(self, default_provider: Optional[Union[Provider, str]] = None) -> None:
        super().__init__()
        self._clients: Dict[str, BaseLLMClient] = {}
        self._default_provider: Optional[str] = _resolve_key(default_provider) if default_provider else None

    def add_client(self, name: Union[Provider, str], client: BaseLLMClient, default: bool = False) -> None:
        key = _resolve_key(name)
        self._clients[key] = client
        if default or not self._default_provider:
            self._default_provider = key

    def remove_client(self, name: Union[Provider, str]) -> None:
        key = _resolve_key(name)
        if key not in self._clients:
            raise KeyError(f"Client '{key}' not found")
        del self._clients[key]
        if self._default_provider == key:
            self._default_provider = next(iter(self._clients), None)

    def get_client(self, name: Optional[Union[Provider, str]] = None) -> BaseLLMClient:
        key = _resolve_key(name) if name else self._default_provider
        if not key:
            raise ValueError("No LLM client registered. Use add_client() first.")
        if key not in self._clients:
            raise KeyError(f"Client '{key}' not found. Available: {list(self._clients.keys())}")
        return self._clients[key]

    @property
    def default_provider(self) -> Optional[str]:
        return self._default_provider

    def set_default_provider(self, name: Union[Provider, str]) -> None:
        key = _resolve_key(name)
        if key not in self._clients:
            raise KeyError(f"Client '{key}' not found. Available: {list(self._clients.keys())}")
        self._default_provider = key

    @property
    def available_profiles(self) -> List[str]:
        return list(self._clients.keys())

    @property
    def available_providers(self) -> List[str]:
        return self.available_profiles

    def completion(
        self,
        messages: Messages,
        model: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[ToolDef]] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        messages = self._normalize_messages(messages)
        client = self.get_client(provider)
        return client.completion(messages, model=model, system=system, tools=tools,
                                 max_tokens=max_tokens, temperature=temperature, **kwargs)

    async def async_completion(
        self,
        messages: Messages,
        model: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[ToolDef]] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        messages = self._normalize_messages(messages)
        client = self.get_client(provider)
        return await client.async_completion(messages, model=model, system=system, tools=tools,
                                             max_tokens=max_tokens, temperature=temperature, **kwargs)

    def stream(
        self,
        messages: Messages,
        model: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[ToolDef]] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        messages = self._normalize_messages(messages)
        client = self.get_client(provider)
        yield from client.stream(messages, model=model, system=system, tools=tools,
                                 max_tokens=max_tokens, temperature=temperature, **kwargs)

    async def async_stream(
        self,
        messages: Messages,
        model: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[ToolDef]] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        messages = self._normalize_messages(messages)
        client = self.get_client(provider)
        async for chunk in client.async_stream(messages, model=model, system=system, tools=tools,
                                               max_tokens=max_tokens, temperature=temperature, **kwargs):
            yield chunk

    def close(self) -> None:
        for client in self._clients.values():
            client.close()

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
