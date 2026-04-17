"""
Abstract LLM client interface and factory.
Adding a new model = implement LLMClientBase + register in LLMFactory.
"""
from abc import ABC, abstractmethod
from typing import AsyncIterator
import os


# Global current model (set by /model/switch endpoint)
_current_model = "minimax-m2.7"


def get_current_model() -> str:
    return _current_model


def set_current_model(model: str) -> None:
    global _current_model
    _current_model = model


class LLMClientBase(ABC):
    """Abstract interface all LLM clients must implement."""

    @abstractmethod
    async def astream(self, messages: list) -> AsyncIterator[dict]:
        """
        Stream LLM chunks. Each yielded dict has:
          - text: str (incremental text)
          - thinking: str | None (thinking block)
        """
        raise NotImplementedError

    @abstractmethod
    async def invoke(self, messages: list) -> dict:
        """
        Invoke LLM and return full response. Returns dict with:
          - text: str
          - thinking: str | None
        """
        raise NotImplementedError


# ── MiniMax / Anthropic-compatible ─────────────────────────────────────────

class MiniMaxClient(LLMClientBase):
    """MiniMax client via LangChain ChatAnthropic."""

    def __init__(self, model: str | None = None):
        from langchain_anthropic import ChatAnthropic
        auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN")
        base_url = os.getenv("ANTHROPIC_BASE_URL")
        model = model or os.getenv("ANTHROPIC_MODEL")
        if not auth_token or not model or not base_url:
            raise ValueError("ANTHROPIC related environment variable is not set")
        self._llm = ChatAnthropic(
            model=model,
            anthropic_api_key=auth_token,
            base_url=base_url,
        )

    async def astream(self, messages: list) -> AsyncIterator[dict]:
        import asyncio
        try:
            async for chunk in self._llm.astream(messages):
                yield _parse_langchain_chunk(chunk)
        except Exception as e:
            print(f"[MiniMaxClient] Stream error: {e}")
            raise

    async def invoke(self, messages: list) -> dict:
        import asyncio
        try:
            response = await asyncio.to_thread(self._llm.invoke, messages)
            return _parse_langchain_chunk(response)
        except Exception as e:
            print(f"[MiniMaxClient] Invoke error: {e}")
            raise


def _parse_langchain_chunk(chunk) -> dict:
    """Parse LangChain (MiniMax/Anthropic) streaming chunk."""
    result = {"text": "", "thinking": None}
    if hasattr(chunk, "content"):
        content = chunk.content
        if isinstance(content, list):
            for block in content:
                if hasattr(block, "type"):
                    if block.type == "thinking" and hasattr(block, "thinking"):
                        result["thinking"] = block.thinking
                    elif block.type == "text" and hasattr(block, "text"):
                        result["text"] += block.text
                elif isinstance(block, dict):
                    if block.get("type") == "thinking":
                        result["thinking"] = block.get("thinking")
                    elif block.get("type") == "text":
                        result["text"] += block.get("text", "")
        elif isinstance(content, str):
            result["text"] = content
    return result


# ── Ollama GLM-5.1 ───────────────────────────────────────────────────────────

class OllamaClient(LLMClientBase):
    """Ollama client for GLM-5.1 and other Ollama models."""

    def __init__(self, model: str = "glm-5.1:cloud"):
        from ollama import AsyncClient
        self._model = model
        self._client = AsyncClient()

    async def astream(self, messages: list) -> AsyncIterator[dict]:
        try:
            # Convert LangChain-style messages to Ollama format
            ollama_messages = _convert_messages(messages)
            stream = await self._client.chat(model=self._model, messages=ollama_messages, stream=True)
            async for chunk in stream:
                yield _parse_ollama_chunk(chunk)
        except Exception as e:
            print(f"[OllamaClient] Stream error: {e}")
            raise

    async def invoke(self, messages: list) -> dict:
        try:
            ollama_messages = _convert_messages(messages)
            response = await self._client.chat(model=self._model, messages=ollama_messages, stream=False)
            return _parse_ollama_chunk(response)
        except Exception as e:
            print(f"[OllamaClient] Invoke error: {e}")
            raise


def _convert_messages(messages: list) -> list[dict]:
    """Convert LangChain messages to Ollama message format."""
    result = []
    for m in messages:
        role = "user"
        if hasattr(m, "type"):
            if m.type == "human":
                role = "user"
            elif m.type == "ai":
                role = "assistant"
            elif m.type == "system":
                role = "system"
        elif hasattr(m, "role"):
            role = m.role
        content = ""
        if hasattr(m, "content"):
            content = m.content
        elif isinstance(m, dict):
            role = m.get("type", "user")
            content = m.get("content", "")
        result.append({"role": role, "content": content})
    return result


def _parse_ollama_chunk(chunk) -> dict:
    """Parse Ollama streaming/non-streaming chunk."""
    result = {"text": "", "thinking": None}
    if hasattr(chunk, "message") and hasattr(chunk.message, "content"):
        result["text"] = chunk.message.content
    elif isinstance(chunk, dict):
        msg = chunk.get("message", {})
        if isinstance(msg, dict):
            result["text"] = msg.get("content", "")
    return result


# ── Factory ─────────────────────────────────────────────────────────────────

_registered_models: dict[str, type[LLMClientBase]] = {
    "minimax-m2.7": MiniMaxClient,
    # "glm-5.1": OllamaClient,  # registered after ollama import confirmed
}


def register_model(model_id: str, client_class: type[LLMClientBase]) -> None:
    _registered_models[model_id] = client_class
    print(f"[LLMFactory] Registered model: {model_id} -> {client_class.__name__}")


def create_llm_client(model: str | None = None) -> LLMClientBase:
    """Factory: create the LLM client for the given model (or current active model)."""
    model = model or _current_model
    if model not in _registered_models:
        raise ValueError(f"Unknown model: {model}. Registered: {list(_registered_models.keys())}")
    return _registered_models[model]()


def is_llm_configured(model: str | None = None) -> bool:
    """Check if the given model (or current active model) is configured."""
    model = model or _current_model
    try:
        client = create_llm_client(model)
        return True
    except (ValueError, Exception):
        return False
