"""
LLM client factory using the external llm_client package.
Supports 4 models: minimax, glm5.1, kimi-k2.6, gemma4-e4b — switch via set_current_model().
"""
from typing import AsyncIterator, Iterator, Optional

from llm_client import (
    LLMClient,
    LLMResponse,
    Message,
    Provider,
    StreamChunk,
    create_anthropic_client,
    create_kimi_client,
    create_mlx_client,
    create_zhipu_client
)

# ── Model ↔ Provider mapping ────────────────────────────────────────────────

MODEL_TO_PROVIDER: dict[str, Provider] = {
    "Minimax": Provider.ANTHROPIC,
    "GLM-5.1": Provider.ZHIPU,
    "Kimi K2.6": Provider.KIMI,
    "Gemma4-e4b": Provider.MLX,
}

# Display names for models (used in group chat and UI)
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "Minimax": "Minimax",
    "GLM-5.1": "GLM-5.1",
    "Kimi K2.6": "Kimi K2.6",
    "Gemma4-e4b": "Gemma4-e4b",
}

_current_model = "Minimax"

# ── Singleton LLMClient ────────────────────────────────────────────────────

_llm_client: LLMClient | None = None


def _init_llm_client() -> LLMClient:
    """Initialise the LLMClient with all 3 providers."""
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    llm = LLMClient(default_provider=Provider.ANTHROPIC)

    # minimax via Anthropic-compatible API
    try:
        llm.add_client(Provider.ANTHROPIC, create_anthropic_client(profile_name="minimax-anthropic"), default=True)
        print("[LLMFactory] Registered: minimax (Anthropic)")
    except KeyError as e:
        print(f"[LLMFactory] Skipped minimax: {e}")

    # glm5.1 via zhipu API
    try:
        llm.add_client(Provider.DOUBAO, create_zhipu_client())
        print("[LLMFactory] Registered: glm5.1")
    except KeyError as e:
        print(f"[LLMFactory] Skipped glm5.1: {e}")

    # kimi-k2.6 via Kimi API
    try:
        llm.add_client(Provider.KIMI, create_kimi_client(profile_name="kimi-k26"))
        print("[LLMFactory] Registered: kimi-k2.6 (Kimi)")
    except KeyError as e:
        print(f"[LLMFactory] Skipped kimi-k2.6: {e}")
        
    # Gemma4 via Mlx Client
    try:
        llm.add_client(Provider.MLX, create_mlx_client())
        print("[LLMFactory] Registered: Gemma-e4b (Mlx)")
    except KeyError as e:
        print(f"[LLMFactory] Skipped Gemma-e4b: {e}")

    _llm_client = llm
    return _llm_client


# ── Public helpers ──────────────────────────────────────────────────────────

def get_current_model() -> str:
    return _current_model


def set_current_model(model: str) -> None:
    global _current_model
    if model not in MODEL_TO_PROVIDER:
        raise ValueError(f"Unknown model: {model}. Available: {list(MODEL_TO_PROVIDER.keys())}")
    _current_model = model
    llm = _init_llm_client()
    llm.set_default_provider(MODEL_TO_PROVIDER[model])
    print(f"[LLMFactory] Switched to: {model} (Provider.{MODEL_TO_PROVIDER[model].value})")


def get_model_provider(model: str) -> Provider:
    """Return the Provider enum for a given model name."""
    if model not in MODEL_TO_PROVIDER:
        raise ValueError(f"Unknown model: {model}. Available: {list(MODEL_TO_PROVIDER.keys())}")
    return MODEL_TO_PROVIDER[model]


def get_available_models() -> list[str]:
    """Return model names whose providers are registered."""
    llm = _init_llm_client()
    return [m for m, p in MODEL_TO_PROVIDER.items() if p in llm._clients]


def get_model_info() -> list[dict]:
    """Return list of {id, display_name} for all available models."""
    llm = _init_llm_client()
    registered = {p for p in Provider}
    return [
        {"id": m, "display_name": MODEL_DISPLAY_NAMES.get(m, m)}
        for m, p in MODEL_TO_PROVIDER.items()
        if p in llm._clients
    ]


# ── ModelLLMClient: thin wrapper routing to a specific model's provider ──────

class ModelLLMClient:
    """Thin wrapper around LLMClient that routes to a specific model's provider.

    Accepts llm_client.Message objects directly.
    Exposes completion/async_completion/stream/async_stream with the system
    prompt extraction needed by some providers (Doubao/GLM).
    """

    def __init__(self, model: str | None = None):
        self._llm = _init_llm_client()
        if model and model in MODEL_TO_PROVIDER:
            self._provider = MODEL_TO_PROVIDER[model]
        else:
            self._provider = None  # use default

    def _extract_system(self, messages: list[Message]) -> tuple[Optional[str], list[Message]]:
        """Extract system messages from the list and return (system_prompt, remaining_messages).

        Some providers (e.g. Doubao/GLM) don't support system messages in the message list
        but accept a separate `system=` parameter.
        """
        system_parts = []
        non_system = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                non_system.append(m)
        system_prompt = "\n\n".join(system_parts) if system_parts else None
        return system_prompt, non_system

    def completion(self, messages: list[Message], system: Optional[str] = None, **kwargs) -> LLMResponse:
        extracted_system, filtered = self._extract_system(messages)
        system_prompt = system or extracted_system
        return self._llm.completion(filtered, system=system_prompt, provider=self._provider, **kwargs)

    async def async_completion(self, messages: list[Message], system: Optional[str] = None, **kwargs) -> LLMResponse:
        extracted_system, filtered = self._extract_system(messages)
        system_prompt = system or extracted_system
        return await self._llm.async_completion(filtered, system=system_prompt, provider=self._provider, **kwargs)

    def stream(self, messages: list[Message], system: Optional[str] = None, **kwargs) -> Iterator[StreamChunk]:
        extracted_system, filtered = self._extract_system(messages)
        system_prompt = system or extracted_system
        return self._llm.stream(filtered, system=system_prompt, provider=self._provider, **kwargs)

    async def async_stream(self, messages: list[Message], system: Optional[str] = None, **kwargs) -> AsyncIterator[StreamChunk]:
        extracted_system, filtered = self._extract_system(messages)
        system_prompt = system or extracted_system
        async for chunk in self._llm.async_stream(filtered, system=system_prompt, provider=self._provider, **kwargs):
            yield chunk


# ── Factory function (unchanged interface) ──────────────────────────────────

def create_llm_client(model: str | None = None) -> ModelLLMClient:
    """Return a client for the given model (or current active model)."""
    return ModelLLMClient(model=model or _current_model)


def is_llm_configured(model: str | None = None) -> bool:
    """Check if the given model (or current active model) is configured."""
    model = model or _current_model
    try:
        llm = _init_llm_client()
        provider = MODEL_TO_PROVIDER.get(model)
        return provider is not None and provider in llm._clients
    except Exception:
        return False
