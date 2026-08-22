"""
LLM client management for the chatbot backend.

Models are identified by their profile name in ~/.llm_client_models.yaml
(e.g. "minimax-anthropic", "zhipu-glm", "kimi", "gemma4"). The provider and
display name for each model are read from that yaml profile — no hardcoded
model→provider or model→display-name mappings live in this repo.

Clients are created lazily on first use, and each model can be enabled or
disabled at runtime (persisted in the settings db), so subscriptions can be
turned on/off from the UI without restarting the server.
"""
import json
from typing import AsyncIterator, Iterator, Optional

from llm_client import (
    LLMClient,
    LLMResponse,
    Message,
    StreamChunk,
    create_anthropic_client,
    create_doubao_client,
    create_kimi_client,
    create_mlx_client,
    create_openai_client,
    create_zhipu_client,
    get_profile,
)

# ── Chat models offered by this app, keyed by yaml profile name ─────────────

CHAT_PROFILES: list[str] = ["minimax-anthropic", "zhipu-glm", "kimi", "gemma4"]

# Code-level dispatch: the yaml profile's `provider` field → its factory.
_PROVIDER_FACTORIES = {
    "anthropic": create_anthropic_client,
    "openai": create_openai_client,
    "zhipu": create_zhipu_client,
    "doubao": create_doubao_client,
    "kimi": create_kimi_client,
    "mlx": create_mlx_client,
}

_ENABLED_MODELS_KEY = "enabled_models"

_current_model: str | None = None

# ── Singleton LLMClient (clients registered by profile name) ────────────────

_llm_client: LLMClient | None = None


def _get_llm() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def _get_profile_or_none(model: str) -> dict | None:
    try:
        return get_profile(model)
    except Exception:
        return None


def _ensure_client(model: str):
    """Lazily create and register the client for a profile. Returns it or None."""
    llm = _get_llm()
    if model in llm._clients:
        return llm._clients[model]
    profile = _get_profile_or_none(model)
    if profile is None:
        print(f"[LLMManager] Skipped {model}: profile not found in config")
        return None
    provider = profile.get("provider")
    factory = _PROVIDER_FACTORIES.get(provider)
    if factory is None:
        print(f"[LLMManager] Skipped {model}: unsupported provider '{provider}'")
        return None
    try:
        client = factory(profile_name=model)
    except Exception as e:
        print(f"[LLMManager] Skipped {model}: {e}")
        return None
    llm.add_client(model, client)
    print(f"[LLMManager] Registered: {model} ({provider})")
    return client


# ── Enabled/disabled state (persisted in the settings db) ───────────────────

def _get_enabled_models() -> list[str]:
    from services.db_service import db_get_setting
    raw = db_get_setting(_ENABLED_MODELS_KEY)
    if raw is None:
        return list(CHAT_PROFILES)
    try:
        enabled = set(json.loads(raw))
    except (json.JSONDecodeError, TypeError):
        return list(CHAT_PROFILES)
    return [m for m in CHAT_PROFILES if m in enabled]


def _save_enabled_models(enabled: list[str]) -> None:
    from services.db_service import db_set_setting
    db_set_setting(_ENABLED_MODELS_KEY, json.dumps(enabled))


def set_model_enabled(model: str, enabled: bool) -> None:
    """Enable or disable a model at runtime (no restart needed)."""
    if model not in CHAT_PROFILES:
        raise ValueError(f"Unknown model: {model}. Available: {CHAT_PROFILES}")
    current = _get_enabled_models()
    if enabled:
        if model in current:
            return
        # Verify the client can actually be created before enabling
        if _ensure_client(model) is None:
            raise ValueError(f"Cannot enable '{model}': failed to create client (check config/keys)")
        current.append(model)
        _save_enabled_models(current)
    else:
        if model not in current:
            return
        if len(current) == 1:
            raise ValueError("Cannot disable the last enabled model")
        current.remove(model)
        _save_enabled_models(current)
        if _current_model == model:
            set_current_model(current[0])


# ── Public helpers ──────────────────────────────────────────────────────────

def get_current_model() -> str:
    global _current_model
    if _current_model is None:
        available = get_available_models()
        _current_model = available[0] if available else CHAT_PROFILES[0]
    return _current_model


def set_current_model(model: str) -> None:
    global _current_model
    available = get_available_models()
    if model not in available:
        raise ValueError(f"Unknown or disabled model: {model}. Available: {available}")
    _current_model = model
    if _ensure_client(model) is not None:
        _get_llm().set_default_provider(model)
    print(f"[LLMManager] Switched to: {model}")


def get_model_provider(model: str) -> str:
    """Return the provider string from the yaml profile (e.g. 'anthropic', 'mlx')."""
    if model not in CHAT_PROFILES:
        raise ValueError(f"Unknown model: {model}. Available: {CHAT_PROFILES}")
    profile = get_profile(model)
    return profile.get("provider", "")


def get_display_name(model: str) -> str:
    """Display name for a model: the `model` field from its yaml profile."""
    profile = _get_profile_or_none(model)
    if profile:
        return profile.get("model") or model
    return model


def get_available_models() -> list[str]:
    """Enabled profiles that exist in the yaml config."""
    return [m for m in _get_enabled_models() if _get_profile_or_none(m) is not None]


def get_model_info() -> list[dict]:
    """Return {id, display_name, provider, configured, enabled} for every chat model."""
    enabled = set(_get_enabled_models())
    info = []
    for m in CHAT_PROFILES:
        profile = _get_profile_or_none(m)
        info.append({
            "id": m,
            "display_name": (profile or {}).get("model") or m,
            "provider": (profile or {}).get("provider"),
            "configured": profile is not None,
            "enabled": m in enabled and profile is not None,
        })
    return info


# ── ModelLLMClient: thin wrapper routing to a specific model's profile ───────

class ModelLLMClient:
    """Thin wrapper around LLMClient that routes to a specific model's profile.

    Accepts llm_client.Message objects directly.
    Exposes completion/async_completion/stream/async_stream with the system
    prompt extraction needed by some providers (Doubao/GLM).
    """

    def __init__(self, model: str | None = None):
        self._llm = _get_llm()
        if model is None:
            self._provider = None  # use default
        elif model in CHAT_PROFILES and _ensure_client(model) is not None:
            self._provider = model
        else:
            raise ValueError(f"Model '{model}' is not available")

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
    return ModelLLMClient(model=model or get_current_model())


def is_llm_configured(model: str | None = None) -> bool:
    """Check if the given model (or current active model) is configured."""
    model = model or get_current_model()
    if model not in CHAT_PROFILES:
        return False
    try:
        return _ensure_client(model) is not None
    except Exception:
        return False
