"""
LLM client factory using the external llm_client package.
Supports 3 models: minimax, glm5.1, kimi-k2.6 — switch via set_current_model().
"""
from typing import AsyncIterator

from llm_client import (
    LLMClient,
    Message,
    Provider,
    StreamEvent,
    create_anthropic_client,
    create_doubao_client,
    create_kimi_client,
    create_mlx_client,
)

# ── Model ↔ Provider mapping ────────────────────────────────────────────────

MODEL_TO_PROVIDER: dict[str, Provider] = {
    "Minimax": Provider.ANTHROPIC,
    "GLM-5.1": Provider.DOUBAO,
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

    # glm5.1 via Doubao API
    try:
        llm.add_client(Provider.DOUBAO, create_doubao_client(profile_name="doubao-glm"))
        print("[LLMFactory] Registered: glm5.1 (Doubao)")
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


# ── LangChain → llm_client Message conversion ──────────────────────────────

def _convert_langchain_messages(messages: list) -> list[Message]:
    """Convert LangChain messages (HumanMessage / AIMessage / SystemMessage)
    to llm_client.Message objects."""
    result: list[Message] = []
    for m in messages:
        role = "user"
        if hasattr(m, "type"):
            role = {"human": "user", "ai": "assistant", "system": "system"}.get(m.type, "user")
        elif hasattr(m, "role"):
            role = m.role
        content = m.content if hasattr(m, "content") else str(m)
        result.append(Message(role=role, content=content))
    return result


# ── Adapter: preserves the {"text", "thinking"} dict interface ──────────────

class _LLMClientAdapter:
    """Wraps llm_client.LLMClient and exposes the astream / invoke interface
    that the rest of the backend expects (LangChain messages in, dicts out)."""

    def __init__(self, model: str | None = None):
        self._llm = _init_llm_client()
        if model and model in MODEL_TO_PROVIDER:
            self._provider = MODEL_TO_PROVIDER[model]
        else:
            self._provider = None  # use default

    def _extract_system(self, llm_messages: list[Message]) -> tuple[Optional[str], list[Message]]:
        """Extract system message from the list and return (system_prompt, remaining_messages).

        Some providers (e.g. Doubao/GLM) don't support system messages in the message list
        but accept a separate `system=` parameter. This method separates them so the caller
        can pass the system prompt via the proper parameter.
        """
        system_parts = []
        non_system = []
        for m in llm_messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                non_system.append(m)
        system_prompt = "\n\n".join(system_parts) if system_parts else None
        return system_prompt, non_system

    async def astream(self, messages: list, system: Optional[str] = None) -> AsyncIterator[dict]:
        llm_messages = _convert_langchain_messages(messages)
        extracted_system, llm_messages = self._extract_system(llm_messages)
        # Caller-provided system takes precedence; otherwise use extracted
        system_prompt = system or extracted_system
        async for chunk in self._llm.async_stream(llm_messages, system=system_prompt, provider=self._provider):
            if chunk.event == StreamEvent.TEXT:
                yield {"text": chunk.data, "thinking": None}
            elif chunk.event == StreamEvent.THINKING:
                yield {"text": "", "thinking": chunk.data}
            elif chunk.event == StreamEvent.DONE:
                return

    def invoke(self, messages: list, system: Optional[str] = None) -> dict:
        llm_messages = _convert_langchain_messages(messages)
        extracted_system, llm_messages = self._extract_system(llm_messages)
        system_prompt = system or extracted_system
        response = self._llm.completion(llm_messages, system=system_prompt, provider=self._provider)
        return {"text": response.content, "thinking": response.thinking or None}


# ── Factory function (unchanged interface) ──────────────────────────────────

def create_llm_client(model: str | None = None) -> _LLMClientAdapter:
    """Return an adapter for the given model (or current active model)."""
    return _LLMClientAdapter(model=model or _current_model)


def is_llm_configured(model: str | None = None) -> bool:
    """Check if the given model (or current active model) is configured."""
    model = model or _current_model
    try:
        llm = _init_llm_client()
        provider = MODEL_TO_PROVIDER.get(model)
        return provider is not None and provider in llm._clients
    except Exception:
        return False
