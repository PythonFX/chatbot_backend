from .models import (
    AzureTokenProvider,
    LLMResponse,
    Message,
    Messages,
    Provider,
    StreamChunk,
    StreamEvent,
    ThinkingBlock,
    ToolDef,
    ToolUse,
    detect_provider,
)
from .base import BaseLLMClient
from .openai_client import OpenAIClient
from .azure_client import AzureClient
from .anthropic_client import AnthropicClient
from .llm_client import LLMClient
from .llm_factory import (
    create_anthropic_client, create_azure_client, create_doubao_client, create_from_profiles,
    create_kimi_client, create_llm_client, create_mlx_client, create_zhipu_client,
    create_openai_client, get_config, get_profile
)

__all__ = [
    "AzureClient",
    "AzureTokenProvider",
    "BaseLLMClient",
    "LLMClient",
    "LLMResponse",
    "Message",
    "Messages",
    "OpenAIClient",
    "AnthropicClient",
    "MlxClient",
    "Provider",
    "StreamChunk",
    "StreamEvent",
    "ThinkingBlock",
    "ToolDef",
    "ToolUse",
    "detect_provider",
    "create_anthropic_client",
    "create_azure_client",
    "create_doubao_client",
    "create_from_profiles",
    "create_kimi_client",
    "create_llm_client",
    "create_zhipu_client",
    "create_mlx_client",
    "create_openai_client",
    "get_config",
    "get_profile",
]


def __getattr__(name: str):
    if name == "MlxClient":
        from .mlx_client import MlxClient

        return MlxClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
