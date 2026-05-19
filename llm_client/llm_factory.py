from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import yaml

from .anthropic_client import AnthropicClient
from .azure_client import AzureClient
from .base import BaseLLMClient
from .llm_client import LLMClient
from .models import Provider
from .openai_client import OpenAIClient

if TYPE_CHECKING:
    from .mlx_client import MlxClient

_DEFAULT_CONFIG_PATH = Path.home() / ".llm_client_models.yaml"

_config: Dict[str, Any] = {}


def _load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    global _config
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        _config = yaml.safe_load(f) or {}
    return _config


def get_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    if not _config:
        _load_config(config_path)
    return _config


def get_profile(name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    cfg = get_config(config_path)
    profiles = cfg.get("profiles", {})
    if name not in profiles:
        raise KeyError(f"Profile '{name}' not found. Available: {list(profiles.keys())}")
    return profiles[name]


def create_anthropic_client(
    profile_name: str = "anthropic",
    timeout: float = 300.0,
    **overrides: Any,
) -> AnthropicClient:
    p = get_profile(profile_name) | overrides
    return AnthropicClient(
        api_key=p["api_key"],
        base_url=p.get("base_url"),
        model=p.get("model"),
        auth_mode=p.get("auth_mode", "x-api-key"),
        thinking=p.get("thinking"),
        timeout=timeout,
    )


def create_openai_client(
    profile_name: str = "openai",
    timeout: float = 300.0,
    **overrides: Any,
) -> OpenAIClient:
    p = get_profile(profile_name) | overrides
    return OpenAIClient(
        api_key=p["api_key"],
        base_url=p.get("base_url"),
        model=p.get("model"),
        timeout=timeout,
    )


def create_azure_client(
    profile_name: str = "azure-openai",
    timeout: float = 300.0,
    **overrides: Any,
) -> AzureClient:
    p = get_profile(profile_name) | overrides

    if p.get("api_key"):
        return AzureClient(
            deployment=p["deployment"],
            endpoint=p["endpoint"],
            api_version=p.get("api_version", "2024-06-01"),
            api_key=p["api_key"],
            timeout=timeout,
        )

    from .llm_helper import get_azure_ad_token

    return AzureClient(
        deployment=p["deployment"],
        endpoint=p["endpoint"],
        api_version=p.get("api_version", "2024-06-01"),
        ad_token_provider=lambda: get_azure_ad_token(profile_name),
        timeout=timeout,
    )


def create_azure_client_ad_token(
    profile_name: str = "azure-openai-ad-token",
    timeout: float = 300.0,
    **overrides: Any,
) -> AzureClient:
    p = get_profile(profile_name) | overrides
    from .llm_helper import get_azure_ad_token
    return AzureClient(
        deployment=p["deployment"],
        endpoint=p["endpoint"],
        api_version=p.get("api_version", "2024-06-01"),
        ad_token_provider=lambda: get_azure_ad_token(profile_name),
        timeout=timeout,
    )


def create_zhipu_client(
    profile_name: str = "zhipu-glm",
    thinking: bool = True,
    timeout: float = 300.0,
    **overrides: Any,
) -> AnthropicClient:
    p = get_profile(profile_name) | overrides
    thinking_config = p.get("thinking")
    if thinking_config is None:
        thinking_config = {"type": "enabled", "budget_tokens": 10000} if thinking else None
    return AnthropicClient(
        api_key=p["api_key"],
        base_url=p.get("base_url"),
        model=p.get("model"),
        auth_mode=p.get("auth_mode", "bearer"),
        thinking=thinking_config,
        timeout=timeout,
    )


def create_doubao_client(
    profile_name: str = "doubao",
    timeout: float = 300.0,
    **overrides: Any,
) -> AnthropicClient:
    p = get_profile(profile_name) | overrides
    return AnthropicClient(
        api_key=p["api_key"],
        base_url=p.get("base_url"),
        model=p.get("model"),
        auth_mode=p.get("auth_mode", "bearer"),
        thinking=p.get("thinking"),
        timeout=timeout,
    )


def create_kimi_client(
    profile_name: str = "kimi",
    thinking: bool = True,
    timeout: float = 300.0,
    **overrides: Any,
) -> AnthropicClient:
    p = get_profile(profile_name) | overrides
    thinking_config = p.get("thinking")
    if thinking_config is None:
        thinking_config = {"type": "enabled", "budget_tokens": 10000} if thinking else None
    return AnthropicClient(
        api_key=p["api_key"],
        base_url=p.get("base_url"),
        model=p.get("model"),
        auth_mode=p.get("auth_mode", "bearer"),
        thinking=thinking_config,
        timeout=timeout,
    )


def create_mlx_client(
    profile_name: str = "gemma4-e4b",
    **overrides: Any,
) -> "MlxClient":
    from .mlx_client import MlxClient

    p = get_profile(profile_name) | overrides
    return MlxClient(
        model_path=p["model_path"],
        enable_thinking=p.get("enable_thinking", True),
    )


def create_llm_client(
    default_provider: Optional[Union[Provider, str]] = None,
    timeout: float = 300.0,
) -> LLMClient:
    cfg = get_config()
    provider = default_provider or cfg.get("default", "anthropic")
    llm = LLMClient(default_provider=provider)

    if provider == Provider.ANTHROPIC:
        llm.add_client(Provider.ANTHROPIC, create_anthropic_client(timeout=timeout), default=True)
    elif provider == Provider.OPENAI:
        llm.add_client(Provider.OPENAI, create_openai_client(timeout=timeout), default=True)
    elif provider == Provider.AZURE:
        llm.add_client(Provider.AZURE, create_azure_client(timeout=timeout), default=True)
    elif provider == Provider.DOUBAO:
        llm.add_client(Provider.DOUBAO, create_doubao_client(timeout=timeout), default=True)
    elif provider == Provider.KIMI:
        llm.add_client(Provider.KIMI, create_kimi_client(timeout=timeout), default=True)
    elif provider == Provider.MLX:
        llm.add_client(Provider.MLX, create_mlx_client(), default=True)
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: {[p.value for p in Provider]}")

    return llm


def create_from_profiles(
    config_path: Optional[str] = None,
    default: Optional[str] = None,
    timeout: float = 300.0,
) -> LLMClient:
    cfg = _load_config(config_path) if config_path else get_config()

    profiles = cfg.get("profiles", {})
    if not profiles:
        raise ValueError("No profiles defined in config file")

    default_name = default or cfg.get("default")
    llm = LLMClient()

    for name, profile in profiles.items():
        provider = profile.get("provider", "")
        profile["_name"] = name
        client = _create_client_from_profile(provider, profile, timeout)
        is_default = name == default_name
        llm.add_client(name, client, default=is_default)

    if not llm.default_provider:
        llm.set_default_provider(next(iter(profiles)))

    return llm


def _create_client_from_profile(
    provider: str,
    profile: Dict[str, Any],
    timeout: float,
) -> BaseLLMClient:
    if provider == "openai":
        return OpenAIClient(
            api_key=profile["api_key"],
            base_url=profile.get("base_url"),
            model=profile.get("model"),
            timeout=timeout,
        )
    elif provider == "anthropic":
        return AnthropicClient(
            api_key=profile["api_key"],
            base_url=profile.get("base_url"),
            model=profile.get("model"),
            auth_mode=profile.get("auth_mode", "x-api-key"),
            thinking=profile.get("thinking"),
            timeout=timeout,
        )
    elif provider == "azure":
        if profile.get("api_key"):
            return AzureClient(
                deployment=profile["deployment"],
                endpoint=profile["endpoint"],
                api_version=profile.get("api_version", "2024-06-01"),
                api_key=profile["api_key"],
                timeout=timeout,
            )

        from .llm_helper import get_azure_ad_token

        profile_name = profile.get("_name", "azure")
        return AzureClient(
            deployment=profile["deployment"],
            endpoint=profile["endpoint"],
            api_version=profile.get("api_version", "2024-06-01"),
            ad_token_provider=lambda pn=profile_name: get_azure_ad_token(pn),
            timeout=timeout,
        )
    elif provider in ("doubao", "kimi"):
        return AnthropicClient(
            api_key=profile["api_key"],
            base_url=profile.get("base_url"),
            model=profile.get("model"),
            auth_mode=profile.get("auth_mode", "bearer"),
            thinking=profile.get("thinking"),
            timeout=timeout,
        )
    elif provider == "mlx":
        from .mlx_client import MlxClient

        return MlxClient(
            model_path=profile["model_path"],
            enable_thinking=profile.get("enable_thinking", True),
        )
    else:
        raise ValueError(f"Unknown provider in profile: {provider}")
