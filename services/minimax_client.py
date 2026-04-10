import os
from langchain_anthropic import ChatAnthropic


def create_minimax_client() -> ChatAnthropic:
    """Create a MiniMax (Anthropic-compatible) client."""
    auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN")
    base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.minimaxi.com/anthropic")
    model = os.getenv("ANTHROPIC_MODEL", "mini-max-m2.7-highspeed")

    if not auth_token:
        raise ValueError("ANTHROPIC_AUTH_TOKEN environment variable is not set")

    return ChatAnthropic(
        model=model,
        anthropic_api_key=auth_token,
        base_url=base_url,
    )


def is_minimax_configured() -> bool:
    """Check if MiniMax is configured."""
    return bool(os.getenv("ANTHROPIC_AUTH_TOKEN"))
