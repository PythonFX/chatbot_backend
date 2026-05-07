import os
from llm_client import AzureClient


def create_azure_chat_openai() -> AzureClient:
    """Create an Azure OpenAI client using llm_client."""
    api_key = os.getenv("OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_API_BASE")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    if not azure_endpoint:
        raise ValueError("AZURE_OPENAI_API_BASE environment variable is not set")

    return AzureClient(
        api_key=api_key,
        endpoint=azure_endpoint,
        api_version=api_version,
        deployment=deployment_name,
    )


def is_azure_configured() -> bool:
    """Check if Azure OpenAI is configured."""
    return bool(os.getenv("AZURE_OPENAI_API_BASE"))
