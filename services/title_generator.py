from llm_client import Message
from services.llm_manager import create_llm_client


def generate_title(first_message: str) -> str:
    """Generate a short title for a conversation based on the first user message."""
    if not first_message.strip():
        return "New Chat"

    cleaned = first_message.strip()
    if len(cleaned) <= 20:
        return cleaned[:50] if cleaned else "New Chat"

    try:
        print(f"[TitleGenerator] Calling LLM for: {first_message[:30]}...")
        llm = create_llm_client()

        response = llm.completion([
            Message(role="system", content="You are a title generator. Given the content, generate a very short title (3-5 words max) that summarizes what the content is about. Only respond with the title, nothing else."),
            Message(role="user", content=first_message),
        ])

        title = response.content.strip()
        title = title.strip('"\'')
        result = title[:50] if title else "New Chat"
        print(f"[TitleGenerator] Final title: {result}")
        return result
    except Exception as e:
        print(f"[TitleGenerator] Error: {e}")
        return "New Chat"
