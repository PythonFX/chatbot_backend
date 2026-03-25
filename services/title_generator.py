from langchain_core.messages import HumanMessage, SystemMessage
from services.minimax_client import create_minimax_client


def generate_title(first_message: str) -> str:
    """Generate a short title for a conversation based on the first user message."""
    if not first_message.strip():
        return "New Chat"

    try:
        print(f"[TitleGenerator] Calling MiniMax for: {first_message[:30]}...")
        llm = create_minimax_client()
        response = llm.invoke([
            SystemMessage(content="You are a title generator. Given a user's first message to a chatbot, generate a very short title (3-5 words max) that summarizes what the conversation is about. Only respond with the title, nothing else."),
            HumanMessage(content=first_message),
        ])
        print(f"[TitleGenerator] Raw response: {response}")
        print(f"[TitleGenerator] Response content: {response.content}")

        # Extract text from response.content which may be a list of blocks
        content = response.content
        if isinstance(content, list):
            title = ""
            for block in content:
                if hasattr(block, "type") and block.type == "text" and hasattr(block, "text"):
                    title = block.text
                    break
                elif isinstance(block, dict) and block.get("type") == "text":
                    title = block.get("text", "")
                    break
        else:
            title = content

        if not title:
            return "New Chat"

        title = str(title).strip()
        # Remove quotes if present
        title = title.strip('"\'')
        result = title[:50] if title else "New Chat"
        print(f"[TitleGenerator] Final title: {result}")
        return result
    except Exception as e:
        print(f"[TitleGenerator] Error: {e}")
        return "New Chat"
