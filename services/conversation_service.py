from __future__ import annotations
import os
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from models.conversation import Conversation, Message


DATA_DIR = Path(__file__).parent.parent / "data" / "conversations"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _get_file_path(conversation_id: str) -> Path:
    return DATA_DIR / f"{conversation_id}.json"


def get_all_conversations() -> list[Conversation]:
    """Get all conversations, ordered by updated_at descending."""
    conversations = []
    for file_path in DATA_DIR.glob("*.json"):
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            conversations.append(Conversation.from_dict(data))
        except Exception:
            continue
    conversations.sort(key=lambda c: c.updated_at, reverse=True)
    return conversations


def get_conversation(conversation_id: str) -> Optional[Conversation]:
    """Get a single conversation by ID."""
    file_path = _get_file_path(conversation_id)
    if not file_path.exists():
        return None
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    return Conversation.from_dict(data)


def create_conversation() -> Conversation:
    """Create a new empty conversation."""
    conversation = Conversation()
    _save_conversation(conversation)
    return conversation


def _save_conversation(conversation: Conversation) -> None:
    """Save a conversation to disk."""
    with open(_get_file_path(conversation.id), "w", encoding="utf-8") as f:
        json.dump(conversation.to_dict(), f, indent=2, ensure_ascii=False)


def update_title(conversation_id: str, title: str) -> Optional[Conversation]:
    """Update the title of a conversation."""
    conversation = get_conversation(conversation_id)
    if not conversation:
        return None
    conversation.title = title.strip()[:100] or "New Chat"
    conversation.updated_at = datetime.utcnow()
    _save_conversation(conversation)
    return conversation


def update_file_ids(conversation_id: str, file_ids: list[str]) -> Optional[Conversation]:
    """Update the linked file IDs of a conversation."""
    conversation = get_conversation(conversation_id)
    if not conversation:
        return None
    conversation.file_ids = file_ids
    conversation.updated_at = datetime.utcnow()
    _save_conversation(conversation)
    return conversation


def add_message(
    conversation_id: str,
    role: str,
    content: str,
    thinking: Optional[str] = None,
    signature: Optional[str] = None,
    type: Optional[str] = None,
    raw_response: Optional[dict] = None,
    complete: bool = True,
) -> Optional[Tuple[Conversation, Message]]:
    """Add a message to a conversation."""
    if content is None:
        raise ValueError("Message content cannot be None")

    conversation = get_conversation(conversation_id)
    if not conversation:
        return None

    message = Message(
        role=str(role),
        content=str(content),
        thinking=thinking,
        signature=signature,
        type=type,
        raw_response=raw_response,
        complete=complete,
    )
    conversation.messages.append(message)
    conversation.updated_at = datetime.utcnow()
    _save_conversation(conversation)
    return conversation, message


def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation."""
    file_path = _get_file_path(conversation_id)
    if not file_path.exists():
        return False
    file_path.unlink()
    return True


def remove_message(conversation_id: str, message_id: str) -> bool:
    """Remove a message from a conversation by message ID."""
    conversation = get_conversation(conversation_id)
    if not conversation:
        return False

    for i, m in enumerate(conversation.messages):
        if m.id == message_id:
            conversation.messages.pop(i)
            conversation.updated_at = datetime.utcnow()
            _save_conversation(conversation)
            return True
    return False


def update_message(
    conversation_id: str,
    message_id: str,
    content: Optional[str] = None,
    thinking: Optional[str] = None,
    complete: Optional[bool] = None,
    rag_contexts: Optional[list[dict]] = None,
) -> Optional[Message]:
    """Update a message's content, thinking, complete status, or rag_contexts."""
    conversation = get_conversation(conversation_id)
    if not conversation:
        return None

    for m in conversation.messages:
        if m.id == message_id:
            if content is not None:
                m.content = content
            if thinking is not None:
                m.thinking = thinking
            if complete is not None:
                m.complete = complete
            if rag_contexts is not None:
                m.rag_contexts = rag_contexts
            conversation.updated_at = datetime.utcnow()
            _save_conversation(conversation)
            return m
    return None


def append_chunk(
    conversation_id: str,
    message_id: str,
    text: str = "",
    thinking: str = "",
) -> Optional[Message]:
    """
    Append content to a message's content/thinking fields.
    Used by stream_manager to incrementally update messages.
    """
    conversation = get_conversation(conversation_id)
    if not conversation:
        return None

    for m in conversation.messages:
        if m.id == message_id:
            if text:
                m.content += text
            if thinking:
                if m.thinking:
                    m.thinking += thinking
                else:
                    m.thinking = thinking
            # Note: don't update updated_at on every chunk to reduce disk writes
            # Only update on meaningful boundaries
            _save_conversation(conversation)
            return m
    return None


def save_conversation(conversation: Conversation) -> None:
    """Public method to save a conversation."""
    conversation.updated_at = datetime.utcnow()
    _save_conversation(conversation)


@dataclass
class SearchResult:
    """Represents a single search match in a message."""
    conversation_id: str
    conversation_title: str
    message_id: str
    role: str
    context_before: str  # Text before the match
    matched_text: str    # The matching text
    context_after: str   # Text after the match
    full_context: str     # Combined context with match highlighted (for display)


def search_messages(query: str, context_length: int = 50) -> list[SearchResult]:
    """
    Search all messages across all conversations for a query string.
    Returns matches with surrounding context, excluding thinking content.

    Args:
        query: The search string to find (case-insensitive)
        context_length: Number of characters to show before/after match

    Returns:
        List of SearchResult objects with surrounding context
    """
    if not query or len(query.strip()) == 0:
        return []

    query_lower = query.lower()
    results: list[SearchResult] = []

    conversations = get_all_conversations()
    for conv in conversations:
        for msg in conv.messages:
            # Skip thinking content as per requirement
            content = msg.content
            if not content:
                continue

            content_lower = content.lower()
            pos = content_lower.find(query_lower)
            if pos == -1:
                continue

            # Extract surrounding context
            start = max(0, pos - context_length)
            end = min(len(content), pos + len(query) + context_length)

            context_before = content[start:pos]
            matched_text = content[pos:pos + len(query)]
            context_after = content[pos + len(query):end]

            # Add ellipsis if there are more characters before/after
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(content) else ""

            # Build full context for display (single line)
            full_context = f"{prefix}{context_before}{matched_text}{context_after}{suffix}"

            results.append(SearchResult(
                conversation_id=conv.id,
                conversation_title=conv.title,
                message_id=msg.id,
                role=msg.role,
                context_before=context_before,
                matched_text=matched_text,
                context_after=context_after,
                full_context=full_context,
            ))

    return results
