from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from models.conversation import Conversation, Message
from services import db_service


DATA_DIR = Path(__file__).parent.parent / "data" / "conversations"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _get_file_path(conversation_id: str) -> Path:
    return DATA_DIR / f"{conversation_id}.json"


def _save_conversation_to_json(conversation: Conversation) -> None:
    with open(_get_file_path(conversation.id), "w", encoding="utf-8") as f:
        json.dump(conversation.to_dict(), f, indent=2, ensure_ascii=False)


def _write_json_safe(conversation: Conversation) -> None:
    try:
        _save_conversation_to_json(conversation)
    except Exception as e:
        print(f"[WARN] Failed to write JSON backup for {conversation.id}: {e}")


def _persist(conversation: Conversation) -> None:
    """Write to DB and JSON backup. Call after every mutation."""
    db_service.db_upsert_conversation(conversation.to_dict())
    _write_json_safe(conversation)


def get_all_conversations() -> list[Conversation]:
    convs = db_service.db_get_all_conversations()
    return [Conversation.from_dict(c) for c in convs]


def get_conversation(conversation_id: str) -> Optional[Conversation]:
    conv = db_service.db_get_conversation(conversation_id)
    if conv is not None:
        return Conversation.from_dict(conv)
    return None


def create_conversation() -> Conversation:
    conv = Conversation()
    conv.created_at = datetime.utcnow()
    conv.updated_at = conv.created_at
    _persist(conv)
    return conv


def update_title(conversation_id: str, title: str) -> Optional[Conversation]:
    conv = get_conversation(conversation_id)
    if not conv:
        return None
    conv.title = title.strip()[:100] or "New Chat"
    conv.updated_at = datetime.utcnow()
    _persist(conv)
    return conv


def set_novel_agent(conversation_id: str, is_novel_agent: bool) -> Optional[Conversation]:
    conv = get_conversation(conversation_id)
    if not conv:
        return None
    conv.is_novel_agent = is_novel_agent
    if not is_novel_agent:
        conv.selected_novel_id = None
    conv.updated_at = datetime.utcnow()
    _persist(conv)
    return conv


def set_selected_novel(conversation_id: str, novel_id: str) -> Optional[Conversation]:
    conv = get_conversation(conversation_id)
    if not conv:
        return None
    conv.selected_novel_id = novel_id
    conv.updated_at = datetime.utcnow()
    _persist(conv)
    return conv


def update_file_ids(conversation_id: str, file_ids: list[str]) -> Optional[Conversation]:
    conv = get_conversation(conversation_id)
    if not conv:
        return None
    conv.file_ids = file_ids
    conv.updated_at = datetime.utcnow()
    _persist(conv)
    return conv


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
    if content is None:
        raise ValueError("Message content cannot be None")

    conv = get_conversation(conversation_id)
    if not conv:
        return None

    now = datetime.utcnow()
    msg = Message(
        role=str(role),
        content=str(content),
        thinking=thinking,
        signature=signature,
        type=type,
        raw_response=raw_response,
        complete=complete,
    )
    msg.created_at = now
    conv.messages.append(msg)
    conv.updated_at = now

    db_service.db_add_message(
        msg.id, conversation_id, msg.role, msg.content, msg.thinking,
        msg.signature, msg.type, msg.raw_response, msg.complete,
        msg.rag_contexts, msg.versions, msg.selected_version_index,
        msg.created_at.isoformat(),
    )
    _write_json_safe(conv)
    return conv, msg


def delete_conversation(conversation_id: str) -> bool:
    db_ok = db_service.db_delete_conversation(conversation_id)
    fp = _get_file_path(conversation_id)
    json_ok = fp.exists() and fp.unlink()
    return db_ok or json_ok


def remove_message(conversation_id: str, message_id: str) -> bool:
    db_service.db_remove_message(message_id)
    conv = get_conversation(conversation_id)
    if not conv:
        return False
    for i, m in enumerate(conv.messages):
        if m.id == message_id:
            conv.messages.pop(i)
            conv.updated_at = datetime.utcnow()
            _persist(conv)
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
    fields = {}
    if content is not None:
        fields["content"] = content
    if thinking is not None:
        fields["thinking"] = thinking
    if complete is not None:
        fields["complete"] = complete
    if rag_contexts is not None:
        fields["rag_contexts"] = rag_contexts
    if not fields:
        return None

    db_service.db_update_message(message_id, **fields)

    conv = get_conversation(conversation_id)
    if not conv:
        return None
    for m in conv.messages:
        if m.id == message_id:
            if content is not None:
                m.content = content
            if thinking is not None:
                m.thinking = thinking
            if complete is not None:
                m.complete = complete
            if rag_contexts is not None:
                m.rag_contexts = rag_contexts
            conv.updated_at = datetime.utcnow()
            _persist(conv)
            return m
    return None


def append_chunk(
    conversation_id: str,
    message_id: str,
    text: str = "",
    thinking: str = "",
) -> Optional[Message]:
    """Append content to a message's content/thinking fields. Writes to both DB and JSON."""
    db_service.db_append_chunk(message_id, text, thinking)

    conv = get_conversation(conversation_id)
    if not conv:
        return None
    for m in conv.messages:
        if m.id == message_id:
            if text:
                m.content += text
            if thinking:
                m.thinking = (m.thinking or "") + thinking
            _write_json_safe(conv)
            return m
    return None


def add_version(
    conversation_id: str,
    message_id: str,
    content: str,
    thinking: Optional[str],
) -> Optional[Message]:
    """Append a new version dict to the message's versions list. Sets selected_version_index to the new index."""
    conv = get_conversation(conversation_id)
    if not conv:
        return None
    for m in conv.messages:
        if m.id == message_id:
            if m.versions is None:
                m.versions = []
            m.versions.append({
                "content": content,
                "thinking": thinking,
                "created_at": datetime.utcnow().isoformat(),
            })
            m.selected_version_index = len(m.versions) - 1
            conv.updated_at = datetime.utcnow()
            _persist(conv)
            db_service.db_update_message(
                message_id,
                versions=m.versions,
                selected_version_index=m.selected_version_index,
            )
            return m
    return None


def select_version(
    conversation_id: str,
    message_id: str,
    version_index: int,
) -> Optional[Message]:
    """Set selected_version_index to the given value."""
    conv = get_conversation(conversation_id)
    if not conv:
        return None
    for m in conv.messages:
        if m.id == message_id:
            if m.versions is None or not (0 <= version_index < len(m.versions)):
                return None
            m.selected_version_index = version_index
            conv.updated_at = datetime.utcnow()
            _persist(conv)
            db_service.db_update_message(
                message_id,
                selected_version_index=version_index,
            )
            return m
    return None


def get_selected_version(
    conversation_id: str,
    message_id: str,
) -> Tuple[str, Optional[str]]:
    """Return (content, thinking) of the selected version, or primary fields if none selected."""
    conv = get_conversation(conversation_id)
    if not conv:
        return "", None
    for m in conv.messages:
        if m.id == message_id:
            if m.selected_version_index is not None and m.versions:
                v = m.versions[m.selected_version_index]
                return v["content"], v.get("thinking")
            return m.content, m.thinking
    return "", None


def save_conversation(conversation: Conversation) -> None:
    _persist(conversation)


@dataclass
class SearchResult:
    conversation_id: str
    conversation_title: str
    message_id: str
    role: str
    context_before: str
    matched_text: str
    context_after: str
    full_context: str


def search_messages(query: str, context_length: int = 50) -> list[SearchResult]:
    if not query or not query.strip():
        return []
    results = db_service.db_search_messages(query, context_length)
    return [SearchResult(**r) for r in results]
