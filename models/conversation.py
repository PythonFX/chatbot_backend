from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Any
import uuid


class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    # Optional fields for assistant messages (MiniMax extended response)
    thinking: Optional[str] = None
    signature: Optional[str] = None
    type: Optional[str] = None
    # Raw response for debugging/forward compatibility
    raw_response: Optional[dict] = None
    # Track if message generation is complete (for streaming)
    complete: bool = True
    # RAG contexts used for this message (list of {file_id, chunk_text, score})
    rag_contexts: Optional[list[dict]] = None
    # Versioning for AI message responses
    versions: Optional[list[dict]] = None
    # Index into versions of selected version; null means primary content/thinking is selected
    selected_version_index: Optional[int] = None

    def to_dict(self) -> dict:
        result = {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
        }
        if self.thinking:
            result["thinking"] = self.thinking
        if self.signature:
            result["signature"] = self.signature
        if self.type:
            result["type"] = self.type
        if self.raw_response:
            result["raw_response"] = self.raw_response
        result["complete"] = self.complete
        if self.rag_contexts:
            result["rag_contexts"] = self.rag_contexts
        if self.versions:
            result["versions"] = self.versions
        if self.selected_version_index is not None:
            result["selected_version_index"] = self.selected_version_index
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            id=data["id"],
            role=data["role"],
            content=data["content"],
            created_at=datetime.fromisoformat(data["created_at"]),
            thinking=data.get("thinking"),
            signature=data.get("signature"),
            type=data.get("type"),
            raw_response=data.get("raw_response"),
            complete=data.get("complete", True),
            rag_contexts=data.get("rag_contexts"),
            versions=data.get("versions"),
            selected_version_index=data.get("selected_version_index"),
        )


class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "New Chat"
    messages: list[Message] = []
    file_ids: list[str] = []  # Linked file IDs for RAG
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    # Novel agent mode
    is_novel_agent: bool = False
    selected_novel_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "messages": [m.to_dict() for m in self.messages],
            "file_ids": self.file_ids,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_novel_agent": self.is_novel_agent,
            "selected_novel_id": self.selected_novel_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        messages = [
            Message.from_dict(m)
            for m in data.get("messages", [])
        ]
        return cls(
            id=data["id"],
            title=data["title"],
            messages=messages,
            file_ids=data.get("file_ids", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            is_novel_agent=data.get("is_novel_agent", False),
            selected_novel_id=data.get("selected_novel_id"),
        )
