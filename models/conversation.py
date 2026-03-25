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
        )


class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "New Chat"
    messages: list[Message] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
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
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )
