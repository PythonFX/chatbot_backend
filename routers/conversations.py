from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from services import conversation_service
from services.title_generator import generate_title

router = APIRouter(prefix="/conversations", tags=["conversations"])


class CreateConversationResponse(BaseModel):
    id: str
    title: str


class RenameRequest(BaseModel):
    title: str


class AddMessageRequest(BaseModel):
    role: str
    content: str


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    created_at: str
    thinking: str | None = None
    type: str | None = None
    complete: bool = True


class ConversationResponse(BaseModel):
    id: str
    title: str
    messages: list[MessageResponse]
    created_at: str
    updated_at: str


@router.get("", response_model=list[ConversationResponse])
async def list_conversations():
    """List all conversations, newest first."""
    conversations = conversation_service.get_all_conversations()
    return [
        ConversationResponse(
            id=c.id,
            title=c.title,
            messages=[
                MessageResponse(
                    id=m.id,
                    role=m.role,
                    content=m.content,
                    created_at=m.created_at.isoformat(),
                    thinking=m.thinking,
                    type=m.type,
                    complete=m.complete,
                )
                for m in c.messages
            ],
            created_at=c.created_at.isoformat(),
            updated_at=c.updated_at.isoformat(),
        )
        for c in conversations
    ]


@router.post("", response_model=CreateConversationResponse)
async def create_conversation():
    """Create a new empty conversation."""
    conversation = conversation_service.create_conversation()
    return CreateConversationResponse(id=conversation.id, title=conversation.title)


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str):
    """Get a single conversation with all messages."""
    conversation = conversation_service.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        messages=[
            MessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                created_at=m.created_at.isoformat(),
                thinking=m.thinking,
                type=m.type,
                complete=m.complete,
            )
            for m in conversation.messages
        ],
        created_at=conversation.created_at.isoformat(),
        updated_at=conversation.updated_at.isoformat(),
    )


@router.patch("/{conversation_id}/title", response_model=ConversationResponse)
async def rename_conversation(conversation_id: str, body: RenameRequest):
    """Rename a conversation."""
    conversation = conversation_service.update_title(conversation_id, body.title)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        messages=[
            MessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                created_at=m.created_at.isoformat(),
                thinking=m.thinking,
                type=m.type,
                complete=m.complete,
            )
            for m in conversation.messages
        ],
        created_at=conversation.created_at.isoformat(),
        updated_at=conversation.updated_at.isoformat(),
    )


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    if not conversation_service.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "ok"}


@router.post("/{conversation_id}/auto-rename", response_model=ConversationResponse)
async def auto_rename_conversation(conversation_id: str):
    """Auto-rename a conversation using LLM based on the first user message."""
    conversation = conversation_service.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Find the first user message
    first_user_message = None
    for msg in conversation.messages:
        if msg.role == "user":
            first_user_message = msg.content
            break

    if not first_user_message:
        raise HTTPException(status_code=400, detail="No user message found to generate title from")

    # Generate title using LLM
    new_title = generate_title(first_user_message)

    # Update the conversation title
    updated = conversation_service.update_title(conversation_id, new_title)
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update title")

    return ConversationResponse(
        id=updated.id,
        title=updated.title,
        messages=[
            MessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                created_at=m.created_at.isoformat(),
                thinking=m.thinking,
                type=m.type,
                complete=m.complete,
            )
            for m in updated.messages
        ],
        created_at=updated.created_at.isoformat(),
        updated_at=updated.updated_at.isoformat(),
    )
