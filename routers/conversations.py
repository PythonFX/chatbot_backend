from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from services import conversation_service
from services.title_generator import generate_title
from services.conversation_service import SearchResult

router = APIRouter(prefix="/conversations", tags=["conversations"])


class CreateConversationResponse(BaseModel):
    id: str
    title: str


class RenameRequest(BaseModel):
    title: str


class AddMessageRequest(BaseModel):
    role: str
    content: str


class SearchResultResponse(BaseModel):
    conversation_id: str
    conversation_title: str
    message_id: str
    role: str
    context_before: str
    matched_text: str
    context_after: str
    full_context: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultResponse]


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
    file_ids: list[str] = []
    created_at: str
    updated_at: str


class UpdateFilesRequest(BaseModel):
    file_ids: list[str]


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


@router.get("/search", response_model=SearchResponse)
async def search_conversations(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(50, ge=1, le=200, description="Max results to return"),
):
    """
    Search all messages across all conversations.
    Returns matches with surrounding context, excluding thinking content.
    """
    results = conversation_service.search_messages(q, context_length=50)
    # Apply limit
    limited_results = results[:limit]
    return SearchResponse(
        query=q,
        results=[
            SearchResultResponse(
                conversation_id=r.conversation_id,
                conversation_title=r.conversation_title,
                message_id=r.message_id,
                role=r.role,
                context_before=r.context_before,
                matched_text=r.matched_text,
                context_after=r.context_after,
                full_context=r.full_context,
            )
            for r in limited_results
        ],
    )


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
        file_ids=conversation.file_ids,
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
        file_ids=conversation.file_ids,
        created_at=conversation.created_at.isoformat(),
        updated_at=conversation.updated_at.isoformat(),
    )


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    if not conversation_service.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "ok"}


@router.patch("/{conversation_id}/files", response_model=ConversationResponse)
async def update_conversation_files(conversation_id: str, body: UpdateFilesRequest):
    """Update the linked files of a conversation."""
    conversation = conversation_service.update_file_ids(conversation_id, body.file_ids)
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
        file_ids=conversation.file_ids,
        created_at=conversation.created_at.isoformat(),
        updated_at=conversation.updated_at.isoformat(),
    )


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
        file_ids=updated.file_ids,
        created_at=updated.created_at.isoformat(),
        updated_at=updated.updated_at.isoformat(),
    )
