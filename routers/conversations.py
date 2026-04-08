from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from services import conversation_service
from services.title_generator import generate_title
from services.conversation_service import SearchResult

router = APIRouter(prefix="/conversations", tags=["conversations"])


def _build_message_responses(messages) -> list:
    return [
        MessageResponse(
            id=m.id,
            role=m.role,
            content=m.content,
            created_at=m.created_at.isoformat(),
            thinking=m.thinking,
            type=m.type,
            complete=m.complete,
            rag_contexts=m.rag_contexts,
        )
        for m in messages
    ]


def _build_conv_response(conv) -> dict:
    return {
        "id": conv.id,
        "title": conv.title,
        "messages": _build_message_responses(conv.messages),
        "file_ids": conv.file_ids,
        "created_at": conv.created_at.isoformat(),
        "updated_at": conv.updated_at.isoformat(),
        "is_novel_agent": conv.is_novel_agent,
        "selected_novel_id": conv.selected_novel_id,
    }


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
    rag_contexts: list[dict] | None = None


class ConversationResponse(BaseModel):
    id: str
    title: str
    messages: list[MessageResponse]
    file_ids: list[str] = []
    created_at: str
    updated_at: str
    is_novel_agent: bool = False
    selected_novel_id: Optional[str] = None


class UpdateFilesRequest(BaseModel):
    file_ids: list[str]


@router.get("", response_model=list[ConversationResponse])
async def list_conversations():
    """List all conversations, newest first."""
    conversations = conversation_service.get_all_conversations()
    return [ConversationResponse(**_build_conv_response(c)) for c in conversations]


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
    results = conversation_service.search_messages(q, context_length=50)
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
    return ConversationResponse(**_build_conv_response(conversation))


@router.patch("/{conversation_id}/title", response_model=ConversationResponse)
async def rename_conversation(conversation_id: str, body: RenameRequest):
    """Rename a conversation."""
    conversation = conversation_service.update_title(conversation_id, body.title)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationResponse(**_build_conv_response(conversation))


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
    return ConversationResponse(**_build_conv_response(conversation))


@router.post("/{conversation_id}/auto-rename", response_model=ConversationResponse)
async def auto_rename_conversation(conversation_id: str):
    """Auto-rename a conversation using LLM based on the first user message."""
    conversation = conversation_service.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    first_user_message = None
    for msg in conversation.messages:
        if msg.role == "user":
            first_user_message = msg.content
            break

    if not first_user_message:
        raise HTTPException(status_code=400, detail="No user message found to generate title from")

    new_title = generate_title(first_user_message)
    updated = conversation_service.update_title(conversation_id, new_title)
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update title")

    return ConversationResponse(**_build_conv_response(updated))


class RagContextsResponse(BaseModel):
    conversation_id: str
    message_id: str
    rag_contexts: list[dict] | None = None


@router.get("/{conversation_id}/messages/{message_id}/rag-contexts", response_model=RagContextsResponse)
async def get_message_rag_contexts(conversation_id: str, message_id: str):
    """
    Get the RAG contexts used for a specific message in a conversation.
    Returns the chunk_text and score for each context retrieved during RAG.
    """
    conversation = conversation_service.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    for m in conversation.messages:
        if m.id == message_id:
            return RagContextsResponse(
                conversation_id=conversation_id,
                message_id=message_id,
                rag_contexts=m.rag_contexts,
            )

    raise HTTPException(status_code=404, detail="Message not found")
