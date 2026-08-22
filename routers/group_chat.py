"""
Group chat API endpoints.

Provides endpoints for creating group chat conversations,
streaming agent responses, and listing available agents.
"""
from __future__ import annotations

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services import conversation_service
from services.llm_manager import get_available_models, get_model_info
from services.group_chat.coordinator import GroupChatCoordinator, create_coordinator_for_conversation

router = APIRouter(prefix="/group-chat", tags=["group-chat"])

# Track active group chat streams for stop functionality
_active_streams: dict[str, asyncio.Event] = {}


class CreateGroupChatRequest(BaseModel):
    agent_ids: list[str]


class GroupChatStreamRequest(BaseModel):
    conversation_id: str
    message: str


@router.post("/create")
async def create_group_chat(request: CreateGroupChatRequest):
    """Create a new group chat conversation with the specified agents."""
    # Validate that requested agents exist and are configured
    available = get_available_models()
    valid_agents = [a for a in request.agent_ids if a in available]
    if not valid_agents:
        raise HTTPException(
            status_code=400,
            detail=f"No valid agents. Available models: {available}",
        )

    conv = conversation_service.create_group_conversation(valid_agents)
    return {
        "id": conv.id,
        "title": conv.title,
        "type": conv.type,
        "agents": conv.agents,
        "created_at": conv.created_at.isoformat(),
        "updated_at": conv.updated_at.isoformat(),
    }


@router.get("/agents")
async def list_available_agents():
    """List all chat models; enabled ones are available for group chat."""
    agents = [
        {"id": m["id"], "name": m["display_name"], "available": m["enabled"]}
        for m in get_model_info()
    ]
    return {"agents": agents}


@router.post("/stream")
async def stream_group_chat(request: GroupChatStreamRequest):
    """Send a message in a group chat and stream agent responses via SSE."""
    conversation = conversation_service.get_conversation(request.conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if conversation.type != "group_chat":
        raise HTTPException(status_code=400, detail="Not a group chat conversation")

    # Create coordinator with the conversation's agents
    coordinator = create_coordinator_for_conversation(conversation.agents)
    if not coordinator:
        raise HTTPException(
            status_code=500,
            detail="No agents available for this group chat",
        )

    # Save the user message
    _, user_msg = conversation_service.add_message(
        conversation_id=request.conversation_id,
        role="user",
        content=request.message,
    )

    # Set up stop event
    stop_event = asyncio.Event()
    _active_streams[request.conversation_id] = stop_event

    async def event_generator():
        try:
            # Build message list for the coordinator
            # Refresh conversation to get the latest state including the new user message
            conv = conversation_service.get_conversation(request.conversation_id)
            if not conv:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Conversation not found'})}\n\n"
                return

            messages = conv.messages

            # Build the user message dict for the coordinator
            user_message_dict = {
                "role": "user",
                "content": request.message,
            }

            async for event in coordinator.process_user_message(messages, user_message_dict):
                if stop_event.is_set():
                    yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
                    break

                if event["type"] == "agent_message_complete":
                    # Save the agent's response to the conversation
                    conversation_service.add_message(
                        conversation_id=request.conversation_id,
                        role="assistant",
                        content=event["content"],
                        thinking=event.get("thinking"),
                        sender_id=event["agent_id"],
                        sender_name=event["agent_name"],
                    )
                    # Don't yield this internal event to the frontend
                    continue

                yield f"data: {json.dumps(event)}\n\n"

            # Generate title if this is the first message
            conv = conversation_service.get_conversation(request.conversation_id)
            if conv and conv.title == "New Chat":
                try:
                    from services.title_generator import generate_title
                    new_title = await asyncio.to_thread(generate_title, request.message)
                    conversation_service.update_title(request.conversation_id, new_title)
                except Exception as e:
                    print(f"[GroupChat] Title generation failed: {e}")

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            _active_streams.pop(request.conversation_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/stop/{conversation_id}")
async def stop_group_chat(conversation_id: str):
    """Stop an ongoing group chat generation."""
    stop_event = _active_streams.get(conversation_id)
    if stop_event:
        stop_event.set()
        return {"status": "ok", "message": "Stop signal sent"}
    return {"status": "ok", "message": "No active stream found"}
