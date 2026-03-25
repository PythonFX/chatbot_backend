import asyncio
import json
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from services import conversation_service
from services.conversation_service import save_conversation
from services.title_generator import generate_title
from services.minimax_client import create_minimax_client, is_minimax_configured

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    conversation_id: str
    message: str


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    content: str
    title: str
    thinking: str | None = None
    type: str | None = None


class RegenerateRequest(BaseModel):
    conversation_id: str
    message_id: str


class RegenerateResponse(BaseModel):
    conversation_id: str
    message_id: str
    content: str
    thinking: str | None = None
    type: str | None = None


# Track stop events per conversation
stop_events: dict[str, asyncio.Event] = {}


def _build_langchain_messages(messages: list) -> list:
    """Convert our message format to LangChain message format."""
    langchain_messages = []
    for m in messages:
        if m.role == "user":
            langchain_messages.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            langchain_messages.append(AIMessage(content=m.content))
        elif m.role == "system":
            langchain_messages.append(SystemMessage(content=m.content))
    return langchain_messages


def _parse_minimax_chunk(chunk) -> dict:
    """Parse a MiniMax streaming chunk and extract relevant info."""
    result = {"text": "", "thinking": None}

    if hasattr(chunk, "content"):
        content = chunk.content
        if isinstance(content, list):
            for block in content:
                if hasattr(block, "type"):
                    if block.type == "thinking" and hasattr(block, "thinking"):
                        result["thinking"] = block.thinking
                    elif block.type == "text" and hasattr(block, "text"):
                        result["text"] += block.text
                elif isinstance(block, dict):
                    if block.get("type") == "thinking":
                        result["thinking"] = block.get("thinking")
                    elif block.get("type") == "text":
                        result["text"] += block.get("text", "")
        elif isinstance(content, str):
            result["text"] = content

    return result


async def _stream_llm(messages: list, conversation_id: str):
    """Stream LLM response and yield chunks."""
    llm = create_minimax_client()
    stop_event = stop_events.get(conversation_id)

    print(f"[Stream] Starting for conversation: {conversation_id}")

    try:
        async for chunk in llm.astream(messages):
            if stop_event and stop_event.is_set():
                print(f"[Stream] Stopped by event for conversation: {conversation_id}")
                yield {"type": "stopped", "text": ""}
                return

            parsed = _parse_minimax_chunk(chunk)

            # Yield thinking separately if present
            if parsed["thinking"]:
                print(f"[Thinking] {parsed['thinking']}")
                yield {
                    "type": "thinking",
                    "thinking": parsed["thinking"],
                }

            # Yield text chunk
            if parsed["text"]:
                print(f"[Text Chunk] {parsed['text']}", end="", flush=True)
                yield {
                    "type": "chunk",
                    "text": parsed["text"],
                }

    except asyncio.CancelledError:
        print(f"[Stream] Cancelled for conversation: {conversation_id}")
        yield {"type": "stopped", "text": ""}
        return


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Send a message and stream AI response using Server-Sent Events."""
    if not is_minimax_configured():
        return StreamingResponse(
            iter([json.dumps({"type": "error", "error": "MiniMax not configured"})]),
            media_type="application/json",
            status_code=500,
        )

    if not request.message.strip():
        return StreamingResponse(
            iter([json.dumps({"type": "error", "error": "Empty message"})]),
            media_type="application/json",
            status_code=400,
        )

    # Get conversation
    conversation = conversation_service.get_conversation(request.conversation_id)
    if not conversation:
        return StreamingResponse(
            iter([json.dumps({"type": "error", "error": "Conversation not found"})]),
            media_type="application/json",
            status_code=404,
        )

    # Cancel any ongoing generation for this conversation
    if request.conversation_id in stop_events:
        stop_events[request.conversation_id].set()

    # Create stop event for this conversation
    stop_events[request.conversation_id] = asyncio.Event()

    # Add user message
    result = conversation_service.add_message(
        request.conversation_id, "user", request.message
    )
    if not result:
        return StreamingResponse(
            iter([json.dumps({"type": "error", "error": "Failed to save message"})]),
            media_type="application/json",
            status_code=500,
        )
    conversation, user_msg = result

    # Generate title if this is the first user message (run in background)
    title = conversation.title
    title_task = None
    if len(conversation.messages) == 1 and conversation.title == "New Chat":
        print(f"[Title] Starting background title generation for: {request.message[:50]}...")

        def run_title_gen():
            return generate_title(request.message)

        title_task = asyncio.create_task(asyncio.to_thread(run_title_gen))

    # Build messages for LLM
    langchain_messages = _build_langchain_messages(conversation.messages)

    # Create assistant message placeholder with complete=False
    placeholder_result = conversation_service.add_message(
        request.conversation_id,
        "assistant",
        "",
        complete=False,
    )
    if not placeholder_result:
        return StreamingResponse(
            iter([json.dumps({"type": "error", "error": "Failed to create message placeholder"})]),
            media_type="application/json",
            status_code=500,
        )
    _, placeholder_msg = placeholder_result
    assistant_msg_id = placeholder_msg.id

    async def generate():
        nonlocal title

        full_text = ""
        full_thinking = ""

        yield f"data: {json.dumps({'type': 'start', 'message_id': assistant_msg_id, 'title': title})}\n\n"

        try:
            async for chunk in _stream_llm(langchain_messages, request.conversation_id):
                if chunk["type"] == "stopped":
                    break

                if chunk["type"] == "thinking":
                    full_thinking += chunk["thinking"]
                    # Update message in storage
                    conversation_service.update_message(
                        request.conversation_id,
                        assistant_msg_id,
                        thinking=full_thinking,
                    )
                    yield f"data: {json.dumps({'type': 'thinking', 'thinking': full_thinking})}\n\n"
                elif chunk["text"]:
                    full_text += chunk["text"]
                    # Update message in storage
                    conversation_service.update_message(
                        request.conversation_id,
                        assistant_msg_id,
                        content=full_text,
                    )
                    yield f"data: {json.dumps({'type': 'chunk', 'text': chunk['text']})}\n\n"

            # Finalize: mark message as complete
            if full_text:
                # Check if title was generated in background
                if title_task and not title_task.done():
                    # Wait for title if still pending
                    new_title = await title_task
                    if new_title and new_title != "New Chat":
                        title = new_title
                        conversation_service.update_title(conversation.id, title)
                        print(f"[Title] Updated after stream: {title}")
                elif title_task:
                    try:
                        new_title = title_task.result()
                        if new_title and new_title != "New Chat":
                            title = new_title
                            conversation_service.update_title(conversation.id, title)
                            print(f"[Title] Updated after stream: {title}")
                    except Exception as e:
                        print(f"[Title] Task error: {e}")

                conversation_service.update_message(
                    request.conversation_id,
                    assistant_msg_id,
                    content=full_text,
                    thinking=full_thinking,
                    complete=True,
                )
                print(f"\n[Complete] Final text ({len(full_text)} chars):\n{full_text}")
                if full_thinking:
                    print(f"[Complete] Final thinking ({len(full_thinking)} chars):\n{full_thinking}")
                yield f"data: {json.dumps({'type': 'done', 'message_id': assistant_msg_id, 'title': title})}\n\n"
            else:
                # Remove empty message if no content
                conversation_service.remove_message(request.conversation_id, assistant_msg_id)
                print(f"\n[Error] Empty response")
                yield f"data: {json.dumps({'type': 'error', 'error': 'Empty response'})}\n\n"

        except asyncio.CancelledError:
            # Mark message as complete even if cancelled (partial content is saved)
            conversation_service.update_message(
                request.conversation_id,
                assistant_msg_id,
                complete=True,
            )
            print(f"\n[Stopped] Partial text ({len(full_text)} chars):\n{full_text}")
            yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
        except Exception as e:
            print(f"\n[Error] {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            if request.conversation_id in stop_events:
                del stop_events[request.conversation_id]

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/stop/{conversation_id}")
async def stop_generation(conversation_id: str):
    """Stop ongoing generation for a conversation."""
    if conversation_id in stop_events:
        stop_events[conversation_id].set()
        return {"status": "cancelled", "conversation_id": conversation_id}
    return {"status": "no_active_generation", "conversation_id": conversation_id}


@router.post("/chat/regenerate", response_model=RegenerateResponse)
async def regenerate_response(request: RegenerateRequest):
    """Regenerate the AI response for a given user message ID (non-streaming)."""
    if not is_minimax_configured():
        raise HTTPException(status_code=500, detail="MiniMax not configured")

    conversation = conversation_service.get_conversation(request.conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Find the user message
    user_msg = None
    user_msg_index = -1
    for i, m in enumerate(conversation.messages):
        if m.id == request.message_id and m.role == "user":
            user_msg = m
            user_msg_index = i
            break

    if not user_msg:
        raise HTTPException(status_code=404, detail="User message not found")

    # Find the corresponding assistant message (should be immediately after)
    assistant_msg = None
    if user_msg_index + 1 < len(conversation.messages):
        next_msg = conversation.messages[user_msg_index + 1]
        if next_msg.role == "assistant":
            assistant_msg = next_msg

    # Build messages up to and including the user message
    langchain_messages = _build_langchain_messages(conversation.messages[:user_msg_index + 1])

    try:
        llm = create_minimax_client()
        response = await asyncio.to_thread(llm.invoke, langchain_messages)

        # Parse the response
        full_text = ""
        thinking = None
        if hasattr(response, "content"):
            parsed = _parse_minimax_chunk(type("Chunk", (), {"content": response.content})())
            full_text = parsed["text"]
            thinking = parsed["thinking"]

    except asyncio.CancelledError:
        raise HTTPException(status_code=499, detail="Generation cancelled")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MiniMax error: {str(e)}")

    if not full_text:
        raise HTTPException(status_code=500, detail="Empty response from MiniMax")

    # Remove old assistant message if exists
    if assistant_msg:
        conversation_service.remove_message(request.conversation_id, assistant_msg.id)

    # Add new AI response
    result = conversation_service.add_message(
        request.conversation_id,
        "assistant",
        full_text,
        thinking=thinking,
        type="text",
    )
    if not result:
        raise HTTPException(status_code=500, detail="Failed to save regenerated response")

    _, ai_msg = result

    return RegenerateResponse(
        conversation_id=request.conversation_id,
        message_id=ai_msg.id,
        content=full_text,
        thinking=thinking,
        type="text",
    )
