import asyncio
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from services import conversation_service
from services.conversation_service import save_conversation
from services.title_generator import generate_title
from services.minimax_client import create_minimax_client, is_minimax_configured
from services.stream_manager import stream_manager
from services.embedding_service import search_chunks_in_files


router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    conversation_id: str
    message: str
    resume: bool = False  # If True, resume existing stream instead of starting new one


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


# Track stop events per conversation (legacy, kept for compatibility)
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


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Send a message and stream AI response using Server-Sent Events.

    The LLM stream runs in a background task that persists even if the
    HTTP connection is closed. Chunks are saved to JSON in real-time.
    """
    if not is_minimax_configured():
        return StreamingResponse(
            iter([json.dumps({"type": "error", "error": "MiniMax not configured"})]),
            media_type="application/json",
            status_code=500,
        )

    # Get conversation
    conversation = conversation_service.get_conversation(request.conversation_id)
    if not conversation:
        return StreamingResponse(
            iter([json.dumps({"type": "error", "error": "Conversation not found"})]),
            media_type="application/json",
            status_code=404,
        )

    # Handle resume mode: re-subscribe to an existing stream (must be before empty message check)
    if request.resume:
        existing_state = stream_manager.get_stream(request.conversation_id)
        if not existing_state:
            return StreamingResponse(
                iter([json.dumps({"type": "error", "error": "No active stream to resume"})]),
                media_type="application/json",
                status_code=404,
            )

        print(f"[Chat] Resuming stream for conversation: {request.conversation_id}")

        async def generate_resume():
            try:
                # Send start event first
                start_data = {
                    'type': 'start',
                    'message_id': existing_state.assistant_msg_id,
                    'title': existing_state.title,
                }
                yield f"data: {json.dumps(start_data)}\n\n"

                # Send accumulated content as first chunk(s) if any
                if existing_state.full_text:
                    yield f"data: {json.dumps({'type': 'chunk', 'text': existing_state.full_text})}\n\n"
                if existing_state.full_thinking:
                    yield f"data: {json.dumps({'type': 'thinking', 'thinking': existing_state.full_thinking})}\n\n"

                while True:
                    if existing_state.is_complete and len(existing_state.chunks) == 0:
                        # Stream is done and no more chunks
                        done_data = {'type': 'done', 'message_id': existing_state.assistant_msg_id, 'title': existing_state.title}
                        yield f"data: {json.dumps(done_data)}\n\n"
                        return

                    chunks = await stream_manager.wait_for_chunks(existing_state, timeout=60.0)
                    for chunk in chunks:
                        if chunk["type"] == "done":
                            # Drain any remaining chunks before exiting
                            continue
                        elif chunk["type"] == "stopped":
                            yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
                            return
                        elif chunk["type"] == "error":
                            return
                        elif chunk["type"] == "chunk":
                            yield f"data: {json.dumps({'type': 'chunk', 'text': chunk['text']})}\n\n"
                    await asyncio.sleep(0)
            except asyncio.CancelledError:
                print(f"[Chat] Resume connection closed for: {request.conversation_id}")
                raise

        return StreamingResponse(
            generate_resume(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-resume requests require a message
    if not request.message.strip():
        return StreamingResponse(
            iter([json.dumps({"type": "error", "error": "Empty message"})]),
            media_type="application/json",
            status_code=400,
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

    # Build messages for LLM
    langchain_messages = _build_langchain_messages(conversation.messages)

    # Search for RAG context if conversation has linked files
    rag_chunks: list[dict] = []
    if conversation.file_ids:
        print(f"[Chat] Conversation has {len(conversation.file_ids)} linked files, searching for RAG context...")
        try:
            rag_chunks = await search_chunks_in_files(
                query=request.message,
                file_ids=conversation.file_ids,
                top_k=5
            )
            if rag_chunks:
                context_parts = []
                for i, chunk in enumerate(rag_chunks):
                    context_parts.append(f"[Context {i+1}] {chunk['chunk_text']}")
                rag_context = "\n\n".join(context_parts)
                rag_system_msg = SystemMessage(
                    content=f"""You have access to the following documents. Use them to answer the user's question if relevant.\n\n{rag_context}\n\nIf the documents don't contain relevant information, say you don't know based on the provided documents."""
                )
                # Insert RAG context as second-to-last position (before user message)
                langchain_messages.insert(-1, rag_system_msg)
                print(f"[Chat] Added {len(rag_chunks)} RAG chunks to context")
            else:
                print(f"[Chat] No relevant chunks found in linked files")
        except Exception as e:
            print(f"[Chat] RAG search error: {e}")

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

    # Get current title (will be updated after stream completes if first message)
    title = conversation.title

    # Start the background stream (this runs independently of HTTP)
    stream_state = stream_manager.start_stream(
        request.conversation_id,
        assistant_msg_id,
        langchain_messages,
        title=title,
        rag_contexts=rag_chunks,
    )

    async def generate():
        nonlocal title

        yield f"data: {json.dumps({'type': 'start', 'message_id': assistant_msg_id, 'title': title})}\n\n"

        try:
            # Wait for chunks from the background stream
            while True:
                # Check if stream is already complete
                if stream_state.is_complete and len(stream_state.chunks) == 0:
                    break

                # Wait for new chunks
                chunks = await stream_manager.wait_for_chunks(stream_state, timeout=60.0)

                for chunk in chunks:
                    if chunk["type"] == "done":
                        # Generate title after stream completes (only for first message)
                        if len(conversation.messages) == 1 and conversation.title == "New Chat":
                            print(f"[Title] Starting title generation after stream for: {request.message[:50]}...")

                            def run_title_gen():
                                return generate_title(request.message)

                            title_task = asyncio.create_task(asyncio.to_thread(run_title_gen))
                            new_title = await title_task
                            if new_title and new_title != "New Chat":
                                title = new_title
                                conversation_service.update_title(conversation.id, title)
                                print(f"[Title] Updated after stream: {title}")

                        yield f"data: {json.dumps({'type': 'done', 'message_id': assistant_msg_id, 'title': title})}\n\n"
                        return

                    elif chunk["type"] == "stopped":
                        yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
                        return

                    elif chunk["type"] == "error":
                        yield f"data: {json.dumps({'type': 'error', 'error': chunk.get('error', 'Unknown error')})}\n\n"
                        return

                    elif chunk["type"] == "chunk":
                        yield f"data: {json.dumps({'type': 'chunk', 'text': chunk['text']})}\n\n"

                    elif chunk["type"] == "thinking":
                        yield f"data: {json.dumps({'type': 'thinking', 'thinking': chunk['thinking']})}\n\n"

                    # Small yield to prevent blocking
                    await asyncio.sleep(0)

        except asyncio.CancelledError:
            # HTTP connection was closed, but the background stream continues!
            print(f"[Chat] HTTP connection closed, background stream continues for: {request.conversation_id}")
            raise

        finally:
            # Don't clean up the stream - let it run in background
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
    # Stop the background stream
    stream_manager.stop_stream(conversation_id)

    # Legacy support
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