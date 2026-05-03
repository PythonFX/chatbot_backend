import asyncio
import json
import re
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from services import conversation_service
from services.conversation_service import save_conversation
from services.title_generator import generate_title
from services.llm_factory import create_llm_client, is_llm_configured, get_available_models, get_current_model
from services.stream_manager import stream_manager
from services.multi_stream_manager import multi_stream_manager
from services.embedding_service import search_chunks_in_files
from services import novel_service


router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    conversation_id: str
    message: str
    resume: bool = False
    deep_qa_mode: bool = False
    multi_model: bool = False


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
    total_versions: int = 0


class SelectVersionRequest(BaseModel):
    conversation_id: str
    version_index: Optional[int] = None


class SelectVersionResponse(BaseModel):
    message_id: str
    content: str
    thinking: Optional[str] = None
    selected_version_index: Optional[int] = None


class GenerateVersionRequest(BaseModel):
    conversation_id: str


class GenerateVersionResponse(BaseModel):
    message_id: str
    content: str
    thinking: Optional[str] = None
    total_versions: int


class RegenerateModelRequest(BaseModel):
    conversation_id: str
    message_id: str
    model: str


class RegenerateModelResponse(BaseModel):
    message_id: str
    content: str
    thinking: Optional[str] = None
    model: str


stop_events: dict[str, asyncio.Event] = {}


def _build_langchain_messages(messages: list, conversation_id: str = "") -> list:
    langchain_messages = []
    for m in messages:
        if m.role == "user":
            langchain_messages.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            content, _ = conversation_service.get_selected_version(conversation_id, m.id)
            langchain_messages.append(AIMessage(content=content))
        elif m.role == "system":
            langchain_messages.append(SystemMessage(content=m.content))
    return langchain_messages


def _build_novel_system_message(novel_data: dict) -> SystemMessage:
    """Format selected novel data into a system message for the LLM."""
    chapters_lines = "\n".join(
        f"{c['index']}. {c['title']}" for c in novel_data["chapters"]
    )
    inspiration_str = json.dumps(novel_data.get("inspiration") or {}, ensure_ascii=False, indent=2)
    return SystemMessage(
        content=f"""你正在使用小说参考资料进行创作分析。
当前参考小说：《{novel_data["title"]}》

【章节列表】
{chapters_lines}

【灵感分析（inspiration.json）】
{inspiration_str}

请结合上述参考内容进行创作。请先阅读章节列表和灵感分析，然后在后续对话中依据这些内容回答用户的问题或提供创作建议。"""
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    if not is_llm_configured():
        return StreamingResponse(
            iter([json.dumps({"type": "error", "error": "MiniMax not configured"})]),
            media_type="application/json",
            status_code=500,
        )

    conversation = conversation_service.get_conversation(request.conversation_id)
    if not conversation:
        return StreamingResponse(
            iter([json.dumps({"type": "error", "error": "Conversation not found"})]),
            media_type="application/json",
            status_code=404,
        )

    # ── Resume mode ──────────────────────────────────────────────────────────
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
                start_data = {
                    'type': 'start',
                    'message_id': existing_state.assistant_msg_id,
                    'title': existing_state.title,
                }
                yield f"data: {json.dumps(start_data)}\n\n"

                if existing_state.full_text:
                    yield f"data: {json.dumps({'type': 'chunk', 'text': existing_state.full_text})}\n\n"
                if existing_state.full_thinking:
                    yield f"data: {json.dumps({'type': 'thinking', 'thinking': existing_state.full_thinking})}\n\n"

                while True:
                    if existing_state.is_complete and len(existing_state.chunks) == 0:
                        done_data = {'type': 'done', 'message_id': existing_state.assistant_msg_id, 'title': existing_state.title}
                        yield f"data: {json.dumps(done_data)}\n\n"
                        return

                    chunks = await stream_manager.wait_for_chunks(existing_state, timeout=60.0)
                    for chunk in chunks:
                        if chunk["type"] == "done":
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

    # ── Non-resume: require a message ────────────────────────────────────────
    if not request.message.strip():
        return StreamingResponse(
            iter([json.dumps({"type": "error", "error": "Empty message"})]),
            media_type="application/json",
            status_code=400,
        )

    if request.conversation_id in stop_events:
        stop_events[request.conversation_id].set()
    stop_events[request.conversation_id] = asyncio.Event()

    msg_text = request.message.strip()

    # ── Novel agent flow ───────────────────────────────────────────────────────
    is_novel = conversation.is_novel_agent
    is_selecting_book = is_novel and conversation.selected_novel_id is None

    # 1) User typed /novel → enter novel agent mode, list books
    if msg_text == "/novel":
        conversation_service.set_novel_agent(request.conversation_id, True)
        novels = novel_service.list_novels()

        async def generate_book_list():
            yield f"data: {json.dumps({'type': 'novel_books', 'books': novels})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'message_id': '', 'title': conversation.title})}\n\n"

        return StreamingResponse(
            generate_book_list(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # 2) User sent a number → book selection
    if is_selecting_book and re.fullmatch(r"\d+", msg_text):
        idx = int(msg_text)
        novels = novel_service.list_novels()
        if idx < 1 or idx > len(novels):
            async def generate_error():
                yield f"data: {json.dumps({'type': 'error', 'error': f'Invalid selection. Please enter a number between 1 and {len(novels)}.'})}\n\n"

            return StreamingResponse(
                generate_error(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        selected_novel = novels[idx - 1]
        novel_data = novel_service.load_novel(selected_novel["id"])
        conversation_service.set_selected_novel(request.conversation_id, selected_novel["id"])
        conversation_service.update_title(request.conversation_id, selected_novel["title"])

        # Save chapters + analysis + inspiration as an assistant message
        chapters_lines = "\n".join(f"{c['index']}. {c['title']}" for c in novel_data["chapters"])
        msg_content = (
            f"**【参考小说：《{novel_data['title']}》】**\n\n"
            f"**Chapters**\n{chapters_lines}\n\n"
        )
        if novel_data.get("analysis"):
            analysis_str = json.dumps(novel_data["analysis"], ensure_ascii=False, indent=2)
            msg_content += f"**Analysis**\n```json\n{analysis_str}\n```\n\n"
        if novel_data.get("inspiration"):
            inspiration_str = json.dumps(novel_data["inspiration"], ensure_ascii=False, indent=2)
            msg_content += f"**Inspiration**\n```json\n{inspiration_str}\n```"
        else:
            msg_content = msg_content.rstrip()
        conversation_service.add_message(
            request.conversation_id, "assistant", msg_content, type="novel_selected"
        )

        async def generate_selected():
            yield f"data: {json.dumps({'type': 'novel_selected', 'book': novel_data})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'message_id': '', 'title': conversation.title})}\n\n"

        return StreamingResponse(
            generate_selected(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Normal chat ───────────────────────────────────────────────────────────
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

    langchain_messages = _build_langchain_messages(conversation.messages, request.conversation_id)

    # Inject novel context for active novel agent conversations
    if is_novel and conversation.selected_novel_id:
        novel_data = novel_service.load_novel(conversation.selected_novel_id)
        if novel_data:
            novel_sys_msg = _build_novel_system_message(novel_data)
            langchain_messages.insert(-1, novel_sys_msg)

    # RAG context
    rag_chunks: list[dict] = []
    if conversation.file_ids and not request.deep_qa_mode:
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
                langchain_messages.insert(-1, rag_system_msg)
                print(f"[Chat] Added {len(rag_chunks)} RAG chunks to context")
        except Exception as e:
            print(f"[Chat] RAG search error: {e}")

    # Create assistant placeholder
    placeholder_result = conversation_service.add_message(
        request.conversation_id,
        "assistant",
        "",
        complete=False,
        is_multi_mode=request.multi_model,
    )
    if not placeholder_result:
        return StreamingResponse(
            iter([json.dumps({"type": "error", "error": "Failed to create message placeholder"})]),
            media_type="application/json",
            status_code=500,
        )
    _, placeholder_msg = placeholder_result
    assistant_msg_id = placeholder_msg.id
    title = conversation.title
    stream_state = None

    async def generate():
        nonlocal title, rag_chunks

        # Deep Q&A mode
        if request.deep_qa_mode and conversation.file_ids:
            print(f"[Chat] Deep Q&A mode: performing multi-step retrieval...")
            try:
                from services.deep_qa_service import deep_qa_retrieve, build_rag_context

                yield f"data: {json.dumps({'type': 'deep_qa_status', 'status': 'processing', 'message': 'Analyzing contexts...'})}\n\n"

                history_messages = conversation.messages[:-1]
                history_text = "\n".join([
                    f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
                    for m in history_messages
                ])

                final_query, selected_contexts = await deep_qa_retrieve(
                    query=request.message,
                    file_ids=conversation.file_ids,
                    conversation_history=history_text,
                    initial_top_k=10
                )

                yield f"data: {json.dumps({'type': 'deep_qa_status', 'status': 'done', 'message': 'Analysis complete'})}\n\n"

                if selected_contexts:
                    rag_context = build_rag_context(selected_contexts)
                    rag_system_msg = SystemMessage(
                        content=f"""You have access to the following documents. Use them to answer the user's question if relevant.\n\n{rag_context}\n\nIf the documents don't contain relevant information, say you don't know based on the provided documents."""
                    )
                    langchain_messages.insert(-1, rag_system_msg)
                    rag_chunks = selected_contexts
                    print(f"[Chat] Deep Q&A: selected {len(rag_chunks)} contexts")
                else:
                    print(f"[Chat] Deep Q&A: no relevant contexts found")
            except Exception as e:
                print(f"[Chat] Deep Q&A error: {e}")
                yield f"data: {json.dumps({'type': 'deep_qa_status', 'status': 'error', 'message': str(e)})}\n\n"

        # Re-fetch conversation in case it was updated by novel_service calls above
        conversation = conversation_service.get_conversation(request.conversation_id)

        # ── Multi-model branch ──────────────────────────────────────────────
        if request.multi_model:
            models = get_available_models()
            if not models:
                yield f"data: {json.dumps({'type': 'error', 'error': 'No models available'})}\n\n"
                return

            multi_state = multi_stream_manager.start_multi_stream(
                request.conversation_id,
                assistant_msg_id,
                langchain_messages,
                models=models,
                title=title,
                rag_contexts=rag_chunks,
            )

            version_map = {m: i for i, m in enumerate(models)}
            yield f"data: {json.dumps({'type': 'multi_start', 'message_id': assistant_msg_id, 'title': title, 'models': models, 'version_map': version_map})}\n\n"

            try:
                while True:
                    if multi_state.all_complete and len(multi_state.chunks) == 0:
                        break

                    chunks = await multi_stream_manager.wait_for_chunks(multi_state, timeout=60.0)

                    for chunk in chunks:
                        if chunk["type"] == "done":
                            conversation = conversation_service.get_conversation(request.conversation_id)
                            if conversation and len(conversation.messages) > 1 and conversation.title == "New Chat":
                                def run_title_gen():
                                    return generate_title(request.message)
                                title_task = asyncio.create_task(asyncio.to_thread(run_title_gen))
                                new_title = await title_task
                                if new_title and new_title != "New Chat":
                                    title = new_title
                                    conversation_service.update_title(conversation.id, title)

                            yield f"data: {json.dumps({'type': 'done', 'message_id': assistant_msg_id, 'title': title, 'models': models})}\n\n"
                            return

                        elif chunk["type"] == "stopped":
                            yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
                            return

                        elif chunk["type"] == "model_done":
                            yield f"data: {json.dumps({'type': 'model_done', 'model': chunk['model'], 'version_index': version_map[chunk['model']]})}\n\n"

                        elif chunk["type"] == "model_error":
                            yield f"data: {json.dumps({'type': 'model_error', 'model': chunk['model'], 'error': chunk['error']})}\n\n"

                        elif chunk["type"] == "error":
                            yield f"data: {json.dumps({'type': 'error', 'error': chunk.get('error', 'Unknown error')})}\n\n"
                            return

                        elif chunk["type"] == "chunk":
                            yield f"data: {json.dumps({'type': 'chunk', 'text': chunk['text'], 'model': chunk['model']})}\n\n"

                        elif chunk["type"] == "thinking":
                            yield f"data: {json.dumps({'type': 'thinking', 'thinking': chunk['thinking'], 'model': chunk['model']})}\n\n"

                        await asyncio.sleep(0)

            except asyncio.CancelledError:
                print(f"[Chat] HTTP connection closed (multi), background stream continues for: {request.conversation_id}")
                raise

            finally:
                if request.conversation_id in stop_events:
                    del stop_events[request.conversation_id]

        # ── Single-model branch ─────────────────────────────────────────────
        else:
            stream_state = stream_manager.start_stream(
                request.conversation_id,
                assistant_msg_id,
                langchain_messages,
                title=title,
                rag_contexts=rag_chunks,
            )

            yield f"data: {json.dumps({'type': 'start', 'message_id': assistant_msg_id, 'title': title})}\n\n"

            try:
                while True:
                    if stream_state.is_complete and len(stream_state.chunks) == 0:
                        break

                    chunks = await stream_manager.wait_for_chunks(stream_state, timeout=60.0)

                    for chunk in chunks:
                        if chunk["type"] == "done":
                            conversation = conversation_service.get_conversation(request.conversation_id)
                            if conversation and len(conversation.messages) > 1 and conversation.title == "New Chat":
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

                        await asyncio.sleep(0)

            except asyncio.CancelledError:
                print(f"[Chat] HTTP connection closed, background stream continues for: {request.conversation_id}")
                raise

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
    stream_manager.stop_stream(conversation_id)
    multi_stream_manager.stop_multi_stream(conversation_id)
    if conversation_id in stop_events:
        stop_events[conversation_id].set()
        return {"status": "cancelled", "conversation_id": conversation_id}
    return {"status": "no_active_generation", "conversation_id": conversation_id}


@router.post("/chat/regenerate", response_model=RegenerateResponse)
async def regenerate_response(request: RegenerateRequest):
    if not is_llm_configured():
        raise HTTPException(status_code=500, detail="MiniMax not configured")

    conversation = conversation_service.get_conversation(request.conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    user_msg = None
    user_msg_index = -1
    for i, m in enumerate(conversation.messages):
        if m.id == request.message_id and m.role == "user":
            user_msg = m
            user_msg_index = i
            break

    if not user_msg:
        raise HTTPException(status_code=404, detail="User message not found")

    assistant_msg = None
    if user_msg_index + 1 < len(conversation.messages):
        next_msg = conversation.messages[user_msg_index + 1]
        if next_msg.role == "assistant":
            assistant_msg = next_msg

    if assistant_msg:
        conversation_service.add_version(
            request.conversation_id,
            assistant_msg.id,
            assistant_msg.content,
            assistant_msg.thinking,
            model=get_current_model(),
        )

    langchain_messages = _build_langchain_messages(conversation.messages[:user_msg_index + 1], request.conversation_id)

    # Inject novel context if in novel agent mode with a selected novel
    if conversation.is_novel_agent and conversation.selected_novel_id:
        novel_data = novel_service.load_novel(conversation.selected_novel_id)
        if novel_data:
            novel_sys_msg = _build_novel_system_message(novel_data)
            langchain_messages.insert(-1, novel_sys_msg)

    try:
        llm = create_llm_client()
        response = await llm.invoke(langchain_messages)

        full_text = response.get("text", "")
        thinking = response.get("thinking")

    except asyncio.CancelledError:
        raise HTTPException(status_code=499, detail="Generation cancelled")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MiniMax error: {str(e)}")

    if not full_text:
        raise HTTPException(status_code=500, detail="Empty response from MiniMax")

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

    total_versions = len(ai_msg.versions) if ai_msg.versions else 0

    return RegenerateResponse(
        conversation_id=request.conversation_id,
        message_id=ai_msg.id,
        content=full_text,
        thinking=thinking,
        type="text",
        total_versions=total_versions,
    )


@router.post("/chat/message/{message_id}/versions", response_model=GenerateVersionResponse)
async def generate_version(message_id: str, request: GenerateVersionRequest):
    """Generate an additional version for an existing assistant message without replacing current content."""
    if not is_llm_configured():
        raise HTTPException(status_code=500, detail="MiniMax not configured")

    conversation = conversation_service.get_conversation(request.conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    target_msg = None
    for m in conversation.messages:
        if m.id == message_id and m.role == "assistant":
            target_msg = m
            break

    if not target_msg:
        raise HTTPException(status_code=404, detail="Assistant message not found")

    # Save current primary as version[0] if no versions exist yet
    if not target_msg.versions:
        conversation_service.add_version(
            request.conversation_id,
            message_id,
            target_msg.content,
            target_msg.thinking,
            model=get_current_model(),
        )

    # Build context: all messages up to and including this assistant message
    user_msg_index = -1
    for i, m in enumerate(conversation.messages):
        if m.id == message_id and m.role == "assistant":
            user_msg_index = i - 1
            break
    if user_msg_index < 0:
        raise HTTPException(status_code=400, detail="Cannot find preceding user message")

    langchain_messages = _build_langchain_messages(
        conversation.messages[:user_msg_index + 1],
        request.conversation_id,
    )

    # Inject novel/RAG context
    if conversation.is_novel_agent and conversation.selected_novel_id:
        novel_data = novel_service.load_novel(conversation.selected_novel_id)
        if novel_data:
            novel_sys_msg = _build_novel_system_message(novel_data)
            langchain_messages.insert(-1, novel_sys_msg)

    try:
        llm = create_llm_client()
        response = await llm.invoke(langchain_messages)
        full_text = response.get("text", "")
        thinking = response.get("thinking")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MiniMax error: {str(e)}")

    if not full_text:
        raise HTTPException(status_code=500, detail="Empty response")

    # Append as a new version (does NOT replace primary)
    updated_msg = conversation_service.add_version(
        request.conversation_id,
        message_id,
        full_text,
        thinking,
        # No model field — single-model versions should not light up a model tab
    )
    # Select the newly generated version (the last one)
    new_version_index = len(updated_msg.versions) - 1
    conversation_service.select_version(request.conversation_id, message_id, new_version_index)

    total_versions = len(updated_msg.versions) if updated_msg.versions else 1

    return GenerateVersionResponse(
        message_id=message_id,
        content=full_text,
        thinking=thinking,
        total_versions=total_versions,
    )


@router.post("/chat/message/{message_id}/versions/stream")
async def generate_version_stream(message_id: str, request: GenerateVersionRequest):
    """Stream-generate a new version for an assistant message via SSE."""
    if not is_llm_configured():
        return StreamingResponse(
            iter([json.dumps({"type": "error", "error": "LLM not configured"})]),
            media_type="application/json",
            status_code=500,
        )

    conversation = conversation_service.get_conversation(request.conversation_id)
    if not conversation:
        return StreamingResponse(
            iter([json.dumps({"type": "error", "error": "Conversation not found"})]),
            media_type="application/json",
            status_code=404,
        )

    target_msg = None
    for m in conversation.messages:
        if m.id == message_id and m.role == "assistant":
            target_msg = m
            break

    if not target_msg:
        return StreamingResponse(
            iter([json.dumps({"type": "error", "error": "Assistant message not found"})]),
            media_type="application/json",
            status_code=404,
        )

    # Save current primary as version[0] if no versions exist yet
    if not target_msg.versions:
        conversation_service.add_version(
            request.conversation_id,
            message_id,
            target_msg.content,
            target_msg.thinking,
            model=get_current_model(),
        )

    # Build context: all messages up to and including this assistant message
    user_msg_index = -1
    for i, m in enumerate(conversation.messages):
        if m.id == message_id and m.role == "assistant":
            user_msg_index = i - 1
            break
    if user_msg_index < 0:
        return StreamingResponse(
            iter([json.dumps({"type": "error", "error": "Cannot find preceding user message"})]),
            media_type="application/json",
            status_code=400,
        )

    langchain_messages = _build_langchain_messages(
        conversation.messages[:user_msg_index + 1],
        request.conversation_id,
    )

    # Inject novel/RAG context
    if conversation.is_novel_agent and conversation.selected_novel_id:
        novel_data = novel_service.load_novel(conversation.selected_novel_id)
        if novel_data:
            novel_sys_msg = _build_novel_system_message(novel_data)
            langchain_messages.insert(-1, novel_sys_msg)

    # Create empty placeholder version (auto-selected by add_version)
    updated_msg = conversation_service.add_version(
        request.conversation_id,
        message_id,
        "",
        None,
        status="generating",
    )
    new_version_index = len(updated_msg.versions) - 1

    async def generate():
        yield f"data: {json.dumps({'type': 'start', 'version_index': new_version_index})}\n\n"

        llm = create_llm_client()
        full_text = ""
        full_thinking = ""

        try:
            async for parsed in llm.astream(langchain_messages):
                if parsed.get("thinking"):
                    thinking_chunk = parsed["thinking"]
                    full_thinking += thinking_chunk
                    conversation_service.append_chunk_to_version(
                        request.conversation_id,
                        message_id,
                        new_version_index,
                        thinking=thinking_chunk,
                    )
                    yield f"data: {json.dumps({'type': 'thinking', 'thinking': thinking_chunk})}\n\n"

                if parsed.get("text"):
                    text_chunk = parsed["text"]
                    full_text += text_chunk
                    conversation_service.append_chunk_to_version(
                        request.conversation_id,
                        message_id,
                        new_version_index,
                        text=text_chunk,
                    )
                    yield f"data: {json.dumps({'type': 'chunk', 'text': text_chunk})}\n\n"

                await asyncio.sleep(0)

            if not full_text:
                conversation_service.update_version(
                    request.conversation_id,
                    message_id,
                    new_version_index,
                    "",
                    full_thinking,
                    status="error",
                    error="Empty response from LLM",
                )
                yield f"data: {json.dumps({'type': 'error', 'error': 'Empty response from LLM'})}\n\n"
                return

            # Mark as success
            conversation_service.update_version(
                request.conversation_id,
                message_id,
                new_version_index,
                full_text,
                full_thinking,
                status="success",
            )

            total_versions = len(updated_msg.versions) if updated_msg.versions else 1
            yield f"data: {json.dumps({'type': 'done', 'message_id': message_id, 'content': full_text, 'thinking': full_thinking, 'total_versions': total_versions})}\n\n"

        except Exception as e:
            conversation_service.update_version(
                request.conversation_id,
                message_id,
                new_version_index,
                full_text,
                full_thinking,
                status="error",
                error=str(e),
            )
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.patch("/chat/message/{message_id}/select-version", response_model=SelectVersionResponse)
async def select_message_version(message_id: str, request: SelectVersionRequest):
    """Update the selected version index for an assistant message."""
    updated_msg = conversation_service.select_version(
        request.conversation_id,
        message_id,
        request.version_index,
    )
    if not updated_msg:
        raise HTTPException(status_code=404, detail="Message or version index not found")

    content, thinking = conversation_service.get_selected_version(
        request.conversation_id,
        message_id,
    )
    return SelectVersionResponse(
        message_id=message_id,
        content=content,
        thinking=thinking,
        selected_version_index=updated_msg.selected_version_index,
    )


@router.post("/chat/regenerate-model", response_model=RegenerateModelResponse)
async def regenerate_model(request: RegenerateModelRequest):
    """Re-generate a single failed model response for a multi-model assistant message."""
    if not is_llm_configured(request.model):
        raise HTTPException(status_code=500, detail=f"Model {request.model} not configured")

    conversation = conversation_service.get_conversation(request.conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Find the assistant message
    assistant_msg = None
    assistant_idx = -1
    for i, m in enumerate(conversation.messages):
        if m.id == request.message_id and m.role == "assistant":
            assistant_msg = m
            assistant_idx = i
            break

    if not assistant_msg:
        raise HTTPException(status_code=404, detail="Assistant message not found")

    # Find preceding user message
    user_msg = None
    for i in range(assistant_idx - 1, -1, -1):
        if conversation.messages[i].role == "user":
            user_msg = conversation.messages[i]
            break

    if not user_msg:
        raise HTTPException(status_code=400, detail="No preceding user message found")

    # Build context up to and including the user message
    langchain_messages = _build_langchain_messages(
        conversation.messages[:assistant_idx],
        request.conversation_id,
    )

    # Inject novel context if in novel agent mode
    if conversation.is_novel_agent and conversation.selected_novel_id:
        novel_data = novel_service.load_novel(conversation.selected_novel_id)
        if novel_data:
            novel_sys_msg = _build_novel_system_message(novel_data)
            langchain_messages.insert(-1, novel_sys_msg)

    # RAG context
    if conversation.file_ids:
        try:
            rag_chunks = await search_chunks_in_files(
                query=user_msg.content,
                file_ids=conversation.file_ids,
                top_k=5,
            )
            if rag_chunks:
                context_parts = []
                for i, chunk in enumerate(rag_chunks):
                    context_parts.append(f"[Context {i+1}] {chunk['chunk_text']}")
                rag_context = "\n\n".join(context_parts)
                rag_system_msg = SystemMessage(
                    content=f"""You have access to the following documents. Use them to answer the user's question if relevant.\n\n{rag_context}\n\nIf the documents don't contain relevant information, say you don't know based on the provided documents."""
                )
                langchain_messages.insert(-1, rag_system_msg)
        except Exception as e:
            print(f"[Chat] RAG search error during model regenerate: {e}")

    try:
        llm = create_llm_client(model=request.model)
        response = await llm.invoke(langchain_messages)
        full_text = response.get("text", "")
        thinking = response.get("thinking")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{request.model} error: {str(e)}")

    if not full_text:
        raise HTTPException(status_code=500, detail="Empty response from model")

    # Find version index by model name
    version_index = None
    if assistant_msg.versions:
        for i, v in enumerate(assistant_msg.versions):
            if v.get("model") == request.model:
                version_index = i
                break

    if version_index is not None:
        # Update existing version in-place
        updated_msg = conversation_service.update_version(
            request.conversation_id,
            request.message_id,
            version_index,
            full_text,
            thinking,
            status="success",
        )
    else:
        # Should not happen, but fallback to adding a new version
        updated_msg = conversation_service.add_version(
            request.conversation_id,
            request.message_id,
            full_text,
            thinking,
            model=request.model,
            is_multi_mode=True,
            status="success",
        )

    if not updated_msg:
        raise HTTPException(status_code=500, detail="Failed to save regenerated response")

    return RegenerateModelResponse(
        message_id=request.message_id,
        content=full_text,
        thinking=thinking,
        model=request.model,
    )
