import asyncio
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from services import conversation_service
from services.llm_manager import create_llm_client
from llm_client import StreamEvent
from services.title_generator import generate_title


@dataclass
class StreamState:
    """State for an active stream."""
    conversation_id: str
    assistant_msg_id: str
    chunks: deque = field(default_factory=deque)  # Received chunks buffer
    event: asyncio.Event = field(default_factory=asyncio.Event)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: Optional[asyncio.Task] = None
    title: str = "New Chat"
    full_text: str = ""
    full_thinking: str = ""
    is_complete: bool = False
    rag_contexts: list[dict] = field(default_factory=list)


class StreamManager:
    """
    Manages background LLM streams that persist even when HTTP connections断开.
    Each conversation has its own background task that continuously receives
    chunks from the LLM and saves them to JSON.
    """

    def __init__(self):
        self._streams: dict[str, StreamState] = {}

    def get_stream(self, conversation_id: str) -> Optional[StreamState]:
        """Get stream state for a conversation."""
        return self._streams.get(conversation_id)

    def is_streaming(self, conversation_id: str) -> bool:
        """Check if a conversation is currently being streamed."""
        state = self._streams.get(conversation_id)
        return state is not None and state.task is not None and not state.task.done()

    def start_stream(
        self,
        conversation_id: str,
        assistant_msg_id: str,
        messages: list,
        title: str = "New Chat",
        rag_contexts: list[dict] | None = None,
    ) -> StreamState:
        """
        Start a background task that continuously receives LLM chunks.
        This task runs independently of HTTP connections.
        """
        # If already streaming, return existing state
        if conversation_id in self._streams:
            state = self._streams[conversation_id]
            if state.task and not state.task.done():
                return state
            # Clean up old completed task
            del self._streams[conversation_id]

        # Create new stream state
        state = StreamState(
            conversation_id=conversation_id,
            assistant_msg_id=assistant_msg_id,
            title=title,
            rag_contexts=rag_contexts or [],
        )
        self._streams[conversation_id] = state

        # Create background task (does NOT inherit from asyncio.CancelledError of HTTP)
        state.task = asyncio.create_task(
            self._run_llm_stream(state, messages)
        )

        print(f"[StreamManager] Started stream for conversation: {conversation_id}")
        return state

    async def _run_llm_stream(self, state: StreamState, messages: list):
        """
        Background task that continuously receives chunks from LLM.
        This runs independently of any HTTP connection.
        """
        llm = create_llm_client()
        full_text = ""
        full_thinking = ""

        try:
            print(f"[StreamManager] Beginning LLM stream for: {state.conversation_id}")
            async for chunk in llm.async_stream(messages):
                if state.stop_event.is_set():
                    print(f"[StreamManager] Stop event detected for: {state.conversation_id}")
                    break

                # Update thinking if present
                if chunk.event == StreamEvent.THINKING:
                    full_thinking += chunk.data
                    conversation_service.append_chunk(
                        state.conversation_id,
                        state.assistant_msg_id,
                        text="",
                        thinking=chunk.data,
                    )
                    state.chunks.append({
                        "type": "thinking",
                        "thinking": chunk.data,
                    })
                    state.event.set()

                # Update text if present
                elif chunk.event == StreamEvent.TEXT:
                    full_text += chunk.data
                    conversation_service.append_chunk(
                        state.conversation_id,
                        state.assistant_msg_id,
                        text=chunk.data,
                        thinking="",
                    )
                    state.full_text = full_text
                    state.full_thinking = full_thinking

                    state.chunks.append({
                        "type": "chunk",
                        "text": chunk.data,
                    })
                    state.event.set()

                elif chunk.event == StreamEvent.DONE:
                    break

                await asyncio.sleep(0)

            # Save partial/complete content
            state.is_complete = True
            state.full_text = full_text
            state.full_thinking = full_thinking

            conversation_service.update_message(
                state.conversation_id,
                state.assistant_msg_id,
                content=full_text,
                thinking=full_thinking,
                complete=True,
                rag_contexts=state.rag_contexts,
            )

            if state.stop_event.is_set():
                state.chunks.append({"type": "stopped"})
                state.event.set()
                print(f"[StreamManager] Stream stopped for: {state.conversation_id} ({len(full_text)} chars)")
            else:
                state.chunks.append({"type": "done"})
                state.event.set()
                print(f"[StreamManager] Stream complete for: {state.conversation_id} ({len(full_text)} chars)")

        except asyncio.CancelledError:
            print(f"[StreamManager] Stream cancelled for: {state.conversation_id}")
            state.is_complete = True
            conversation_service.update_message(
                state.conversation_id,
                state.assistant_msg_id,
                content=full_text,
                thinking=full_thinking,
                complete=True,
            )
            state.chunks.append({"type": "stopped"})
            state.event.set()
            raise

        except Exception as e:
            print(f"[StreamManager] Stream error for {state.conversation_id}: {e}")
            state.is_complete = True
            state.full_text = full_text
            state.full_thinking = full_thinking
            # Save partial content before the error
            conversation_service.update_message(
                state.conversation_id,
                state.assistant_msg_id,
                content=full_text,
                thinking=full_thinking,
                complete=True,
                rag_contexts=state.rag_contexts,
            )
            state.chunks.append({"type": "error", "error": str(e)})
            state.event.set()

    def stop_stream(self, conversation_id: str) -> bool:
        """Stop an active stream."""
        state = self._streams.get(conversation_id)
        if not state:
            return False

        state.stop_event.set()
        print(f"[StreamManager] Stop signal sent for: {conversation_id}")
        return True

    async def wait_for_chunks(self, state: StreamState, timeout: float = 60.0):
        """
        Wait for new chunks to arrive.
        Returns list of new chunks since last call.
        """
        try:
            await asyncio.wait_for(state.event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        state.event.clear()

        # Drain all accumulated chunks
        chunks = list(state.chunks)
        state.chunks.clear()
        return chunks

    def cleanup(self, conversation_id: str):
        """Clean up stream state after it's no longer needed."""
        if conversation_id in self._streams:
            del self._streams[conversation_id]


# Global stream manager instance
stream_manager = StreamManager()