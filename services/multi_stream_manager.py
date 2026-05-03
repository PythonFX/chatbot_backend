"""
Multi-model stream manager: runs multiple LLM streams concurrently
and merges their results into versions of a single assistant message.
"""
import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from services import conversation_service
from services.llm_factory import create_llm_client, get_available_models, get_current_model
from services.title_generator import generate_title


@dataclass
class ModelStreamState:
    """State for a single model's stream within a multi-model session."""
    model: str
    full_text: str = ""
    full_thinking: str = ""
    is_complete: bool = False
    is_stopped: bool = False
    error: Optional[str] = None


@dataclass
class MultiStreamState:
    """State for a multi-model stream session."""
    conversation_id: str
    assistant_msg_id: str
    models: list[str] = field(default_factory=list)
    model_states: dict[str, ModelStreamState] = field(default_factory=dict)
    chunks: deque = field(default_factory=deque)
    event: asyncio.Event = field(default_factory=asyncio.Event)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: Optional[asyncio.Task] = None
    title: str = "New Chat"
    all_complete: bool = False
    rag_contexts: list[dict] = field(default_factory=list)


class MultiStreamManager:
    """Manages concurrent LLM streams across multiple models."""

    def __init__(self):
        self._streams: dict[str, MultiStreamState] = {}

    def get_stream(self, conversation_id: str) -> Optional[MultiStreamState]:
        return self._streams.get(conversation_id)

    def is_streaming(self, conversation_id: str) -> bool:
        state = self._streams.get(conversation_id)
        return state is not None and state.task is not None and not state.task.done()

    def start_multi_stream(
        self,
        conversation_id: str,
        assistant_msg_id: str,
        messages: list,
        models: list[str],
        title: str = "New Chat",
        rag_contexts: list[dict] | None = None,
    ) -> MultiStreamState:
        if conversation_id in self._streams:
            state = self._streams[conversation_id]
            if state.task and not state.task.done():
                return state
            del self._streams[conversation_id]

        model_states = {
            m: ModelStreamState(model=m) for m in models
        }
        state = MultiStreamState(
            conversation_id=conversation_id,
            assistant_msg_id=assistant_msg_id,
            models=models,
            model_states=model_states,
            title=title,
            rag_contexts=rag_contexts or [],
        )
        self._streams[conversation_id] = state

        state.task = asyncio.create_task(
            self._run_multi_stream(state, messages)
        )
        print(f"[MultiStreamManager] Started multi-stream for: {conversation_id} models={models}")
        return state

    async def _run_single_model(self, state: MultiStreamState, model: str, messages: list):
        """Run a single model's stream and push chunks to the shared queue."""
        ms = state.model_states[model]
        llm = create_llm_client(model=model)
        full_text = ""
        full_thinking = ""

        try:
            async for parsed in llm.astream(messages):
                if state.stop_event.is_set():
                    ms.is_stopped = True
                    break

                if parsed.get("thinking"):
                    full_thinking += parsed["thinking"]
                    ms.full_thinking = full_thinking
                    state.chunks.append({
                        "type": "thinking",
                        "thinking": parsed["thinking"],
                        "model": model,
                    })
                    state.event.set()

                if parsed.get("text"):
                    full_text += parsed["text"]
                    ms.full_text = full_text
                    state.chunks.append({
                        "type": "chunk",
                        "text": parsed["text"],
                        "model": model,
                    })
                    state.event.set()

                await asyncio.sleep(0)

            ms.is_complete = True
            ms.full_text = full_text
            ms.full_thinking = full_thinking

            state.chunks.append({
                "type": "model_done",
                "model": model,
            })
            state.event.set()

            print(f"[MultiStreamManager] Model {model} complete: {len(full_text)} chars")

        except asyncio.CancelledError:
            ms.is_complete = True
            ms.is_stopped = True
            state.chunks.append({
                "type": "model_done",
                "model": model,
            })
            state.event.set()
            raise

        except Exception as e:
            ms.is_complete = True
            ms.error = str(e)
            state.chunks.append({
                "type": "model_error",
                "model": model,
                "error": str(e),
            })
            state.event.set()
            print(f"[MultiStreamManager] Model {model} error: {e}")

    async def _run_multi_stream(self, state: MultiStreamState, messages: list):
        """Background task that runs all model streams concurrently."""
        try:
            await asyncio.gather(*[
                self._run_single_model(state, model, messages)
                for model in state.models
            ])

            # All models done -- merge results into versions
            self._merge_results(state)

            state.all_complete = True
            state.chunks.append({"type": "done"})
            state.event.set()

            print(f"[MultiStreamManager] All models complete for: {state.conversation_id}")

        except asyncio.CancelledError:
            # On cancellation, still merge whatever we have
            self._merge_results(state)
            state.all_complete = True
            state.chunks.append({"type": "stopped"})
            state.event.set()
            print(f"[MultiStreamManager] Multi-stream cancelled for: {state.conversation_id}")
            raise

    def _merge_results(self, state: MultiStreamState):
        """Merge all model results into versions of the primary assistant message."""
        first_success_content = None
        first_success_thinking = None
        first_success_index = None

        for i, model in enumerate(state.models):
            ms = state.model_states[model]
            has_content = bool(ms.full_text or ms.full_thinking)

            if ms.error:
                # Save failed version with error status so the tag persists
                conversation_service.add_version(
                    state.conversation_id,
                    state.assistant_msg_id,
                    "",
                    None,
                    model=model,
                    is_multi_mode=True,
                    status="error",
                    error=ms.error,
                )
            elif has_content:
                conversation_service.add_version(
                    state.conversation_id,
                    state.assistant_msg_id,
                    ms.full_text,
                    ms.full_thinking,
                    model=model,
                    is_multi_mode=True,
                    status="success",
                )
                if first_success_content is None:
                    first_success_content = ms.full_text
                    first_success_thinking = ms.full_thinking
                    first_success_index = i
            else:
                # Empty but not an error — skip
                continue

        # Write first successful model's content to primary fields
        if first_success_content is not None:
            conversation_service.update_message(
                state.conversation_id,
                state.assistant_msg_id,
                content=first_success_content,
                thinking=first_success_thinking,
                complete=True,
                rag_contexts=state.rag_contexts,
            )

        # Select first successful version by default, or version 0 if all failed
        default_index = first_success_index if first_success_index is not None else 0
        conversation_service.select_version(
            state.conversation_id,
            state.assistant_msg_id,
            default_index,
        )

    def stop_multi_stream(self, conversation_id: str) -> bool:
        state = self._streams.get(conversation_id)
        if not state:
            return False
        state.stop_event.set()
        print(f"[MultiStreamManager] Stop signal sent for: {conversation_id}")
        return True

    async def wait_for_chunks(self, state: MultiStreamState, timeout: float = 60.0):
        try:
            await asyncio.wait_for(state.event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        state.event.clear()

        chunks = list(state.chunks)
        state.chunks.clear()
        return chunks

    def cleanup(self, conversation_id: str):
        if conversation_id in self._streams:
            del self._streams[conversation_id]


# Global instance
multi_stream_manager = MultiStreamManager()
