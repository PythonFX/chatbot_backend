from __future__ import annotations

import asyncio
import re
import threading
from functools import partial
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, List, Optional

from mlx_lm.sample_utils import make_sampler

from .base import BaseLLMClient
from .models import LLMResponse, Message, Messages, StreamChunk, StreamEvent, ToolDef

THINK_START = "<|channel>thought"
THINK_END = "<channel|>"
CONTROL_TOKENS_RE = re.compile(r"<\|?turn\|?>")
DEFAULT_TEMPERATURE = 0.2
MAX_TOKEN = 8192


class MlxClient(BaseLLMClient):
    def __init__(
        self,
        model_path: str,
        enable_thinking: bool = True,
    ) -> None:
        super().__init__(model=model_path)
        self._model_path = Path(model_path)
        self._enable_thinking = enable_thinking
        self._model = None
        self._tokenizer = None
        self._load_lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            from mlx_lm.utils import load_model, load_tokenizer

            self._model, config = load_model(self._model_path, strict=False)
            self._tokenizer = load_tokenizer(self._model_path, eos_token_ids=config.get("eos_token_id"))

    @staticmethod
    def _strip_control_tokens(text: str) -> str:
        return CONTROL_TOKENS_RE.sub("", text)

    @staticmethod
    def _parse_response(text: str) -> tuple[str, str]:
        if THINK_START in text and THINK_END in text:
            start = text.index(THINK_START) + len(THINK_START)
            end = text.index(THINK_END)
            thinking = MlxClient._strip_control_tokens(text[start:end])
            answer = MlxClient._strip_control_tokens(text[end + len(THINK_END):])
            return thinking, answer
        return "", MlxClient._strip_control_tokens(text)

    def _build_prompt(self, messages: List[Message], enable_thinking: bool) -> str:
        dict_messages = [{"role": m.role, "content": m.content} for m in messages]
        return self._tokenizer.apply_chat_template(
            dict_messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )

    def completion(
        self,
        messages: Messages,
        model: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[ToolDef]] = None,
        max_tokens: int = MAX_TOKEN,
        temperature: float = DEFAULT_TEMPERATURE,
        **kwargs: Any,
    ) -> LLMResponse:
        messages = self._normalize_messages(messages)
        from mlx_lm import generate

        self._load()

        if system:
            messages = [Message(role="system", content=system)] + messages

        enable_thinking = kwargs.pop("enable_thinking", self._enable_thinking)
        prompt = self._build_prompt(messages, enable_thinking)
        sampler = make_sampler(temp=temperature)
        raw = generate(
            self._model, self._tokenizer, prompt=prompt,
            max_tokens=max_tokens, sampler=sampler, verbose=False,
        )
        thinking, content = self._parse_response(raw)
        return LLMResponse(content=content, thinking=thinking, stop_reason="end_turn")

    async def async_completion(
        self,
        messages: Messages,
        model: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[ToolDef]] = None,
        max_tokens: int = MAX_TOKEN,
        temperature: float = DEFAULT_TEMPERATURE,
        **kwargs: Any,
    ) -> LLMResponse:
        messages = self._normalize_messages(messages)
        loop = asyncio.get_event_loop()
        func = partial(
            self.completion, messages, model=model, system=system,
            tools=tools, max_tokens=max_tokens, temperature=temperature, **kwargs,
        )
        return await loop.run_in_executor(None, func)

    def stream(
        self,
        messages: Messages,
        model: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[ToolDef]] = None,
        max_tokens: int = MAX_TOKEN,
        temperature: float = DEFAULT_TEMPERATURE,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        messages = self._normalize_messages(messages)
        from mlx_lm import stream_generate

        self._load()

        if system:
            messages = [Message(role="system", content=system)] + messages

        enable_thinking = kwargs.pop("enable_thinking", self._enable_thinking)
        prompt = self._build_prompt(messages, enable_thinking)
        sampler = make_sampler(temp=temperature)

        buffer = ""
        in_thinking = False
        thinking_done = False

        for chunk in stream_generate(
            self._model, self._tokenizer, prompt=prompt,
            max_tokens=max_tokens, sampler=sampler,
        ):
            buffer += chunk.text

            while True:
                if not thinking_done and not in_thinking and THINK_START in buffer:
                    before = buffer[: buffer.index(THINK_START)]
                    if before:
                        yield StreamChunk(event=StreamEvent.TEXT, data=self._strip_control_tokens(before))
                    buffer = buffer[buffer.index(THINK_START) + len(THINK_START) :]
                    in_thinking = True
                    continue

                if in_thinking and THINK_END in buffer:
                    before = buffer[: buffer.index(THINK_END)]
                    if before:
                        yield StreamChunk(event=StreamEvent.THINKING, data=self._strip_control_tokens(before))
                    buffer = buffer[buffer.index(THINK_END) + len(THINK_END) :]
                    in_thinking = False
                    thinking_done = True
                    continue

                if not in_thinking and not thinking_done:
                    if buffer.startswith("<|"):
                        break
                    safe_end = len(buffer)
                    for marker in ("<|channel>", "<channel|>", "<turn|>"):
                        idx = buffer.find(marker)
                        if idx != -1:
                            safe_end = min(safe_end, idx)
                    if safe_end > 0:
                        yield StreamChunk(event=StreamEvent.TEXT, data=self._strip_control_tokens(buffer[:safe_end]))
                        buffer = buffer[safe_end:]
                    break

                if in_thinking:
                    safe_end = len(buffer)
                    idx = buffer.find("<channel|>")
                    if idx != -1:
                        safe_end = idx
                    if safe_end > 0:
                        yield StreamChunk(event=StreamEvent.THINKING, data=self._strip_control_tokens(buffer[:safe_end]))
                        buffer = buffer[safe_end:]
                    break

                if thinking_done:
                    yield StreamChunk(event=StreamEvent.TEXT, data=self._strip_control_tokens(buffer))
                    buffer = ""
                    break

                break

        if buffer:
            yield StreamChunk(event=StreamEvent.TEXT, data=self._strip_control_tokens(buffer))
        yield StreamChunk(event=StreamEvent.DONE)

    async def async_stream(
        self,
        messages: Messages,
        model: Optional[str] = None,
        system: Optional[str] = None,
        tools: Optional[List[ToolDef]] = None,
        max_tokens: int = MAX_TOKEN,
        temperature: float = DEFAULT_TEMPERATURE,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        messages = self._normalize_messages(messages)
        sync_iter = self.stream(
            messages, model=model, system=system,
            tools=tools, max_tokens=max_tokens, temperature=temperature, **kwargs,
        )
        loop = asyncio.get_event_loop()
        while True:
            try:
                chunk = await loop.run_in_executor(None, next, sync_iter)
            except StopIteration:
                break
            yield chunk

    def close(self) -> None:
        self._model = None
        self._tokenizer = None

    async def aclose(self) -> None:
        self.close()
