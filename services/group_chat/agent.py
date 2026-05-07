"""
Group chat agent: wraps an LLM model with evaluate and respond capabilities.

Each agent participates in a group chat by:
1. Evaluating whether it should respond to new messages
2. Generating a response when the coordinator approves

The evaluate step uses prompt-based structured output (JSON) to decide
whether to respond, with what priority, and how much to say.
This can be upgraded to native tool calling when the LLM client supports it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from services.llm_factory import create_llm_client, _LLMClientAdapter, _convert_langchain_messages, MODEL_DISPLAY_NAMES

from llm_client import Message


EVALUATE_SYSTEM_PROMPT = """\
You are {name}, an AI assistant participating in a group chat with the user and other AI assistants.

Your task: Evaluate whether you should respond to the latest message in the conversation.

Guidelines:
- If another assistant already gave a correct and complete answer, you probably don't need to respond.
- If you have a DIFFERENT perspective, additional information, or a correction to make, you should respond.
- If the user's question is in your area of expertise, you should respond.
- If you strongly disagree with a previous answer, respond with high priority.

Respond with ONLY a JSON object (no markdown, no explanation outside the JSON):
{{
  "should_respond": true/false,
  "reason": "brief explanation of your decision",
  "priority": 1-5,
  "desired_length": "brief" or "moderate" or "detailed"
}}

Priority levels:
1 = low (minor addition, optional)
2 = below average (small clarification)
3 = moderate (different perspective worth sharing)
4 = high (important correction or significant addition)
5 = urgent (critical error in previous answer that must be corrected)

Length levels:
- brief: 1-2 sentences
- moderate: 1 paragraph
- detailed: multiple paragraphs with thorough explanation
"""

RESPOND_SYSTEM_PROMPT = """\
You are {name}, an AI assistant participating in a group chat.

You are responding because: {reason}

Guidelines:
- Be {length_instruction}.
- Address the user's question directly.
- If correcting a previous answer, be specific about what was wrong.
- If adding to a previous answer, acknowledge what was already said and add your contribution.
- Use your unique perspective and strengths as {name}.
- Do not repeat what another assistant already said unless you're correcting it.
"""


@dataclass
class EvaluationResult:
    agent_id: str
    agent_name: str
    should_respond: bool
    reason: str
    priority: int  # 1-5
    desired_length: str  # "brief", "moderate", "detailed"

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "should_respond": self.should_respond,
            "reason": self.reason,
            "priority": self.priority,
            "desired_length": self.desired_length,
        }


class GroupChatAgent:
    def __init__(self, model_id: str, display_name: Optional[str] = None):
        self.model_id = model_id
        self.display_name = display_name or MODEL_DISPLAY_NAMES.get(model_id, model_id)
        self._llm: Optional[_LLMClientAdapter] = None

    @property
    def llm(self) -> _LLMClientAdapter:
        if self._llm is None:
            self._llm = create_llm_client(model=self.model_id)
        return self._llm

    def _build_conversation_messages(self, messages: list, new_message: Optional[dict] = None) -> list[Message]:
        """Convert conversation messages to llm_client Message format for evaluation."""
        result: list[Message] = []
        for m in messages:
            role = m.role if hasattr(m, "role") else m.get("role", "user")
            content = m.content if hasattr(m, "content") else m.get("content", "")

            # For group chat, prefix agent messages with their name
            sender_id = m.sender_id if hasattr(m, "sender_id") else m.get("sender_id")
            sender_name = m.sender_name if hasattr(m, "sender_name") else m.get("sender_name")

            if role == "assistant" and sender_name:
                content = f"[{sender_name}]: {content}"
            elif role == "user":
                content = f"[User]: {content}"

            result.append(Message(role="user" if role == "user" else "assistant", content=content))

        if new_message:
            role = new_message.get("role", "user")
            content = new_message.get("content", "")
            sender_name = new_message.get("sender_name")
            if role == "assistant" and sender_name:
                content = f"[{sender_name}]: {content}"
            elif role == "user":
                content = f"[User]: {content}"
            result.append(Message(role="user" if role == "user" else "assistant", content=content))

        return result

    async def evaluate(self, messages: list, new_message: Optional[dict] = None) -> EvaluationResult:
        """Ask this agent whether it should respond to the conversation."""
        system_prompt = EVALUATE_SYSTEM_PROMPT.format(name=self.display_name)
        conv_messages = self._build_conversation_messages(messages, new_message)

        # Add system prompt as first message
        all_messages = [Message(role="system", content=system_prompt)] + conv_messages

        try:
            response = self.llm.invoke(all_messages)
            text = response.get("text", "").strip()
            return self._parse_evaluation(text)
        except Exception as e:
            print(f"[GroupChatAgent:{self.model_id}] Evaluate error: {e}")
            return EvaluationResult(
                agent_id=self.model_id,
                agent_name=self.display_name,
                should_respond=False,
                reason=f"Evaluation failed: {str(e)}",
                priority=1,
                desired_length="brief",
            )

    def _parse_evaluation(self, text: str) -> EvaluationResult:
        """Parse the LLM's JSON evaluation response."""
        # Try to extract JSON from the response
        # The LLM might wrap it in markdown code blocks
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if not json_match:
            print(f"[GroupChatAgent:{self.model_id}] No JSON found in evaluation: {text[:200]}")
            return EvaluationResult(
                agent_id=self.model_id,
                agent_name=self.display_name,
                should_respond=False,
                reason="Failed to parse evaluation",
                priority=1,
                desired_length="brief",
            )

        try:
            data = json.loads(json_match.group())
            return EvaluationResult(
                agent_id=self.model_id,
                agent_name=self.display_name,
                should_respond=bool(data.get("should_respond", False)),
                reason=str(data.get("reason", "")),
                priority=max(1, min(5, int(data.get("priority", 1)))),
                desired_length=data.get("desired_length", "moderate") if data.get("desired_length") in ("brief", "moderate", "detailed") else "moderate",
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"[GroupChatAgent:{self.model_id}] Parse error: {e}, text: {text[:200]}")
            return EvaluationResult(
                agent_id=self.model_id,
                agent_name=self.display_name,
                should_respond=False,
                reason=f"Parse error: {str(e)}",
                priority=1,
                desired_length="brief",
            )

    async def respond(self, messages: list, evaluation: EvaluationResult) -> AsyncIterator[dict]:
        """Generate a streaming response after coordinator approval."""
        length_instruction = {
            "brief": "concise (1-2 sentences)",
            "moderate": "moderate length (about 1 paragraph)",
            "detailed": "detailed and thorough (multiple paragraphs)",
        }.get(evaluation.desired_length, "moderate length")

        system_prompt = RESPOND_SYSTEM_PROMPT.format(
            name=self.display_name,
            reason=evaluation.reason,
            length_instruction=length_instruction,
        )

        conv_messages = self._build_conversation_messages(messages)
        all_messages = [Message(role="system", content=system_prompt)] + conv_messages

        try:
            async for chunk in self.llm.astream(all_messages):
                yield chunk
        except Exception as e:
            print(f"[GroupChatAgent:{self.model_id}] Respond error: {e}")
            yield {"text": f"[Error generating response: {str(e)}]", "thinking": None}
