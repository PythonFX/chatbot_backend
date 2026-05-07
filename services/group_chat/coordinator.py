"""
Group chat coordinator: orchestrates the multi-agent conversation flow.

The coordinator manages the evaluation -> selection -> response loop:
1. All agents evaluate the new message
2. Coordinator selects the highest-priority agent to speak
3. Selected agent generates a response (streaming)
4. Re-evaluation with the new message
5. Repeat until no agent wants to speak or max turns reached
"""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Optional

from services.group_chat.agent import GroupChatAgent, EvaluationResult
from services.llm_factory import is_llm_configured, get_available_models, MODEL_DISPLAY_NAMES


class GroupChatCoordinator:
    def __init__(
        self,
        agents: list[GroupChatAgent],
        max_turns: int = 3,
        evaluation_timeout: float = 30.0,
    ):
        self.agents = agents
        self.max_turns = max_turns
        self.evaluation_timeout = evaluation_timeout

    async def process_user_message(
        self,
        conversation_messages: list,
        user_message: dict,
    ) -> AsyncIterator[dict]:
        """Main loop: evaluate -> select -> respond -> repeat.

        Yields SSE event dicts for the frontend.
        """
        # Add the user message to the conversation for context
        all_messages = list(conversation_messages)
        all_messages.append(user_message)

        for turn in range(self.max_turns):
            # Yield round start
            yield {
                "type": "round_start",
                "round": turn + 1,
            }

            # Evaluate all agents concurrently
            evaluations = await self._evaluate_all(all_messages)

            # Yield evaluation results for UI feedback
            for ev in evaluations:
                yield {
                    "type": "evaluation",
                    **ev.to_dict(),
                }

            # Select speaker: highest priority among those who want to respond
            speaker = self._select_speaker(evaluations)

            if speaker is None:
                yield {"type": "round_end", "reason": "no_speaker"}
                break

            # Announce speaker
            yield {
                "type": "agent_speaking",
                "agent_id": speaker.agent_id,
                "agent_name": speaker.agent_name,
            }

            # Generate response (streaming)
            agent = self._get_agent(speaker.agent_id)
            if agent is None:
                yield {"type": "error", "message": f"Agent {speaker.agent_id} not found"}
                break

            full_text = ""
            full_thinking = ""
            async for chunk in agent.respond(all_messages, speaker):
                text = chunk.get("text", "")
                thinking = chunk.get("thinking")
                if thinking:
                    full_thinking += thinking
                    yield {
                        "type": "thinking",
                        "agent_id": speaker.agent_id,
                        "thinking": thinking,
                    }
                if text:
                    full_text += text
                    yield {
                        "type": "chunk",
                        "agent_id": speaker.agent_id,
                        "text": text,
                    }

            # Agent done
            yield {
                "type": "agent_done",
                "agent_id": speaker.agent_id,
            }

            # Add the agent's response to the conversation for next round
            agent_message = {
                "role": "assistant",
                "content": full_text,
                "sender_id": speaker.agent_id,
                "sender_name": speaker.agent_name,
            }
            if full_thinking:
                agent_message["thinking"] = full_thinking
            all_messages.append(agent_message)

            # Yield the complete message data so the router can save it
            yield {
                "type": "agent_message_complete",
                "agent_id": speaker.agent_id,
                "agent_name": speaker.agent_name,
                "content": full_text,
                "thinking": full_thinking or None,
            }

        else:
            # Max turns reached
            yield {"type": "round_end", "reason": "max_turns"}

        yield {"type": "done"}

    async def _evaluate_all(self, messages: list) -> list[EvaluationResult]:
        """Evaluate all agents concurrently with a timeout."""
        tasks = [agent.evaluate(messages) for agent in self.agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        evaluations = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                evaluations.append(EvaluationResult(
                    agent_id=self.agents[i].model_id,
                    agent_name=self.agents[i].display_name,
                    should_respond=False,
                    reason=f"Evaluation error: {str(result)}",
                    priority=1,
                    desired_length="brief",
                ))
            else:
                evaluations.append(result)

        return evaluations

    def _select_speaker(self, evaluations: list[EvaluationResult]) -> Optional[EvaluationResult]:
        """Select the highest-priority agent that wants to speak."""
        candidates = [e for e in evaluations if e.should_respond]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.priority)

    def _get_agent(self, agent_id: str) -> Optional[GroupChatAgent]:
        """Find an agent by its model ID."""
        for agent in self.agents:
            if agent.model_id == agent_id:
                return agent
        return None


def create_coordinator_for_conversation(agent_ids: list[str]) -> Optional[GroupChatCoordinator]:
    """Create a coordinator with agents for the given model IDs.

    Only includes agents whose models are configured and available.
    Returns None if no agents are available.
    """
    agents = []
    for model_id in agent_ids:
        if is_llm_configured(model=model_id):
            display_name = MODEL_DISPLAY_NAMES.get(model_id, model_id)
            agents.append(GroupChatAgent(model_id=model_id, display_name=display_name))
        else:
            print(f"[GroupChat] Skipping unconfigured model: {model_id}")

    if not agents:
        return None

    return GroupChatCoordinator(agents=agents)
