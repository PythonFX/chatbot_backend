"""
Deep Q&A Service

Implements a multi-step RAG pipeline for deep question answering:
1. Query rewriting based on conversation history
2. Extended context retrieval (±1 context)
3. Essential score evaluation via LLM
4. Context selection respecting LLM context window
"""

import json
import os
import re
from typing import Optional

from llm_client import Message
from services.embedding_service import get_file_chunks
from services.llm_manager import create_llm_client


LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "200000"))
# Rough estimate: 1 token ≈ 4 characters for Chinese/English mixed content
CHARS_PER_TOKEN = 4
MAX_CONTEXT_CHARS = LLM_CONTEXT_WINDOW * CHARS_PER_TOKEN


def _build_conversation_history_text(messages: list) -> str:
    """Build conversation history string for query rewriting prompt."""
    history_parts = []
    for msg in messages:
        role = "User" if msg.role == "user" else "Assistant"
        history_parts.append(f"{role}: {msg.content}")
    return "\n".join(history_parts)


async def rewrite_query(conversation_history: str, current_query: str) -> tuple[bool, str]:
    """
    Step 1: Rewrite the query if needed based on conversation history.

    Args:
        conversation_history: Combined text of previous messages
        current_query: The user's current query

    Returns:
        Tuple of (needs_rewrite, rewritten_query)
        - If needs_rewrite is False, rewritten_query is empty/invalid
    """
    prompt = f"""You are a query rewriting assistant. Given the conversation history and the current query, determine if the query needs to be rewritten.

A query needs rewriting if:
- It contains pronouns (he, she, it, they, this, that, etc.) referring to previous context
- It references something mentioned earlier in the conversation
- The meaning would be unclear without the conversation history
- The query is a follow-up that relies on implicit subject from history

CONVERSATION HISTORY:
{conversation_history if conversation_history else "(No previous messages)"}

CURRENT QUERY:
{current_query}

Respond in JSON format only, without any markdown or explanation:
- If rewriting needed: {{"need_rewrite_query": true, "query": "[rewritten_query]"}}
- If no rewriting needed: {{"need_rewrite_query": false}}
"""

    try:
        llm = create_llm_client()
        response = await llm.async_completion([Message(role="user", content=prompt)])

        content = response.content
        # Extract JSON from response
        json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            need_rewrite = data.get("need_rewrite_query", False)
            query = data.get("query", "") if need_rewrite else ""
            return need_rewrite, query
    except Exception as e:
        print(f"[DeepQA] Query rewrite error: {e}")

    return False, ""


def get_extended_contexts(initial_contexts: list[dict]) -> list[dict]:
    """
    Step 2 (partial): Extend contexts by ±1 chunk index.

    For each context in initial_contexts (which has file_id and chunk_index from search),
    extend to include adjacent chunks (±1).

    Args:
        initial_contexts: List of dicts with file_id and chunk_index from search

    Returns:
        List of dicts with file_id, chunk_index, chunk_text, and original_index
    """
    if not initial_contexts:
        return []

    # Collect all needed (file_id, chunk_index) pairs to fetch
    # Include ±1 for each found chunk
    needed_chunks: set[tuple] = set()
    for ctx in initial_contexts:
        file_id = ctx.get("file_id")
        chunk_index = ctx.get("chunk_index", 0)
        # Add the chunk itself and its neighbors
        needed_chunks.add((file_id, chunk_index))
        needed_chunks.add((file_id, chunk_index - 1))
        needed_chunks.add((file_id, chunk_index + 1))

    # Group by file_id for efficient retrieval
    chunks_by_file: dict = {}
    for file_id, _ in needed_chunks:
        if file_id not in chunks_by_file:
            chunks_by_file[file_id] = get_file_chunks(file_id)

    # Build extended contexts list
    extended: list[dict] = []
    seen_indices: set[tuple] = set()  # (file_id, chunk_index)

    # Process in order: for each initial context's index, include it and neighbors
    for ctx in initial_contexts:
        file_id = ctx.get("file_id")
        chunk_index = ctx.get("chunk_index", 0)

        # Get all chunks for this file
        file_chunks = chunks_by_file.get(file_id, [])

        # Find indices to include: chunk_index-1, chunk_index, chunk_index+1
        indices_to_include = {chunk_index - 1, chunk_index, chunk_index + 1}

        for chunk in file_chunks:
            idx = chunk.get("index")
            if (file_id, idx) in seen_indices:
                continue
            if idx in indices_to_include:
                extended.append({
                    "file_id": file_id,
                    "chunk_index": idx,
                    "chunk_text": chunk.get("text", ""),
                })
                seen_indices.add((file_id, idx))

    # Sort by chunk_index within each file
    extended.sort(key=lambda x: (x["file_id"], x["chunk_index"]))

    return extended


async def get_essential_scores(query: str, extended_contexts: list[dict]) -> list[dict]:
    """
    Step 2 (scoring): Ask LLM to rate essential score (0-5) for each context.

    Args:
        query: The (possibly rewritten) user query
        extended_contexts: List of dicts with chunk_text and chunk_index

    Returns:
        List of dicts with chunk_index, essential_score, and chunk_text
    """
    if not extended_contexts:
        return []

    # Build context sections for the prompt
    context_sections = []
    for i, ctx in enumerate(extended_contexts):
        context_sections.append(
            f"[CONTEXT_INDEX: [{i}]]\n{ctx.get('chunk_text', '')}"
        )

    context_block = "\n\n".join(context_sections)

    prompt = f"""Based on query to determine whether each piece of context is essential to answer user's question or not.
Respond only in below json format, without markdown tags or any explanation, essential score is from 0 to 5, 5 means most essential:
[
  {{
    "index": index_a,
    "essential_score": score
  }},
  {{
    "index": index_b,
    "essential_score": score
  }}
]

From here starts the query and the contexts, for each piece of context, it starts with [CONTEXT_INDEX: [index_a]], index_a is its index number,

[QUERY]:
{query}

[CONTEXT]:
{context_block}
"""

    try:
        llm = create_llm_client()
        response = await llm.async_completion([Message(role="user", content=prompt)])

        content = response.content

        # Extract JSON array from response
        json_match = re.search(r'\[[\s\S]*\]', content)
        if json_match:
            scores_data = json.loads(json_match.group())
            # Map scores back to contexts
            for score_item in scores_data:
                idx = score_item.get("index")
                score = score_item.get("essential_score", 0)
                if 0 <= idx < len(extended_contexts):
                    extended_contexts[idx]["essential_score"] = score

        # Ensure all contexts have a score
        for ctx in extended_contexts:
            if "essential_score" not in ctx:
                ctx["essential_score"] = 0

    except Exception as e:
        print(f"[DeepQA] Essential score error: {e}")
        # Set all scores to 1 on error (include all)
        for ctx in extended_contexts:
            ctx["essential_score"] = 1

    return extended_contexts


def select_final_contexts(
    contexts_with_scores: list[dict],
    context_window_size: Optional[int] = None
) -> list[dict]:
    """
    Step 2 (selection): Select contexts with score > 0, respecting context window.

    Args:
        contexts_with_scores: List of dicts with essential_score and chunk_text
        context_window_size: Optional override for max context size in chars

    Returns:
        List of selected contexts (filtered by score > 0, ordered by original position)
    """
    if not contexts_with_scores:
        return []

    max_chars = (context_window_size or LLM_CONTEXT_WINDOW) * CHARS_PER_TOKEN

    # Filter contexts with score > 0
    scored_contexts = [
        ctx for ctx in contexts_with_scores
        if ctx.get("essential_score", 0) > 0
    ]

    if not scored_contexts:
        return []

    # Calculate total characters
    total_chars = sum(len(ctx.get("chunk_text", "")) for ctx in scored_contexts)

    if total_chars <= max_chars:
        # All contexts with score > 0 fit in context window
        return scored_contexts

    # Need to select subset - keep highest scores while maintaining order
    # Sort by score descending, but track original positions
    indexed_contexts = [
        (i, ctx) for i, ctx in enumerate(scored_contexts)
    ]
    # Sort by score (descending), then by original index (ascending)
    indexed_contexts.sort(key=lambda x: (-x[1].get("essential_score", 0), x[0]))

    # Greedily select highest scoring contexts until we fit
    selected: list[dict] = []
    current_chars = 0

    for original_idx, ctx in indexed_contexts:
        ctx_text = ctx.get("chunk_text", "")
        ctx_chars = len(ctx_text)

        if current_chars + ctx_chars <= max_chars:
            selected.append(ctx)
            current_chars += ctx_chars
        else:
            # Check if adding this context would exceed, but maybe some later one fits
            # For simplicity, we stop when we can't fit the current one
            # (could be improved with more sophisticated selection)
            remaining = max_chars - current_chars
            if remaining >= 100:  # Only add if we have space for at least 100 chars
                selected.append(ctx)
                current_chars = max_chars  # No more room
                break

    # Re-sort by original position to maintain context coherence
    selected.sort(key=lambda x: (
        x.get("file_id", ""),
        x.get("chunk_index", 0)
    ))

    return selected


async def deep_qa_retrieve(
    query: str,
    file_ids: list[str],
    conversation_history: str = "",
    initial_top_k: int = 10
) -> tuple[str, list[dict]]:
    """
    Full Deep Q&A retrieval pipeline.

    Args:
        query: User's current query
        file_ids: List of file IDs to search in
        conversation_history: Combined text of previous messages
        initial_top_k: Number of initial contexts to retrieve

    Returns:
        Tuple of (final_query, selected_contexts)
        - final_query: The (possibly rewritten) query
        - selected_contexts: List of selected context dicts with chunk_text
    """
    from services.embedding_service import search_chunks_in_files

    # Step 1: Query rewrite
    need_rewrite, rewritten_query = await rewrite_query(conversation_history, query)
    final_query = rewritten_query if need_rewrite else query

    print(f"[DeepQA] Query rewrite: need_rewrite={need_rewrite}, query={final_query[:50]}...")

    # Step 2: Initial similarity search
    initial_contexts = await search_chunks_in_files(
        query=final_query,
        file_ids=file_ids,
        top_k=initial_top_k
    )

    if not initial_contexts:
        print("[DeepQA] No initial contexts found")
        return final_query, []

    print(f"[DeepQA] Found {len(initial_contexts)} initial contexts")

    # Step 3: Extend contexts by ±1
    extended_contexts = get_extended_contexts(initial_contexts)
    print(f"[DeepQA] Extended to {len(extended_contexts)} contexts")

    # Step 4: Get essential scores
    scored_contexts = await get_essential_scores(final_query, extended_contexts)

    # Step 5: Select final contexts
    selected = select_final_contexts(scored_contexts)
    print(f"[DeepQA] Selected {len(selected)} contexts")

    return final_query, selected


def build_rag_context(contexts: list[dict]) -> str:
    """Build RAG context string from selected contexts."""
    if not contexts:
        return ""

    context_parts = []
    for i, ctx in enumerate(contexts):
        context_parts.append(f"[Context {i+1}] {ctx.get('chunk_text', '')}")

    return "\n\n".join(context_parts)
