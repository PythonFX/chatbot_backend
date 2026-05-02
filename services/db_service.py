"""
SQLite data layer for conversations and messages.
All write operations go through conversation_service which mirrors to both DB and JSON.
"""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "conversations.db"


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _row_to_conv(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "title": row["title"],
        "file_ids": json.loads(row["file_ids"]) if row["file_ids"] else [],
        "is_novel_agent": bool(row["is_novel_agent"]),
        "selected_novel_id": row["selected_novel_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "messages": [],
    }


def _row_to_msg(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "conversation_id": row["conversation_id"],
        "role": row["role"],
        "content": row["content"],
        "thinking": row["thinking"],
        "signature": row["signature"],
        "type": row["type"],
        "raw_response": json.loads(row["raw_response"]) if row["raw_response"] else None,
        "complete": bool(row["complete"]),
        "rag_contexts": json.loads(row["rag_contexts"]) if row["rag_contexts"] else None,
        "versions": json.loads(row["versions"]) if row["versions"] else None,
        "selected_version_index": row["selected_version_index"],
        "is_multi_mode": bool(row["is_multi_mode"]) if "is_multi_mode" in row.keys() else False,
        "created_at": row["created_at"],
    }


# ── Schema ────────────────────────────────────────────────────────────────────

def init_tables() -> None:
    conn = _get_db()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New Chat',
                file_ids TEXT NOT NULL DEFAULT '[]',
                is_novel_agent INTEGER NOT NULL DEFAULT 0,
                selected_novel_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                thinking TEXT,
                signature TEXT,
                type TEXT,
                raw_response TEXT,
                complete INTEGER NOT NULL DEFAULT 1,
                rag_contexts TEXT,
                versions TEXT,
                selected_version_index INTEGER,
                is_multi_mode INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);
        """)
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    init_tables()


# ── Internal upsert ────────────────────────────────────────────────────────────

def db_upsert_conversation(conv: dict) -> None:
    conn = _get_db()
    try:
        conn.execute("""
            INSERT OR REPLACE INTO conversations
            (id, title, file_ids, is_novel_agent, selected_novel_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            conv["id"],
            conv["title"],
            json.dumps(conv.get("file_ids", [])),
            int(conv.get("is_novel_agent", False)),
            conv.get("selected_novel_id"),
            conv["created_at"],
            conv["updated_at"],
        ))
        conn.commit()
    finally:
        conn.close()


def db_add_message_raw(msg: dict) -> None:
    conn = _get_db()
    try:
        conn.execute("""
            INSERT OR REPLACE INTO messages
            (id, conversation_id, role, content, thinking, signature, type,
             raw_response, complete, rag_contexts, versions, selected_version_index, is_multi_mode, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            msg["id"],
            msg["conversation_id"],
            msg["role"],
            msg["content"],
            msg.get("thinking"),
            msg.get("signature"),
            msg.get("type"),
            json.dumps(msg["raw_response"]) if msg.get("raw_response") else None,
            int(msg.get("complete", True)),
            json.dumps(msg["rag_contexts"]) if msg.get("rag_contexts") else None,
            json.dumps(msg.get("versions")) if msg.get("versions") else None,
            msg.get("selected_version_index"),
            int(msg.get("is_multi_mode", False)),
            msg["created_at"],
        ))
        conn.commit()
    finally:
        conn.close()


# ── Public CRUD ───────────────────────────────────────────────────────────────

def db_get_all_conversations() -> list[dict]:
    conn = _get_db()
    try:
        cur = conn.execute("SELECT * FROM conversations ORDER BY updated_at DESC")
        return [_row_to_conv(r) for r in cur.fetchall()]
    finally:
        conn.close()


def db_get_conversation(conv_id: str) -> Optional[dict]:
    conn = _get_db()
    try:
        cur = conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,))
        row = cur.fetchone()
        if row is None:
            return None
        conv = _row_to_conv(row)
        cur2 = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at",
            (conv_id,)
        )
        conv["messages"] = [_row_to_msg(r) for r in cur2.fetchall()]
        return conv
    finally:
        conn.close()


def db_create_conversation(conv_id: str, created_at: str, updated_at: str) -> dict:
    conn = _get_db()
    try:
        conn.execute("""
            INSERT INTO conversations (id, title, file_ids, is_novel_agent, selected_novel_id, created_at, updated_at)
            VALUES (?, 'New Chat', '[]', 0, NULL, ?, ?)
        """, (conv_id, created_at, updated_at))
        conn.commit()
    finally:
        conn.close()
    return db_get_conversation(conv_id)


def db_update_title(conv_id: str, title: str, updated_at: str) -> None:
    conn = _get_db()
    try:
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, updated_at, conv_id)
        )
        conn.commit()
    finally:
        conn.close()


def db_set_novel_agent(conv_id: str, is_novel: bool, updated_at: str) -> None:
    conn = _get_db()
    try:
        conn.execute(
            "UPDATE conversations SET is_novel_agent = ?, updated_at = ? WHERE id = ?",
            (int(is_novel), updated_at, conv_id)
        )
        conn.commit()
    finally:
        conn.close()


def db_set_selected_novel(conv_id: str, novel_id: str, updated_at: str) -> None:
    conn = _get_db()
    try:
        conn.execute(
            "UPDATE conversations SET selected_novel_id = ?, updated_at = ? WHERE id = ?",
            (novel_id, updated_at, conv_id)
        )
        conn.commit()
    finally:
        conn.close()


def db_update_file_ids(conv_id: str, file_ids: list, updated_at: str) -> None:
    conn = _get_db()
    try:
        conn.execute(
            "UPDATE conversations SET file_ids = ?, updated_at = ? WHERE id = ?",
            (json.dumps(file_ids), updated_at, conv_id)
        )
        conn.commit()
    finally:
        conn.close()


def db_add_message(
    msg_id: str,
    conv_id: str,
    role: str,
    content: str,
    thinking: Optional[str],
    signature: Optional[str],
    msg_type: Optional[str],
    raw_response: Optional[dict],
    complete: bool,
    rag_contexts: Optional[list],
    created_at: str,
    versions: Optional[list] = None,
    selected_version_index: Optional[int] = None,
    is_multi_mode: bool = False,
) -> dict:
    conn = _get_db()
    try:
        conn.execute("""
            INSERT INTO messages
            (id, conversation_id, role, content, thinking, signature, type,
             raw_response, complete, rag_contexts, versions, selected_version_index, is_multi_mode, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            msg_id, conv_id, role, content, thinking, signature, msg_type,
            json.dumps(raw_response) if raw_response else None,
            int(complete),
            json.dumps(rag_contexts) if rag_contexts else None,
            json.dumps(versions) if versions else None,
            selected_version_index,
            int(is_multi_mode),
            created_at,
        ))
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (created_at, conv_id)
        )
        conn.commit()
    finally:
        conn.close()
    return {
        "id": msg_id, "conversation_id": conv_id, "role": role,
        "content": content, "thinking": thinking, "signature": signature,
        "type": msg_type, "raw_response": raw_response,
        "complete": complete, "rag_contexts": rag_contexts,
        "versions": versions, "selected_version_index": selected_version_index,
        "is_multi_mode": is_multi_mode,
        "created_at": created_at,
    }


def db_delete_conversation(conv_id: str) -> bool:
    conn = _get_db()
    try:
        cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def db_remove_message(msg_id: str) -> bool:
    conn = _get_db()
    try:
        cur = conn.execute("DELETE FROM messages WHERE id = ?", (msg_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def db_update_message(msg_id: str, **fields) -> Optional[dict]:
    allowed = {"content", "thinking", "complete", "rag_contexts", "versions", "selected_version_index", "is_multi_mode"}
    sets, args = [], []
    for k, v in fields.items():
        if k in allowed:
            if k in ("complete", "is_multi_mode"):
                v = int(v)
            elif k in ("rag_contexts", "versions"):
                v = json.dumps(v) if v else None
            sets.append(f"{k} = ?")
            args.append(v)
    if not sets:
        return None
    args.append(msg_id)
    conn = _get_db()
    try:
        conn.execute(f"UPDATE messages SET {', '.join(sets)} WHERE id = ?", args)
        conn.commit()
        cur = conn.execute("SELECT * FROM messages WHERE id = ?", (msg_id,))
        row = cur.fetchone()
        return _row_to_msg(row) if row else None
    finally:
        conn.close()


def db_append_chunk(msg_id: str, text: str = "", thinking: str = "") -> None:
    conn = _get_db()
    try:
        cur = conn.execute("SELECT content, thinking FROM messages WHERE id = ?", (msg_id,))
        row = cur.fetchone()
        if not row:
            return
        conn.execute(
            "UPDATE messages SET content = ?, thinking = ? WHERE id = ?",
            ((row["content"] or "") + text, (row["thinking"] or "") + thinking, msg_id)
        )
        conn.commit()
    finally:
        conn.close()


def db_sync_conversation(conv: dict) -> None:
    """Full upsert: replace all messages for a conversation."""
    db_upsert_conversation(conv)
    conn = _get_db()
    try:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv["id"],))
        for msg in conv.get("messages", []):
            db_add_message_raw(msg)
    finally:
        conn.close()


def db_search_messages(query: str, context_length: int = 50) -> list[dict]:
    conn = _get_db()
    try:
        cur = conn.execute(
            """SELECT m.*, c.title as conversation_title
               FROM messages m
               JOIN conversations c ON m.conversation_id = c.id
               WHERE m.content LIKE ? AND m.role != 'system'
               ORDER BY m.created_at DESC LIMIT 100""",
            (f"%{query}%",)
        )
        results = []
        for row in cur.fetchall():
            content = row["content"] or ""
            ql = query.lower()
            pos = content.lower().find(ql)
            start = max(0, pos - context_length)
            end = min(len(content), pos + len(query) + context_length)
            results.append({
                "conversation_id": row["conversation_id"],
                "conversation_title": row["conversation_title"],
                "message_id": row["id"],
                "role": row["role"],
                "context_before": content[start:pos],
                "matched_text": content[pos:pos + len(query)],
                "context_after": content[pos + len(query):end],
                "full_context": f"{('...' if start > 0 else '')}{content[start:pos + len(query)]}{('...' if end < len(content) else '')}",
            })
        return results
    finally:
        conn.close()
