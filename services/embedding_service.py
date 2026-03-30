import os
import json
import sqlite3
import numpy as np
from pathlib import Path
from typing import Optional
import re

import faiss
import fitz  # PyMuPDF for PDF text extraction
from docx import Document

# Ollama embedding configuration - qwen3-embedding-4b outputs 2560-dim vectors
EMBEDDING_DIM = 2560  # Will be verified when first embedding is received

# Sentence delimiters for chunking
SENTENCE_DELIMITERS = r"(?<=[。！？.!?])\s*"
# Maximum characters in a single chunk (independent chunks are used for sentences exceeding this)
MAX_CHUNK_SIZE = 400

# Testing flag - limits PDF processing to first N pages when True
TESTING = True
TESTING_MAX_PAGES = 5

# Directory paths
DATA_DIR = Path(__file__).parent.parent / "data"
FILES_DIR = DATA_DIR / "files"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
DB_PATH = DATA_DIR / "embeddings.db"

# Ensure directories exist
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory FAISS index
_vector_index: Optional[faiss.IndexFlatIP] = None
_index_loaded: bool = False


def _get_db() -> sqlite3.Connection:
    """Get SQLite connection, creating tables if needed."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS file_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            vector_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(file_id, chunk_index)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS loaded_files (
            file_id TEXT PRIMARY KEY,
            chunk_count INTEGER NOT NULL,
            loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def _get_index() -> faiss.IndexFlatIP:
    """Get or create the FAISS index."""
    global _vector_index, _index_loaded
    if _vector_index is None:
        _vector_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    return _vector_index


def _get_file_embeddings_dir(file_id: str) -> Path:
    """Get directory for a file's embeddings."""
    d = EMBEDDINGS_DIR / file_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _extract_text_from_file(file_path: Path, file_type: str) -> str:
    """Extract text content from a file based on its type."""
    text = ""

    if file_type == "pdf":
        with fitz.open(file_path) as doc:
            max_pages = TESTING_MAX_PAGES if TESTING else len(doc)
            for page in doc[:max_pages]:
                text += page.get_text() + "\n"

    elif file_type in ("doc", "docx"):
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"

    elif file_type == "txt":
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            text = f.read()

    elif file_type == "json":
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
            text = json.dumps(data, indent=2, ensure_ascii=False)

    return text.strip()


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving sentence-ending punctuation."""
    sentences = re.split(SENTENCE_DELIMITERS, text)
    return [s.strip() for s in sentences if s.strip()]


def _chunk_text(text: str) -> list[str]:
    """
    Split text into chunks using two-pass sentence-based chunking.

    Pass 1: Identify sentences that exceed MAX_CHUNK_SIZE (independent chunks).
    Pass 2: Build chunks, stopping when hitting an independent chunk.
            The part before the independent chunk ends as a complete chunk.
    """
    if not text:
        return []

    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    # Pass 1: mark independent sentences (those exceeding MAX_CHUNK_SIZE)
    independent_indices: set[int] = set()
    for i, sent in enumerate(sentences):
        if len(sent) > MAX_CHUNK_SIZE:
            independent_indices.add(i)

    # Pass 2: build chunks
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for i, sent in enumerate(sentences):
        if i in independent_indices:
            # Flush current chunk if any
            if current:
                chunks.append(" ".join(current))
                current = []
                current_size = 0
            # This sentence becomes its own independent chunk
            chunks.append(sent)
        else:
            sent_len = len(sent)
            # Check if adding this sentence would exceed MAX_CHUNK_SIZE
            if current_size + sent_len + (1 if current else 0) > MAX_CHUNK_SIZE and current:
                # Flush and start new chunk
                chunks.append(" ".join(current))
                current = [sent]
                current_size = sent_len
            else:
                current.append(sent)
                current_size += sent_len + (1 if len(current) > 1 else 0)

    # Don't forget the last chunk
    if current:
        chunks.append(" ".join(current))

    return chunks


async def _call_embedding_api(texts: list[str]) -> tuple[list[list[float]], int]:
    """Call Ollama embedding API. Returns (embeddings, dimension)."""
    import httpx

    ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("EMBEDDING_MODEL", "qwen3-embedding-4b")

    url = f"{ollama_base}/api/embeddings"

    async with httpx.AsyncClient() as client:
        embeddings = []
        dim = None
        for text in texts:
            response = await client.post(
                url,
                json={"model": model, "prompt": text},
                timeout=60.0,
            )
            if response.status_code != 200:
                raise Exception(f"Ollama embedding error: {response.status_code} - {response.text}")
            data = response.json()
            embedding = data.get("embedding", [])
            embeddings.append(embedding)
            if dim is None:
                dim = len(embedding)

    return embeddings, dim


async def process_file_embedding(file_id: str, file_path: Path, file_type: str) -> None:
    """Process a file: extract text, create embeddings, save to disk and SQLite."""
    try:
        from services.file_service import update_file_status

        update_file_status(file_id, status="processing", progress=0)

        # Extract text from file (all types use same extraction, then chunk)
        text = _extract_text_from_file(file_path, file_type)
        if not text:
            update_file_status(file_id, status="error", error="No text content extracted")
            return

        text_preview = text[:200] + "..." if len(text) > 200 else text
        chunks = _chunk_text(text)

        if not chunks:
            update_file_status(file_id, status="error", error="Failed to chunk text")
            return

        # Chunking complete — 10%
        update_file_status(file_id, status="processing", progress=10)

        # Embed chunks one by one to track progress
        # 10% to 95% is distributed across chunks (85% total)
        import httpx
        ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("EMBEDDING_MODEL", "qwen3-embedding-4b")
        url = f"{ollama_base}/api/embeddings"

        embeddings = []
        embedding_dim = None
        chunk_count = len(chunks)
        progress_per_chunk = 85.0 / chunk_count if chunk_count > 0 else 0

        async with httpx.AsyncClient() as client:
            for i, chunk_text in enumerate(chunks):
                response = await client.post(
                    url,
                    json={"model": model, "prompt": chunk_text},
                    timeout=60.0,
                )
                if response.status_code != 200:
                    raise Exception(f"Ollama embedding error: {response.status_code} - {response.text}")
                data = response.json()
                embedding = data.get("embedding", [])
                embeddings.append(embedding)
                if embedding_dim is None:
                    embedding_dim = len(embedding)

                # Update progress: 10% + (i+1)/chunk_count * 85%
                prog = int(10 + (i + 1) / chunk_count * 85)
                update_file_status(file_id, status="processing", progress=prog)

        # All embeddings done — 95%, now save to disk/DB
        update_file_status(file_id, status="processing", progress=95)

        # Save embeddings to disk and SQLite
        emb_dir = _get_file_embeddings_dir(file_id)
        conn = _get_db()
        cur = conn.cursor()

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Save vector as .npy file
            vector_path = emb_dir / f"chunk_{i}.npy"
            np.save(vector_path, np.array(embedding, dtype=np.float32))

            # Save chunk text
            chunk_path = emb_dir / f"chunk_{i}.json"
            with open(chunk_path, "w", encoding="utf-8") as f:
                json.dump({"text": chunk, "index": i}, f, ensure_ascii=False)

            # Record in SQLite
            cur.execute("""
                INSERT OR REPLACE INTO file_embeddings (file_id, chunk_index, chunk_text, vector_path)
                VALUES (?, ?, ?, ?)
            """, (file_id, i, chunk, str(vector_path)))

        conn.commit()
        conn.close()

        # Create FAISS index file for this file
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        index_path = emb_dir / "index.faiss"
        index = faiss.IndexFlatIP(embedding_dim)
        faiss.write_index(index, str(index_path))
        # Add vectors to the index
        temp_index = faiss.read_index(str(index_path))
        temp_index.add(vectors)
        faiss.write_index(temp_index, str(index_path))

        update_file_status(
            file_id,
            status="ready",
            progress=100,
            chunk_count=len(chunks),
            text_preview=text_preview,
        )

    except Exception as e:
        from services.file_service import update_file_status
        update_file_status(file_id, status="error", error=str(e))


def get_file_chunks(file_id: str) -> list[dict]:
    """Get all chunks for a file. Returns list of {index, text}."""
    conn = _get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT chunk_index, chunk_text FROM file_embeddings WHERE file_id = ? ORDER BY chunk_index",
        (file_id,)
    )
    rows = cur.fetchall()
    conn.close()
    return [{"index": row["chunk_index"], "text": row["chunk_text"]} for row in rows]


def load_file_embeddings(file_id: str) -> bool:
    """
    Load a file's embeddings from disk into the in-memory FAISS index.
    Returns True if successful, False if file not found.
    """
    global _index_loaded, _vector_index

    emb_dir = EMBEDDINGS_DIR / file_id
    if not emb_dir.exists():
        return False

    index_path = emb_dir / "index.faiss"
    if not index_path.exists():
        return False

    try:
        # Check if already loaded
        conn = _get_db()
        cur = conn.cursor()
        cur.execute("SELECT chunk_count FROM loaded_files WHERE file_id = ?", (file_id,))
        row = cur.fetchone()
        conn.close()

        if row:
            # Already loaded, just return True
            return True

        # Read vectors from .npy files
        conn = _get_db()
        cur = conn.cursor()
        cur.execute("SELECT vector_path FROM file_embeddings WHERE file_id = ? ORDER BY chunk_index", (file_id,))
        rows = cur.fetchall()
        conn.close()

        if not rows:
            return False

        vectors = []
        for row in rows:
            vec = np.load(row["vector_path"])
            vectors.append(vec)

        vectors_array = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors_array)

        # Get or create the global index
        if _vector_index is None:
            dim = vectors_array.shape[1]
            _vector_index = faiss.IndexFlatIP(dim)

        _vector_index.add(vectors_array)

        # Record as loaded
        conn = _get_db()
        cur = conn.cursor()
        cur.execute("INSERT OR REPLACE INTO loaded_files (file_id, chunk_count) VALUES (?, ?)",
                    (file_id, len(vectors)))
        conn.commit()
        conn.close()

        _index_loaded = True
        return True

    except Exception as e:
        print(f"Error loading file embeddings: {e}")
        return False


def unload_file_embeddings(file_id: str) -> None:
    """
    Unload a file's embeddings from the in-memory index.
    Note: FAISS doesn't support removal, so we just mark it as unloaded.
    """
    conn = _get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM loaded_files WHERE file_id = ?", (file_id,))
    conn.commit()
    conn.close()

    # Reset index if no files loaded
    global _index_loaded, _vector_index
    conn = _get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM loaded_files")
    count = cur.fetchone()[0]
    conn.close()

    if count == 0:
        _vector_index = None
        _index_loaded = False


def delete_embeddings(file_id: str) -> None:
    """Delete embeddings for a file from disk and SQLite."""
    import shutil

    # Remove from disk
    emb_dir = EMBEDDINGS_DIR / file_id
    if emb_dir.exists():
        shutil.rmtree(emb_dir)

    # Remove from SQLite
    conn = _get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM file_embeddings WHERE file_id = ?", (file_id,))
    cur.execute("DELETE FROM loaded_files WHERE file_id = ?", (file_id,))
    conn.commit()
    conn.close()

    # Unload if was loaded
    unload_file_embeddings(file_id)


async def search_similar_files(query: str, top_k: int = 5) -> list[dict]:
    """Search for files similar to the query text."""
    global _vector_index, _index_loaded

    # If no index loaded, rebuild from all files
    if not _index_loaded or _vector_index is None or _vector_index.ntotal == 0:
        await _rebuild_index_from_disk()

    # Get query embedding
    query_embedding, _ = await _call_embedding_api([query])
    query_vector = np.array(query_embedding, dtype=np.float32)
    faiss.normalize_L2(query_vector)

    # Search
    index = _get_index()
    if index.ntotal == 0:
        return []

    k = min(top_k * 3, index.ntotal)
    distances, indices = index.search(query_vector.reshape(1, -1), k)

    # Get file info from SQLite
    conn = _get_db()
    cur = conn.cursor()

    file_scores: dict = {}
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue

        cur.execute("""
            SELECT fe.file_id, fe.chunk_text, fe.chunk_index
            FROM file_embeddings fe
            ORDER BY fe.file_id, fe.chunk_index
            LIMIT 1 OFFSET ?
        """, (idx,))
        row = cur.fetchone()

        if not row:
            continue

        file_id = row["file_id"]
        if file_id not in file_scores:
            file_scores[file_id] = {"score": dist, "matched_chunks": []}
        else:
            file_scores[file_id]["score"] = max(file_scores[file_id]["score"], dist)

        file_scores[file_id]["matched_chunks"].append({
            "text": row["chunk_text"],
            "score": float(dist),
        })

    # Get file metadata
    results = []
    from services.file_service import get_file_metadata

    for file_id, data in sorted(file_scores.items(), key=lambda x: x[1]["score"], reverse=True)[:top_k]:
        file_meta = get_file_metadata(file_id)
        if not file_meta:
            continue

        results.append({
            "file_id": file_id,
            "filename": file_meta["filename"],
            "file_type": file_meta["file_type"],
            "size": file_meta["size"],
            "uploaded_at": file_meta["uploaded_at"],
            "score": float(data["score"]),
            "matched_chunks": data["matched_chunks"][:2],
        })

    conn.close()
    return results


async def search_chunks_in_files(query: str, file_ids: list[str], top_k: int = 5) -> list[dict]:
    """
    Search for similar chunks within specific files.
    Used for RAG - retrieves relevant context from linked files.
    Returns list of dicts with chunk_text and score.
    """
    global _vector_index, _index_loaded

    if not file_ids:
        return []

    # If no index loaded, rebuild from all files
    if not _index_loaded or _vector_index is None or _vector_index.ntotal == 0:
        await _rebuild_index_from_disk()

    # Get query embedding
    query_embedding, _ = await _call_embedding_api([query])
    query_vector = np.array(query_embedding, dtype=np.float32)
    faiss.normalize_L2(query_vector)

    # Search
    index = _get_index()
    if index.ntotal == 0:
        return []

    k = min(top_k * 3, index.ntotal)
    distances, indices = index.search(query_vector.reshape(1, -1), k)

    # Get chunks from SQLite, filtered by file_ids
    conn = _get_db()
    cur = conn.cursor()

    # Get all chunks for the specified files
    placeholders = ','.join('?' * len(file_ids))
    cur.execute(f"""
        SELECT fe.file_id, fe.chunk_text, fe.chunk_index
        FROM file_embeddings fe
        WHERE fe.file_id IN ({placeholders})
        ORDER BY fe.file_id, fe.chunk_index
    """, file_ids)

    file_chunks = {}
    for row in cur.fetchall():
        if row["file_id"] not in file_chunks:
            file_chunks[row["file_id"]] = {}
        file_chunks[row["file_id"]][row["chunk_index"]] = row["chunk_text"]

    # Get global index ordering
    cur.execute("""
        SELECT fe.file_id, fe.chunk_text
        FROM file_embeddings fe
        ORDER BY ROWID
    """)
    all_chunks = cur.fetchall()

    # Map global index to chunks
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(all_chunks):
            continue

        row = all_chunks[idx]
        file_id = row["file_id"]

        # Only include if in requested file_ids
        if file_id not in file_ids:
            continue

        results.append({
            "file_id": file_id,
            "chunk_text": row["chunk_text"],
            "score": float(dist),
        })

    # Sort by score and dedupe
    seen = set()
    deduped = []
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        if r["chunk_text"] not in seen:
            seen.add(r["chunk_text"])
            deduped.append(r)

    conn.close()
    return deduped[:top_k]


async def _rebuild_index_from_disk() -> None:
    """Rebuild the in-memory FAISS index from all embeddings on disk."""
    global _vector_index, _index_loaded

    _vector_index = faiss.IndexFlatIP(EMBEDDING_DIM)

    conn = _get_db()
    cur = conn.cursor()
    cur.execute("SELECT file_id FROM loaded_files")
    loaded_files = [row["file_id"] for row in cur.fetchall()]
    conn.close()

    for file_id in loaded_files:
        await _load_single_file_index(file_id)

    _index_loaded = True


def load_all_embeddings() -> dict:
    """
    Load all ready files' embeddings into the in-memory FAISS index.
    Returns dict with counts of loaded files and chunks.
    """
    global _index_loaded, _vector_index

    _vector_index = faiss.IndexFlatIP(EMBEDDING_DIM)

    conn = _get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT fe.file_id, COUNT(*) as chunk_count
        FROM file_embeddings fe
        INNER JOIN (
            SELECT file_id FROM file_embeddings
            GROUP BY file_id
        ) f ON fe.file_id = f.file_id
        GROUP BY fe.file_id
    """)
    file_chunks = {row["file_id"]: row["chunk_count"] for row in cur.fetchall()}
    conn.close()

    total_chunks = 0
    loaded_files = 0

    for file_id, chunk_count in file_chunks.items():
        if _load_single_file_index_internal(file_id):
            total_chunks += chunk_count
            loaded_files += 1

    _index_loaded = True

    return {
        "loaded_files": loaded_files,
        "total_chunks": total_chunks,
    }


def _load_single_file_index_internal(file_id: str) -> bool:
    """Internal function to load a single file's embeddings (no async)."""
    global _vector_index

    emb_dir = EMBEDDINGS_DIR / file_id
    if not emb_dir.exists():
        return False

    try:
        conn = _get_db()
        cur = conn.cursor()
        cur.execute("SELECT vector_path FROM file_embeddings WHERE file_id = ? ORDER BY chunk_index", (file_id,))
        rows = cur.fetchall()
        conn.close()

        if not rows:
            return False

        vectors = []
        for row in rows:
            vec = np.load(row["vector_path"])
            vectors.append(vec)

        if vectors:
            vectors_array = np.array(vectors, dtype=np.float32)
            faiss.normalize_L2(vectors_array)
            if _vector_index is None:
                _vector_index = faiss.IndexFlatIP(EMBEDDING_DIM)
            _vector_index.add(vectors_array)

        # Mark as loaded in DB
        conn = _get_db()
        cur = conn.cursor()
        cur.execute("INSERT OR REPLACE INTO loaded_files (file_id, chunk_count) VALUES (?, ?)",
                    (file_id, len(vectors)))
        conn.commit()
        conn.close()

        return True

    except Exception as e:
        print(f"Error loading file index {file_id}: {e}")
        return False


async def _load_single_file_index(file_id: str) -> None:
    """Load a single file's embeddings into the global index."""
    emb_dir = EMBEDDINGS_DIR / file_id
    index_path = emb_dir / "index.faiss"

    if not index_path.exists():
        return

    try:
        file_index = faiss.read_index(str(index_path))
        index = _get_index()

        # Get vectors from the file index
        # We need to reconstruct vectors from stored .npy files
        conn = _get_db()
        cur = conn.cursor()
        cur.execute("SELECT vector_path FROM file_embeddings WHERE file_id = ? ORDER BY chunk_index", (file_id,))
        rows = cur.fetchall()
        conn.close()

        vectors = []
        for row in rows:
            vec = np.load(row["vector_path"])
            vectors.append(vec)

        if vectors:
            vectors_array = np.array(vectors, dtype=np.float32)
            faiss.normalize_L2(vectors_array)
            index.add(vectors_array)

    except Exception as e:
        print(f"Error loading file index {file_id}: {e}")
