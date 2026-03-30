import os
import json
import sqlite3
import numpy as np
from pathlib import Path
from typing import Optional, Literal
import re
from dataclasses import dataclass, field

import faiss
import pdfplumber
from docx import Document

# Ollama embedding configuration - qwen3-embedding-4b outputs 2560-dim vectors
EMBEDDING_DIM = 2560  # Will be verified when first embedding is received


@dataclass
class PDFElement:
    """Represents a element in a PDF document."""
    type: Literal["heading", "paragraph", "list_item", "table"]
    text: str
    level: int = 0  # heading level (1-6), 0 for non-headings
    font_size: float = 0
    page_num: int = 0


@dataclass
class PDFSection:
    """Represents a section with heading and its content."""
    heading: Optional[PDFElement] = None
    content: list[PDFElement] = field(default_factory=list)
    children: list["PDFSection"] = field(default_factory=list)

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
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

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


def _is_heading_line(line_text: str, font_size: float, prev_font_size: float,
                      line_bold: bool, all_font_sizes: list[float]) -> bool:
    """Determine if a line is a heading based on multiple signals."""
    if not line_text.strip():
        return False

    # Skip very short lines (likely noise)
    if len(line_text.strip()) < 2:
        return False

    # Skip lines that are mostly numbers or punctuation
    alpha_ratio = sum(c.isalpha() for c in line_text) / max(len(line_text), 1)
    if alpha_ratio < 0.3:
        return False

    # Heading indicators:
    # 1. Font size is larger than median font size in document
    median_size = sorted(all_font_sizes)[len(all_font_sizes) // 2] if all_font_sizes else 12
    # 2. Font size is notably larger than previous line
    # 3. Text is bold
    # 4. Line is short (headings tend to be shorter than paragraphs)
    is_large_font = font_size > median_size * 1.2 if median_size else font_size > 14
    is_significantly_larger = font_size > prev_font_size * 1.3 if prev_font_size > 0 else False
    is_short_line = len(line_text.strip()) < 100
    is_bold = line_bold

    # Count heading-like keywords at start
    heading_keywords = [
        "第", "章", "节", "条", "款", "项", "目",
        "chapter", "section", "article", "part",
        "概述", "简介", "前言", "摘要", "目录",
        "定义", "原理", "方法", "步骤", "流程",
        "总结", "结论", "参考", "附录",
    ]
    starts_with_keyword = any(line_text.strip().startswith(kw) for kw in heading_keywords)

    score = 0
    if is_large_font:
        score += 2
    if is_significantly_larger:
        score += 2
    if is_bold:
        score += 1
    if is_short_line:
        score += 1
    if starts_with_keyword:
        score += 3

    return score >= 3


def _extract_pdf_with_structure(file_path: Path) -> list[PDFElement]:
    """
    Extract text from PDF while preserving heading structure.
    Uses font size analysis to identify headings at different levels.
    """
    elements: list[PDFElement] = []

    with pdfplumber.open(file_path) as pdf:
        all_font_sizes: list[float] = []

        # First pass: collect all font sizes
        for page in pdf.pages:
            chars = page.chars
            for char in chars:
                if char.get("size", 0) > 0:
                    all_font_sizes.append(char["size"])

        if not all_font_sizes:
            # Fallback to plain text extraction
            return [_extract_pdf_plain_text(file_path)]

        # Second pass: extract elements with structure
        prev_element: Optional[PDFElement] = None

        for page_num, page in enumerate(pdf.pages, start=1):
            # Use chars to reconstruct lines with font info
            chars = page.chars
            if not chars:
                continue

            # Group chars by vertical position (y coordinate) to get lines
            from collections import defaultdict
            y_groups: dict[float, list] = defaultdict(list)

            for char in chars:
                top = round(char.get("top", 0), 1)
                y_groups[top].append(char)

            # Sort by vertical position (top to bottom)
            sorted_ys = sorted(y_groups.keys())

            for y in sorted_ys:
                line_chars = sorted(y_groups[y], key=lambda c: c.get("x0", 0))

                # Detect and inject spaces where needed
                line_text = _reconstruct_line_text(line_chars)

                if not line_text:
                    continue

                # Get font properties for this line
                font_sizes = [c.get("size", 0) for c in line_chars if c.get("size", 0) > 0]
                font_size = max(font_sizes) if font_sizes else 0

                # Check if bold (many PDFs don't expose this clearly)
                # Use font name as proxy
                font_names = [c.get("fontname", "").lower() for c in line_chars]
                is_bold = any("bold" in fn or "heavy" in fn for fn in font_names)

                prev_font_size = prev_element.font_size if prev_element else 0

                # Determine if this is a heading
                if _is_heading_line(line_text, font_size, prev_font_size, is_bold, all_font_sizes):
                    # Determine heading level based on font size relative to median
                    median_size = sorted(all_font_sizes)[len(all_font_sizes) // 2]
                    size_ratio = font_size / median_size if median_size > 0 else 1.0

                    if size_ratio >= 2.0:
                        level = 1
                    elif size_ratio >= 1.5:
                        level = 2
                    elif size_ratio >= 1.2:
                        level = 3
                    else:
                        level = 4

                    elements.append(PDFElement(
                        type="heading",
                        text=line_text,
                        level=level,
                        font_size=font_size,
                        page_num=page_num
                    ))
                else:
                    elements.append(PDFElement(
                        type="paragraph",
                        text=line_text,
                        level=0,
                        font_size=font_size,
                        page_num=page_num
                    ))

                prev_element = elements[-1]

    # Post-process: merge consecutive paragraphs and clean up
    return elements


def _reconstruct_line_text(line_chars: list[dict]) -> str:
    """
    Reconstruct text from a line of PDF characters.
    Detects and injects missing spaces between words.
    """
    if not line_chars:
        return ""

    # Sort by x position
    sorted_chars = sorted(line_chars, key=lambda c: c.get("x0", 0))
    text_parts = []

    for i, char in enumerate(sorted_chars):
        char_text = char.get("text", "")
        if not char_text:
            continue

        text_parts.append(char_text)

        # Check if we need to inject a space after this char
        if i < len(sorted_chars) - 1:
            next_char = sorted_chars[i + 1]
            gap = next_char.get("x0", 0) - char.get("x1", 0)

            # Get font sizes for gap check
            curr_size = char.get("size", 0)
            next_size = next_char.get("size", 0)

            # If gap is significant (> 0.3 * char width) and not already a space
            # AND both chars have similar font sizes (same text block)
            if gap > 0 and curr_size > 0 and next_size > 0:
                # Estimate character width from font size
                avg_size = (curr_size + next_size) / 2
                char_width = avg_size * 0.6  # rough estimate

                # If gap is larger than ~30% of estimated char width, inject space
                if gap > char_width * 0.3:
                    # Only inject if the next char doesn't already have a leading space
                    # and it's not punctuation (which naturally has spacing)
                    next_text = next_char.get("text", "")
                    if next_text not in " \t\n.,;:!?。！？、；：""''（）【】":
                        text_parts.append(" ")

    return "".join(text_parts)


def _extract_pdf_plain_text(file_path: Path) -> PDFElement:
    """Fallback: simple PDF text extraction."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return PDFElement(type="paragraph", text=text.strip(), page_num=1)


def _chunk_text_by_structure(elements: list[PDFElement],
                              chunk_size: int = 400,
                              overlap: int = 50) -> list[str]:
    """
    Chunk elements while respecting heading and sentence boundaries.
    Never cuts a sentence or word in half.
    """
    if not elements:
        return []

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_size = 0
    heading_context: list[str] = []

    def get_heading_prefix() -> str:
        if not heading_context:
            return ""
        return " > ".join(heading_context[-2:]) + ": "

    def flush_chunk() -> None:
        nonlocal current_chunk, current_size
        if current_chunk:
            prefix = get_heading_prefix()
            chunk_text = prefix + " ".join(current_chunk)
            chunks.append(chunk_text.strip())
            current_chunk = []
            current_size = 0

    def add_text_to_chunk(text: str) -> None:
        """Add text to current chunk. Sentences are atomic units — never split mid-sentence."""
        nonlocal current_chunk, current_size

        text = text.strip()
        if not text:
            return

        sentences = _split_into_sentences(text)

        for sent in sentences:
            sent_len = len(sent)

            if current_size + sent_len + 1 > chunk_size and current_chunk:
                # Doesn't fit, flush and start new chunk with this sentence
                flush_chunk()
                current_chunk = [sent]
                current_size = sent_len
            else:
                current_chunk.append(sent)
                current_size += sent_len + 1

    for element in elements:
        text = element.text

        if element.type == "heading":
            if current_chunk:
                flush_chunk()

            level = element.level
            heading_context = heading_context[:level - 1]
            heading_context.append(text)

            add_text_to_chunk(text)
        else:
            add_text_to_chunk(text)

    if current_chunk:
        flush_chunk()

    return chunks


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving sentence-ending punctuation."""
    sentences = re.split(r"(?<=[。！？.!?])\s*", text)
    return [s.strip() for s in sentences if s.strip()]


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """
    Split plain text into chunks respecting sentence boundaries.
    Sentences are atomic — never cut mid-sentence or mid-word.
    """
    if not text:
        return []

    sentences = _split_into_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for sent in sentences:
        sent_len = len(sent)

        if current_size + sent_len + 1 > chunk_size and current:
            chunks.append(" ".join(current))
            current = [sent]
            current_size = sent_len
        else:
            current.append(sent)
            current_size += sent_len + 1

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

        update_file_status(file_id, status="processing")

        # Extract text and chunk based on file type
        if file_type == "pdf":
            # Use structure-aware extraction for PDFs
            elements = _extract_pdf_with_structure(file_path)
            if not elements:
                update_file_status(file_id, status="error", error="No text content extracted")
                return

            # Build full text for preview
            text = " ".join(e.text for e in elements if e.type == "paragraph")
            text_preview = text[:200] + "..." if len(text) > 200 else text

            # Chunk with structure awareness
            chunks = _chunk_text_by_structure(elements)
        else:
            # Use plain text extraction for other file types
            text = _extract_text_from_file(file_path, file_type)
            if not text:
                update_file_status(file_id, status="error", error="No text content extracted")
                return

            text_preview = text[:200] + "..." if len(text) > 200 else text
            chunks = _chunk_text(text)

        if not chunks:
            update_file_status(file_id, status="error", error="Failed to chunk text")
            return

        # Get embeddings
        embeddings, embedding_dim = await _call_embedding_api(chunks)

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
