import os
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from services.file_service import (
    get_all_files,
    get_file_metadata,
    save_file_metadata,
    delete_file_record,
    DATA_DIR as FILES_DATA_DIR,
)
from services.embedding_service import (
    process_file_embedding,
    delete_embeddings,
    search_similar_files,
    load_file_embeddings,
    load_all_embeddings,
    unload_file_embeddings,
    get_file_chunks,
)

router = APIRouter(prefix="/files", tags=["files"])


class FileMetadataResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    size: int
    uploaded_at: str
    status: str  # "processing", "ready", "error"
    progress: Optional[int] = None  # 0-100, None when not processing
    error: Optional[str] = None
    chunk_count: Optional[int] = None
    text_preview: Optional[str] = None


class FileListResponse(BaseModel):
    files: list[FileMetadataResponse]


class UploadResponse(BaseModel):
    id: str
    filename: str
    status: str
    message: str


class SearchFilesRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchFilesResponse(BaseModel):
    query: str
    results: list[dict]


@router.get("", response_model=FileListResponse)
async def list_files():
    """List all uploaded files with their metadata."""
    files = get_all_files()
    return FileListResponse(
        files=[
            FileMetadataResponse(
                id=f["id"],
                filename=f["filename"],
                file_type=f["file_type"],
                size=f["size"],
                uploaded_at=f["uploaded_at"],
                status=f.get("status", "unknown"),
                progress=f.get("progress"),
                error=f.get("error"),
                chunk_count=f.get("chunk_count"),
                text_preview=f.get("text_preview"),
            )
            for f in files
        ]
    )


@router.get("/{file_id}", response_model=FileMetadataResponse)
async def get_file(file_id: str):
    """Get metadata for a specific file."""
    metadata = get_file_metadata(file_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")
    return FileMetadataResponse(
        id=metadata["id"],
        filename=metadata["filename"],
        file_type=metadata["file_type"],
        size=metadata["size"],
        uploaded_at=metadata["uploaded_at"],
        status=metadata.get("status", "unknown"),
        progress=metadata.get("progress"),
        error=metadata.get("error"),
        chunk_count=metadata.get("chunk_count"),
        text_preview=metadata.get("text_preview"),
    )


class FileChunksResponse(BaseModel):
    file_id: str
    chunks: list[dict]


@router.get("/{file_id}/chunks", response_model=FileChunksResponse)
async def get_chunks(file_id: str):
    """Get all chunks for a specific file."""
    metadata = get_file_metadata(file_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")

    if metadata.get("status") != "ready":
        raise HTTPException(status_code=400, detail="File not ready yet")

    chunks = get_file_chunks(file_id)
    return FileChunksResponse(file_id=file_id, chunks=chunks)


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file for processing.
    Supported types: pdf, doc, docx, txt, json
    """
    # Validate file type
    allowed_types = {"pdf", "doc", "docx", "txt", "json"}
    file_ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""

    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Supported: {', '.join(allowed_types)}"
        )

    # Generate file ID
    import uuid
    file_id = str(uuid.uuid4())

    # Create upload directory
    upload_dir = FILES_DATA_DIR / file_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save the uploaded file
    file_path = upload_dir / file.filename
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Create metadata
    metadata = {
        "id": file_id,
        "filename": file.filename,
        "file_type": file_ext,
        "size": len(content),
        "uploaded_at": datetime.utcnow().isoformat(),
        "status": "processing",
        "progress": 0,
        "original_path": str(file_path),
        "error": None,
        "chunk_count": None,
        "text_preview": None,
    }

    # Save metadata
    save_file_metadata(metadata)

    # Start async embedding processing
    asyncio.create_task(process_file_embedding(file_id, file_path, file_ext))

    return UploadResponse(
        id=file_id,
        filename=file.filename,
        status="processing",
        message="File uploaded successfully. Processing started."
    )


@router.delete("/{file_id}")
async def delete_file(file_id: str):
    """Delete a file and its embeddings."""
    # Delete from disk
    file_dir = FILES_DATA_DIR / file_id
    if file_dir.exists():
        import shutil
        shutil.rmtree(file_dir)

    # Delete embeddings
    delete_embeddings(file_id)

    # Delete metadata
    if not delete_file_record(file_id):
        raise HTTPException(status_code=404, detail="File not found")

    return {"status": "ok", "file_id": file_id}


@router.post("/search", response_model=SearchFilesResponse)
async def search_files(request: SearchFilesRequest):
    """
    Search uploaded files by semantic similarity.
    Returns files with the most similar content to the query.
    """
    results = await search_similar_files(request.query, request.top_k)
    return SearchFilesResponse(
        query=request.query,
        results=results
    )


class LoadAllResponse(BaseModel):
    loaded_files: int
    total_chunks: int
    status: str


class LoadFileResponse(BaseModel):
    file_id: str
    status: str
    message: str


@router.post("/load-all", response_model=LoadAllResponse)
async def load_all():
    """
    Load all uploaded files' embeddings into memory.
    Call this when starting a session so the assistant can access all files.
    """
    result = load_all_embeddings()
    return LoadAllResponse(
        loaded_files=result["loaded_files"],
        total_chunks=result["total_chunks"],
        status="loaded"
    )


@router.post("/load/{file_id}", response_model=LoadFileResponse)
async def load_file(file_id: str):
    """Load a specific file's embeddings into memory."""
    metadata = get_file_metadata(file_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")

    if metadata.get("status") != "ready":
        raise HTTPException(status_code=400, detail="File not ready, still processing")

    success = load_file_embeddings(file_id)
    if success:
        return LoadFileResponse(
            file_id=file_id,
            status="loaded",
            message="Embeddings loaded into memory"
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to load embeddings")


@router.post("/unload/{file_id}")
async def unload_file(file_id: str):
    """Unload a file's embeddings from memory."""
    metadata = get_file_metadata(file_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")

    unload_file_embeddings(file_id)
    return {"file_id": file_id, "status": "unloaded"}


@router.post("/reembed/{file_id}")
async def reembed_file(file_id: str):
    """
    Re-process a file's embeddings with current chunking settings.
    Use this when chunk_size has changed or to fix improperly chunked files.
    """
    metadata = get_file_metadata(file_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")

    if metadata.get("status") == "processing":
        raise HTTPException(status_code=400, detail="File is still being processed")

    # Delete existing embeddings
    delete_embeddings(file_id)

    # Get file path and type
    original_path = metadata.get("original_path")
    file_type = metadata.get("file_type")
    if not original_path:
        raise HTTPException(status_code=500, detail="File path not found in metadata")

    from pathlib import Path
    file_path = Path(original_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")

    # Update status
    from services.file_service import update_file_status
    update_file_status(file_id, status="processing", progress=0)

    # Start re-embedding
    asyncio.create_task(process_file_embedding(file_id, file_path, file_type))

    return {"file_id": file_id, "status": "processing", "message": "Re-embedding started"}
