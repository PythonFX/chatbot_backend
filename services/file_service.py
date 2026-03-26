import os
import json
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data" / "files"
DATA_DIR.mkdir(parents=True, exist_ok=True)

METADATA_FILE = DATA_DIR / "metadata.json"


def _load_metadata() -> dict:
    """Load all file metadata from the JSON file."""
    if not METADATA_FILE.exists():
        return {}
    with open(METADATA_FILE, encoding="utf-8") as f:
        return json.load(f)


def _save_metadata(metadata: dict) -> None:
    """Save all file metadata to the JSON file."""
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def get_all_files() -> list:
    """Get all uploaded files, ordered by uploaded_at descending."""
    all_metadata = _load_metadata()
    files = list(all_metadata.values())
    files.sort(key=lambda f: f.get("uploaded_at", ""), reverse=True)
    return files


def get_file_metadata(file_id: str) -> Optional[dict]:
    """Get metadata for a specific file."""
    all_metadata = _load_metadata()
    return all_metadata.get(file_id)


def save_file_metadata(metadata: dict) -> None:
    """Save or update metadata for a file."""
    all_metadata = _load_metadata()
    file_id = metadata["id"]
    all_metadata[file_id] = metadata
    _save_metadata(all_metadata)


def update_file_status(file_id: str, status: str, **kwargs) -> Optional[dict]:
    """Update the status and other fields of a file."""
    all_metadata = _load_metadata()
    if file_id not in all_metadata:
        return None

    all_metadata[file_id]["status"] = status
    for key, value in kwargs.items():
        all_metadata[file_id][key] = value

    _save_metadata(all_metadata)
    return all_metadata[file_id]


def delete_file_record(file_id: str) -> bool:
    """Delete a file's metadata record."""
    all_metadata = _load_metadata()
    if file_id not in all_metadata:
        return False
    del all_metadata[file_id]
    _save_metadata(all_metadata)
    return True
