"""
Novel loading service.
Reads reference novels from backend/data/novels/{novel_id}/
Each novel folder contains:
  - 第x章 *.txt  → chapter text files
  - inspiration.json → brain analysis / migration ideas
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data" / "novels"

# Regex to parse "第1章 xxx.txt" or "第01章 xxx.txt"
CHAPTER_PATTERN = re.compile(r"^第(\d+)章\s+(.+)\.txt$")


def _ensure_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def list_novels() -> list[dict]:
    """
    Scan all sub-folders under data/novels/ and return a lightweight list.
    """
    novels_dir = _ensure_dir()
    result = []
    for folder in sorted(novels_dir.iterdir()):
        if not folder.is_dir():
            continue
        result.append({
            "id": folder.name,
            "title": folder.name,
        })
    return result


def load_novel(novel_id: str) -> Optional[dict]:
    """
    Load all chapters and inspiration for a given novel folder name.

    Returns:
        {
            "id": str,
            "title": str,
            "chapters": [{"index": int, "title": str, "filename": str}, ...],
            "inspiration": dict | None,
        }
    or None if the folder doesn't exist.
    """
    novels_dir = _ensure_dir()
    folder = novels_dir / novel_id
    if not folder.is_dir():
        return None

    chapters = []
    for file in folder.iterdir():
        if file.suffix != ".txt":
            continue
        m = CHAPTER_PATTERN.match(file.name)
        if not m:
            continue
        chapters.append({
            "index": int(m.group(1)),
            "title": f"第{m.group(1)}章 {m.group(2)}",
            "filename": file.name,
        })

    chapters.sort(key=lambda c: c["index"])

    # Load analysis.json if present (shown before inspiration)
    analysis_path = folder / "analysis.json"
    analysis = None
    if analysis_path.exists():
        try:
            with open(analysis_path, encoding="utf-8") as f:
                analysis = json.load(f)
        except Exception:
            pass

    # Load inspiration.json if present
    inspiration_path = folder / "inspiration.json"
    inspiration = None
    if inspiration_path.exists():
        try:
            with open(inspiration_path, encoding="utf-8") as f:
                inspiration = json.load(f)
        except Exception:
            pass

    return {
        "id": novel_id,
        "title": folder.name,
        "chapters": chapters,
        "analysis": analysis,
        "inspiration": inspiration,
    }
