"""Prompt library storage: load/save, path safety and atomic writes."""
import json
import logging
import os
import re
import shutil
import threading
import time

import folder_paths

from .constants import LIBRARY_EMPTY
from .parsing import slugify

logger = logging.getLogger("llm_prompt_studio")

__all__ = [
    "safe_path_in_output",
    "resolve_library_path",
    "load_library",
    "library_scenes",
    "save_prompt_to_library",
]

# Set to False to allow library/save paths outside the ComfyUI output directory.
RESTRICT_PATHS_TO_OUTPUT = True

# Per-library-path locks so concurrent saves (many queued prompts, or the Smart Save
# auto-save running alongside a manual save) serialize their read-modify-write instead of
# clobbering each other's entries. Guarded by a global lock when first created.
_library_locks = {}
_library_locks_guard = threading.Lock()


def _lock_for(library_path: str) -> threading.Lock:
    with _library_locks_guard:
        return _library_locks.setdefault(library_path, threading.Lock())


def _atomic_replace(tmp_path: str, library_path: str, attempts: int = 10):
    # os.replace is atomic, but on Windows a MoveFileEx can transiently fail with
    # ERROR_SHARING_VIOLATION (PermissionError) if another handle still lingers on the
    # destination. A short bounded retry lets the holder release before we give up.
    last = None
    for _ in range(attempts):
        try:
            os.replace(tmp_path, library_path)
            return
        except OSError as e:
            last = e
            time.sleep(0.01)
    raise last


def safe_path_in_output(path: str) -> str:
    """Resolve a path and ensure it stays inside the ComfyUI output directory."""
    out_dir = folder_paths.get_output_directory()
    base = os.path.realpath(out_dir)
    p = (path or "").strip()
    candidate = os.path.join(base, p) if p else base
    candidate = os.path.realpath(candidate)
    if RESTRICT_PATHS_TO_OUTPUT and candidate != base \
            and not candidate.startswith(base + os.sep):
        raise ValueError(f"Path '{path}' is outside the output directory")
    return candidate


def resolve_library_path(library_path: str) -> str:
    lib = (library_path or "").strip() or "llm_prompt_studio_library.json"
    return safe_path_in_output(lib)


def load_library(library_path: str) -> list:
    if not os.path.exists(library_path):
        return []
    try:
        with open(library_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def library_scenes(library_path: str = "") -> list:
    entries = load_library(resolve_library_path(library_path))
    names = [str(e.get("name")) for e in entries
             if isinstance(e, dict) and e.get("name")]
    return names or [LIBRARY_EMPTY]


def save_prompt_to_library(library_path: str, scene_name: str,
                            positive: str, negative: str,
                            face_positive: str = "", face_negative: str = ""):
    """Appends a prompt to the JSON library with duplicate check.
    Returns (name, added): added=False if the same positive already exists.

    The read-modify-write is serialized by a per-path lock and written atomically via a
    temp file + ``os.replace``; the previous file is kept as a ``.bak`` for cheap recovery."""
    # Defense in depth: the callers already resolve via resolve_library_path(), but this
    # function must never write outside the output directory on its own.
    library_path = safe_path_in_output(library_path)
    parent = os.path.dirname(library_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    lock = _lock_for(library_path)
    with lock:
        entries = []
        if os.path.exists(library_path):
            try:
                with open(library_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    entries = data
            except (json.JSONDecodeError, OSError):
                entries = []
        for e in entries:
            if isinstance(e, dict) and \
                    str(e.get("prompt", "")).strip() == positive.strip() and \
                    str(e.get("negative_prompt", "")).strip() == negative.strip() and \
                    str(e.get("face_positive", "")).strip() == face_positive.strip() and \
                    str(e.get("face_negative", "")).strip() == face_negative.strip():
                return str(e.get("name", "")), False
        max_idx = 0
        for e in entries:
            if isinstance(e, dict):
                m = re.match(r"(\d+)", str(e.get("name", "")))
                if m:
                    max_idx = max(max_idx, int(m.group(1)))
        slug = scene_name.strip() or slugify(positive)
        name = f"{max_idx + 1:03d}_{slug}"
        entries.append({
            "name": name,
            "prompt": positive,
            "negative_prompt": negative,
            "face_positive": face_positive,
            "face_negative": face_negative,
        })
        # Keep the previous version as a backup before replacing it atomically.
        if os.path.exists(library_path):
            try:
                shutil.copyfile(library_path, library_path + ".bak")
            except OSError:
                pass
        tmp_path = library_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        _atomic_replace(tmp_path, library_path)
    logger.debug("Saved scene '%s' to library %s", name, library_path)
    return name, True