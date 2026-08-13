"""Style-preset management for the Writer node.

Presets ship as ``presets_default.json`` (package root) and are copied to a user-editable
file in the ComfyUI output directory (``llm_prompt_studio_presets.json``) on first run.
The user file is the source of truth: it is never overwritten by a refresh, and a broken
JSON falls back to the defaults. Schema is versioned via ``schema_version`` so older user
files are migrated forward transparently.
"""
import json
import os
import shutil
from typing import Dict, List, Optional

CURRENT_SCHEMA_VERSION = "1.0.0"

_PRESETS_DEFAULT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "presets_default.json")
_USER_FILE_NAME = "llm_prompt_studio_presets.json"


def get_user_presets_path() -> str:
    """Path to the user-editable presets file in the ComfyUI output directory."""
    try:
        import folder_paths
        out_dir = folder_paths.get_output_directory()
    except Exception:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(out_dir, _USER_FILE_NAME)


def _load_defaults() -> Dict:
    try:
        with open(_PRESETS_DEFAULT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"schema_version": CURRENT_SCHEMA_VERSION, "presets": []}


def ensure_presets_file():
    """Create the user presets file from defaults if it does not exist."""
    user_path = get_user_presets_path()
    if not os.path.exists(user_path):
        try:
            shutil.copy(_PRESETS_DEFAULT_FILE, user_path)
        except OSError:
            pass


def _migrate(data: Dict) -> Dict:
    """Backfill missing top-level and per-preset fields; bump schema version."""
    if not isinstance(data, dict):
        data = {}
    data.setdefault("schema_version", CURRENT_SCHEMA_VERSION)
    presets = data.get("presets")
    if not isinstance(presets, list):
        presets = []
    migrated = []
    for p in presets:
        if not isinstance(p, dict):
            continue
        p.setdefault("id", p.get("name", "").lower().replace(" ", "_"))
        p.setdefault("name", p.get("id", "preset"))
        p.setdefault("description", "")
        p.setdefault("system_prompt", "")
        p.setdefault("style_tags_positive", [])
        p.setdefault("style_tags_negative", [])
        p.setdefault("disabled_in_no_negative_mode", False)
        migrated.append(p)
    data["presets"] = migrated
    data["schema_version"] = CURRENT_SCHEMA_VERSION
    return data


def load_presets() -> Dict:
    """Load presets from the user file, migrating and falling back to defaults on error."""
    ensure_presets_file()
    user_path = get_user_presets_path()
    try:
        with open(user_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return _load_defaults()
    return _migrate(data)


def get_preset_names() -> List[str]:
    """Names for the Writer's style_preset combobox."""
    return [p["name"] for p in load_presets().get("presets", [])]


def get_preset_by_name(name: str) -> Optional[Dict]:
    """Return a preset dict by its display name (the combobox stores names, not ids)."""
    if not name:
        return None
    for p in load_presets().get("presets", []):
        if p.get("name") == name:
            return p
    return None


def apply_preset_to_prompts(preset: Dict, positive: str, negative: str,
                            no_negative: bool = False):
    """Append the preset's style tags to the positive (and negative, unless no-negative)."""
    pos_tags = preset.get("style_tags_positive") or []
    if pos_tags:
        positive = f"{positive}, {', '.join(pos_tags)}" if positive else ", ".join(pos_tags)
    if not no_negative:
        neg_tags = preset.get("style_tags_negative") or []
        if neg_tags:
            negative = f"{negative}, {', '.join(neg_tags)}" if negative else ", ".join(neg_tags)
    return positive, negative


def reset_to_defaults():
    """Delete the user file and restore the shipped defaults."""
    user_path = get_user_presets_path()
    if os.path.exists(user_path):
        try:
            os.remove(user_path)
        except OSError:
            pass
    ensure_presets_file()
