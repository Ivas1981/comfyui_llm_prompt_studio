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
        return {"schema_version": CURRENT_SCHEMA_VERSION, "presets": [], "defaults": {}}


def load_presets_raw() -> Dict:
    """Read the combined presets file (defaults + presets) without side effects.

    Prefers the user-editable copy in the ComfyUI output directory, falling back to the
    shipped ``presets_default.json``. Does NOT create or overwrite the user file.

    The returned data is normalized in memory via :func:`_migrate` (missing top-level
    and per-preset fields backfilled, schema version bumped) so callers never have to
    handle a stale shape. The migration is NOT written to disk — :func:`load_presets`
    is what persists it on first run.
    """
    user_path = get_user_presets_path()
    data = None
    if os.path.exists(user_path):
        try:
            with open(user_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = None
    if not isinstance(data, dict):
        data = _load_defaults()
    return _migrate(data)


def get_defaults() -> Dict:
    """Return the base default prompts (Writer/Scene Builder/Critic/Describe/face)."""
    data = load_presets_raw()
    defaults = data.get("defaults")
    if not isinstance(defaults, dict):
        # Backfill from the shipped file so an older user file still gets base prompts.
        return _load_defaults().get("defaults", {})
    return defaults


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
    if not isinstance(data.get("defaults"), dict):
        data["defaults"] = _load_defaults().get("defaults", {})
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
        p.setdefault("system_prompt_no_negative", "")
        p.setdefault("style_tags_positive", [])
        p.setdefault("style_tags_negative", [])
        p.setdefault("disabled_in_no_negative_mode", False)
        p.setdefault("category", "")
        # Normalize types: a hand-edited JSON may carry a wrong type (e.g. a string
        # where a list is expected). setdefault only fills missing keys, so coerce
        # explicitly to avoid a crash later in apply_preset_to_prompts.
        for _tag_key in ("style_tags_positive", "style_tags_negative"):
            _tags = p.get(_tag_key)
            if not isinstance(_tags, list):
                p[_tag_key] = [_tags] if _tags else []
        if not isinstance(p.get("disabled_in_no_negative_mode"), bool):
            p["disabled_in_no_negative_mode"] = bool(p.get("disabled_in_no_negative_mode"))
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
    """Labels for the Writer's style_preset combobox.

    Each preset is shown as ``"<category> > <name>"`` when it carries a category, so the
    (now ~50) presets are grouped in the dropdown. Presets without a category keep their
    bare ``name``. The bare name remains the stored identity (see :func:`get_preset_by_name`)."""
    out = []
    for p in load_presets().get("presets", []):
        cat = p.get("category") or ""
        name = p.get("name", "")
        out.append(f"{cat} > {name}" if cat else name)
    return out


def get_preset_by_name(name: str) -> Optional[Dict]:
    """Return a preset dict by its display name (the combobox stores labels/names, not ids).

    Matches both the bare preset ``name`` (old saved workflows) and the categorized
    ``"<category> > <name>"`` label produced by :func:`get_preset_names`, so a workflow
    saved before categorization still resolves after "Reload presets"."""
    if not name:
        return None
    presets = load_presets().get("presets", [])
    for p in presets:
        if p.get("name") == name:
            return p
    bare = name.split("> ")[-1].strip() if "> " in name else name
    for p in presets:
        if p.get("name") == bare:
            return p
    return None


def get_architecture_guidance() -> Dict:
    """Architecture-specific prompt guidance from ``presets_default.json``.

    Keyed by canonical architecture (``sdxl``, ``sd15``, ``pony``, ``illustrious``,
    ``flux``, ``sd3``, ``unknown``). Each entry may carry ``system_addendum`` (appended to
    the system prompt), ``default_negative`` (appended to the negative in standard mode) and
    ``force_no_negative`` (bool). User-editable via the presets file."""
    data = load_presets_raw()
    return data.get("architecture_guidance", {})


def append_negative_tags(negative: str, additions: str) -> str:
    """Append architecture ``default_negative`` tokens to an existing negative prompt.

    Mirrors :func:`apply_preset_to_prompts` for negatives: tokens are split on commas,
    trimmed, deduplicated against the existing negative (and within the additions), and
    joined back. Empty input is returned unchanged."""
    if not additions:
        return negative
    existing = {t.strip().lower() for t in negative.split(",") if t.strip()}
    new = []
    for tok in additions.split(","):
        tok = tok.strip()
        if tok and tok.lower() not in existing:
            existing.add(tok.lower())
            new.append(tok)
    if not new:
        return negative
    return (negative + ", " + ", ".join(new)) if negative.strip() else ", ".join(new)


def apply_preset_to_prompts(preset: Dict, positive: str, negative: str,
                            no_negative: bool = False):
    """Append the preset's style tags to the positive (and negative, unless no-negative)."""
    pos_tags = preset.get("style_tags_positive") or []
    if not isinstance(pos_tags, list):
        pos_tags = [pos_tags] if pos_tags else []
    if pos_tags:
        positive = f"{positive}, {', '.join(pos_tags)}" if positive else ", ".join(pos_tags)
    if not no_negative:
        neg_tags = preset.get("style_tags_negative") or []
        if not isinstance(neg_tags, list):
            neg_tags = [neg_tags] if neg_tags else []
        if neg_tags:
            negative = f"{negative}, {', '.join(neg_tags)}" if negative else ", ".join(neg_tags)
    return positive, negative


def apply_preset_style(preset: Dict, positive: str, negative: str,
                       face_positive: str = "", face_negative: str = "",
                       no_negative: bool = False):
    """Apply a style preset's tags to the main prompts AND the face prompts.

    The selected style should influence ``face_positive``/``face_negative`` too (Q4): the
    same style tokens that shape the scene are appended to the per-face prompts so a face
    inpainted by FaceDetailer matches the chosen style. Empty face strings are left empty
    (e.g. when face prompts are disabled), so no tags are injected into a blank prompt."""
    pos_tags = preset.get("style_tags_positive") or []
    if not isinstance(pos_tags, list):
        pos_tags = [pos_tags] if pos_tags else []
    if pos_tags:
        joined = ", ".join(pos_tags)
        positive = f"{positive}, {joined}" if positive else joined
        if face_positive:
            face_positive = f"{face_positive}, {joined}"
    if not no_negative:
        neg_tags = preset.get("style_tags_negative") or []
        if not isinstance(neg_tags, list):
            neg_tags = [neg_tags] if neg_tags else []
        if neg_tags:
            joined = ", ".join(neg_tags)
            negative = f"{negative}, {joined}" if negative else joined
            if face_negative:
                face_negative = f"{face_negative}, {joined}"
    return positive, negative, face_positive, face_negative


def reset_to_defaults():
    """Delete the user file and restore the shipped defaults."""
    user_path = get_user_presets_path()
    if os.path.exists(user_path):
        try:
            os.remove(user_path)
        except OSError:
            pass
    ensure_presets_file()
