"""Styles folder loader and system-prompt assembler for the Writer node.

The styles system lives in ``Styles/`` (package root):

    Styles/system_prompts.json   - base prompts + nsfw/prompt_format/negative fragments
    Styles/<direction>.json      - style presets (each file is one direction/category)

On first run the whole ``Styles/`` folder is mirrored into the ComfyUI output directory
as ``llm_prompt_studio_styles/`` (variant B): the user copy is never overwritten, so
hand edits survive a refresh. ``reload_styles()`` re-reads from disk; ``reset_styles()``
deletes the user copy and falls back to the shipped ``Styles/``.

A style preset may ``extends`` another id (inheritance): the child merges
``system_prompt``/``system_prompt_suffix`` and unions style tags, up to depth 3.

The three generation toggles (negative_prompt, face_prompt, nsfw) and the prompt_format
choice are ORTHOGONAL and applied at assembly time in :func:`build_system_prompt`; they are
never stored per preset.
"""
import json
import os
import shutil
from typing import Dict, List, Optional

CURRENT_SCHEMA_VERSION = "2.0.0"
_STYLES_DIR_NAME = "Styles"
_USER_DIR_NAME = "llm_prompt_studio_styles"

# Cache of the loaded styles (invalidated by reload_styles / reset_styles).
_cache: Dict = {}


def _package_styles_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), _STYLES_DIR_NAME)


def _user_styles_dir() -> str:
    try:
        import folder_paths
        out_dir = folder_paths.get_output_directory()
    except Exception:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(out_dir, _USER_DIR_NAME)


def _active_styles_dir() -> str:
    """The folder we read from: the user copy if it exists, else the shipped Styles/."""
    user = _user_styles_dir()
    if os.path.isdir(user) and os.listdir(user):
        return user
    return _package_styles_dir()


def ensure_styles_dir():
    """Mirror the shipped Styles/ into the user output dir if it does not yet exist."""
    user = _user_styles_dir()
    if not os.path.isdir(user) or not os.listdir(user):
        try:
            shutil.copytree(_package_styles_dir(), user)
        except (OSError, FileExistsError):
            pass


def reset_styles():
    """Delete the user styles copy and drop the in-memory cache."""
    user = _user_styles_dir()
    if os.path.isdir(user):
        try:
            shutil.rmtree(user)
        except OSError:
            pass
    _cache.clear()


def reload_styles():
    """Force a fresh read from disk on the next access."""
    _cache.clear()


def get_styles_dir() -> str:
    """Return the live styles folder (user copy if present, else the shipped Styles/).

    Shown by the UI 'Copy styles path' button so the user knows which file is actually read.
    """
    return _active_styles_dir()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_all() -> Dict:
    if _cache:
        return _cache
    root = _active_styles_dir()
    data: Dict = {
        "system_prompts": {},
        "presets": [],
        "by_id": {},
    }
    sp_path = os.path.join(root, "system_prompts.json")
    if os.path.exists(sp_path):
        try:
            with open(sp_path, "r", encoding="utf-8") as f:
                data["system_prompts"] = json.load(f)
        except (OSError, json.JSONDecodeError):
            data["system_prompts"] = {}
    for fn in sorted(os.listdir(root)):
        if not fn.endswith(".json") or fn == "system_prompts.json":
            continue
        path = os.path.join(root, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        presets = content.get("presets") if isinstance(content, dict) else None
        if not isinstance(presets, list):
            continue
        for p in presets:
            if not isinstance(p, dict) or not p.get("id"):
                continue
            p.setdefault("name", p["id"])
            p.setdefault("category", "")
            p.setdefault("description", "")
            p.setdefault("system_prompt", "")
            p.setdefault("system_prompt_no_negative", "")
            p.setdefault("style_tags_positive", [])
            p.setdefault("style_tags_negative", [])
            p.setdefault("blend_note", p.get("name", ""))
            p.setdefault("disabled_in_no_negative_mode", False)
            data["presets"].append(p)
            data["by_id"][p["id"]] = p
    _cache.update(data)
    return data


def get_system_prompts() -> Dict:
    """Return the base prompt fragments. Falls back to the legacy presets file."""
    data = _load_all()
    if data.get("system_prompts"):
        return data["system_prompts"]
    try:
        from .presets import get_defaults
    except ImportError:  # top-level (tests)
        from presets import get_defaults  # type: ignore
    return get_defaults()


def get_all_styles() -> List[Dict]:
    return _load_all().get("presets", [])


def get_style_by_id(style_id: str) -> Optional[Dict]:
    if not style_id:
        return None
    return _load_all().get("by_id", {}).get(style_id)


def get_style_labels() -> List[str]:
    """Combobox entries: 'Category / Name' (or bare Name), plus a '— none —' default."""
    out = ["— none —"]
    for p in get_all_styles():
        cat = p.get("category") or ""
        name = p.get("name", "")
        out.append(f"{cat} / {name}" if cat else name)
    return out


def get_style_by_label(label: str) -> Optional[Dict]:
    if not label or label == "— none —":
        return None
    bare = label.split("/ ", 1)[-1].strip() if "/ " in label else label.strip()
    for p in get_all_styles():
        if p.get("name") == label or p.get("name") == bare:
            return p
    return None


def resolve_style_token(token: str) -> Optional[Dict]:
    """Resolve a blend/style token that may be an id or a 'Category / Name' label."""
    token = (token or "").strip()
    if not token or token == "— none —":
        return None
    s = resolve_style(token)
    if s:
        return s
    lbl = get_style_by_label(token)
    if lbl:
        return resolve_style(lbl.get("id"))
    return None


# ---------------------------------------------------------------------------
# Inheritance resolution
# ---------------------------------------------------------------------------
def resolve_style(style_id: str, _depth: int = 0) -> Optional[Dict]:
    """Return a preset with ``extends`` flattened (system_prompt + suffix merge, tag union)."""
    if _depth > 3:
        return None
    raw = get_style_by_id(style_id)
    if not raw:
        return None
    parent_id = raw.get("extends")
    if not parent_id:
        return dict(raw)
    parent = resolve_style(parent_id, _depth + 1)
    if not parent:
        return dict(raw)
    merged = dict(parent)
    merged["id"] = raw["id"]
    merged["name"] = raw.get("name", parent.get("name"))
    merged["category"] = raw.get("category", parent.get("category"))
    merged["description"] = raw.get("description", parent.get("description"))
    base_sp = (parent.get("system_prompt") or "")
    child_sp = (raw.get("system_prompt") or "")
    merged["system_prompt"] = (base_sp + "\n\n" + child_sp).strip() if child_sp else base_sp
    suffix = raw.get("system_prompt_suffix")
    if suffix:
        merged["system_prompt"] = (merged["system_prompt"] + "\n\n" + suffix).strip()
    # Union of tags (preserving parent order, child overrides/extends).
    for key in ("style_tags_positive", "style_tags_negative"):
        p_tags = list(parent.get(key) or [])
        c_tags = list(raw.get(key) or [])
        seen = {t.lower() for t in p_tags}
        for t in c_tags:
            if t.lower() not in seen:
                p_tags.append(t)
                seen.add(t.lower())
        merged[key] = p_tags
    # Child fields win for scalars unless empty.
    for k in ("system_prompt_no_negative", "blend_note", "disabled_in_no_negative_mode"):
        if raw.get(k) not in (None, "", False):
            merged[k] = raw.get(k)
    return merged


# ---------------------------------------------------------------------------
# System-prompt assembly
# ---------------------------------------------------------------------------
def build_system_prompt(preset: Optional[Dict] = None, *,
                        nsfw: bool = False,
                        prompt_format: str = "natural",
                        negative_prompt: bool = True,
                        face_prompt: bool = False,
                        blend_styles: str = "",
                        architecture: str = "",
                        base: Optional[str] = None) -> str:
    """Assemble the full system prompt the Writer sends to the local model.

    Order (mirrors Plans/styles_system_redesign.md):
      base writer_system -> general craft rules -> style.system_prompt -> face
      -> nsfw -> prompt_format -> negative(on/off) -> blend notes
      -> architecture addendum -> reasoning hint
    """
    sp = get_system_prompts()
    parts: List[str] = []

    # 1) Base prompt engineer instruction (format-agnostic, JSON envelope).
    parts.append(base if base is not None else sp.get("writer_system", ""))

    # 1b) Shared general craft rules (applied once, so presets stay DRY).
    eng = sp.get("engineering_rules", "")
    if eng:
        parts.append(eng)

    # 2) Style preset (choose no-negative variant when negative prompt is off).
    if preset:
        if negative_prompt:
            style_text = preset.get("system_prompt") or ""
        else:
            style_text = preset.get("system_prompt_no_negative") or preset.get("system_prompt") or ""
        if style_text:
            parts.append(style_text)

    # 3) Face prompt instruction.
    if face_prompt:
        parts.append(sp.get("face_instruction", ""))
    else:
        parts.append(sp.get("face_prompt", {}).get("off", ""))

    # 4) NSFW policy.
    parts.append(sp.get("nsfw_policy", {}).get("on" if nsfw else "off", ""))

    # 5) Prompt format fragment.
    fmt = sp.get("prompt_format", {}).get(prompt_format)
    if not fmt:
        fmt = sp.get("prompt_format", {}).get("natural", "")
    if fmt:
        parts.append(fmt)

    # 6) Negative handling.
    parts.append(sp.get("negative", {}).get("off" if not negative_prompt else "on", ""))

    # 7) Blend notes (union of blend_note from each referenced style).
    blend_ids = [s.strip() for s in (blend_styles or "").split(",") if s.strip()]
    blend_notes = []
    for bid in blend_ids[:3]:
        if bid == (preset or {}).get("id"):
            continue
        bp = resolve_style_token(bid)
        if not bp:
            continue
        name = bp.get("name", bid)
        note = (bp.get("blend_note") or "").strip()
        if not note or note.lower() == name.lower():
            # fall back to a short descriptor drawn from the style's own prompt
            sp_text = (bp.get("system_prompt") or "").strip()
            note = sp_text.split("\n")[0].strip() if sp_text else name
            if len(note) > 200:
                note = note[:197] + "..."
        blend_notes.append(f"- {name}: {note}")
    if blend_notes:
        header = (
            "BLEND STYLES: fuse ALL of the following styles into ONE coherent image "
            "description. Every listed style MUST be clearly and visibly present in the "
            "positive prompt \u2014 combine their defining visual traits, do not merely name "
            "them, and never drop any."
        )
        parts.append(header + "\n" + "\n".join(blend_notes))

    # 8) Architecture addendum (e.g. SD1.5 short tags, Flux natural language, Pony booru).
    arch = (architecture or "").strip().lower()
    if arch:
        guidance = sp.get("architecture_guidance", {}).get(arch, {})
        addendum = guidance.get("system_addendum")
        if addendum:
            parts.append(addendum)

    # 9) Reasoning hint keeps the model finishing with the JSON object.
    rh = sp.get("reasoning_hint", "")
    if rh and "ALWAYS finish your reply with the complete JSON object" not in parts[-1]:
        parts.append(rh)

    return "\n\n".join(p for p in parts if p and p.strip())
