import logging
import os

logger = logging.getLogger("llm_prompt_studio")

# Base default prompts live in ``Styles/system_prompts.json`` (mirrored to a user-editable
# ``llm_prompt_studio_styles/`` folder in the ComfyUI output directory). If the Styles folder
# is unavailable we fall back to the legacy ``presets_default.json`` via ``presets.get_defaults``.
try:
    from ..styles import get_system_prompts
except ImportError:  # top-level (tests)
    from styles import get_system_prompts  # type: ignore

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_prompts() -> dict:
    sp = get_system_prompts()
    if isinstance(sp, dict) and sp:
        return sp
    try:
        from ..presets import get_defaults
    except ImportError:
        from presets import get_defaults  # type: ignore
    return get_defaults() or {}


_P = _load_prompts()

if not _P:
    logger.warning("Default prompts missing — default prompts will be empty.")

_REASONING_HINT = _P.get("reasoning_hint", "")
REASONING_HINT = _REASONING_HINT
_FINISH_HINT = "ALWAYS finish your reply with the complete JSON object."


def _with_hint(text: str) -> str:
    """Append the reasoning hint and JSON finish instruction exactly once.

    Older prompt files embedded the finish sentence in ``writer_system`` (and in
    ``reasoning_hint``), which produced duplicates in the assembled system prompt.
    """
    out = (text or "").strip()
    hint = (_REASONING_HINT or "").strip()
    if hint:
        if _FINISH_HINT in hint and _FINISH_HINT in out:
            hint = hint.replace(" and " + _FINISH_HINT, "")
            hint = hint.replace(_FINISH_HINT, "").strip()
        if hint and hint not in out:
            out = (out + " " + hint).strip() if out else hint
    if _FINISH_HINT not in out:
        out = (out + " " + _FINISH_HINT).strip() if out else _FINISH_HINT
    return out


def _fallback(text: str, key: str) -> str:
    """Use ``text`` if present, else the legacy ``presets_default.json`` key, else empty."""
    if text:
        return text
    try:
        from ..presets import get_defaults
    except ImportError:
        from presets import get_defaults  # type: ignore
    return (get_defaults() or {}).get(key, "")


# The new no-negative handling is per-style (each style carries system_prompt_no_negative)
# and runtime-driven by prompt_mode, so the base writer system is shared.
DEFAULT_SYSTEM = _with_hint(_P.get("writer_system", ""))
DEFAULT_SYSTEM_NO_NEGATIVE = _with_hint(
    _P.get("writer_system_no_negative")
    or _fallback("", "writer_system_no_negative")
    or _P.get("writer_system", ""))
FACE_PROMPT_INSTRUCTION = _P.get("face_instruction", "")
FACE_PROMPT_INSTRUCTION_NO_NEGATIVE = (
    _P.get("face_instruction_no_negative")
    or _fallback("", "face_instruction_no_negative")
    or _P.get("face_instruction", ""))
DEFAULT_CRITIC = _with_hint(_P.get("critic_system", _fallback("", "critic_system")))
DEFAULT_DESCRIBE = (_P.get("describe", _fallback("", "describe")) or "")
DEFAULT_COMPOSER = _with_hint(_P.get("composer", _fallback("", "composer")))
DEFAULT_COMPOSER_NO_NEGATIVE = _with_hint(
    _P.get("composer_no_negative")
    or _fallback("", "composer_no_negative")
    or _P.get("composer", _fallback("", "composer")))
