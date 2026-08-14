import json
import logging
import os

logger = logging.getLogger("llm_prompt_studio")

# Base default prompts (Writer, Scene Builder, Critic, Describe, face instructions) and
# style presets all live in a single file: ``presets_default.json`` (copied to a
# user-editable ``llm_prompt_studio_presets.json`` in the ComfyUI output directory on
# first run). The package-root file is the shipped source of truth; the user file
# overrides it. See ``presets.py`` for loading/migration.
from ..presets import get_defaults

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Note for stage-1 Scene Builder (plain prose, not JSON): if the model reasons before
# answering we must still receive the description itself, never the reasoning trace.
_DESCRIBE_REASONING_HINT = (
    " If you use reasoning, keep it to 2-3 sentences and end your reply with the "
    "description itself — never output JSON here.")


def _load_prompts() -> dict:
    data = get_defaults()
    return data if isinstance(data, dict) else {}


_P = _load_prompts()

if not _P:
    logger.warning("Default prompts missing from presets file — default prompts will be empty.")

_REASONING_HINT = _P.get("reasoning_hint", "")
REASONING_HINT = _REASONING_HINT


def _with_hint(text: str) -> str:
    return (text or "") + _REASONING_HINT


DEFAULT_SYSTEM = _with_hint(_P.get("writer_system", ""))
FACE_PROMPT_INSTRUCTION = _P.get("face_instruction", "")
DEFAULT_CRITIC = _with_hint(_P.get("critic_system", ""))
DEFAULT_DESCRIBE = (_P.get("describe", "") or "") + _DESCRIBE_REASONING_HINT
DEFAULT_COMPOSER = _with_hint(_P.get("composer", ""))

# No-negative (distilled) variants. If a key is missing (e.g. an outdated presets file),
# fall back to the standard prompt so the node never runs with an empty system prompt.
DEFAULT_SYSTEM_NO_NEGATIVE = _with_hint(
    _P.get("writer_system_no_negative") or _P.get("writer_system", ""))
DEFAULT_COMPOSER_NO_NEGATIVE = _with_hint(
    _P.get("composer_no_negative") or _P.get("composer", ""))
FACE_PROMPT_INSTRUCTION_NO_NEGATIVE = (
    _P.get("face_instruction_no_negative") or _P.get("face_instruction", ""))
