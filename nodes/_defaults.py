import json
import logging
import os

logger = logging.getLogger("llm_prompt_studio")

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROMPTS_FILE = os.path.join(_PKG_ROOT, "prompts.json")

# Note for stage-1 Scene Builder (plain prose, not JSON): if the model reasons before
# answering we must still receive the description itself, never the reasoning trace.
_DESCRIBE_REASONING_HINT = (
    " If you use reasoning, keep it to 2-3 sentences and end your reply with the "
    "description itself — never output JSON here.")


def _load_prompts():
    try:
        with open(_PROMPTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


_P = _load_prompts()

if not _P:
    logger.warning("prompts.json not found or invalid — default prompts will be empty.")

_REASONING_HINT = _P.get("reasoning_hint", "")


def _with_hint(text: str) -> str:
    return (text or "") + _REASONING_HINT


DEFAULT_SYSTEM = _with_hint(_P.get("writer_system", ""))
FACE_PROMPT_INSTRUCTION = _P.get("face_instruction", "")
DEFAULT_CRITIC = _with_hint(_P.get("critic_system", ""))
DEFAULT_DESCRIBE = (_P.get("describe", "") or "") + _DESCRIBE_REASONING_HINT
DEFAULT_COMPOSER = _with_hint(_P.get("composer", ""))

# No-negative (distilled) variants. If a key is missing (e.g. an outdated prompts.json),
# fall back to the standard prompt so the node never runs with an empty system prompt.
DEFAULT_SYSTEM_NO_NEGATIVE = _with_hint(
    _P.get("writer_system_no_negative") or _P.get("writer_system", ""))
DEFAULT_COMPOSER_NO_NEGATIVE = _with_hint(
    _P.get("composer_no_negative") or _P.get("composer", ""))
FACE_PROMPT_INSTRUCTION_NO_NEGATIVE = (
    _P.get("face_instruction_no_negative") or _P.get("face_instruction", ""))
