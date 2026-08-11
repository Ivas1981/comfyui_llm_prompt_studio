import json
import logging
import os

logger = logging.getLogger("llm_prompt_studio")

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROMPTS_FILE = os.path.join(_PKG_ROOT, "prompts.json")


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

DEFAULT_SYSTEM = _P.get("writer_system", "") + _REASONING_HINT
FACE_PROMPT_INSTRUCTION = _P.get("face_instruction", "")
DEFAULT_CRITIC = _P.get("critic_system", "") + _REASONING_HINT
DEFAULT_DESCRIBE = _P.get("describe", "")
DEFAULT_COMPOSER = _P.get("composer", "") + _REASONING_HINT