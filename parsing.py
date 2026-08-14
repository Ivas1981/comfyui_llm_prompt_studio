"""JSON extraction and parsing for LLM responses, plus string normalization."""
import json
import re

__all__ = [
    "slugify",
    "parse_prompt_json",
    "parse_critic_json",
]


def slugify(text: str, max_words: int = 6) -> str:
    # [^\W_] keeps unicode word letters/digits (Cyrillic, CJK, …) but not underscores, so
    # non-ASCII prompts get a meaningful slug; CJK collapses to one token (acceptable).
    words = re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)
    return "_".join(words[:max_words]) or "scene"


def _iter_brace_objects(text: str):
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        # Inside a JSON string: only the escape char and the closing quote matter,
        # so braces within string values don't disturb the depth counter.
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start:i + 1]
                    start = -1


def _extract_json_dict(text: str, prefer_key: str):
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and prefer_key in obj:
            return obj
    except (json.JSONDecodeError, TypeError):
        pass
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.S):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and prefer_key in obj:
                return obj
        except (json.JSONDecodeError, TypeError):
            continue
    for cand in _iter_brace_objects(text):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and prefer_key in obj:
            return obj
    return None


def _salvage_partial_prompt(text: str):
    """Salvage positive/negative and face fields by quotes even from a truncated answer."""
    def grab(key):
        m = re.search(r'"' + key + r'"\s*:\s*"', text)
        if not m:
            return ""
        start = m.end()
        buf = []
        i = start
        while i < len(text):
            c = text[i]
            if c == "\\" and i + 1 < len(text):
                buf.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                return "".join(buf)
            buf.append(c)
            i += 1
        return "".join(buf)
    pos = grab("positive")
    neg = grab("negative")
    face_pos = grab("face_positive")
    face_neg = grab("face_negative")
    if pos or neg:
        return pos, neg, "", face_pos, face_neg
    return None


def parse_prompt_json(text: str, allow_plain_text_fallback: bool = True):
    """Returns (positive, negative, scene_name, face_positive, face_negative).

    When the reply is neither JSON nor salvageable, ``allow_plain_text_fallback=True`` (the
    default, used by the Critic/Scene Builder) treats the raw text as the positive prompt.
    ``allow_plain_text_fallback=False`` (used by the Writer) returns the empty tuple instead,
    so a model refusal / non-prompt reply becomes an explicit error rather than a silent, bad
    prompt."""
    obj = _extract_json_dict(text, "positive")
    if obj:
        result = (str(obj.get("positive", "")).strip(),
                  str(obj.get("negative", "")).strip(),
                  str(obj.get("scene_name", "")).strip(),
                  str(obj.get("face_positive", "")).strip(),
                  str(obj.get("face_negative", "")).strip())
    else:
        salv = _salvage_partial_prompt(text)
        if salv:
            result = salv
        elif allow_plain_text_fallback:
            fallback = text.strip()
            if fallback and len(fallback) <= 2000:
                result = (fallback, "", "", "", "")
            else:
                result = ("", "", "", "", "")
        else:
            result = ("", "", "", "", "")
    try:
        from .debug import log_parse_attempt  # package context
    except ImportError:
        from debug import log_parse_attempt  # type: ignore  # top-level (tests) context
    log_parse_attempt(text, obj is not None, obj)
    return result


def parse_critic_json(text: str):
    """Returns (score, verdict, revision_notes)."""
    obj = _extract_json_dict(text, "score")
    if obj:
        try:
            score = round(float(obj.get("score", -1)))
        except (TypeError, ValueError):
            score = -1
        return score, str(obj.get("verdict", "")), str(obj.get("revision_notes", ""))
    fallback = text.strip()
    if fallback and len(fallback) <= 2000:
        return -1, "", fallback
    return -1, "", ""


def find_missing_fields(parsed_tuple, require_face: bool = False,
                        require_negative: bool = True,
                        require_face_negative: bool = True):
    """Return the names of empty critical fields in a `parse_prompt_json` tuple.

    `parsed_tuple` is (positive, negative, scene_name, face_positive, face_negative).
    `positive` and `scene_name` are always critical. `negative` is critical when
    `require_negative` is True. `require_face`/`require_face_negative` are accepted for API
    compatibility but face fields are intentionally NOT treated as missing: the prompts
    explicitly allow empty `face_positive`/`face_negative` when no face is present, and
    empty-vs-absent is indistinguishable after JSON parsing, so retrying on them is futile
    (the Writer/Scene Builder fall back to the main prompts instead)."""
    positive, negative, scene_name, _fp, _fn = parsed_tuple
    critical = ["positive", "scene_name"]
    if require_negative:
        critical.append("negative")
    values = {
        "positive": positive,
        "negative": negative,
        "scene_name": scene_name,
    }
    return [name for name in critical if not str(values[name]).strip()]