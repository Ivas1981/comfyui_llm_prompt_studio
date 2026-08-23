"""Debug logging for LLM Prompt Studio nodes and HTTP layer.

Logging is OFF by default. Set ``DEBUG_LEVEL`` below (and restart ComfyUI) to activate:
  - "OFF"     -> no log file is ever created, all logging calls are no-ops (cheap).
  - "MINIMAL" -> node enter / exit / error only.
  - "FULL"    -> everything above plus HTTP request/response and parse attempts.

API keys (Authorization headers, api_key fields) and base64 image blobs are never written
to the log. Log lines are rotated when they reach ``LOG_MAX_SIZE_MB``.
"""
import contextlib
import json
import logging
import os
import re
import threading
import time
import traceback
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

# --- Configuration (edit here; requires a ComfyUI restart to take effect) ----------
DEBUG_LEVEL = "OFF"            # "OFF" | "MINIMAL" | "FULL"
LOG_MAX_SIZE_MB = 50           # rotate when the log reaches this size
LOG_BACKUP_COUNT = 1           # number of rotated backups to keep
# ---------------------------------------------------------------------------------

_LEVEL_RANK = {"OFF": 0, "MINIMAL": 1, "FULL": 2}
_MINIMAL_EVENTS = ("NODE_ENTER", "NODE_EXIT", "NODE_ERROR")

_logger: Optional[logging.Logger] = None
_lock = threading.Lock()

_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")


def debug_active() -> bool:
    return DEBUG_LEVEL != "OFF"


def _get_logger() -> Optional[logging.Logger]:
    """Lazily create the rotating file logger (only once a real record is emitted)."""
    global _logger
    if _logger is not None:
        return _logger
    if DEBUG_LEVEL == "OFF":
        return None
    try:
        import folder_paths
        out_dir = folder_paths.get_output_directory()
    except Exception:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(out_dir, "llm_prompt_studio.log")
    logger = logging.getLogger("llm_prompt_studio.debug")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        try:
            handler = RotatingFileHandler(
                log_path, maxBytes=LOG_MAX_SIZE_MB * 1024 * 1024,
                backupCount=LOG_BACKUP_COUNT, encoding="utf-8")
            handler.setFormatter(logging.Formatter(
                "[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"))
            logger.addHandler(handler)
        except OSError:
            return None
    _logger = logger
    return _logger


def mask_api_key(value: str) -> str:
    """Return ``***...<last 4 chars>`` for a secret, or a generic mask if too short."""
    if value is None:
        return ""
    value = str(value)
    if not value:
        return value
    if len(value) <= 4:
        return "***"
    return "***..." + value[-4:]


def _mask_headers(headers: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not headers:
        return {}
    out = {}
    for k, v in headers.items():
        if k.lower() == "authorization" and isinstance(v, str) and v.startswith("Bearer "):
            out[k] = "Bearer " + mask_api_key(v[len("Bearer "):])
        else:
            out[k] = v
    return out


def _looks_like_b64(s: str) -> bool:
    s = s.strip()
    if not s or len(s) < 50:
        return False
    if not _BASE64_RE.match(s):
        return False
    # Real base64 is padded ('=') or its length is a multiple of 4 once whitespace is
    # stripped. This drops ordinary long alphanumeric text that merely matches the alphabet.
    core = re.sub(r"\s", "", s)
    return "=" in core or len(core) % 4 == 0


def _redact_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in ("authorization", "api_key", "apikey"):
                out[k] = mask_api_key(str(v)) if v else v
            elif isinstance(v, str) and _looks_like_b64(v):
                out[k] = f"<base64 {len(v)} chars>"
            elif isinstance(v, (dict, list)):
                out[k] = _redact_obj(v)
            else:
                out[k] = v
        return out
    if isinstance(obj, list):
        return [_redact_obj(x) for x in obj]
    return obj


def _truncate(data: Dict[str, Any], max_len: int = 200) -> Dict[str, Any]:
    out = {}
    for k, v in data.items():
        if isinstance(v, str) and len(v) > max_len:
            out[k] = v[:max_len] + f"... ({len(v)} chars)"
        elif isinstance(v, (dict, list)):
            out[k] = _redact_obj(v)
        else:
            out[k] = v
    return out


def log(level: str, event_type: str, node_id: str, data: Dict[str, Any]):
    if DEBUG_LEVEL == "OFF":
        return
    if DEBUG_LEVEL == "MINIMAL" and event_type not in _MINIMAL_EVENTS:
        return
    logger = _get_logger()
    if logger is None:
        return
    try:
        payload = json.dumps(_redact_obj(data), ensure_ascii=False, default=str)
    except Exception:
        payload = "<unserializable data>"
    with _lock:
        logger.log(getattr(logging, level, logging.INFO),
                   f"[{event_type}] [{node_id}] {payload}")


def log_node_enter(node_name: str, node_id: Any, inputs: Dict[str, Any]):
    log("INFO", "NODE_ENTER", str(node_id), {"node": node_name, **_truncate(inputs)})


def log_node_exit(node_name: str, node_id: Any, outputs: Dict[str, Any], duration_ms: float):
    log("INFO", "NODE_EXIT", str(node_id),
        {"node": node_name, "duration_ms": round(duration_ms, 1), **_truncate(outputs)})


def log_error(node_id: Any, exception: Exception, tb: str):
    log("ERROR", "NODE_ERROR", str(node_id), {"error": str(exception), "traceback": tb})


def log_type_mismatch(node_id: Any, input_name: str, source_type: Any, dest_type: Any,
                      note: str = ""):
    """Log a sampler/scheduler (or other combo) type-contract violation.

    ComfyUI validates a prompt link by comparing the source node's output TYPE tuple
    against the destination node's input TYPE tuple for exact equality. Two different
    tuple objects (even with equal contents) make the whole prompt fail to validate
    with a ``Return type mismatch`` error before any node runs. This event surfaces
    such mismatches early so they are not mistaken for a generation/runtime failure.
    """
    def _repr(t: Any) -> str:
        try:
            if isinstance(t, (list, tuple)):
                return "(%d) %s" % (len(t), ", ".join(map(str, t))[:120])
            return str(t)
        except Exception:
            return repr(t)

    log("ERROR", "TYPE_MISMATCH", str(node_id), {
        "input": input_name,
        "source_type": _repr(source_type),
        "dest_type": _repr(dest_type),
        "note": note,
    })


def log_http_request(method: str, url: str, headers: Optional[Dict[str, Any]], body: Any):
    log("DEBUG", "HTTP_REQUEST", "-",
        {"method": method, "url": url, "headers": _mask_headers(headers),
         "body": _redact_obj(body)})


def log_http_response(status_code: int, duration_ms: float, body_size: int):
    log("DEBUG", "HTTP_RESPONSE", "-",
        {"status": status_code, "duration_ms": round(duration_ms, 1), "body_size": body_size})


def log_parse_attempt(raw_text: str, success: bool, extracted: Optional[Dict]):
    snippet = raw_text if len(raw_text) <= 500 else raw_text[:500] + "..."
    log("DEBUG", "PARSE_ATTEMPT", "-",
        {"success": success, "raw_len": len(raw_text),
         "extracted": _redact_obj(extracted) if extracted else None, "raw": snippet})


@contextlib.contextmanager
def node_span(node_name: str, node_id: Any, inputs: Optional[Dict[str, Any]] = None):
    """Context manager that logs node enter/exit and any exception (for utility nodes).

    Usage::

        with node_span("Smart Save", unique_id, {"approved": approved}):
            ... node body ...
    """
    _t0 = time.time()
    log_node_enter(node_name, node_id, inputs or {})
    try:
        yield
    except Exception as e:
        log_error(node_id, e, traceback.format_exc())
        raise
    finally:
        log_node_exit(node_name, node_id, {}, (time.time() - _t0) * 1000)
