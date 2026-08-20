"""VRAM release helpers bridging ComfyUI model management and LM Studio.

This module is intentionally ComfyUI-optional: it must import and run in
standalone tests where ``comfy.model_management`` is absent, and never raise on
import. The dependency direction is one-way: ``nodes/* -> vram.py -> lm_http.py``
— ``lm_http.py`` must NOT import this module, to keep the import graph acyclic.
"""
import ipaddress
import os
from urllib.parse import urlparse

from . import lm_http
from .debug import debug_active, log


logger = __import__("logging").getLogger("llm_prompt_studio")


# Global kill switch (mirrors the env-var parsing style in lm_http.py:53).
_KEEP_LOADED = os.getenv("LLM_PROMPT_STUDIO_KEEP_MODEL_LOADED", "False").lower() \
    in ("1", "true", "yes")


def _model_management():
    """Lazily import ``comfy.model_management``; return None when unavailable.

    ``comfy`` itself is stubbed in tests without ``model_management``, so both the
    import and the attribute access must be guarded."""
    try:
        import comfy.model_management as mm  # type: ignore
        if getattr(mm, "unload_all_models", None) is None:
            return None
        return mm
    except (ImportError, AttributeError):
        return None


def coerce_bool_widget(value, default=True):
    """Coerce a widget value to bool, treating None/"" as the *default*.

    Saved workflows can feed a stray empty string for a boolean widget (the
    positional ``widgets_values`` overflow). ``""`` is falsy, which would silently
    disable a default-True feature, so we map it back to the default instead."""
    if value is None or value == "":
        return default
    return bool(value)


def free_vram_bytes(device=None):
    """Best-effort free-VRAM in bytes, or None when not measurable."""
    mm = _model_management()
    if mm is None:
        return None
    try:
        if hasattr(mm, "get_free_memory"):
            free = mm.get_free_memory(device)
            if isinstance(free, (int, float)):
                return free
    except Exception:
        return None
    return None


def log_vram(stage: str, note: str = ""):
    """Emit a free-VRAM line at INFO plus a DEBUG-level VRAM event when enabled."""
    try:
        free = free_vram_bytes()
    except Exception:
        free = None
    if not isinstance(free, (int, float)):
        free = None
    if free is None:
        msg = f"[VRAM][{stage}] free=unknown" + (f" ({note})" if note else "")
    else:
        gib = free / (1024 ** 3)
        msg = f"[VRAM][{stage}] free={free} bytes ({gib:.2f} GiB)" + (f" ({note})" if note else "")
    logger.info(msg)
    if debug_active():
        log("DEBUG", "VRAM", stage, {"note": note, "free_bytes": free})


def is_local_server(server_url) -> bool:
    """True when the LM Studio host shares this machine's GPU.

    Localhost, ``*.localhost`` and loopback IPs are local; everything else (LAN /
    remote) shares no GPU and must not trigger ComfyUI-side model eviction."""
    try:
        parsed = urlparse(str(server_url or "").strip())
    except ValueError:
        return False
    host = parsed.hostname
    if not host:
        return False
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(ip.is_loopback)


def unload_comfy_models(reason: str) -> bool:
    """Evict every ComfyUI model from VRAM into CPU RAM. Never raises."""
    mm = _model_management()
    if mm is None:
        return False
    try:
        mm.unload_all_models()
        soft = getattr(mm, "soft_empty_cache", None)
        if callable(soft):
            soft()
        return True
    except Exception as e:  # noqa: BLE001 — releasing VRAM is best-effort
        logger.warning("[VRAM] ComfyUI model eviction failed (%s): %s", reason, e)
        return False


def release_enabled() -> bool:
    """False when the global keep-loaded kill switch is set."""
    return not _KEEP_LOADED


def mark_keep_loaded(server_url: str, keep: bool):
    """Pin (or unpin) a server so release logic skips / performs its unload.

    Delegates to ``lm_http.mark_keep_loaded`` so the node layer can call it through
    the single ``vram`` import it already has."""
    return lm_http.mark_keep_loaded(server_url, keep)


def prepare_for_llm(server_url: str, note: str = ""):
    """Evict ComfyUI models before loading the LLM (loopback hosts only)."""
    if not release_enabled():
        return
    if not is_local_server(server_url):
        return
    log_vram("before-llm-load", note)
    unload_comfy_models(note or "prepare-for-llm")
    log_vram("after-comfy-unload", note)


def release_after_llm(slot, server_url, api_key, note=""):
    """Unload the just-used LM Studio model and confirm VRAM is freed."""
    if slot:
        logger.info("[VRAM] Releasing LM Studio model for %s (slot=%s)", server_url, slot)
    else:
        logger.info("[VRAM] Releasing LM Studio model for %s", server_url)
    log_vram("before-llm-release", note)
    try:
        lm_http.release_model(server_url, api_key, slot)
        lm_http.wait_until_unloaded(server_url, api_key)
    except Exception as e:  # noqa: BLE001 — release is an optimisation, never fatal
        logger.warning("[VRAM] LM Studio release failed (%s): %s", note or server_url, e)
    log_vram("after-llm-release", note)


def release_before_sample():
    """Release every seen LM Studio server before ComfyUI samples the diffusion model.

    Does zero network I/O when no server has been seen in this process (so
    sampler-only users are never probed), and skips any server the user pinned
    keep-loaded via the node widget / env var."""
    servers = lm_http.seen_servers()
    if not servers:
        return
    keep = set(lm_http.keep_loaded_servers())
    for server_url in servers:
        if server_url in keep:
            logger.info("[VRAM] Skipping release of %s (kept loaded)", server_url)
            continue
        release_after_llm(None, server_url, "", "before-sample")
