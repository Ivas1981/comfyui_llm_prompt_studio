"""HTTP client for LM Studio / OpenAI-compatible servers, model cache and SSRF guard."""
import hashlib
import ipaddress
import json
import logging
import os
import re
import time
from collections import OrderedDict
from typing import Any, Dict, Optional
from urllib.parse import quote, urlparse

import requests


from .constants import PLACEHOLDER, PLACEHOLDER_EMPTY
from .debug import debug_active, log, log_http_request, log_http_response

logger = logging.getLogger("llm_prompt_studio")

# Nothing is logged when DEBUG_LEVEL is "OFF"; importing debug here is cheap and guarded.
__all__ = [
    "ALLOW_PUBLIC_SERVER_URLS",
    "DEFAULT_SERVER",
    "validate_server_url",
    "fetch_models",
    "cache_models",
    "get_cached_models",
    "cached_model_list",
    "remember_model",
    "looks_like_vision",
    "model_supports_vision",
    "resolve_vision",
    "model_supports_reasoning",
    "map_reasoning_level",
    "unload_all_loaded",
    "maybe_unload_old",
    "load_model",
    "ensure_model_loaded",
    "chat_completion",
    "server_status",
    "seen_servers",
    "keep_loaded_servers",
    "mark_keep_loaded",
    "release_model",
    "wait_until_unloaded",
]

# Server+model pairs for which a v1 load option key was rejected (e.g. `gpu_offload` on
# builds that only accept `gpuOffload`). Remembered so a subsequent load drops the key up
# front instead of round-tripping the same 400. Per-process cache; harmless to keep.
_rejected_v1_keys = {}

# Max times we strip a rejected key from the v1 load body and retry before giving up.
_MAX_REJECTED_RETRIES = 8

# Read allow-public flag from environment for runtime configuration.
ALLOW_PUBLIC_SERVER_URLS = os.getenv("LLM_PROMPT_STUDIO_ALLOW_PUBLIC", "False").lower() in ("1", "true", "yes")

# Opt-in global LLM-response cache. OFF by default; enable with
# LLM_PROMPT_STUDIO_LLM_CACHE=true. When on, identical chat_completion requests (same server,
# model, messages and sampling params, excluding api_key) return the cached text instead of
# hitting the server again. It sits BELOW the per-node `reuse_last_prompt` cache in writer.py:
# both coexist. Bounded LRU (~256 entries) to bound memory.
_LLM_CACHE_ENABLED = os.getenv("LLM_PROMPT_STUDIO_LLM_CACHE", "false").lower() in ("1", "true", "yes")
_LLM_CACHE_MAX = 256
_llm_response_cache = OrderedDict()

# cache "last loaded model" to unload the old one on switch: {slot: fingerprint}
_last_loaded = {}
# The single model currently resident on a given server (managed by ensure_model_loaded /
# unload_all_loaded within this process). LM Studio holds one model at a time for our use, so
# this is the authoritative "what is actually loaded" truth that lets us unload models left
# resident by a DIFFERENT node/slot before loading the selected one.
_server_loaded = {}
# model instance ids returned by the v1 load endpoint, keyed for precise unload: {(server, model): id}
_model_instances = {}
# Servers this process has interacted with (for VRAM release). Keyed by NORMALIZED server
# URL (not disk-backed like _static_keys, so existing installs never probe localhost on a
# sample). Cleared on release so a stale server is not probed after the model is gone.
_seen_servers = set()
# Servers the user asked to keep loaded (release_vram_after_run=False / env var) — release
# logic must skip these.
_keep_loaded = set()
# cache of model lists, KEYED BY NORMALIZED SERVER URL (api_key is intentionally NOT part
# of the key or the disk cache: model lists don't depend on it, and we must not persist
# secrets in plaintext). {server_url: (models, timestamp)}
_model_cache = {}
CACHE_TTL = 60  # seconds (was 10)
_MODEL_CACHE_MAX = 32

DEFAULT_SERVER = "http://localhost:1234/v1"
# Server URLs whose list is disk-backed and must NOT be auto-refetched; "Refresh models"
# is the source of truth. This keeps INPUT_TYPES non-blocking while still letting a saved
# workflow load with its previously selected model already present in the combo.
_static_keys = set()


def _normalize_server(url: str) -> str:
    """Canonicalize a server URL for cache keys (drop trailing slash, default to localhost)."""
    u = (url or "").strip()
    if not u:
        u = DEFAULT_SERVER
    return u.rstrip("/")


def _model_cache_path():
    # Cache lives in the package directory: always resolvable at import and at runtime,
    # independent of ComfyUI's output-directory resolution (which can differ between
    # import time and request time and break the read after a restart).
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "llm_prompt_studio_models_cache.json")


def _read_disk_cache():
    """Return the per-server model cache from disk: {normalized_server_url: [model_ids]}.

    Migrates the legacy flat-list cache (a plain JSON array, or the old copy that lived in
    the ComfyUI output dir) into the DEFAULT_SERVER entry so future reads are consistent."""
    candidates = [_model_cache_path()]
    # Backwards-compat: also read the old cache that previously lived in the output dir.
    try:
        import folder_paths
        candidates.append(os.path.join(folder_paths.get_output_directory(),
                                        "llm_prompt_studio_models_cache.json"))
    except ImportError:
        # folder_paths not available outside ComfyUI — it's fine, we skip the old path.
        pass
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            servers = {str(k): list(v) for k, v in data.items()
                       if isinstance(v, list) and v}
            if servers:
                # Promote a legacy output-dir list cache into the package-location dict.
                if path != _model_cache_path() and DEFAULT_SERVER not in servers:
                    _persist_models(DEFAULT_SERVER, servers.get(DEFAULT_SERVER, []))
                return servers
        if isinstance(data, list) and data and data not in ([PLACEHOLDER], [PLACEHOLDER_EMPTY]):
            # Old flat-list cache: treat it as the default server's list and migrate.
            if path != _model_cache_path():
                _persist_models(DEFAULT_SERVER, list(data))
            return {DEFAULT_SERVER: list(data)}
    return {}


def _persist_models(server_url: str, models):
    if not models or models in ([PLACEHOLDER], [PLACEHOLDER_EMPTY]):
        return
    server_url = _normalize_server(server_url)
    cache = _read_disk_cache()
    existing = cache.get(server_url) or []
    merged = list(dict.fromkeys(list(existing) + list(models)))
    cache[server_url] = merged
    try:
        with open(_model_cache_path(), "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except OSError:
        pass


def remember_model(model: str, server_url: str = DEFAULT_SERVER):
    """Persist a single model id so it stays valid/selectable offline.

    Called whenever a node actually uses a model, so a workflow reloaded after the
    server went away still validates its previously-run model instead of ComfyUI
    rejecting it with 'Value not in list'. The model is recorded under its server so it
    never leaks into a different server's combo."""
    if not model or model.startswith("—"):
        return
    _persist_models(server_url, [model])


def _load_disk_cache():
    disk = _read_disk_cache()
    for srv, models in disk.items():
        _store_model_cache(srv, list(models))
        _static_keys.add(srv)


def cached_model_list(server_url: str = DEFAULT_SERVER):
    """Return the persisted model list for `server_url` (non-blocking), seeding the
    in-memory cache. Falls back to the default-server list when this server has none yet."""
    disk = _read_disk_cache()
    srv = _normalize_server(server_url)
    models = disk.get(srv)
    if models:
        _store_model_cache(srv, list(models))
        _static_keys.add(srv)
        return list(models)
    if srv != DEFAULT_SERVER and disk.get(DEFAULT_SERVER):
        return list(disk[DEFAULT_SERVER])
    return None


VISION_NAME_HINTS = (
    # generic substrings found in vision-capable model ids
    "vl", "vlm", "vision", "visual",
    "llava", "llama-3.2-vision", "llama-3.1-vision",
    "qwen2-vl", "qwen2.5-vl", "qwen-vl", "qvq",
    "gemma-3", "gemma3",
    "pixtral", "mistral-vl",
    "minicpm-v", "minicpmo", "minicpm-v2",
    "internvl", "deepseek-vl", "cogvlm", "cogagent",
    "idefics", "smolvlm", "molmo", "aya-vision", "xcomposer", "glm-4v",
    "mantis", "ovis", "janus", "embo", "moondream",
    "phi-3-vision", "phi-3.5-vision", "florence",
)


def validate_server_url(url: str) -> str:
    """Basic SSRF guard: allow http(s) and, by default, only local/private hosts."""
    parsed = urlparse(str(url).strip())
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"server_url scheme '{parsed.scheme}' is not allowed (use http or https)")
    host = parsed.hostname
    if not host:
        raise ValueError("server_url has no host")
    if ALLOW_PUBLIC_SERVER_URLS:
        return url
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        return url
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError(
            f"server_url host '{host}' is not a local address. "
            "Set LLM_PROMPT_STUDIO_ALLOW_PUBLIC=True in the environment to allow remote hosts.")
    if ip.is_loopback or ip.is_private or ip.is_link_local:
        return url
    raise ValueError(
        f"server_url host '{host}' is public. "
        "Set LLM_PROMPT_STUDIO_ALLOW_PUBLIC=True in the environment to allow it."
    )


def _parse_native_models(data) -> list:
    """Extract model identifiers from a native ``GET /api/v1/models`` envelope.

    Handles both the documented ``{data: [...]}`` shape and a legacy ``{models: [...]}``
    shape. Per entry it prefers ``key`` (the canonical native id) and falls back to ``id``,
    matching the resolution logic in :func:`_model_matches` so listed ids line up with the
    ids used for chat/load."""
    if not isinstance(data, dict):
        return []
    models = data.get("data")
    if not isinstance(models, list):
        models = data.get("models")
    if not isinstance(models, list):
        return []
    ids = []
    for m in models:
        if not isinstance(m, dict):
            continue
        # Skip embedding models: they are not chat models and must not appear in the
        # model combo (LM Studio's native list mixes them in with a "type": "embedding").
        if m.get("type") == "embedding":
            continue
        mid = m.get("key") or m.get("id")
        if mid:
            ids.append(str(mid))
    return ids


def fetch_models(server_url: str, api_key: str = "", timeout: int = 5) -> list:
    server_url = validate_server_url(server_url)
    # Record that this process talked to this server, so VRAM release can find it
    # without probing localhost on every sample. Normalized (not raw) spelling.
    _seen_servers.add(_normalize_server(server_url))
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    base = _server_root(server_url)
    # Prefer LM Studio's native model list (LMStudioAPI.md §15: "Lists models via native
    # GET /api/v1/models"). It returns capability-rich entries keyed by `key`, consistent
    # with the identifiers used by chat/load and _model_matches.
    try:
        resp = requests.get(f"{base}/api/v1/models", headers=headers, timeout=timeout)
        resp.raise_for_status()
        models = _parse_native_models(resp.json())
        if models:
            return models
    except (requests.RequestException, ValueError):
        logger.debug("Native /api/v1/models unavailable for %s; falling back to /v1/models",
                     server_url)
    # Fallback for pre-0.4.0 servers that only expose the OpenAI-compatible list.
    try:
        resp = requests.get(f"{server_url.rstrip('/')}/models", headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Failed to fetch models from %s: %s", server_url, e)
        raise
    try:
        data = resp.json()
    except ValueError:
        snippet = (resp.text or '')[:1000]
        logger.error("Invalid JSON returned by %s: %s", server_url, snippet)
        raise RuntimeError(f"Invalid JSON from model server {server_url}: {snippet}")
    return [m.get("id") for m in data.get("data", []) if isinstance(m, dict) and m.get("id")
            and m.get("type") != "embedding"]


def _store_model_cache(server_url: str, models):
    _model_cache[_normalize_server(server_url)] = (models, time.time())
    if len(_model_cache) > _MODEL_CACHE_MAX:
        _model_cache.pop(next(iter(_model_cache)))  # drop oldest


_load_disk_cache()


def cache_models(server_url: str, api_key: str, models: list):
    if models:
        srv = _normalize_server(server_url)
        # Persist under the REFRESHED server (not the default) so each server's combo stays
        # scoped to its own models — fixing the cross-server model mix-up where a Refresh of
        # server B would overwrite server A's combo. The api_key is deliberately not stored.
        _store_model_cache(srv, list(models))
        _persist_models(srv, models)
        _static_keys.add(srv)


def get_cached_models(server_url=DEFAULT_SERVER, api_key="", allow_fetch=True, timeout=5):
    srv = _normalize_server(server_url)
    cached = _model_cache.get(srv)
    # Fresh cache within TTL — return it
    if cached and time.time() - cached[1] < CACHE_TTL:
        return list(cached[0])
    # Disk-backed / known list: never auto-refetch; Refresh is the source of truth.
    # This lets INPUT_TYPES return the previously known models without hitting the network.
    if srv in _static_keys:
        if cached:
            return list(cached[0])
        # In-memory entry evicted (LRU) but the list is still "known": re-read disk for
        # this server instead of falling back to the unavailable placeholder.
        disk = _read_disk_cache().get(srv)
        return list(disk) if disk else [PLACEHOLDER]
    # Expired cache — try to re-fetch, fall back to stale only on error
    if cached:
        if not allow_fetch:
            return list(cached[0])
        try:
            models = fetch_models(server_url, api_key, timeout=timeout) or [PLACEHOLDER_EMPTY]
            _store_model_cache(srv, models)
            _persist_models(srv, models)
            _static_keys.add(srv)
            return models
        except Exception:
            return list(cached[0])
    # No cache at all — fetch or signal unavailable
    if allow_fetch:
        try:
            models = fetch_models(server_url, api_key, timeout=timeout) or [PLACEHOLDER_EMPTY]
            _store_model_cache(srv, models)
            _persist_models(srv, models)
            _static_keys.add(srv)
            return models
        except Exception:
            # Negative-cache the failure so we don't block on every INPUT_TYPES call while
            # the server is down; it expires with CACHE_TTL and a manual Refresh still works.
            _store_model_cache(srv, [PLACEHOLDER])
            return [PLACEHOLDER]
    return [PLACEHOLDER]


def looks_like_vision(model_id: str) -> bool:
    low = model_id.lower()
    return any(h in low for h in VISION_NAME_HINTS)


def _model_matches(entry: dict, model_id: str) -> bool:
    """Case-insensitive match of a v1 /api/v1/models entry against `model_id`.

    Tries the entry's `key`, `id`, any `loaded_instances[].id`, and `variants`."""
    low = model_id.lower()
    if not low:
        return False
    candidates = [entry.get("key"), entry.get("id")]
    for inst in entry.get("loaded_instances") or []:
        if isinstance(inst, dict):
            candidates.append(inst.get("id"))
        else:
            candidates.append(inst)
    candidates.extend(entry.get("variants") or [])
    return any(str(c).lower() == low for c in candidates if c)


def model_supports_vision(server_url: str, api_key: str, model_id: str,
                          timeout: int = 5) -> Optional[bool]:
    """Best-effort authoritative vision capability from LM Studio's native endpoint.

    Queries ``GET {server}/api/v1/models`` and returns
    ``entry["capabilities"]["vision"]`` (bool) when the model is present, else ``None``.

    Never raises: network errors, parse errors, missing ``/api/v1/models`` and 404 all
    return ``None`` so callers fall back to the name heuristic without regressing on
    servers that lack the native endpoint."""
    try:
        base = _server_root(validate_server_url(server_url))
        resp = requests.get(f"{base}/api/v1/models",
                            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                            timeout=timeout)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    models = data.get("data") if isinstance(data, dict) else None
    if models is None:
        models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return None
    for entry in models:
        if not isinstance(entry, dict):
            continue
        if _model_matches(entry, model_id):
            caps = entry.get("capabilities") or {}
            vision = caps.get("vision")
            if isinstance(vision, bool):
                return vision
            # capabilities present but vision key absent -> unknown, keep looking
            if caps:
                return None
    return None


def resolve_vision(server_url: str, api_key: str, model_id: str) -> bool:
    """Authoritative vision decision: the server probe wins in BOTH directions.

    A server-reported bool (true or false) overrides the name heuristic; only when the
    probe is unavailable (``None`` — old server, network error, 404) do we fall back to
    ``looks_like_vision`` so behavior never regresses."""
    server = model_supports_vision(server_url, api_key, model_id)
    if server is not None:
        return server
    return looks_like_vision(model_id)


# Cache of reasoning capability probes: {(server, model): allowed_options_or_None}.
# allowed_options is the list of reasoning levels a model accepts (e.g. ["off","on"]),
# or None when the model exposes no reasoning configuration at all (template thinking).
_reasoning_cap_cache = {}


def model_supports_reasoning(server_url: str, api_key: str, model_id: str,
                             timeout: int = 5):
    """Best-effort reasoning capability probe from LM Studio's native endpoint.

    Queries ``GET {server}/api/v1/models`` and reads
    ``entry["capabilities"]["reasoning"]`` for ``model_id``. Returns one of:

    * a list of allowed reasoning option strings (e.g. ``["off", "on"]``),
    * ``[]`` when reasoning is exposed but allows no extra levels,
    * ``None`` when the model exposes no reasoning configuration (the server may
      still think via its template — we simply omit the param).

    Never raises: network errors, parse errors, missing ``/api/v1/models`` and 404
    all return ``None`` so callers fall back to sending nothing rather than guessing."""
    key = (server_url, model_id)
    if key in _reasoning_cap_cache:
        return _reasoning_cap_cache[key]
    result = None
    try:
        base = _server_root(validate_server_url(server_url))
        resp = requests.get(f"{base}/api/v1/models",
                            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                            timeout=timeout)
    except requests.RequestException:
        _reasoning_cap_cache[key] = None
        return None
    if resp.status_code != 200:
        _reasoning_cap_cache[key] = None
        return None
    try:
        data = resp.json()
    except ValueError:
        _reasoning_cap_cache[key] = None
        return None
    models = data.get("data") if isinstance(data, dict) else None
    if models is None:
        models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        _reasoning_cap_cache[key] = None
        return None
    for entry in models:
        if not isinstance(entry, dict):
            continue
        if not _model_matches(entry, model_id):
            continue
        caps = entry.get("capabilities") or {}
        reasoning = caps.get("reasoning")
        if reasoning is None:
            # capabilities present but no reasoning key -> unknown for this model.
            result = None
            break
        if isinstance(reasoning, dict):
            allowed = reasoning.get("allowed_options")
            if isinstance(allowed, list):
                result = [str(a) for a in allowed]
            else:
                # exposed but no explicit list -> treat as configurable, default off/on.
                result = ["off", "on"]
        elif reasoning is True:
            result = ["off", "on"]
        else:
            result = []
        break
    _reasoning_cap_cache[key] = result
    return result


def map_reasoning_level(level, allowed_options):
    """Map the widget's reasoning level to a value the model accepts.

    Widget levels are ``off`` / ``low`` / ``medium`` / ``high`` / ``on``. ``allowed_options``
    is the list from :func:`model_supports_reasoning` (or ``None`` when the model exposes no
    reasoning configuration). Returns the allowed string to send, or ``None`` to OMIT the
    parameter entirely (no reasoning config, or only ``off`` is available and we want off)."""
    allowed = allowed_options
    if not isinstance(allowed, list) or not allowed:
        # Model exposes no reasoning configuration: never send the param.
        return None
    allowed_lc = [str(a).lower() for a in allowed]
    if "off" not in allowed_lc and "on" not in allowed_lc:
        # Listed values are neither off nor on (unexpected) — safest to omit.
        return None
    lvl = str(level or "off").lower()
    # off is always acceptable when present.
    off = "off" if "off" in allowed_lc else None
    on = "on" if "on" in allowed_lc else None
    if lvl == "off":
        return off if off is not None else on
    if lvl in ("on",) or lvl in ("low", "medium", "high"):
        # Prefer the strongest allowed level (high if listed, else on) so the user's
        # "think harder" intent is honored on models that support gradations.
        if "high" in allowed_lc:
            return "high" if "high" in allowed else on
        if on is not None:
            return on
        return off
    # Unknown level -> omit (server decides), but if only off is allowed return off.
    return off if off is not None else on


def maybe_unload_old(slot: str, server_url: str, new_model: str):
    """Unload the previously-loaded model before a new one loads.

    The unload targets the exact v1 ``instance_id`` captured from a successful load (in
    ``_model_instances``). 404 / already-gone ids are treated as idempotent success. When no
    instance id is known we probe the v1 route by model name; only if that route is missing
    (404 / connection error) do we fall back to the legacy ``/api/v0/models/unload`` — a
    modern server answers 200 "Unexpected endpoint", which we do NOT treat as a failure.

    ``_last_loaded[slot]`` may hold a fingerprint tuple (see ``ensure_model_loaded``); the
    model name is taken from its first element. This function only performs the unload — the
    caller is responsible for recording the new load state."""
    old = _last_loaded.get(slot)
    old_model = old[0] if isinstance(old, tuple) else old
    if (not old_model or old_model == new_model
            or str(old_model).startswith("—")):
        return
    try:
        base = _server_root(server_url)
        instance_id = _model_instances.get((server_url, old_model))
        if instance_id:
            # v1 unload of the exact instance; 404 (already gone) is success.
            resp = requests.post(f"{base}/api/v1/models/unload",
                                 json={"instance_id": instance_id}, timeout=5)
            if resp.status_code < 400 or resp.status_code == 404:
                logger.debug("Unloaded previous model '%s' (instance %s)",
                             old_model, instance_id)
            else:
                logger.debug("Unload of '%s' returned HTTP %s: %s", old_model,
                             resp.status_code, (resp.text or "")[:200])
        else:
            # No stored instance id: try the v1 route by model name (old servers handle it).
            # A 404 means the v1 route is missing -> fall back to legacy v0. A 2xx
            # "Unexpected endpoint" is an old server's no-op success, not an error.
            resp = None
            try:
                resp = requests.post(f"{base}/api/v1/models/unload",
                                     json={"model": old_model}, timeout=5)
            except requests.RequestException:
                resp = None
            if resp is None or resp.status_code == 404:
                requests.post(f"{base}/api/v0/models/unload",
                              json={"model": old_model}, timeout=5)
            elif resp.status_code >= 400:
                logger.debug("v1 unload of '%s' failed (HTTP %s): %s", old_model,
                             resp.status_code, (resp.text or "")[:200])
    except requests.RequestException as e:
        logger.debug("Could not unload previous model '%s': %s", old_model, e)
    # Record the new model name so a same-model check (and the legacy string-state path)
    # behaves correctly. ensure_model_loaded overwrites this with the full fingerprint on a
    # successful load, so the config-fingerprint compare stays authoritative.
    _last_loaded[slot] = new_model


def unload_all_loaded(server_url: str, api_key: str = "",
                      except_model: Optional[str] = None):
    """Unload every model currently resident on the server so the next load starts clean.

    This is the "check for and unload already-loaded models before loading the selected one"
    step: it enumerates loaded instances via the native ``GET /api/v1/models`` (each entry's
    ``loaded_instances[].id``) and unloads each with ``POST /api/v1/models/unload`` using the
    exact ``instance_id``. ``except_model`` (when given) is spared so a model that is already
    the intended one is not needlessly evicted. 404 / already-gone ids are idempotent success.

    Never raises: list/parse/network errors are logged and ignored so the subsequent load can
    still proceed. Used by ``ensure_model_loaded`` to evict models left resident by other
    nodes/slots before the selected model is loaded."""
    try:
        base = _server_root(validate_server_url(server_url))
    except ValueError as e:
        logger.debug("unload_all_loaded: invalid server %s: %s", server_url, e)
        return
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = requests.get(f"{base}/api/v1/models", headers=headers, timeout=5)
    except requests.RequestException as e:
        logger.debug("unload_all_loaded: could not list models on %s: %s", server_url, e)
        return
    if resp.status_code != 200:
        return
    try:
        data = resp.json()
    except ValueError:
        return
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list):
        models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return
    except_lc = except_model.lower() if except_model else None
    for entry in models:
        if not isinstance(entry, dict):
            continue
        # Skip the model we intend to keep loaded (matched by native key or id).
        if except_lc:
            key = entry.get("key") or entry.get("id")
            if key and str(key).lower() == except_lc:
                continue
        for inst in (entry.get("loaded_instances") or []):
            inst_id = inst.get("id") if isinstance(inst, dict) else inst
            if not inst_id:
                continue
            try:
                r = requests.post(f"{base}/api/v1/models/unload",
                                  json={"instance_id": inst_id},
                                  headers=headers, timeout=5)
                if r.status_code < 400 or r.status_code == 404:
                    logger.debug("Unloaded resident model instance %s", inst_id)
                else:
                    logger.debug("unload_all_loaded: unload of %s returned HTTP %s: %s",
                                 inst_id, r.status_code, (r.text or "")[:200])
            except requests.RequestException as e:
                logger.debug("unload_all_loaded: could not unload instance %s: %s",
                             inst_id, e)


# HTTP status codes worth retrying: transient server hiccups / rate limits.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _is_transient(status, exc):
    if exc is not None:
        return True  # connection error / timeout — always worth a retry
    return status in _RETRYABLE_STATUS


def _server_root(server_url: str) -> str:
    """Strip a trailing ``/v1`` so we can build native ``/api/v1/*`` endpoints from a
    server_url that already ends in ``/v1`` (the ComfyUI default)."""
    base = server_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    return base


def _decode_error(resp):
    """Return the `error` object of a JSON error response, or None."""
    if resp is None:
        return None
    try:
        data = resp.json()
    except Exception:
        return None
    if isinstance(data, dict):
        return data.get("error")
    return None


def _is_unrecognized_keys(resp) -> bool:
    """True when an LM Studio error rejects one or more request body keys by name
    (code `unrecognized_keys` and friends). Used to make loads resilient to per-server
    schema differences: a rejected optional key is dropped and the load retried."""
    err = _decode_error(resp)
    if not isinstance(err, dict):
        return False
    code = err.get("code")
    if code in ("unrecognized_keys", "unknown_keys", "invalid_keys",
                "unrecognized_key", "unknown_key"):
        return True
    msg = str(err.get("message", "") or "").lower()
    return "unrecognized" in msg or "unknown key" in msg


def _parse_rejected_keys(resp, body_keys) -> set:
    """Extract the rejected body-key names from an `unrecognized_keys` 400 response.

    Tries an explicit `keys`/`key` list in the error object first, then falls back to
    quoted identifiers in the message and any body-key name that appears verbatim."""
    err = _decode_error(resp)
    keys: set = set()
    if isinstance(err, dict):
        listed = err.get("keys") or err.get("unrecognized_keys") or err.get("key")
        if isinstance(listed, str):
            listed = [listed]
        if isinstance(listed, list):
            for k in listed:
                if isinstance(k, str):
                    keys.add(k)
    if not keys and isinstance(err, dict):
        msg = str(err.get("message", "") or "")
        for m in re.findall(r"['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]", msg):
            keys.add(m)
        for k in body_keys:
            if re.search(r"\b" + re.escape(k) + r"\b", msg):
                keys.add(k)
    return keys


def _try_post(path: str, body: dict, headers: dict, timeout: int, retries: int, backoff: float):
    """POST JSON and return (ok, last_error, response-or-None).

    Retries transient failures with exponential backoff; non-retryable client errors
    (e.g. 400/401/404) stop immediately so the caller can try the next endpoint."""
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(path, json=body, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            last_err = str(e)
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
                continue
            return False, last_err, None
        if resp.status_code < 400:
            return True, None, resp
        last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
        if _is_transient(resp.status_code, None) and attempt < retries - 1:
            time.sleep(backoff * (2 ** attempt))
            continue
        return False, last_err, resp
    return False, last_err, None


def load_model(server_url: str, api_key: str, model: str,
               context_length: int = 8192, gpu_offload: float = 1.0,
               timeout: int = 300, retries: int = 3, backoff: float = 1.0,
               flash_attention: Optional[bool] = None,
               offload_kv_cache_to_gpu: Optional[bool] = None,
               eval_batch_size: Optional[int] = None,
               num_experts: Optional[int] = None,
               echo_load_config: bool = True) -> bool:
    """Load a model on the LM Studio server with the given context length / GPU offload.

    Tries LM Studio's native ``/api/v1/models/load`` endpoint first (it supports flash
    attention, KV-cache GPU offload, eval batch size and MoE experts), then falls back to the
    legacy ``/v1/models/{id}/load`` and ``/api/v0/models/{id}/load`` endpoints which honor
    ``gpuOffload`` but ignore the newer options. Returns True if any endpoint confirmed the
    load (HTTP < 400).

    Transient failures (HTTP 429/500/502/503/504, connection errors, timeouts) are retried
    with exponential backoff per endpoint."""
    server_url = validate_server_url(server_url)
    base = _server_root(server_url)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    identifier = quote(model, safe="")

    # Native v1 load body. All optional v1 keys are snake_case per the LM Studio docs;
    # we send the full set and let the resilient retry below drop any key a particular
    # server rejects (e.g. `gpu_offload` on builds that only accept `gpuOffload`, or
    # servers without flash-attention / KV-offload support). New options are only included
    # when set. Keys previously rejected for this server+model are dropped up front so we
    # don't repeat a known-bad request.
    rejected = set(_rejected_v1_keys.get((server_url, model), set()))
    v1_body = {"model": model, "context_length": int(context_length), "gpu_offload": gpu_offload}
    if flash_attention is not None:
        v1_body["flash_attention"] = bool(flash_attention)
    if offload_kv_cache_to_gpu is not None:
        v1_body["offload_kv_cache_to_gpu"] = bool(offload_kv_cache_to_gpu)
    if eval_batch_size is not None:
        v1_body["eval_batch_size"] = int(eval_batch_size)
    if num_experts is not None:
        v1_body["num_experts"] = int(num_experts)
    if echo_load_config:
        v1_body["echo_load_config"] = True
    v1_body = {k: v for k, v in v1_body.items() if k not in rejected}

    # Legacy body (camelCase) used by the older endpoints; ignores the new options.
    legacy_body = {"contextLength": int(context_length), "gpuOffload": gpu_offload, "seed": -1}

    # 1) Native v1 endpoint (best effort — supports flash attention / kv offload).
    ok, err, resp = _try_post(f"{base}/api/v1/models/load", v1_body, headers,
                              timeout, retries, backoff)
    if ok:
        _record_load_config(server_url, model, resp)
        return True

    # Resilient retry: some LM Studio builds reject optional parameters by name with a
    # 400 `unrecognized_keys` error (e.g. `gpu_offload`). Parse the rejected key names,
    # drop them from the body, and retry — looping until nothing is rejected or the body
    # stabilizes. This makes loads succeed on servers that don't support a given key
    # instead of hard-failing the whole load.
    if resp is not None and resp.status_code == 400 and _is_unrecognized_keys(resp):
        body = dict(v1_body)
        seen = set(rejected)
        for _ in range(_MAX_REJECTED_RETRIES):
            dropped = set()
            for k in _parse_rejected_keys(resp, set(body.keys())):
                if k in body:
                    body.pop(k, None)
                    dropped.add(k)
                seen.add(k)
            _rejected_v1_keys.setdefault((server_url, model), set()).update(seen)
            if not dropped:
                break
            ok, err, resp = _try_post(f"{base}/api/v1/models/load", body, headers,
                                      timeout, retries, backoff)
            if ok:
                _record_load_config(server_url, model, resp)
                return True
            if not (resp is not None and resp.status_code == 400
                    and _is_unrecognized_keys(resp)):
                break

    # A modern server's legacy route is a no-op that answers 200 ("Unexpected endpoint"),
    # which would mask the real v1 rejection. Only fall through to the legacy endpoints on
    # a connection error (resp is None) or 404 (an old server without the v1 load route).
    v1_rejected = resp is not None and 400 <= resp.status_code < 500 and resp.status_code != 404
    if v1_rejected:
        logger.warning(
            "LM Studio rejected loading model '%s' (HTTP %s): %s",
            model, resp.status_code, (resp.text or "")[:300])
        return False

    # 2) Legacy endpoints (honor gpuOffload; the new options are silently ignored).
    for path in (f"{base}/v1/models/{identifier}/load",
                 f"{base}/api/v0/models/{identifier}/load"):
        ok, err, resp = _try_post(path, legacy_body, headers, timeout, retries, backoff)
        if ok:
            logger.info("Loaded model '%s' (context=%s, gpuOffload=%s) in %.1fs",
                        model, context_length, gpu_offload, 0.0)
            return True

    logger.warning("LM Studio could not load model '%s': %s", model, err)
    return False


def _record_load_config(server_url: str, model: str, resp):
    """Persist the instance id / echoed load_config from a successful v1 load (debug only)."""
    try:
        data = resp.json() if resp is not None else {}
    except Exception:
        return
    if not isinstance(data, dict):
        return
    instance_id = data.get("instance_id")
    load_config = data.get("load_config", {})
    if instance_id:
        _model_instances[(server_url, model)] = instance_id
    if debug_active():
        log("DEBUG", "MODEL_LOAD", model,
            {"instance_id": instance_id, "load_config": load_config})


def ensure_model_loaded(slot: str, server_url: str, api_key: str, model: str,
                        context_length: int = 8192, gpu_offload: float = 1.0,
                        flash_attention: Optional[bool] = None,
                        offload_kv_cache_to_gpu: Optional[bool] = None,
                        eval_batch_size: Optional[int] = None,
                        num_experts: Optional[int] = None,
                        echo_load_config: bool = True):
    """Make sure `model` is the one loaded on the server before we call it.

    Before loading, it unloads *every* model currently resident on the server (including ones
    left loaded by a different node/slot) so VRAM holds only the selected model — the requested
    behavior of "check for and unload already-loaded models before loading the selected one".

    It skips the unload+reload when the requested model is already loaded with the identical
    config (tracked server-scoped, since LM Studio serves one model at a time). A changed
    config (context_length / gpu_offload / flash-attention / KV-offload / batch-size / experts)
    forces a reload. A failed load is not recorded as "loaded", so the next run will retry."""
    if not model or model.startswith("—"):
        return
    # Record that this process loaded from this server (normalized spelling), so the
    # VRAM-release logic can find it later without probing localhost on every sample.
    _seen_servers.add(_normalize_server(server_url))
    # Remember this model so a saved workflow that uses it stays valid (and selectable)
    # even when the server is offline later and the combo would otherwise be the placeholder.
    remember_model(model, server_url)
    # Fingerprint the requested load config so a changed context_length / gpu_offload /
    # flash-attention / KV-offload / batch-size / experts forces a reload even when the same
    # model is currently loaded.
    fingerprint = (model, context_length, gpu_offload, flash_attention,
                   offload_kv_cache_to_gpu, eval_batch_size, num_experts)
    # Server-scoped truth: if the requested model is already resident with the exact config,
    # there is nothing to do — avoid a needless unload+reload.
    if _server_loaded.get(server_url) == fingerprint:
        _last_loaded[slot] = fingerprint
        return
    # Evict every resident model (other nodes' included) before loading the selected one.
    unload_all_loaded(server_url, api_key)
    if not load_model(server_url, api_key, model, context_length, gpu_offload,
                      flash_attention=flash_attention,
                      offload_kv_cache_to_gpu=offload_kv_cache_to_gpu,
                      eval_batch_size=eval_batch_size,
                      num_experts=num_experts,
                      echo_load_config=echo_load_config):
        _server_loaded[server_url] = None  # load failed: allow a retry on the next run
        _last_loaded[slot] = None
        logger.warning(
            "Model '%s' for slot '%s' was not loaded. The next LLM call will fail "
            "until the model is available (check that LM Studio is running and the "
            "model id is correct).", model, slot)
    else:
        _server_loaded[server_url] = fingerprint  # record the exact loaded config
        _last_loaded[slot] = fingerprint


def server_status(server_url: str, api_key: str = "", timeout: int = 5) -> dict:
    """Describe the local LM Studio server's availability and what is loaded.

    Returns ``{"reachable": bool, "loaded_models": [ids], "error": str|None}``. Never
    raises: any failure (unreachable host, timeout, non-200, bad JSON) yields
    ``reachable=False`` with the reason in ``error`` so the front-end can show a clear
    "server down" indicator without special-casing exceptions."""
    try:
        base = _server_root(validate_server_url(server_url))
    except ValueError as e:
        return {"reachable": False, "loaded_models": [], "error": str(e)}
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = requests.get(f"{base}/api/v1/models", headers=headers, timeout=timeout)
    except requests.RequestException as e:
        return {"reachable": False, "loaded_models": [], "error": str(e)}
    if resp.status_code != 200:
        return {"reachable": False, "loaded_models": [],
                "error": f"HTTP {resp.status_code}"}
    try:
        data = resp.json()
    except ValueError:
        return {"reachable": False, "loaded_models": [], "error": "invalid JSON from server"}
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list):
        models = data.get("models") if isinstance(data, dict) else None
    loaded = []
    if isinstance(models, list):
        for entry in models:
            if not isinstance(entry, dict):
                continue
            if entry.get("loaded_instances"):
                key = entry.get("key") or entry.get("id")
                if key:
                    loaded.append(str(key))
    return {"reachable": True, "loaded_models": loaded, "error": None}


# ---------------------------------------------------------------------------
# VRAM release support (consumed by vram.py / the node layer).
# ---------------------------------------------------------------------------

def seen_servers() -> set:
    """Servers this process has loaded from / listed models on (normalized URLs)."""
    return set(_seen_servers)


def keep_loaded_servers() -> set:
    """Servers the user pinned to stay loaded (release must skip these)."""
    return set(_keep_loaded)


def mark_keep_loaded(server_url: str, keep: bool):
    """Pin (or unpin) a server so release logic skips / performs its unload."""
    norm = _normalize_server(server_url)
    if keep:
        _keep_loaded.add(norm)
    else:
        _keep_loaded.discard(norm)


def release_model(server_url: str, api_key: str = "", slot=None) -> bool:
    """Unload the LM Studio model for a server and invalidate all cached load state.

    Clears ``_server_loaded`` / ``_last_loaded`` / ``_model_instances`` (both the raw
    and normalized URL spellings) so a subsequent ``ensure_model_loaded`` does NOT
    skip the reload believing the model is still resident. Never raises."""
    norm = _normalize_server(server_url)
    has_state = (
        server_url in _server_loaded or norm in _server_loaded
        or any(k for k in _last_loaded if k.startswith(norm + "::") or k == norm)
        or any(k[0] == server_url or k[0] == norm for k in _model_instances)
    )
    # Avoid any network I/O when this server was never loaded (so release-only users are
    # never probed). The unload itself is best-effort and never raises.
    if has_state:
        try:
            unload_all_loaded(server_url, api_key)
        except Exception as e:  # noqa: BLE001 — release is best-effort
            logger.debug("release_model: unload_all_loaded failed for %s: %s", server_url, e)
    # Invalidate cache so the next node reloads. Slots are ``f"{server}::writer"`` etc.,
    # so drop every _last_loaded entry whose key starts with the normalized server.
    _server_loaded.pop(server_url, None)
    _server_loaded.pop(norm, None)
    if slot is not None:
        _last_loaded.pop(slot, None)
    else:
        # Slots are built as ``f"{server_url}::writer"`` from whatever server_url string the
        # node passed (which may carry a trailing slash). Compare both the raw and the
        # normalized server spelling so a release keyed by the normalized URL still finds
        # slots left by a raw one.
        for k in list(_last_loaded):
            k_server = k.split("::", 1)[0]
            if k_server == server_url or k_server == norm or k == server_url or k == norm:
                _last_loaded.pop(k, None)
    for k in list(_model_instances):
        if _normalize_server(k[0]) == norm or k[0] == server_url:
            _model_instances.pop(k, None)
    return True


def wait_until_unloaded(server_url: str, api_key: str = "", timeout: float = 10.0,
                        interval: float = 0.25) -> bool:
    """Poll ``server_status`` until no model is loaded, or the timeout elapses.

    Returns the outcome (True = confirmed empty); never raises. LM Studio frees VRAM
    asynchronously, so a brief poll is needed before ComfyUI allocates GPU memory."""
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            try:
                status = server_status(server_url, api_key)
            except Exception:  # noqa: BLE001 — yielding failure is fine, keep polling
                status = {"loaded_models": ["?"]}
            if not status.get("loaded_models"):
                return True
            time.sleep(interval)
    except Exception:  # noqa: BLE001 — timeout / interruption: treat as "not confirmed"
        return False
    return False


def _serialize_message_content(content):
    """Convert message content that may be a list (multimodal parts) into a plain text string
    acceptable to OpenAI-style chat/completions endpoints.

    Strategy: join text parts; for image_url parts include the URL (data URL is preserved).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if not isinstance(part, dict):
                out.append(str(part))
                continue
            typ = part.get("type")
            if typ == "text":
                out.append(part.get("text", ""))
            elif typ == "image_url":
                img = part.get("image_url", {})
                url = img.get("url") if isinstance(img, dict) else None
                out.append(f"[Image: {url}]")
            else:
                out.append(str(part))
        return "\n".join([s for s in out if s])
    return str(content)


def _enrich_http_error(model: str, status: int, text: str, has_images: bool) -> str:
    """Turn a raw LM Studio HTTP error into an actionable message.

    Surfaces the two most common user-facing failures with clear guidance:
      * image/vision rejection ("does not support image inputs"), and
      * model load failure ("exited before becoming healthy").
    Falls back to the generic message otherwise. Best-effort: never raises."""
    low = (text or "").lower()
    if status >= 400 and has_images:
        if (("image" in low or "vision" in low or "visual" in low)
                and ("not support" in low or "input" in low)):
            return (
                f"Model '{model}' does not support image inputs. This node requires a "
                f"vision-capable model (Qwen2.5-VL, LLaVA, Gemma-3/4, etc.). "
                f"Pick a vision model or disable vision_check.")
    if status >= 400 and ("failed to load model" in low or "exited before becoming healthy" in low):
        return (
            f"Model '{model}' failed to load on the server (check VRAM and the model "
            f"file): {text[:300]}")
    return None


def _llm_cache_key(server_url, api_key, model, messages, temperature, max_tokens, seed,
                    reasoning, repeat_penalty, top_k, top_p, min_p, presence_penalty,
                    response_format):
    """Stable hash for a chat_completion request. ``api_key`` is intentionally excluded so the
    cache is shared across key-less local calls; every other request-shaping arg is included.
    Multimodal message parts are serialized via their data-URL string for a stable key."""
    def _norm_content(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return [(
                p.get("type"),
                p.get("text", ""),
                (p.get("image_url", {}) or {}).get("url", ""),
            ) for p in content if isinstance(p, dict)]
        return content

    payload = {
        "server_url": server_url,
        "model": model,
        "messages": [(m.get("role"), _norm_content(m.get("content", "")))
                     for m in messages if isinstance(m, dict)],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "reasoning": reasoning,
        "repeat_penalty": repeat_penalty,
        "top_k": top_k,
        "top_p": top_p,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "response_format": response_format,
    }
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _llm_cache_store(key, text):
    """Store a successful response in the bounded LRU cache (no-op when disabled/unkeyed)."""
    if not key or not _LLM_CACHE_ENABLED:
        return
    _llm_response_cache[key] = text
    _llm_response_cache.move_to_end(key)
    while len(_llm_response_cache) > _LLM_CACHE_MAX:
        _llm_response_cache.popitem(last=False)


def _llm_cache_clear():
    """Drop all cached responses (used by tests and on config change)."""
    _llm_response_cache.clear()


def chat_completion(server_url, api_key, model, messages,
                     temperature, max_tokens, timeout=600, seed=None,
                     reasoning="off", repeat_penalty=1.0,
                     top_k=None, top_p=None, min_p=None,
                     presence_penalty=None,
                     frequency_penalty=None, response_format=None) -> str:
    """LM Studio chat completion that prefers the native ``/api/v1/chat`` endpoint.

    The native endpoint (which supports ``reasoning``, ``repeat_penalty``,
    ``top_k``/``top_p``/``min_p`` and vision) is the single path for text, vision and
    reasoning. ``store=False`` keeps the server stateless. If the native call fails —
    including a vision rejection — it gracefully falls back to the OpenAI-compatible
    ``/chat/completions`` path so the node still returns text. JSON-Schema structured
    output (Phase 5, optional) would route here unconditionally."""
    server_url = validate_server_url(server_url)
    has_images = any(
        isinstance(m.get("content"), list)
        for m in messages if isinstance(m, dict))

    # --- Global LLM-response cache (opt-in) --------------------------------------
    # Identical requests (server, model, messages, sampling params; api_key excluded) hit the
    # cache instead of the network. Disabled by default; only active when _LLM_CACHE_ENABLED.
    cache_key = None
    if _LLM_CACHE_ENABLED:
        cache_key = _llm_cache_key(server_url, api_key, model, messages, temperature, max_tokens,
                                   seed, reasoning, repeat_penalty, top_k, top_p, min_p,
                                   presence_penalty, response_format)
        if cache_key in _llm_response_cache:
            logger.debug("LLM cache HIT for model '%s' (skipping network).", model)
            return _llm_response_cache[cache_key]

    # Prefer the native /api/v1/chat endpoint for ALL requests (text, vision, reasoning):
    # it is the project's base integration. A JSON-Schema `response_format` (structured
    # output) is forced onto the OpenAI /chat/completions path, which supports json_schema
    # reliably; the native path is not used for structured output so we never combine it
    # with image (vision) input. For multimodal requests we fall back to the OpenAI
    # /chat/completions path if native vision is rejected, since not every local server's
    # native vision has been confirmed end-to-end.
    prefer_native = True
    if response_format is not None:
        prefer_native = False

    if prefer_native:
        try:
            result = _chat_v1(server_url, api_key, model, messages, temperature, max_tokens,
                              timeout=timeout, seed=seed, reasoning=reasoning,
                              repeat_penalty=repeat_penalty, top_k=top_k, top_p=top_p,
                              min_p=min_p,
                              presence_penalty=presence_penalty,
                              frequency_penalty=frequency_penalty)
            _llm_cache_store(cache_key, result)
            return result
        except Exception as e:  # noqa: BLE001 — graceful fallback to OpenAI-compatible path
            logger.warning("Native /api/v1/chat failed (%s); falling back to OpenAI path.", e)
            # Vision rejection on the native path is the classic reason to fall back here.
            if has_images:
                logger.debug("Falling back to OpenAI /chat/completions for vision request.")

    # --- OpenAI-compatible path (specialized fallback / structured output) -------------
    body = {"model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens}
    if seed is not None:
        body["seed"] = seed
    # Forward the sampling params the node layer already computed (native-v1 equivalents).
    # Most OpenAI-compatible servers accept them; a few builds reject `min_p` / `reasoning`
    # / the penalty keys with a 400 `unrecognized_keys`, in which case the retry block below
    # drops the rejected key(s) so a single unsupported param never fails the whole call.
    if top_p is not None:
        body["top_p"] = top_p
    if top_k is not None:
        body["top_k"] = top_k
    if repeat_penalty is not None:
        body["repeat_penalty"] = repeat_penalty
    if min_p is not None:
        body["min_p"] = min_p
    if reasoning is not None:
        body["reasoning"] = reasoning
    if presence_penalty is not None:
        body["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        body["frequency_penalty"] = frequency_penalty
    if response_format is not None:
        body["response_format"] = response_format
    started = time.time()
    # Some LM Studio builds reject `presence_penalty` / `frequency_penalty` / `min_p` /
    # `reasoning` on the OpenAI path with a 400 `unrecognized_keys`. Drop the rejected
    # keys and retry once so a single unsupported param never fails the whole call
    # (mirrors the native v1 retry).
    optional_keys = ["presence_penalty", "frequency_penalty", "min_p", "reasoning"]
    dropped = False
    for _ in range(2):
        try:
            resp = requests.post(
                f"{server_url.rstrip('/')}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                timeout=timeout)
        except requests.RequestException as e:
            logger.error("Could not reach LM Studio (%s): %s", server_url, e)
            raise RuntimeError(f"Could not reach LM Studio ({server_url}): {e}")
        if (not dropped and resp.status_code == 400
                and _is_unrecognized_keys(resp)):
            rejected = _parse_rejected_keys(resp, set(body.keys()))
            to_drop = [k for k in rejected if k in optional_keys]
            if to_drop:
                for k in to_drop:
                    body.pop(k, None)
                dropped = True
                logger.debug("OpenAI path dropped unsupported key(s) %s; retrying.",
                             sorted(to_drop))
                continue
        break
    if resp.status_code >= 400:
        txt = resp.text or ""
        snippet = txt[:1000] if len(txt) > 1000 else txt
        logger.error("LM Studio HTTP %s for model '%s': %s", resp.status_code, model, snippet)
        enriched = _enrich_http_error(model, resp.status_code, txt, has_images)
        if enriched is not None:
            raise RuntimeError(enriched)
        raise RuntimeError(
            f"LM Studio returned HTTP {resp.status_code} for model '{model}': "
            f"{snippet}")
    # Defensive JSON handling: avoid KeyError/IndexError when server returns unexpected body
    try:
        j = resp.json()
    except ValueError:
        snippet = (resp.text or '')[:1000]
        logger.error("LM Studio returned non-JSON body for model '%s': %s", model, snippet)
        raise RuntimeError(f"LM Studio returned non-JSON response: {snippet}")
    choices = j.get("choices") if isinstance(j, dict) else None
    if not choices or not isinstance(choices, list) or not choices:
        snippet = (resp.text or '')[:1000]
        logger.error("LM Studio reply missing choices for model '%s': %s", model, snippet)
        raise RuntimeError(f"LM Studio returned unexpected response shape: {snippet}")
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not msg or not isinstance(msg, dict):
        snippet = (resp.text or '')[:1000]
        logger.error("LM Studio reply missing message for model '%s': %s", model, snippet)
        raise RuntimeError(f"LM Studio returned unexpected response shape: {snippet}")
    content = msg.get("content") or ""
    if not content.strip():
        content = msg.get("reasoning_content") or msg.get("reasoning") or ""
    last = messages[-1].get("content", "") if messages else ""
    logger.debug("LLM call to '%s' completed in %.1fs (%d chars): %s",
                 model, time.time() - started, len(content),
                 _serialize_message_content(last))

    _llm_cache_store(cache_key, content)
    return content


def _aggregate_v1_output(result: Dict) -> str:
    """Join the ``content`` of all ``message``-type items in a v1 chat ``result``/response."""
    output = result.get("output", []) if isinstance(result, dict) else []
    parts = []
    for item in output:
        if isinstance(item, dict) and item.get("type") == "message":
            parts.append(item.get("content", ""))
    return "".join(parts)


def _build_native_chat_input(messages):
    """Convert OpenAI-style ``messages`` into LM Studio's native ``/api/v1/chat`` shape.

    Returns ``(system_prompt, input_parts)`` where ``system_prompt`` is the joined text of
    all ``system``-role messages (native API takes it as a top-level string) and
    ``input_parts`` is a list of typed content parts:

    * ``{"type": "text", "content": "..."}`` for string or ``text`` message parts;
    * ``{"type": "image", "data_url": "data:image/...;base64,..."}`` for ``image_url`` parts
      (the full data URL string is passed through — LM Studio does NOT want just the base64).

    The native chat endpoint rejects OpenAI ``messages`` as ``input`` ("Invalid discriminator
    value. Expected 'text' | 'image'"), so every message must be flattened into this schema."""
    system_parts = []
    input_parts = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append(content if isinstance(content, str) else str(content))
            continue
        if isinstance(content, str):
            if content:
                input_parts.append({"type": "text", "content": content})
            continue
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    input_parts.append({"type": "text", "content": str(part)})
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    text = part.get("text", "")
                    if text:
                        input_parts.append({"type": "text", "content": text})
                elif ptype == "image_url":
                    img = part.get("image_url", {})
                    url = img.get("url") if isinstance(img, dict) else None
                    if url:
                        input_parts.append({"type": "image", "data_url": url})
                # Unknown part types are skipped (native API only knows text/image).
            continue
        if content:
            input_parts.append({"type": "text", "content": str(content)})
    system_prompt = "\n".join(p for p in system_parts if p) or None
    return system_prompt, input_parts


def _is_reasoning_rejection(text: str) -> bool:
    """True when a 400 error is specifically about an unsupported/unknown ``reasoning`` setting.

    Some models expose no reasoning configuration and reject the param with messages like
    'does not expose reasoning configuration' or 'Reasoning setting ... is not supported'."""
    low = (text or "").lower()
    return ("reasoning" in low
            and ("does not expose reasoning" in low
                 or ("not support" in low and "reasoning" in low)
                 or ("is not supported" in low and "reasoning" in low)
                 or ("invalid" in low and "reasoning" in low)))


def _post_v1_chat(url, headers, payload, timeout):
    """POST to the native ``/api/v1/chat`` endpoint, resilient to ``unrecognized_keys`` 400s.

    Mirrors :func:`load_model`: if the server rejects optional body keys (e.g. ``seed`` on
    builds that don't accept it), the rejected keys are dropped from the payload and the
    request is retried, so a single unsupported parameter no longer fails the whole call and
    forces a fallback to the OpenAI path. Other 400s are returned unchanged for the caller's
    reasoning-rejection / error-enrichment handling. Connection errors propagate so the
    caller's reachability guard can turn them into a clear ``RuntimeError``."""
    body = dict(payload)
    last_resp = None
    for _ in range(_MAX_REJECTED_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        except requests.RequestException:
            raise
        last_resp = resp
        if resp.status_code == 400 and _is_unrecognized_keys(resp):
            dropped = set()
            for k in _parse_rejected_keys(resp, set(body.keys())):
                if k in body:
                    body.pop(k, None)
                    dropped.add(k)
            if not dropped:
                break
            logger.debug("Native /api/v1/chat dropped unrecognized key(s) %s; retrying.",
                         sorted(dropped))
            continue
        break
    return last_resp


def _chat_v1(server_url, api_key, model, messages, temperature, max_tokens,
              timeout=600, seed=None, reasoning="off",
              repeat_penalty=1.0, top_k=None, top_p=None, min_p=None,
              skip_reasoning=False, presence_penalty=None,
              frequency_penalty=None) -> str:
    """Call LM Studio's native ``/api/v1/chat`` endpoint (v1-only features).

    Builds the *native* request shape (top-level ``system_prompt`` + typed ``input`` parts),
    adds ``store=False`` so the server keeps no conversation state, and includes ``reasoning``
    only when the model actually supports it (mapped to an allowed value). If the server still
    rejects the ``reasoning`` param, the call is retried once without it."""
    base = _server_root(server_url)
    url = f"{base}/api/v1/chat"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Whether the request carries image (vision) inputs — used to surface a clear
    # "model does not support image inputs" error on vision rejection.
    has_images = any(
        isinstance(m.get("content"), list)
        for m in messages if isinstance(m, dict))

    system_prompt, input_parts = _build_native_chat_input(messages)

    # Decide whether to send reasoning. When skip_reasoning (one-shot fallback) or the model
    # exposes no reasoning config, omit the param entirely.
    send_reasoning = None
    if not skip_reasoning:
        allowed = model_supports_reasoning(server_url, api_key, model)
        send_reasoning = map_reasoning_level(reasoning, allowed)

    payload = {
        "model": model,
        "input": input_parts,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "repeat_penalty": repeat_penalty,
        "store": False,
    }
    if system_prompt is not None:
        payload["system_prompt"] = system_prompt
    if send_reasoning is not None:
        payload["reasoning"] = send_reasoning
    if seed is not None:
        payload["seed"] = seed
    if top_k is not None:
        payload["top_k"] = top_k
    if top_p is not None:
        payload["top_p"] = top_p
    if min_p is not None:
        payload["min_p"] = min_p
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        payload["frequency_penalty"] = frequency_penalty

    log_http_request("POST", url, headers, payload)
    started = time.time()

    try:
        resp = _post_v1_chat(url, headers, payload, timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"Could not reach LM Studio ({url}): {e}")
    if resp.status_code >= 400:
        snippet = (resp.text or "")[:1000]
        if (not skip_reasoning) and _is_reasoning_rejection(resp.text or ""):
            logger.debug("Reasoning rejected by '%s'; retrying without reasoning.", model)
            return _chat_v1(server_url, api_key, model, messages, temperature, max_tokens,
                            timeout=timeout, seed=seed, reasoning=reasoning,
                            repeat_penalty=repeat_penalty, top_k=top_k, top_p=top_p,
                            min_p=min_p, skip_reasoning=True,
                            presence_penalty=presence_penalty,
                            frequency_penalty=frequency_penalty)
        enriched = _enrich_http_error(model, resp.status_code, resp.text or "", has_images)
        if enriched is not None:
            raise RuntimeError(enriched)
        raise RuntimeError(
            f"LM Studio returned HTTP {resp.status_code} for model '{model}': {snippet}")
    try:
        data = resp.json()
    except ValueError:
        snippet = (resp.text or "")[:1000]
        raise RuntimeError(f"LM Studio returned non-JSON response: {snippet}")
    log_http_response(resp.status_code, time.time() - started, len(resp.text or ""))
    result = _aggregate_v1_output(data)
    # A thinking model with reasoning enabled can return an empty `message` (all text lives
    # in the `reasoning` blob). Retry once without reasoning so JSON-extracting callers
    # (Writer / Scene Builder) still get their answer instead of an empty/parse error.
    if (not skip_reasoning) and send_reasoning is not None and not result.strip():
        out_items = data.get("output", []) if isinstance(data, dict) else []
        has_reasoning = any(isinstance(o, dict) and o.get("type") == "reasoning"
                           and (o.get("content") or "").strip() for o in out_items)
        if has_reasoning:
            logger.debug("Reasoning model returned empty message; retrying without reasoning.")
            return _chat_v1(server_url, api_key, model, messages, temperature, max_tokens,
                            timeout=timeout, seed=seed, reasoning=reasoning,
                            repeat_penalty=repeat_penalty, top_k=top_k, top_p=top_p,
                            min_p=min_p, skip_reasoning=True,
                            presence_penalty=presence_penalty,
                            frequency_penalty=frequency_penalty)
    return result



