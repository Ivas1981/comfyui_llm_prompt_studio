"""HTTP client for LM Studio / OpenAI-compatible servers, model cache and SSRF guard."""
import ipaddress
import json
import logging
import os
import time
from urllib.parse import quote, urlparse

import requests

from .constants import PLACEHOLDER, PLACEHOLDER_EMPTY

logger = logging.getLogger("llm_prompt_studio")

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
    "maybe_unload_old",
    "load_model",
    "ensure_model_loaded",
    "chat_completion",
]

# Read allow-public flag from environment for runtime configuration.
ALLOW_PUBLIC_SERVER_URLS = os.getenv("LLM_PROMPT_STUDIO_ALLOW_PUBLIC", "False").lower() in ("1", "true", "yes")

# cache "last loaded model" to unload the old one on switch: {slot: model_id}
_last_loaded = {}
# cache of model lists: {(server_url, api_key): (models, timestamp)}
_model_cache = {}
CACHE_TTL = 60  # seconds (was 10)
_MODEL_CACHE_MAX = 32

DEFAULT_SERVER = "http://localhost:1234/v1"
# Keys whose list is disk-backed and must NOT be auto-refetched; "Refresh models" is the
# source of truth. This keeps INPUT_TYPES non-blocking while still letting a saved
# workflow load with its previously selected model already present in the combo.
_static_keys = set()


def _model_cache_path():
    # Cache lives in the package directory: always resolvable at import and at runtime,
    # independent of ComfyUI's output-directory resolution (which can differ between
    # import time and request time and break the read after a restart).
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "llm_prompt_studio_models_cache.json")


def _read_disk_models():
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
        if isinstance(data, list) and data and data not in ([PLACEHOLDER], [PLACEHOLDER_EMPTY]):
            # Migrate to the new location so future reads are consistent.
            if path != _model_cache_path():
                _persist_models(data)
            return list(data)
    return None


def _persist_models(models):
    if not models or models in ([PLACEHOLDER], [PLACEHOLDER_EMPTY]):
        return
    # Merge with the on-disk list so the cache accumulates every model that has ever
    # been available. This keeps a saved workflow's model selectable (and valid) even
    # when the server is offline and only a stale list is on disk.
    existing = _read_disk_models() or []
    merged = list(dict.fromkeys(list(existing) + list(models)))
    try:
        with open(_model_cache_path(), "w", encoding="utf-8") as f:
            json.dump(merged, f)
    except OSError:
        pass


def remember_model(model: str):
    """Persist a single model id so it stays valid/selectable offline.

    Called whenever a node actually uses a model, so a workflow reloaded after the
    server went away still validates its previously-run model instead of ComfyUI
    rejecting it with 'Value not in list'."""
    if not model or model.startswith("—"):
        return
    _persist_models([model])


def _load_disk_cache():
    disk = _read_disk_models()
    if disk:
        _model_cache[(DEFAULT_SERVER, "")] = (disk, time.time())
        _static_keys.add((DEFAULT_SERVER, ""))


def cached_model_list():
    """Return the persisted model list (non-blocking), seeding the in-memory cache."""
    disk = _read_disk_models()
    if disk:
        _store_model_cache((DEFAULT_SERVER, ""), disk)
        _static_keys.add((DEFAULT_SERVER, ""))
        return disk
    return None


_load_disk_cache()

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


def fetch_models(server_url: str, api_key: str = "", timeout: int = 5) -> list:
    server_url = validate_server_url(server_url)
    resp = requests.get(f"{server_url.rstrip('/')}/models",
                        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                        timeout=timeout)
    resp.raise_for_status()
    return [m["id"] for m in resp.json().get("data", [])]


def _store_model_cache(key, models):
    _model_cache[key] = (models, time.time())
    if len(_model_cache) > _MODEL_CACHE_MAX:
        _model_cache.pop(next(iter(_model_cache)))  # drop oldest


def cache_models(server_url: str, api_key: str, models: list):
    if models:
        key = (server_url, api_key)
        _store_model_cache(key, list(models))
        # Only persist for the default-server key used by combo_models(); persisting a
        # custom server's models under the default key would overwrite the real
        # default-server list and poison INPUT_TYPES on the next start.
        if key == (DEFAULT_SERVER, ""):
            _persist_models(models)
            _static_keys.add(key)


def get_cached_models(server_url=DEFAULT_SERVER, api_key="", allow_fetch=True):
    key = (server_url, api_key)
    cached = _model_cache.get(key)
    # Fresh cache within TTL — return it
    if cached and time.time() - cached[1] < CACHE_TTL:
        return list(cached[0])
    # Disk-backed / default list: never auto-refetch; Refresh is the source of truth.
    # This lets INPUT_TYPES return the previously known models without hitting the network.
    if key in _static_keys:
        return list(cached[0]) if cached else [PLACEHOLDER]
    # Expired cache — try to re-fetch, fall back to stale only on error
    if cached:
        if not allow_fetch:
            return list(cached[0])
        try:
            models = fetch_models(server_url, api_key) or [PLACEHOLDER_EMPTY]
            _store_model_cache(key, models)
            if key == (DEFAULT_SERVER, ""):
                _persist_models(models)
                _static_keys.add(key)
            return models
        except Exception:
            return list(cached[0])
    # No cache at all — fetch or signal unavailable
    if allow_fetch:
        try:
            models = fetch_models(server_url, api_key) or [PLACEHOLDER_EMPTY]
            _store_model_cache(key, models)
            if key == (DEFAULT_SERVER, ""):
                _persist_models(models)
                _static_keys.add(key)
            return models
        except Exception:
            return [PLACEHOLDER]
    return [PLACEHOLDER]


def looks_like_vision(model_id: str) -> bool:
    low = model_id.lower()
    return any(h in low for h in VISION_NAME_HINTS)


def maybe_unload_old(slot: str, server_url: str, new_model: str):
    old = _last_loaded.get(slot)
    if old and old != new_model and not old.startswith("—"):
        try:
            base = server_url.rstrip("/")
            if base.endswith("/v1"):
                base = base[:-3]
            requests.post(f"{base}/api/v0/models/unload", json={"model": old}, timeout=5)
            logger.debug("Requested unload of previous model '%s'", old)
        except requests.RequestException as e:
            logger.debug("Could not unload previous model '%s': %s", old, e)
    _last_loaded[slot] = new_model


# HTTP status codes worth retrying: transient server hiccups / rate limits.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _is_transient(status, exc):
    if exc is not None:
        return True  # connection error / timeout — always worth a retry
    return status in _RETRYABLE_STATUS


def load_model(server_url: str, api_key: str, model: str,
                context_length: int = 8192, gpu_offload: float = 1.0,
                timeout: int = 300, retries: int = 3, backoff: float = 1.0) -> bool:
    """Load a model on the LM Studio server with the given context length / GPU offload.

    Returns True if the server confirmed the load (HTTP < 400). Falls back from the v1 to
    the v0 load endpoint. GPU offload is a 0.0–1.0 fraction (1.0 = max).

    Transient failures (HTTP 429/500/502/503/504, connection errors, timeouts) are retried
    with exponential backoff: `retries` attempts per endpoint, pausing `backoff` seconds
    before the 2nd try, `backoff * 2` before the 3rd, and so on."""
    server_url = validate_server_url(server_url)
    base = server_url.rstrip("/")
    body = {"contextLength": int(context_length), "gpuOffload": gpu_offload, "seed": -1}
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    identifier = quote(model, safe="")
    last_err = None
    for path in (f"{base}/v1/models/{identifier}/load",
                  f"{base}/api/v0/models/{identifier}/load"):
        for attempt in range(retries):
            started = time.time()
            try:
                resp = requests.post(path, json=body, headers=headers, timeout=timeout)
            except requests.RequestException as e:
                last_err = str(e)
                if _is_transient(None, e) and attempt < retries - 1:
                    logger.debug("Load attempt %d/%d for '%s' failed (%s); retrying",
                                 attempt + 1, retries, model, e)
                    time.sleep(backoff * (2 ** attempt))
                    continue
                break
            if resp.status_code < 400:
                logger.info("Loaded model '%s' (context=%s, gpuOffload=%s) in %.1fs",
                            model, context_length, gpu_offload, time.time() - started)
                return True
            last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if _is_transient(resp.status_code, None) and attempt < retries - 1:
                logger.debug("Load attempt %d/%d for '%s' returned %d; retrying",
                             attempt + 1, retries, model, resp.status_code)
                time.sleep(backoff * (2 ** attempt))
                continue
            break  # non-retryable (e.g. 400/401/404) — try the next endpoint
    logger.warning("LM Studio could not load model '%s': %s", model, last_err)
    return False


def ensure_model_loaded(slot: str, server_url: str, api_key: str, model: str,
                        context_length: int = 8192, gpu_offload: float = 1.0):
    """Make sure `model` is the one loaded on the server before we call it.

    Skips work when the same model is already loaded for this slot, otherwise unloads the
    previous model and loads the requested one with the requested parameters. A failed load is
    not recorded as "loaded", so the next run will retry."""
    if not model or model.startswith("—"):
        return
    # Remember this model so a saved workflow that uses it stays valid (and selectable)
    # even when the server is offline later and the combo would otherwise be the placeholder.
    remember_model(model)
    if _last_loaded.get(slot) == model:
        return  # already loaded for this slot — avoid a needless reload
    maybe_unload_old(slot, server_url, model)  # unloads the previous model
    if not load_model(server_url, api_key, model, context_length, gpu_offload):
        _last_loaded[slot] = None  # load failed: allow a retry on the next run
        logger.warning(
            "Model '%s' for slot '%s' was not loaded. The next LLM call will fail "
            "until the model is available (check that LM Studio is running and the "
            "model id is correct).", model, slot)


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


def chat_completion(server_url, api_key, model, messages,
                    temperature, max_tokens, timeout=600, seed=None) -> str:
    server_url = validate_server_url(server_url)
    # Send the messages exactly as given — including multimodal (image_url) content, which
    # OpenAI-compatible servers expect in the request body. Serialize only for the debug log.
    body = {"model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens}
    if seed is not None:
        body["seed"] = seed
    started = time.time()
    try:
        resp = requests.post(
            f"{server_url.rstrip('/')}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=timeout)
    except requests.RequestException as e:
        logger.error("Could not reach LM Studio (%s): %s", server_url, e)
        raise RuntimeError(f"Could not reach LM Studio ({server_url}): {e}")
    if resp.status_code >= 400:
        txt = resp.text or ""
        snippet = txt[:1000] if len(txt) > 1000 else txt
        logger.error("LM Studio HTTP %s for model '%s': %s", resp.status_code, snippet)
        raise RuntimeError(
            f"LM Studio returned HTTP {resp.status_code} for model '{model}': "
            f"{snippet}")
    msg = resp.json()["choices"][0]["message"]
    content = msg.get("content") or ""
    if not content.strip():
        content = msg.get("reasoning_content") or msg.get("reasoning") or ""
    last = messages[-1].get("content", "") if messages else ""
    logger.debug("LLM call to '%s' completed in %.1fs (%d chars): %s",
                 model, time.time() - started, len(content),
                 _serialize_message_content(last))
    return content
