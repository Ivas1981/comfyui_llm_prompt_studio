"""HTTP client for LM Studio / OpenAI-compatible servers, model cache and SSRF guard."""
import ipaddress
import json
import logging
import os
import time
from typing import Any, Dict, Optional
from urllib.parse import quote, urlparse

import threading

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
    "maybe_unload_old",
    "load_model",
    "ensure_model_loaded",
    "chat_completion",
]

# Abort a stalled streaming response if no SSE event arrives for this many seconds.
STREAM_WATCHDOG_SEC = 30

# Read allow-public flag from environment for runtime configuration.
ALLOW_PUBLIC_SERVER_URLS = os.getenv("LLM_PROMPT_STUDIO_ALLOW_PUBLIC", "False").lower() in ("1", "true", "yes")

# cache "last loaded model" to unload the old one on switch: {slot: model_id}
_last_loaded = {}
# model instance ids returned by the v1 load endpoint, keyed for precise unload: {(server, model): id}
_model_instances = {}
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
    try:
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
    return [m.get("id") for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]


def _store_model_cache(key, models):
    _model_cache[key] = (models, time.time())
    if len(_model_cache) > _MODEL_CACHE_MAX:
        _model_cache.pop(next(iter(_model_cache)))  # drop oldest


def cache_models(server_url: str, api_key: str, models: list):
    if models:
        key = (server_url, api_key)
        _store_model_cache(key, list(models))
        # Always persist to the default-server cache that combo_models() reads. combo_models()
        # is keyed to DEFAULT_SERVER regardless of the node's server_url, so persisting the
        # refreshed list here (even for a custom server_url) keeps the server-side INPUT_TYPES
        # combo populated. This is what lets a queued prompt pass "Value not in list" validation
        # without a ComfyUI restart after a Refresh.
        _persist_models(models)
        _static_keys.add((DEFAULT_SERVER, ""))


def get_cached_models(server_url=DEFAULT_SERVER, api_key="", allow_fetch=True, timeout=5):
    key = (server_url, api_key)
    cached = _model_cache.get(key)
    # Fresh cache within TTL — return it
    if cached and time.time() - cached[1] < CACHE_TTL:
        return list(cached[0])
    # Disk-backed / default list: never auto-refetch; Refresh is the source of truth.
    # This lets INPUT_TYPES return the previously known models without hitting the network.
    if key in _static_keys:
        if cached:
            return list(cached[0])
        # In-memory entry evicted (LRU) but the list is still "known": re-read disk
        # instead of falling back to the unavailable placeholder.
        disk = _read_disk_models()
        return list(disk) if disk else [PLACEHOLDER]
    # Expired cache — try to re-fetch, fall back to stale only on error
    if cached:
        if not allow_fetch:
            return list(cached[0])
        try:
            models = fetch_models(server_url, api_key, timeout=timeout) or [PLACEHOLDER_EMPTY]
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
            models = fetch_models(server_url, api_key, timeout=timeout) or [PLACEHOLDER_EMPTY]
            _store_model_cache(key, models)
            if key == (DEFAULT_SERVER, ""):
                _persist_models(models)
                _static_keys.add(key)
            return models
        except Exception:
            # Negative-cache the failure so we don't block on every INPUT_TYPES call while
            # the server is down; it expires with CACHE_TTL and a manual Refresh still works.
            _store_model_cache(key, [PLACEHOLDER])
            return [PLACEHOLDER]
    return [PLACEHOLDER]


def looks_like_vision(model_id: str) -> bool:
    low = model_id.lower()
    return any(h in low for h in VISION_NAME_HINTS)


def maybe_unload_old(slot: str, server_url: str, new_model: str):
    old = _last_loaded.get(slot)
    if old and old != new_model and not old.startswith("—"):
        try:
            base = _server_root(server_url)
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


def _server_root(server_url: str) -> str:
    """Strip a trailing ``/v1`` so we can build native ``/api/v1/*`` endpoints from a
    server_url that already ends in ``/v1`` (the ComfyUI default)."""
    base = server_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    return base


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

    # Native v1 load body (snake_case per LM Studio docs). New options are only included when set.
    v1_body = {"model": model, "context_length": int(context_length), "gpuOffload": gpu_offload}
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

    # Legacy body (camelCase) used by the older endpoints; ignores the new options.
    legacy_body = {"contextLength": int(context_length), "gpuOffload": gpu_offload, "seed": -1}

    # 1) Native v1 endpoint (best effort — supports flash attention / kv offload).
    ok, err, resp = _try_post(f"{base}/api/v1/models/load", v1_body, headers,
                              timeout, retries, backoff)
    if ok:
        _record_load_config(server_url, model, resp)
        return True

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
    if not load_model(server_url, api_key, model, context_length, gpu_offload,
                      flash_attention=flash_attention,
                      offload_kv_cache_to_gpu=offload_kv_cache_to_gpu,
                      eval_batch_size=eval_batch_size,
                      num_experts=num_experts,
                      echo_load_config=echo_load_config):
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
                    temperature, max_tokens, timeout=600, seed=None,
                    stream=False, reasoning="off", repeat_penalty=1.0,
                    top_k=None, top_p=None, min_p=None, on_delta=None) -> str:
    """OpenAI-compatible chat completion with optional LM Studio v1-native features.

    The native ``/api/v1/chat`` endpoint (which supports ``reasoning``, ``repeat_penalty``,
    ``top_k``/``top_p``/``min_p`` and streaming) is used automatically when ``stream=True`` or
    any v1-only parameter is set. Multimodal (image) requests always use the OpenAI
    ``/chat/completions`` path for maximum compatibility — in that case the v1-only options are
    ignored. If a streaming request fails, it gracefully falls back to a normal non-streaming
    call so the node still returns text.

    ``on_delta`` (callable[str]) receives each streamed content chunk when streaming."""
    server_url = validate_server_url(server_url)
    has_images = any(
        isinstance(m.get("content"), list)
        for m in messages if isinstance(m, dict))
    # Route to the v1-native endpoint only when a v1-only feature is requested and the
    # request is not multimodal (the OpenAI path is required for images).
    use_v1 = (stream or reasoning not in (None, "off") or top_k is not None
              or top_p is not None or min_p is not None
              or (repeat_penalty is not None and repeat_penalty != 1.0)) and not has_images
    if use_v1:
        try:
            return _chat_v1(server_url, api_key, model, messages, temperature, max_tokens,
                            timeout=timeout, seed=seed, stream=stream, reasoning=reasoning,
                            repeat_penalty=repeat_penalty, top_k=top_k, top_p=top_p,
                            min_p=min_p, on_delta=on_delta)
        except Exception as e:  # noqa: BLE001 — graceful fallback for streaming failures
            if stream:
                logger.warning("Streaming failed (%s); falling back to non-streaming.", e)
                stream = False
                use_v1 = False
            else:
                raise

    # --- OpenAI-compatible path (default; unchanged behavior) -------------------------
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
        logger.error("LM Studio HTTP %s for model '%s': %s", resp.status_code, model, snippet)
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
    return content


def _aggregate_v1_output(result: Dict) -> str:
    """Join the ``content`` of all ``message``-type items in a v1 chat ``result``/response."""
    output = result.get("output", []) if isinstance(result, dict) else []
    parts = []
    for item in output:
        if isinstance(item, dict) and item.get("type") == "message":
            parts.append(item.get("content", ""))
    return "".join(parts)


def _chat_v1(server_url, api_key, model, messages, temperature, max_tokens,
             timeout=600, seed=None, stream=False, reasoning="off",
             repeat_penalty=1.0, top_k=None, top_p=None, min_p=None, on_delta=None) -> str:
    """Call LM Studio's native ``/api/v1/chat`` endpoint (v1-only features + streaming)."""
    base = _server_root(server_url)
    url = f"{base}/api/v1/chat"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "input": messages,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "reasoning": reasoning,
        "repeat_penalty": repeat_penalty,
        "stream": stream,
    }
    if seed is not None:
        payload["seed"] = seed
    if top_k is not None:
        payload["top_k"] = top_k
    if top_p is not None:
        payload["top_p"] = top_p
    if min_p is not None:
        payload["min_p"] = min_p

    log_http_request("POST", url, headers, payload)
    started = time.time()
    if not stream:
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as e:
            raise RuntimeError(f"Could not reach LM Studio ({url}): {e}")
        if resp.status_code >= 400:
            snippet = (resp.text or "")[:1000]
            raise RuntimeError(
                f"LM Studio returned HTTP {resp.status_code} for model '{model}': {snippet}")
        try:
            data = resp.json()
        except ValueError:
            snippet = (resp.text or "")[:1000]
            raise RuntimeError(f"LM Studio returned non-JSON response: {snippet}")
        log_http_response(resp.status_code, time.time() - started, len(resp.text or ""))
        return _aggregate_v1_output(data)

    # Streaming: consume the SSE event stream.
    try:
        resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"Could not reach LM Studio ({url}): {e}")
    if resp.status_code >= 400:
        snippet = (resp.text or "")[:1000]
        raise RuntimeError(
            f"LM Studio returned HTTP {resp.status_code} for model '{model}': {snippet}")
    return _consume_sse(resp, on_delta, started, url, headers)


def _consume_sse(resp, on_delta, started, url, headers) -> str:
    """Parse an LM Studio ``/api/v1/chat`` SSE stream and return the final aggregated text.

    Emits ``message.delta`` content to ``on_delta`` (for live UI). A no-activity watchdog
    aborts a stalled stream; a server ``error`` event raises; ``chat.end`` returns the result."""
    full: list = []
    event_type = None
    stop = threading.Event()
    watchdog = None

    def _arm():
        nonlocal watchdog
        if watchdog is not None:
            watchdog.cancel()
        if not stop.is_set():
            watchdog = threading.Timer(STREAM_WATCHDOG_SEC, _stall)
            watchdog.daemon = True
            watchdog.start()

    def _stall():
        stop.set()
        try:
            resp.close()
        except Exception:
            pass

    _arm()
    try:
        for raw in resp.iter_lines(decode_unicode=True):
            if stop.is_set():
                raise RuntimeError(
                    f"Streaming stalled (no data for {STREAM_WATCHDOG_SEC}s) from {url}")
            if not raw:
                continue
            _arm()
            line = raw.strip()
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_str = line[len("data:"):].strip()
                if not data_str:
                    continue
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                data["type"] = event_type or data.get("type")
                et = data.get("type")
                if et == "message.delta":
                    chunk = data.get("content", "")
                    full.append(chunk)
                    if on_delta:
                        try:
                            on_delta(chunk)
                        except Exception:
                            pass
                elif et == "error":
                    raise RuntimeError(f"LM Studio streaming error: {data}")
                elif et == "chat.end":
                    return _aggregate_v1_output(data.get("result", {}))
        return "".join(full)
    finally:
        stop.set()
        if watchdog is not None:
            watchdog.cancel()
        try:
            resp.close()
        except Exception:
            pass
