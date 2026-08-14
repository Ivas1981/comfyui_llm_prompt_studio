import os
import sys
import json

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PKG)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from comfyui_llm_prompt_studio import lm_http  # noqa: E402
from comfyui_llm_prompt_studio.constants import PLACEHOLDER  # noqa: E402


def _real_persist(cache_file):
    def persist(server_url, models):
        if not models:
            return
        srv = lm_http._normalize_server(server_url)
        cache = lm_http._read_disk_cache()
        existing = cache.get(srv) or []
        merged = list(dict.fromkeys(list(existing) + list(models)))
        cache[srv] = merged
        with open(str(cache_file), "w", encoding="utf-8") as f:
            json.dump(cache, f)
    return persist


def test_per_server_isolation(tmp_path, monkeypatch):
    cache_file = tmp_path / "models_cache.json"
    monkeypatch.setattr(lm_http, "_model_cache_path", lambda: str(cache_file))
    monkeypatch.setattr(lm_http, "_persist_models", _real_persist(cache_file))
    lm_http._model_cache.clear()
    lm_http._static_keys.clear()

    srvA = "http://server-a:1234/v1"
    srvB = "http://server-b:1234/v1"
    lm_http.cache_models(srvA, "", ["a1", "a2"])
    lm_http.cache_models(srvB, "", ["b1"])

    # cached_model_list is scoped per server — no cross-pollution.
    assert sorted(lm_http.cached_model_list(srvA)) == ["a1", "a2"]
    assert lm_http.cached_model_list(srvB) == ["b1"]
    assert "b1" not in lm_http.cached_model_list(srvA)

    # The on-disk cache is keyed by server URL (api_key is NOT persisted).
    disk = json.loads(cache_file.read_text(encoding="utf-8"))
    assert sorted(disk[srvA]) == ["a1", "a2"]
    assert disk[srvB] == ["b1"]


def test_cache_models_normalizes_trailing_slash(tmp_path, monkeypatch):
    cache_file = tmp_path / "models_cache.json"
    monkeypatch.setattr(lm_http, "_model_cache_path", lambda: str(cache_file))
    monkeypatch.setattr(lm_http, "_persist_models", _real_persist(cache_file))
    lm_http._model_cache.clear()
    lm_http._static_keys.clear()

    lm_http.cache_models("http://host:1234/v1/", "", ["m1"])
    # A query with/without the trailing slash hits the same cached entry.
    assert lm_http.cached_model_list("http://host:1234/v1") == ["m1"]


def test_combo_models_does_not_use_network(monkeypatch):
    # combo_models must read only the persisted cache (Refresh is the source of truth) and
    # must never call fetch_models. Point the cache file at a nonexistent path so disk is empty.
    monkeypatch.setattr(lm_http, "_model_cache_path", lambda: os.devnull)
    lm_http._model_cache.clear()
    lm_http._static_keys.clear()

    def boom(*a, **k):
        raise AssertionError("INPUT_TYPES must not hit the network")

    monkeypatch.setattr(lm_http, "fetch_models", boom)

    # Cached list is empty -> None, and fetch_models was never invoked.
    assert lm_http.cached_model_list("http://x:1234/v1") is None


def test_combo_models_returns_placeholder_when_empty(monkeypatch):
    monkeypatch.setattr(lm_http, "_model_cache_path", lambda: os.devnull)
    lm_http._model_cache.clear()
    lm_http._static_keys.clear()
    from comfyui_llm_prompt_studio.combos import combo_models
    assert combo_models("http://x:1234/v1") == [PLACEHOLDER]
