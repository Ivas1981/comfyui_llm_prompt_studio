import os
import sys

# `lm_http` uses package-relative imports, so import it through the package rather than
# as a bare top-level module. The conftest already puts the package root on sys.path;
# add the parent so `comfyui_llm_prompt_studio` resolves as a package.
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PKG)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from comfyui_llm_prompt_studio import lm_http  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

LOCAL_V1 = "http://localhost:1234/v1"
NATIVE_MODELS_URL = "http://localhost:1234/api/v1/models"


def _ok_response(status=200, text="{}"):
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    try:
        resp.json.return_value = json.loads(text) if text.strip() else {}
    except ValueError:
        # Mirror a real requests.Response: .json() raises when the body isn't JSON.
        resp.json.side_effect = ValueError("No JSON could be decoded")
    resp.headers = {"Content-Type": "application/json"}
    resp.ok = status < 400
    resp.status = status
    return resp


import pytest  # noqa: E402
import json  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_state():
    # Avoid cross-test contamination of the global load caches and rejected-key memory.
    saved = dict(lm_http._last_loaded)
    lm_http._last_loaded.clear()
    lm_http._rejected_v1_keys.clear()
    lm_http._model_instances.clear()
    lm_http._model_cache.clear()
    lm_http._static_keys.clear()
    lm_http._server_loaded.clear()
    lm_http._seen_servers.clear()
    lm_http._keep_loaded.clear()
    yield
    lm_http._last_loaded.clear()
    lm_http._rejected_v1_keys.clear()
    lm_http._model_instances.clear()
    lm_http._model_cache.clear()
    lm_http._static_keys.clear()
    lm_http._server_loaded.clear()
    lm_http._seen_servers.clear()
    lm_http._keep_loaded.clear()
    lm_http._llm_response_cache.clear()


def test_validate_server_url_allows_local():
    assert lm_http.validate_server_url("http://localhost:1234/v1") == "http://localhost:1234/v1"
    assert lm_http.validate_server_url("http://127.0.0.1:1234/v1") == "http://127.0.0.1:1234/v1"


def test_validate_server_url_rejects_public():
    with pytest.raises(ValueError):
        lm_http.validate_server_url("http://1.2.3.4/v1")


def test_fetch_models_returns_ids():
    # Native /api/v1/models envelope: prefer `key`, fall back to `id`.
    resp = _ok_response(text=json.dumps(
        {"data": [{"key": "a", "id": "a-id"}, {"id": "b"}]}))
    with patch("requests.get", return_value=resp) as get:
        models = lm_http.fetch_models(LOCAL_V1)
    assert models == ["a", "b"]
    # The native endpoint is queried first, not the OpenAI /v1/models list.
    assert get.call_args_list[0][0][0] == NATIVE_MODELS_URL


def test_fetch_models_prefers_native_key_over_id():
    # When an entry carries both, the canonical `key` wins so listed ids match chat/load.
    resp = _ok_response(text=json.dumps(
        {"data": [{"key": "canonical", "id": "alt-id"}]}))
    with patch("requests.get", return_value=resp):
        assert lm_http.fetch_models(LOCAL_V1) == ["canonical"]


def test_fetch_models_falls_back_to_openai_list():
    # A pre-0.4.0 server that lacks the native endpoint falls through to /v1/models.
    native = _ok_response(status=404, text="not found")
    openai = _ok_response(text=json.dumps({"data": [{"id": "legacy"}]}))
    with patch("requests.get", side_effect=[native, openai]) as get:
        models = lm_http.fetch_models(LOCAL_V1)
    assert models == ["legacy"]
    # First call native, second the OpenAI-compatible fallback.
    assert get.call_args_list[0][0][0] == NATIVE_MODELS_URL
    assert get.call_args_list[1][0][0].endswith("/v1/models")


def test_fetch_models_handles_nonjson_native_then_openai():
    # Both the native probe and the OpenAI fallback returning non-JSON must surface a
    # RuntimeError (not a bare ValueError) to the caller.
    resp = _ok_response(text="<html>error</html>")
    resp.json.side_effect = ValueError("No JSON")
    with patch("requests.get", return_value=resp):
        with pytest.raises(RuntimeError):
            lm_http.fetch_models(LOCAL_V1)



def test_fetch_models_filters_embedding_models():
    # Embedding models are not chat models and must not appear in the model combo, even
    # when LM Studio's native /api/v1/models list mixes them in with a "type": "embedding".
    resp = _ok_response(text=json.dumps({"data": [
        {"key": "chat-model", "id": "chat-model"},
        {"key": "embed-model", "type": "embedding"},
    ]}))
    with patch("requests.get", return_value=resp):
        assert lm_http.fetch_models(LOCAL_V1) == ["chat-model"]


def test_fetch_models_filters_embedding_models_openai_fallback():
    native = _ok_response(status=404, text="not found")
    openai = _ok_response(text=json.dumps({"data": [
        {"id": "chat-model"},
        {"id": "embed-model", "type": "embedding"},
    ]}))
    with patch("requests.get", side_effect=[native, openai]):
        assert lm_http.fetch_models(LOCAL_V1) == ["chat-model"]


def test_chat_completion_handles_nonjson_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.text = "<html>something went wrong</html>"
    resp.json.side_effect = ValueError("No JSON")
    with patch("requests.post", return_value=resp):
        with pytest.raises(RuntimeError):
            lm_http.chat_completion(LOCAL_V1, "", "m", [], 0.7, 100)


def test_chat_completion_handles_missing_choices():
    # Native-first: a 400 with no usable body fails on the native path and the OpenAI
    # fallback, raising RuntimeError. (An empty 200 body now yields "" instead of erroring,
    # so we exercise the real failure path with a 400.)
    resp = _ok_response(status=400, text=json.dumps({"error": {"message": "bad request"}}))
    with patch("requests.post", return_value=resp):
        with pytest.raises(RuntimeError):
            lm_http.chat_completion(LOCAL_V1, "", "m", [], 0.7, 100)


def test_chat_completion_openai_fallback_forwards_sampling_params():
    # When the native path fails, the OpenAI fallback must still forward the sampling
    # params the node layer computed (top_p/top_k/repeat_penalty) so behavior matches the
    # native path; min_p stays native-only and must be absent.
    native_err = RuntimeError("native transport down")
    ok = _ok_response(status=200,
                      text=json.dumps({"choices": [{"message": {"content": "x"}}]}))
    with patch("requests.post", side_effect=[native_err, ok]) as post:
        out = lm_http.chat_completion(LOCAL_V1, "", "m",
                                      [{"role": "user", "content": "hi"}], 0.7, 100,
                                      top_p=0.9, top_k=40, repeat_penalty=1.1)
    assert out == "x"
    # Second POST is the OpenAI-compatible fallback.
    body = post.call_args_list[1][1]["json"]
    assert body["top_p"] == 0.9
    assert body["top_k"] == 40
    assert body["repeat_penalty"] == 1.1
    assert "min_p" not in body


def test_post_v1_chat_drops_unrecognized_optional_key_and_retries():
    # A server that rejects an OPTIONAL key (e.g. top_k) must not fail the native call: the
    # key is dropped and the request retried once, successfully.
    reject = _ok_response(status=400,
                          text='{"error":{"code":"unrecognized_keys",'
                               '"message":"Unrecognized key(s) in object: \'top_k\'"}}')
    ok = _ok_response(status=200, text=json.dumps(
        {"output": [{"type": "message", "content": "ok"}]}))
    with patch("requests.post", side_effect=[reject, ok]) as post:
        resp = lm_http._post_v1_chat("http://x/api/v1/chat", {},
                                     {"model": "m", "seed": 0, "top_k": 40}, 5)
    assert resp.status_code == 200
    # The retry request omitted the rejected optional key.
    assert "top_k" not in post.call_args_list[1][1]["json"]


def test_post_v1_chat_keeps_protected_seed_when_rejected():
    # A USER-INTENT key (`seed`) must NOT be silently dropped when the native endpoint
    # rejects it: instead the call keeps failing so the caller (chat_completion) can fall
    # back to the OpenAI-compatible route that actually honors seed. Without this protection,
    # seed control would silently do nothing.
    reject = _ok_response(status=400,
                          text='{"error":{"code":"unrecognized_keys",'
                               '"message":"Unrecognized key(s) in object: \'seed\'"}}')
    with patch("requests.post", return_value=reject) as post:
        resp = lm_http._post_v1_chat("http://x/api/v1/chat", {},
                                     {"model": "m", "seed": 7}, 5,
                                     protected_keys={"seed"})
    # seed stays in the body and the call is NOT retried into a success.
    assert resp.status_code == 400
    assert post.call_count == 1
    assert post.call_args_list[0][1]["json"]["seed"] == 7


def test_post_v1_chat_passes_through_non_key_errors():
    # A 400 that is NOT an unrecognized_keys rejection is returned unchanged (the caller's
    # reasoning-rejection / error-enrichment logic decides what to do), with no extra retry.
    err = _ok_response(status=400, text=json.dumps({"error": {"message": "bad request"}}))
    with patch("requests.post", return_value=err) as post:
        resp = lm_http._post_v1_chat("http://x/api/v1/chat", {},
                                     {"model": "m", "seed": 0, "top_k": 40}, 5)
    assert resp.status_code == 400
    assert post.call_count == 1


def test_chat_completion_falls_back_to_openai_when_native_rejects_seed():
    # Native /api/v1/chat rejects `seed` (unrecognized_keys) -> the call is NOT retried into a
    # success with seed silently dropped; instead chat_completion falls back to the
    # OpenAI-compatible /v1/chat/completions endpoint, which honors `seed`. Regression test
    # for seed control appearing broken while the model ignored the seed entirely.
    reject = _ok_response(status=400,
                          text='{"error":{"code":"unrecognized_keys",'
                               '"message":"Unrecognized key(s) in object: \'seed\'"}}')
    oai_ok = _ok_response(status=200, text=json.dumps({
        "choices": [{"message": {"content": "openai result"}}]}))
    with patch("requests.get", return_value=_models_response([{"key": "m", "capabilities": {}}])), \
         patch("requests.post", side_effect=[reject, oai_ok]) as post:
        out = lm_http.chat_completion(LOCAL_V1, "", "m",
                                      [{"role": "user", "content": "hi"}], 0.7, 100,
                                      seed=7)
    assert out == "openai result"
    # The fallback POST is the OpenAI /chat/completions route and still carries the seed.
    oai_calls = [c for c in post.call_args_list if "chat/completions" in c[0][0]]
    assert oai_calls, "expected a fallback to /chat/completions"
    assert oai_calls[0][1]["json"]["seed"] == 7


def test_load_model_success_v1():
    resp = _ok_response(status=200)
    with patch("requests.post", return_value=resp) as post:
        ok = lm_http.load_model(LOCAL_V1, "", "m", backoff=0)
    assert ok is True
    # First attempted endpoint is the native v1 load route.
    assert post.call_args_list[0][0][0].endswith("/api/v1/models/load")


def test_load_model_native_url_has_no_double_v1_prefix():
    # Regression for the bug where server_url ending in /v1 produced
    # /v1/api/v1/models/load (a silent no-op). The native endpoint must be
    # /api/v1/models/load without the duplicated /v1 prefix.
    resp = _ok_response(status=200)
    with patch("requests.post", return_value=resp) as post:
        ok = lm_http.load_model(LOCAL_V1, "", "m", backoff=0)
    assert ok is True
    url = post.call_args_list[0][0][0]
    assert "/v1/api/v1/" not in url
    assert url == "http://localhost:1234/api/v1/models/load"



def test_load_model_fallback_to_v0():
    native = _ok_response(status=404, text="not found")
    legacy = _ok_response(status=404, text="not found")
    v0 = _ok_response(status=200)
    with patch("requests.post", side_effect=[native, legacy, v0]) as post:
        ok = lm_http.load_model(LOCAL_V1, "", "m", retries=1, backoff=0)
    assert ok is True
    # native 404, legacy /v1 404, then v0.
    assert post.call_count == 3
    assert post.call_args_list[2][0][0].endswith("/api/v0/models/m/load")


def test_load_model_retries_transient_then_succeeds():
    resp503 = _ok_response(status=503, text="unavailable")
    resp200 = _ok_response(status=200)
    with patch("requests.post", side_effect=[resp503, resp200]) as post:
        ok = lm_http.load_model(LOCAL_V1, "", "m", retries=3, backoff=0)
    assert ok is True
    # native tried twice (503 then 200); legacy/v0 never needed.
    assert post.call_count == 2
    assert post.call_args_list[0][0][0].endswith("/api/v1/models/load")


def test_load_model_fails_after_retries(caplog):
    resp = _ok_response(status=503, text="unavailable")
    with patch("requests.post", return_value=resp):
        ok = lm_http.load_model(LOCAL_V1, "", "m", retries=2, backoff=0)
    assert ok is False
    # v1 retried, then v0 retried: 2 attempts per endpoint.
    assert caplog.text.count("could not load model") >= 1


def test_maybe_unload_old_posts_unload_for_previous():
    lm_http._last_loaded["slotX"] = "oldmodel"
    with patch("requests.post", return_value=_ok_response(status=200)) as post:
        lm_http.maybe_unload_old("slotX", LOCAL_V1, "newmodel")
    assert lm_http._last_loaded["slotX"] == "newmodel"
    # With no known instance id, the unload is issued on the v1 route by model name.
    unload_calls = [c for c in post.call_args_list
                    if c[0][0].endswith("/api/v1/models/unload")]
    assert unload_calls, "expected a v1 unload request for the previous model"
    assert unload_calls[0].kwargs["json"] == {"model": "oldmodel"}


def test_maybe_unload_old_url_has_no_double_v1_prefix():
    # Regression: the v1 unload URL must be /api/v1/models/unload, not /v1/api/v1/...,
    # when server_url ends in /v1. Uses _server_root() (same helper as load_model). With no
    # known instance id the first attempt is the v1 route by model name.
    lm_http._last_loaded["slotZ"] = "oldmodel"
    with patch("requests.post", return_value=_ok_response(status=200)) as post:
        lm_http.maybe_unload_old("slotZ", LOCAL_V1, "newmodel")
    url = post.call_args_list[0][0][0]
    assert "/v1/api/v1/" not in url
    assert url == "http://localhost:1234/api/v1/models/unload"


def test_maybe_unload_old_skips_when_same_model():
    lm_http._last_loaded["slotY"] = "same"
    with patch("requests.post") as post:
        lm_http.maybe_unload_old("slotY", LOCAL_V1, "same")
    assert post.call_count == 0
    assert lm_http._last_loaded["slotY"] == "same"


def test_ensure_model_loaded_marks_loaded_on_success():
    # unload_all_loaded lists models (none resident here) then the load happens; the loaded
    # state is recorded both per-slot and server-scoped.
    with patch("requests.get", return_value=_ok_response(status=404)), \
         patch.object(lm_http, "load_model", return_value=True) as load:
        lm_http.ensure_model_loaded("s", LOCAL_V1, "", "m")
    load.assert_called_once()
    fp = ("m", 8192, 1.0, None, None, None, None)
    # The loaded state is now a config fingerprint (model + load params), not the bare name.
    assert lm_http._last_loaded["s"] == fp
    assert lm_http._server_loaded[LOCAL_V1] == fp


def test_ensure_model_loaded_skips_when_already_loaded():
    # Server-scoped truth: same fingerprint already resident -> no unload/reload at all.
    fp = ("m", 8192, 1.0, None, None, None, None)
    lm_http._server_loaded[LOCAL_V1] = fp
    with patch("requests.get") as get, \
         patch.object(lm_http, "load_model") as load:
        lm_http.ensure_model_loaded("s", LOCAL_V1, "", "m")
    load.assert_not_called()
    get.assert_not_called()  # no model listing needed when we skip


def test_ensure_model_loaded_nulls_state_on_failure(caplog):
    with patch("requests.get", return_value=_ok_response(status=404)), \
         patch.object(lm_http, "load_model", return_value=False) as load:
        lm_http.ensure_model_loaded("s", LOCAL_V1, "", "m")
    load.assert_called_once()
    # Failure leaves state as None so the next run will retry; warning is logged.
    assert lm_http._last_loaded["s"] is None
    assert lm_http._server_loaded[LOCAL_V1] is None
    assert "was not loaded" in caplog.text


def test_ensure_model_loaded_unloads_others_before_loading():
    # A model resident under a DIFFERENT slot must be evicted before the selected one loads.
    other = _ok_response(status=200, text=json.dumps({"data": [
        {"key": "other", "loaded_instances": [{"id": "inst-other"}]},
    ]}))
    with patch("requests.get", return_value=other), \
         patch("requests.post", return_value=_ok_response(status=200)) as post, \
         patch.object(lm_http, "load_model", return_value=True) as load:
        lm_http.ensure_model_loaded("s", LOCAL_V1, "", "m")
    load.assert_called_once()
    unloads = [c for c in post.call_args_list
               if c[0][0].endswith("/api/v1/models/unload")]
    assert unloads, "expected the resident 'other' model to be unloaded first"
    assert unloads[0].kwargs["json"] == {"instance_id": "inst-other"}
    # The selected model is now the server-scoped resident.
    assert lm_http._server_loaded[LOCAL_V1] == ("m", 8192, 1.0, None, None, None, None)


def test_maybe_unload_old_uses_known_instance_id():
    # When the v1 load returned an instance_id, the unload must target it via the v1 route.
    lm_http._model_instances[(LOCAL_V1, "oldmodel")] = "inst-7"
    lm_http._last_loaded["slotI"] = "oldmodel"
    with patch("requests.post", return_value=_ok_response(status=200)) as post:
        lm_http.maybe_unload_old("slotI", LOCAL_V1, "newmodel")
    unload = [c for c in post.call_args_list
              if c[0][0].endswith("/api/v1/models/unload")]
    assert unload, "expected a v1 unload by instance_id"
    assert unload[0].kwargs["json"] == {"instance_id": "inst-7"}


def test_maybe_unload_old_stale_instance_id_tolerated():
    # A 404 (already gone) on the v1 unload is treated as idempotent success, not an error.
    lm_http._model_instances[(LOCAL_V1, "oldmodel")] = "gone"
    lm_http._last_loaded["slotJ"] = "oldmodel"
    resp404 = _ok_response(status=404, text="not found")
    with patch("requests.post", return_value=resp404) as post:
        lm_http.maybe_unload_old("slotJ", LOCAL_V1, "newmodel")
    unload = [c for c in post.call_args_list
              if c[0][0].endswith("/api/v1/models/unload")]
    assert unload
    assert unload[0].kwargs["json"] == {"instance_id": "gone"}


def test_ensure_model_loaded_reloads_on_config_change():
    # Changing context_length changes the fingerprint -> unload + reload.
    with patch("requests.get", return_value=_ok_response(status=404)), \
         patch.object(lm_http, "load_model", return_value=True) as load:
        lm_http.ensure_model_loaded("s", LOCAL_V1, "", "m", context_length=16384)
    load.assert_called_once()
    # The new fingerprint reflects the changed context_length.
    assert lm_http._last_loaded["s"] == ("m", 16384, 1.0, None, None, None, None)
    assert lm_http._server_loaded[LOCAL_V1] == ("m", 16384, 1.0, None, None, None, None)


def test_ensure_model_loaded_old_string_state_forces_reload():
    # A pre-fingerprint string value compares unequal to the tuple and triggers a one-time reload.
    lm_http._last_loaded["s"] = "m"
    with patch("requests.get", return_value=_ok_response(status=404)), \
         patch.object(lm_http, "load_model", return_value=True) as load:
        lm_http.ensure_model_loaded("s", LOCAL_V1, "", "m")
    load.assert_called_once()


def test_ensure_model_loaded_returns_true_on_success():
    # A4: the function now reports success so callers can stop before a doomed LLM call.
    with patch("requests.get", return_value=_ok_response(status=404)), \
         patch.object(lm_http, "load_model", return_value=True):
        assert lm_http.ensure_model_loaded("s", LOCAL_V1, "", "m") is True


def test_ensure_model_loaded_returns_false_on_failure():
    with patch("requests.get", return_value=_ok_response(status=404)), \
         patch.object(lm_http, "load_model", return_value=False):
        assert lm_http.ensure_model_loaded("s", LOCAL_V1, "", "m") is False


def test_ensure_model_loaded_returns_true_for_placeholder():
    # A placeholder (no real model selected) is treated as "nothing to load" -> success.
    assert lm_http.ensure_model_loaded("s", LOCAL_V1, "", "— none —") is True


def test_ensure_model_loaded_returns_true_when_already_loaded():
    fp = ("m", 8192, 1.0, None, None, None, None)
    lm_http._server_loaded[LOCAL_V1] = fp
    with patch("requests.get") as get, \
         patch.object(lm_http, "load_model") as load:
        assert lm_http.ensure_model_loaded("s", LOCAL_V1, "", "m") is True
    load.assert_not_called()
    get.assert_not_called()




# ---------------------------------------------------------------------------
# Global LLM-response cache (opt-in, off by default).
# ---------------------------------------------------------------------------

def test_llm_cache_hit_avoids_second_network_call():
    # With the cache enabled, an identical repeat request is served from cache (one POST only).
    lm_http._LLM_CACHE_ENABLED = True
    lm_http._llm_response_cache.clear()
    try:
        ok = _ok_response(status=200,
                          text=json.dumps({"output": [{"type": "message",
                                                       "content": "hello"}]}))
        with patch("requests.get",
                   return_value=_models_response([{"key": "m", "capabilities": {}}])), \
             patch("requests.post", return_value=ok) as post:
            out1 = lm_http.chat_completion(LOCAL_V1, "", "m",
                                           [{"role": "user", "content": "hi"}], 0.7, 100)
            out2 = lm_http.chat_completion(LOCAL_V1, "", "m",
                                           [{"role": "user", "content": "hi"}], 0.7, 100)
        assert out1 == out2 == "hello"
        assert post.call_count == 1
    finally:
        lm_http._LLM_CACHE_ENABLED = False
        lm_http._llm_response_cache.clear()


def test_llm_cache_disabled_always_hits_network():
    # Default-off cache: identical requests still hit the network every time.
    assert lm_http._LLM_CACHE_ENABLED is False
    with patch("requests.get",
               return_value=_models_response([{"key": "m", "capabilities": {}}])), \
         patch("requests.post",
               return_value=_ok_response(status=200,
                                         text=json.dumps({"output": [{"type": "message",
                                                                      "content": "x"}]}))) as post:
        lm_http.chat_completion(LOCAL_V1, "", "m", [{"role": "user", "content": "hi"}], 0.7, 100)
        lm_http.chat_completion(LOCAL_V1, "", "m", [{"role": "user", "content": "hi"}], 0.7, 100)
    assert post.call_count == 2


def test_llm_cache_distinct_requests_not_merged():
    # Different messages -> different keys -> a real second call (not a cache hit).
    lm_http._LLM_CACHE_ENABLED = True
    lm_http._llm_response_cache.clear()
    try:
        ok = _ok_response(status=200,
                          text=json.dumps({"output": [{"type": "message",
                                                       "content": "x"}]}))
        with patch("requests.get",
                   return_value=_models_response([{"key": "m", "capabilities": {}}])), \
             patch("requests.post", return_value=ok) as post:
            lm_http.chat_completion(LOCAL_V1, "", "m", [{"role": "user", "content": "a"}], 0.7, 100)
            lm_http.chat_completion(LOCAL_V1, "", "m", [{"role": "user", "content": "b"}], 0.7, 100)
        assert post.call_count == 2
    finally:
        lm_http._LLM_CACHE_ENABLED = False
        lm_http._llm_response_cache.clear()


# ---------------------------------------------------------------------------
# unload_all_loaded: evict every resident model before loading the selected one.
# ---------------------------------------------------------------------------

def test_unload_all_loaded_unloads_every_instance():
    # Two loaded models (distinct instances) must both be unloaded by instance_id.
    models = _ok_response(status=200, text=json.dumps({"data": [
        {"key": "a", "loaded_instances": [{"id": "inst-a"}]},
        {"key": "b", "loaded_instances": [{"id": "inst-b"}]},
    ]}))
    with patch("requests.get", return_value=models) as get, \
         patch("requests.post", return_value=_ok_response(status=200)) as post:
        lm_http.unload_all_loaded(LOCAL_V1, "")
    assert get.call_args_list[0][0][0] == NATIVE_MODELS_URL
    unloads = [c for c in post.call_args_list
               if c[0][0].endswith("/api/v1/models/unload")]
    assert len(unloads) == 2
    assert {u.kwargs["json"]["instance_id"] for u in unloads} == {"inst-a", "inst-b"}


def test_unload_all_loaded_skips_except_model():
    # except_model is spared so an already-selected model is not needlessly evicted.
    models = _ok_response(status=200, text=json.dumps({"data": [
        {"key": "keep", "loaded_instances": [{"id": "inst-keep"}]},
        {"key": "drop", "loaded_instances": [{"id": "inst-drop"}]},
    ]}))
    with patch("requests.get", return_value=models), \
         patch("requests.post", return_value=_ok_response(status=200)) as post:
        lm_http.unload_all_loaded(LOCAL_V1, "", except_model="keep")
    unloads = [c for c in post.call_args_list
               if c[0][0].endswith("/api/v1/models/unload")]
    assert len(unloads) == 1
    assert unloads[0].kwargs["json"] == {"instance_id": "inst-drop"}


def test_unload_all_loaded_handles_errors_silently():
    # Network / HTTP / parse errors must never raise; the function simply does nothing.
    with patch("requests.get", side_effect=__import__("requests").RequestException("down")):
        lm_http.unload_all_loaded(LOCAL_V1, "")  # no raise
    with patch("requests.get", return_value=_ok_response(status=500)):
        lm_http.unload_all_loaded(LOCAL_V1, "")  # no raise
    with patch("requests.get", return_value=_ok_response(text="<html>")):
        lm_http.unload_all_loaded(LOCAL_V1, "")  # no raise (non-JSON)


def test_get_cached_models_fetches_when_empty():
    # When the cache is empty and fetching is allowed, a running server populates the list
    # (so a saved workflow validates without a manual Refresh). Uses a non-default key to
    # avoid perturbing the default-server static-key state.
    url = "http://127.0.0.1:9999/v1"
    resp = _ok_response(text=json.dumps({"data": [{"id": "real-model"}]}))
    with patch("requests.get", return_value=resp):
        models = lm_http.get_cached_models(url, "", allow_fetch=True, timeout=1)
    assert models == ["real-model"]


# ---------------------------------------------------------------------------
# Vision capability detection (server-authoritative).
# ---------------------------------------------------------------------------

def _models_response(models):
    return _ok_response(text=json.dumps({"data": models}))


def test_model_supports_vision_true():
    models = [{"key": "qwen2.5-vl-7b", "capabilities": {"vision": True}}]
    with patch("requests.get", return_value=_models_response(models)) as get:
        assert lm_http.model_supports_vision(LOCAL_V1, "", "qwen2.5-vl-7b") is True
    assert get.call_args[0][0] == NATIVE_MODELS_URL


def test_model_supports_vision_false():
    # Server truth wins over the name heuristic: a "vl" name can be text-only.
    models = [{"key": "qwen2.5-vl-7b", "capabilities": {"vision": False}}]
    with patch("requests.get", return_value=_models_response(models)):
        assert lm_http.model_supports_vision(LOCAL_V1, "", "qwen2.5-vl-7b") is False


def test_model_supports_vision_absent_capabilities_is_none():
    # capabilities present but no vision key -> unknown (None), do not guess.
    models = [{"key": "gemma4-12b", "capabilities": {}}]
    with patch("requests.get", return_value=_models_response(models)):
        assert lm_http.model_supports_vision(LOCAL_V1, "", "gemma4-12b") is None


def test_model_supports_vision_model_not_listed_is_none():
    models = [{"key": "other-model", "capabilities": {"vision": True}}]
    with patch("requests.get", return_value=_models_response(models)):
        assert lm_http.model_supports_vision(LOCAL_V1, "", "gemma4-12b") is None


def test_model_supports_vision_matches_by_id():
    models = [{"id": "gemma4-12b", "capabilities": {"vision": True}}]
    with patch("requests.get", return_value=_models_response(models)):
        assert lm_http.model_supports_vision(LOCAL_V1, "", "gemma4-12b") is True


def test_model_supports_vision_matches_by_loaded_instance_id():
    models = [{"key": "k", "loaded_instances": [{"id": "gemma4-12b-instance"}],
               "capabilities": {"vision": True}}]
    with patch("requests.get", return_value=_models_response(models)):
        assert lm_http.model_supports_vision(LOCAL_V1, "", "gemma4-12b-instance") is True


def test_model_supports_vision_matches_by_variant():
    models = [{"key": "k", "variants": ["gemma4-12b-q4"], "capabilities": {"vision": True}}]
    with patch("requests.get", return_value=_models_response(models)):
        assert lm_http.model_supports_vision(LOCAL_V1, "", "gemma4-12b-q4") is True


def test_model_supports_vision_network_error_is_none():
    with patch("requests.get", side_effect=__import__("requests").RequestException("down")):
        assert lm_http.model_supports_vision(LOCAL_V1, "", "m") is None


def test_model_supports_vision_non200_is_none():
    with patch("requests.get", return_value=_ok_response(status=404)):
        assert lm_http.model_supports_vision(LOCAL_V1, "", "m") is None


def test_model_supports_vision_nonjson_is_none():
    with patch("requests.get", return_value=_ok_response(text="<html>")):
        assert lm_http.model_supports_vision(LOCAL_V1, "", "m") is None


def test_resolve_vision_server_true_overrides_name():
    # The name heuristic says False for "m", but the server truth (True) must win.
    models = [{"key": "m", "capabilities": {"vision": True}}]
    with patch("requests.get", return_value=_models_response(models)):
        assert lm_http.resolve_vision(LOCAL_V1, "", "m") is True


def test_resolve_vision_server_false_overrides_name():
    # Name looks like vision but server says no -> respect the server (fixes qwen2.5-vl false positive).
    models = [{"key": "qwen2.5-vl-7b", "capabilities": {"vision": False}}]
    with patch("requests.get", return_value=_models_response(models)):
        assert lm_http.resolve_vision(LOCAL_V1, "", "qwen2.5-vl-7b") is False


def test_resolve_vision_none_falls_back_to_name():
    # Server unreachable -> fall back to the name heuristic (qwen2.5-vl is a vision name).
    with patch("requests.get", side_effect=__import__("requests").RequestException("down")):
        assert lm_http.resolve_vision(LOCAL_V1, "", "qwen2.5-vl-7b") is True


# ---------------------------------------------------------------------------
# load_model v1 body key + legacy gating.
# ---------------------------------------------------------------------------

def test_load_model_v1_body_uses_snake_case_gpu_offload():
    resp = _ok_response(status=200)
    with patch("requests.post", return_value=resp) as post:
        lm_http.load_model(LOCAL_V1, "", "m", backoff=0)
    body = post.call_args_list[0][1]["json"]
    assert body["gpu_offload"] == 1.0
    assert "gpuOffload" not in body


def test_load_model_legacy_body_keeps_camel_case_gpuOffload():
    native = _ok_response(status=404, text="not found")
    legacy = _ok_response(status=200)
    with patch("requests.post", side_effect=[native, legacy]) as post:
        lm_http.load_model(LOCAL_V1, "", "m", retries=1, backoff=0)
    # second call is the legacy /v1 endpoint (legacy_body uses gpuOffload)
    body = post.call_args_list[1][1]["json"]
    assert body["gpuOffload"] == 1.0
    assert "gpu_offload" not in body


def test_load_model_v1_400_unrecognized_key_is_dropped_and_retried():
    # A server that rejects an optional key (e.g. gpu_offload) by name must not hard-fail:
    # load_model strips the rejected key and retries the v1 endpoint. The key is dropped
    # (not re-sent as camelCase), so the retry body omits it entirely and the load succeeds
    # on the v1 route rather than falling through to legacy.
    snake = _ok_response(status=400,
                         text='{"error":{"code":"unrecognized_keys",'
                              '"message":"Unrecognized key(s) in object: \'gpu_offload\'"}}')
    ok = _ok_response(status=200)
    with patch("requests.post", side_effect=[snake, ok]) as post:
        result = lm_http.load_model(LOCAL_V1, "", "m", retries=1, backoff=0)
    assert result is True
    # Both attempts went to the native v1 endpoint; legacy was never called.
    assert post.call_count == 2
    assert all(c[0][0].endswith("/api/v1/models/load") for c in post.call_args_list)
    # The rejected key is gone from the retry (and the camelCase alias is never invented).
    retry_body = post.call_args_list[1][1]["json"]
    assert "gpu_offload" not in retry_body
    assert "gpuOffload" not in retry_body


def test_load_model_v1_400_unrecognized_keys_list_dropped():
    # The error may also list offending keys under an explicit "keys" field.
    snake = _ok_response(status=400,
                         text='{"error":{"code":"unrecognized_keys",'
                              '"keys":["offload_kv_cache_to_gpu"]}}')
    ok = _ok_response(status=200)
    with patch("requests.post", side_effect=[snake, ok]) as post:
        result = lm_http.load_model(LOCAL_V1, "", "m",
                                    offload_kv_cache_to_gpu=True, retries=1, backoff=0)
    assert result is True
    retry_body = post.call_args_list[1][1]["json"]
    assert "offload_kv_cache_to_gpu" not in retry_body


def test_load_model_v1_400_non_gpu_error_does_not_fall_through():
    # A genuine v1 rejection that is NOT a gpu-key issue must return False and must NOT call
    # the legacy route (which would answer 200 "Unexpected endpoint" and mask the failure).
    native = _ok_response(status=400, text="invalid model identifier")
    legacy = _ok_response(status=200)  # would be the false success
    with patch("requests.post", side_effect=[native, legacy]) as post:
        ok = lm_http.load_model(LOCAL_V1, "", "m", retries=1, backoff=0)
    assert ok is False
    # Only the native v1 endpoint was attempted; legacy was never called.
    assert post.call_count == 1
    assert post.call_args_list[0][0][0].endswith("/api/v1/models/load")


def test_load_model_v1_404_attempts_legacy():
    # 404 from the v1 route means an old server without that endpoint -> legacy is tried.
    native = _ok_response(status=404, text="not found")
    legacy = _ok_response(status=200)
    v0 = _ok_response(status=404)
    with patch("requests.post", side_effect=[native, legacy, v0]) as post:
        ok = lm_http.load_model(LOCAL_V1, "", "m", retries=1, backoff=0)
    assert ok is True
    # native 404 falls through to the legacy /v1 endpoint, which succeeds (no v0 needed).
    assert post.call_count == 2
    assert post.call_args_list[1][0][0].endswith("/v1/models/m/load")


# ---------------------------------------------------------------------------
# chat_completion error enrichment.
# ---------------------------------------------------------------------------

def test_chat_completion_enriches_vision_rejection():
    msg = ("The provided messages contain images, but model does not support image inputs.")
    resp = _ok_response(status=400, text=json.dumps({"error": {"message": msg}}))
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Describe."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,xxx"}},
    ]}]
    with patch("requests.post", return_value=resp):
        with pytest.raises(RuntimeError) as exc:
            lm_http.chat_completion(LOCAL_V1, "", "m", messages, 0.7, 100)
    assert "does not support image inputs" in str(exc.value)
    assert "vision-capable model" in str(exc.value)


def test_chat_completion_generic_error_when_no_images():
    resp = _ok_response(status=400, text=json.dumps({"error": {"message": "bad request"}}))
    messages = [{"role": "user", "content": "hello"}]
    with patch("requests.post", return_value=resp):
        with pytest.raises(RuntimeError) as exc:
            lm_http.chat_completion(LOCAL_V1, "", "m", messages, 0.7, 100)
    assert "does not support image inputs" not in str(exc.value)
    assert "HTTP 400" in str(exc.value)


def test_chat_completion_enriches_model_load_failure():
    msg = ('Failed to load model "m". Error: Engine protocol runtime llama-server '
           "exited before becoming healthy. exitCode=3221226505")
    resp = _ok_response(status=400, text=json.dumps({"error": {"message": msg}}))
    messages = [{"role": "user", "content": "hello"}]
    with patch("requests.post", return_value=resp):
        with pytest.raises(RuntimeError) as exc:
            lm_http.chat_completion(LOCAL_V1, "", "m", messages, 0.7, 100)
    assert "failed to load on the server" in str(exc.value)
    assert "exited before becoming healthy" in str(exc.value)


# ---------------------------------------------------------------------------
# B4: native /api/v1/models entry cache + architecture / param_count helpers.
# ---------------------------------------------------------------------------

def test_get_model_entry_matches_and_caches():
    models = _models_response([{"key": "gemma-3-12b", "architecture": "gemma3",
                                "params_string": "12B", "loaded_instances": []}])
    with patch("requests.get", return_value=models):
        entry = lm_http.get_model_entry(LOCAL_V1, "", "gemma-3-12b")
    assert entry is not None
    assert entry["architecture"] == "gemma3"
    # A second call within the TTL must not re-hit the network (cache returns the object).
    with patch("requests.get", side_effect=AssertionError("should be cached")):
        assert lm_http.get_model_entry(LOCAL_V1, "", "gemma-3-12b") is entry


def test_get_model_entry_not_listed_returns_none():
    models = _models_response([{"key": "other"}])
    with patch("requests.get", return_value=models):
        assert lm_http.get_model_entry(LOCAL_V1, "", "missing") is None


def test_get_model_entry_network_error_returns_none():
    with patch("requests.get", side_effect=__import__("requests").RequestException("down")):
        assert lm_http.get_model_entry(LOCAL_V1, "", "m") is None


def test_model_architecture_and_param_count():
    entry = {"key": "m", "architecture": "gemma3", "params_string": "26B-A4B"}
    with patch.object(lm_http, "get_model_entry", return_value=entry):
        assert lm_http.model_architecture(LOCAL_V1, "", "m") == "gemma3"
        assert lm_http.model_param_count(LOCAL_V1, "", "m") == 4.0
    # MoE suffix absent -> single number.
    with patch.object(lm_http, "get_model_entry", return_value={"params_string": "7B"}):
        assert lm_http.model_param_count(LOCAL_V1, "", "m") == 7.0
    # No entry -> None (graceful fallback, no regression on non-LM-Studio servers).
    with patch.object(lm_http, "get_model_entry", return_value=None):
        assert lm_http.model_architecture(LOCAL_V1, "", "m") is None
        assert lm_http.model_param_count(LOCAL_V1, "", "m") is None


def test_invalidate_model_entry_cache():
    with patch("requests.get", return_value=_models_response([{"key": "m"}])):
        assert lm_http.get_model_entry(LOCAL_V1, "", "m") is not None
    lm_http.invalidate_model_entry_cache(LOCAL_V1, "m")
    with patch("requests.get", side_effect=AssertionError("should refetch after invalidation")):
        with pytest.raises(AssertionError):
            lm_http.get_model_entry(LOCAL_V1, "", "m")
