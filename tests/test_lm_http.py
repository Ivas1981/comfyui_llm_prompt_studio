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
    # Avoid cross-test contamination of the global "last loaded" cache.
    saved = dict(lm_http._last_loaded)
    lm_http._last_loaded.clear()
    yield
    lm_http._last_loaded.clear()
    lm_http._last_loaded.update(saved)


def test_validate_server_url_allows_local():
    assert lm_http.validate_server_url("http://localhost:1234/v1") == "http://localhost:1234/v1"
    assert lm_http.validate_server_url("http://127.0.0.1:1234/v1") == "http://127.0.0.1:1234/v1"


def test_validate_server_url_rejects_public():
    with pytest.raises(ValueError):
        lm_http.validate_server_url("http://1.2.3.4/v1")


def test_fetch_models_returns_ids():
    resp = _ok_response(text=json.dumps({"data": [{"id": "a"}, {"id": "b"}]}))
    with patch("requests.get", return_value=resp) as get:
        models = lm_http.fetch_models(LOCAL_V1)
    assert models == ["a", "b"]
    get.assert_called_once()


def test_fetch_models_handles_nonjson():
    resp = _ok_response(text="<html>error</html>")
    resp.json.side_effect = ValueError("No JSON")
    with patch("requests.get", return_value=resp):
        with pytest.raises(RuntimeError):
            lm_http.fetch_models(LOCAL_V1)


def test_chat_completion_handles_nonjson_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.text = "<html>something went wrong</html>"
    resp.json.side_effect = ValueError("No JSON")
    with patch("requests.post", return_value=resp):
        with pytest.raises(RuntimeError):
            lm_http.chat_completion(LOCAL_V1, "", "m", [], 0.7, 100)


def test_chat_completion_handles_missing_choices():
    resp = MagicMock()
    resp.status_code = 200
    resp.text = json.dumps({"ok": True})
    resp.json.return_value = {"ok": True}
    with patch("requests.post", return_value=resp):
        with pytest.raises(RuntimeError):
            lm_http.chat_completion(LOCAL_V1, "", "m", [], 0.7, 100)


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
    with patch("requests.post") as post:
        lm_http.maybe_unload_old("slotX", LOCAL_V1, "newmodel")
    assert lm_http._last_loaded["slotX"] == "newmodel"
    # An unload request for the previous model was issued.
    unload_calls = [c for c in post.call_args_list
                    if c[0][0].endswith("/api/v0/models/unload")]
    assert unload_calls, "expected an unload request for the previous model"
    assert unload_calls[0].kwargs["json"] == {"model": "oldmodel"}


def test_maybe_unload_old_url_has_no_double_v1_prefix():
    # Regression: the unload URL must be /api/v0/models/unload, not /v1/api/v0/...,
    # when server_url ends in /v1. Uses _server_root() (same helper as load_model).
    lm_http._last_loaded["slotZ"] = "oldmodel"
    with patch("requests.post") as post:
        lm_http.maybe_unload_old("slotZ", LOCAL_V1, "newmodel")
    url = post.call_args_list[0][0][0]
    assert "/v1/api/v0/" not in url
    assert url == "http://localhost:1234/api/v0/models/unload"


def test_maybe_unload_old_skips_when_same_model():
    lm_http._last_loaded["slotY"] = "same"
    with patch("requests.post") as post:
        lm_http.maybe_unload_old("slotY", LOCAL_V1, "same")
    assert post.call_count == 0
    assert lm_http._last_loaded["slotY"] == "same"


def test_ensure_model_loaded_marks_loaded_on_success():
    with patch.object(lm_http, "load_model", return_value=True) as load:
        lm_http.ensure_model_loaded("s", LOCAL_V1, "", "m")
    load.assert_called_once()
    assert lm_http._last_loaded["s"] == "m"


def test_ensure_model_loaded_skips_when_already_loaded():
    lm_http._last_loaded["s"] = "m"
    with patch.object(lm_http, "load_model") as load:
        lm_http.ensure_model_loaded("s", LOCAL_V1, "", "m")
    load.assert_not_called()


def test_ensure_model_loaded_nulls_state_on_failure(caplog):
    lm_http._last_loaded["s"] = "prev"
    with patch.object(lm_http, "load_model", return_value=False) as load:
        lm_http.ensure_model_loaded("s", LOCAL_V1, "", "m")
    load.assert_called_once()
    # Failure leaves state as None so the next run will retry; warning is logged.
    assert lm_http._last_loaded["s"] is None
    assert "was not loaded" in caplog.text


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


def test_load_model_v1_400_does_not_fall_through_to_legacy():
    # Modern server rejects the v1 load (400) but the legacy route would have answered 200
    # ("Unexpected endpoint"). A genuine rejection must return False and must NOT call legacy.
    native = _ok_response(status=400, text="Unrecognized key(s): 'gpuOffload'")
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
