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
