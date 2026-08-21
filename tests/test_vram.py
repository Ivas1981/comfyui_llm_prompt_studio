import os
import sys

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PKG)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from comfyui_llm_prompt_studio import lm_http, vram  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402
import pytest  # noqa: E402

LOCAL_V1 = "http://localhost:1234/v1"
LAN_V1 = "http://192.168.1.50:1234/v1"
NATIVE_MODELS_URL = "http://localhost:1234/api/v1/models"


def _ok(status=200, text="{}"):
    r = MagicMock()
    r.status_code = status
    r.text = text
    try:
        r.json.return_value = __import__("json").loads(text) if text.strip() else {}
    except ValueError:
        r.json.side_effect = ValueError("no json")
    r.headers = {"Content-Type": "application/json"}
    return r


@pytest.fixture(autouse=True)
def _reset_state():
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


# ---------------------------------------------------------------------------
# release_model
# ---------------------------------------------------------------------------

def test_release_model_unloads_every_instance():
    lm_http._last_loaded[LOCAL_V1 + "::writer"] = ("m", 8192, 1.0, None, None, None, None)
    listed = _ok(text=__import__("json").dumps({"data": [
        {"key": "m", "loaded_instances": [{"id": "inst-1"}, {"id": "inst-2"}]},
    ]}))
    with patch("requests.get", return_value=listed) as get, \
         patch("requests.post", return_value=_ok()) as post:
        lm_http.release_model(LOCAL_V1, "")
    assert get.call_args_list[0][0][0] == NATIVE_MODELS_URL
    unloads = [c for c in post.call_args_list if c[0][0].endswith("/api/v1/models/unload")]
    assert len(unloads) == 2
    assert unloads[0].kwargs["json"] == {"instance_id": "inst-1"}
    assert unloads[1].kwargs["json"] == {"instance_id": "inst-2"}


def test_release_model_clears_all_state_and_reload_issues_fresh_load():
    fp = ("m", 8192, 1.0, None, None, None, None)
    # Pre-mark as loaded so ensure_model_loaded would otherwise skip.
    lm_http._server_loaded[LOCAL_V1] = fp
    lm_http._last_loaded[LOCAL_V1 + "::writer"] = fp
    lm_http._model_instances[(LOCAL_V1, "m")] = "inst-x"
    listed = _ok(text=__import__("json").dumps({"data": [
        {"key": "m", "loaded_instances": [{"id": "inst-x"}]},
    ]}))
    with patch("requests.get", return_value=listed), \
         patch("requests.post", return_value=_ok()):
        lm_http.release_model(LOCAL_V1, "")
    # State fully invalidated.
    assert LOCAL_V1 not in lm_http._server_loaded
    assert lm_http._last_loaded.get(LOCAL_V1 + "::writer", "GONE") == "GONE"
    assert (LOCAL_V1, "m") not in lm_http._model_instances
    # A subsequent ensure_model_loaded must NOT skip the reload (regression trap).
    with patch("requests.get", return_value=_ok(status=404)), \
         patch.object(lm_http, "load_model", return_value=True) as load:
        lm_http.ensure_model_loaded("s", LOCAL_V1, "", "m")
    load.assert_called_once()


def test_release_model_no_http_when_nothing_loaded():
    with patch("requests.get") as get, \
         patch("requests.post") as post:
        lm_http.release_model(LOCAL_V1, "")
    get.assert_not_called()
    post.assert_not_called()
    # Must not raise.
    assert True


def test_release_model_clears_both_url_spellings():
    # _server_loaded is keyed by the raw URL passed to ensure_model_loaded; make sure a
    # release keyed by the normalized URL also drops the raw spelling.
    raw = LOCAL_V1 + "/"
    fp = ("m", 8192, 1.0, None, None, None, None)
    lm_http._server_loaded[raw] = fp
    lm_http._last_loaded[raw + "::writer"] = fp
    lm_http._model_instances[(raw, "m")] = "inst-x"
    listed = _ok(text=__import__("json").dumps({"data": []}))
    with patch("requests.get", return_value=listed), \
         patch("requests.post", return_value=_ok()):
        lm_http.release_model(raw, "")
    assert raw not in lm_http._server_loaded
    assert lm_http._last_loaded.get(raw + "::writer", "GONE") == "GONE"
    assert (raw, "m") not in lm_http._model_instances


# ---------------------------------------------------------------------------
# wait_until_unloaded
# ---------------------------------------------------------------------------

def test_wait_until_unloaded_true_when_second_poll_empty():
    statuses = [{"reachable": True, "loaded_models": ["m"]},
                {"reachable": True, "loaded_models": []}]
    with patch.object(lm_http, "server_status", side_effect=statuses), \
         patch.object(lm_http.time, "sleep") as sleep:
        ok = lm_http.wait_until_unloaded(LOCAL_V1, "")
    assert ok is True
    sleep.assert_called()


def test_wait_until_unloaded_false_on_timeout_no_raise():
    with patch.object(lm_http, "server_status",
                      return_value={"reachable": True, "loaded_models": ["m"]}), \
         patch.object(lm_http.time, "sleep"):
        ok = lm_http.wait_until_unloaded(LOCAL_V1, "", timeout=0.01, interval=0.001)
    assert ok is False


# ---------------------------------------------------------------------------
# vram.release_before_sample (LM Studio only; ComfyUI models are never evicted)
# ---------------------------------------------------------------------------

def test_release_before_sample_no_http_when_no_server_seen():
    with patch("requests.get") as get, patch("requests.post") as post:
        vram.release_before_sample()
    get.assert_not_called()
    post.assert_not_called()


def test_release_before_sample_skips_kept_server():
    lm_http._seen_servers.add(LOCAL_V1.rstrip("/"))
    lm_http._keep_loaded.add(LOCAL_V1.rstrip("/"))
    with patch("requests.get") as get, patch("requests.post") as post:
        vram.release_before_sample()
    get.assert_not_called()
    post.assert_not_called()


def test_release_before_sample_releases_non_kept_server():
    norm = LOCAL_V1.rstrip("/")
    lm_http._seen_servers.add(norm)
    # not kept -> release path runs; release_model is a no-op for no state, so no HTTP,
    # but ensure it is invoked (proves the branch is taken) by spying on it.
    with patch.object(lm_http, "release_model") as rm, \
         patch.object(lm_http, "wait_until_unloaded", return_value=True):
        vram.release_before_sample()
    rm.assert_called_once_with(norm, "", None)


# ---------------------------------------------------------------------------
# release_enabled (global kill switch) and coercion
# ---------------------------------------------------------------------------

def test_keep_loaded_env_disables_release():
    vram._KEEP_LOADED = True
    try:
        assert vram.release_enabled() is False
    finally:
        vram._KEEP_LOADED = False


def test_coerce_bool_widget():
    assert vram.coerce_bool_widget("") is True
    assert vram.coerce_bool_widget(None) is True
    assert vram.coerce_bool_widget(False) is False
    assert vram.coerce_bool_widget(True) is True
    assert vram.coerce_bool_widget("") is True  # default True


def test_seen_servers_populated_by_load_and_fetch():
    lm_http._seen_servers.clear()
    with patch("requests.get", return_value=_ok(status=404)), \
         patch.object(lm_http, "load_model", return_value=True):
        lm_http.ensure_model_loaded("s", LOCAL_V1, "", "m")
    assert LOCAL_V1.rstrip("/") in lm_http.seen_servers()
    assert LOCAL_V1.rstrip("/") in lm_http._seen_servers
