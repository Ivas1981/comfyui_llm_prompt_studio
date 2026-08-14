import json
import os
import sys

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PKG)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from comfyui_llm_prompt_studio import lm_http  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402
import pytest  # noqa: E402
import json  # noqa: E402


def _ok_response(status=200, text="{}"):
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    try:
        resp.json.return_value = json.loads(text) if text.strip() else {}
    except ValueError:
        resp.json.side_effect = ValueError("No JSON could be decoded")
    resp.headers = {"Content-Type": "application/json"}
    resp.ok = status < 400
    resp.status = status
    return resp


LOCAL_V1 = "http://localhost:1234/v1"


def _sse_response(events):
    resp = MagicMock()
    resp.status_code = 200
    resp.text = ""
    lines = []
    for ev in events:
        lines.append(f"event: {ev['event']}")
        lines.append(f"data: {ev['data']}")
        lines.append("")  # blank line terminates an SSE event
    resp.iter_lines.return_value = iter(lines)
    return resp


def test_v1_chat_non_streaming_uses_native_endpoint():
    data = {"output": [{"type": "message", "content": "hello world"}]}
    resp = MagicMock()
    resp.status_code = 200
    resp.text = json.dumps(data)
    resp.json.return_value = data
    with patch("requests.post", return_value=resp) as post:
        out = lm_http.chat_completion(LOCAL_V1, "", "m",
                                      [{"role": "user", "content": "hi"}],
                                      0.7, 100, reasoning="low")
    assert out == "hello world"
    # URL must be derived from the server root (strip the trailing /v1).
    assert post.call_args_list[0][0][0].endswith("/api/v1/chat")
    assert "/v1/api/v1/chat" not in post.call_args_list[0][0][0]


def test_v1_chat_streaming_aggregates_and_pushes():
    events = [
        {"event": "message.delta",
         "data": json.dumps({"type": "message.delta", "content": "Hello "})},
        {"event": "message.delta",
         "data": json.dumps({"type": "message.delta", "content": "world"})},
        {"event": "chat.end",
         "data": json.dumps({"type": "chat.end",
                             "result": {"output": [{"type": "message",
                                                    "content": "Hello world"}]}})},
    ]
    resp = _sse_response(events)
    chunks = []
    with patch("requests.post", return_value=resp):
        out = lm_http.chat_completion(LOCAL_V1, "", "m",
                                      [{"role": "user", "content": "hi"}],
                                      0.7, 100, stream=True, on_delta=chunks.append)
    assert out == "Hello world"
    assert chunks == ["Hello ", "world"]


def test_v1_streaming_error_event_raises():
    events = [{"event": "error",
               "data": json.dumps({"type": "error", "message": "boom"})}]
    resp = _sse_response(events)
    with patch("requests.post", return_value=resp):
        with pytest.raises(RuntimeError):
            lm_http.chat_completion(LOCAL_V1, "", "m",
                                   [{"role": "user", "content": "hi"}],
                                   0.7, 100, stream=True)


def test_v1_streaming_falls_back_on_transport_error():
    # A connection error during streaming must fall back to a normal non-streaming call
    # (which the OpenAI path serves from the patched responses).
    ok = MagicMock()
    ok.status_code = 200
    ok.text = json.dumps({"choices": [{"message": {"content": "fallback text"}}]})
    ok.json.return_value = {"choices": [{"message": {"content": "fallback text"}}]}
    with patch("requests.post", side_effect=[RuntimeError("connection reset"), ok]) as post:
        out = lm_http.chat_completion(LOCAL_V1, "", "m",
                                      [{"role": "user", "content": "hi"}],
                                      0.7, 100, stream=True)
    assert out == "fallback text"
    # First attempt was the native streaming endpoint, second the OpenAI fallback.
    assert post.call_args_list[0][0][0].endswith("/api/v1/chat")
    assert post.call_args_list[1][0][0].endswith("/chat/completions")


def test_streaming_failure_resets_then_delivers_full_text():
    # On a streaming failure, on_reset must fire (so the UI clears stale partial text) and the
    # non-streaming fallback's full text is delivered exactly once via on_delta.
    ok = MagicMock()
    ok.status_code = 200
    ok.text = json.dumps({"choices": [{"message": {"content": "full fallback result"}}]})
    ok.json.return_value = {"choices": [{"message": {"content": "full fallback result"}}]}
    reset = MagicMock()
    deltas = []
    with patch("requests.post", side_effect=[RuntimeError("stream died"), ok]):
        out = lm_http.chat_completion(LOCAL_V1, "", "m",
                                      [{"role": "user", "content": "hi"}],
                                      0.7, 100, stream=True,
                                      on_delta=deltas.append, on_reset=reset)
    assert out == "full fallback result"
    reset.assert_called_once()
    # The complete fallback text is shown once; no partial left behind.
    assert deltas == ["full fallback result"]


# ---------------------------------------------------------------------------
# Native /api/v1/chat request construction (system_prompt + input parts).
# ---------------------------------------------------------------------------

def test_v1_chat_builds_native_input_with_system_prompt_and_image():
    # Native chat must NOT send OpenAI `messages` as `input`; it must split system into a
    # top-level system_prompt and render the rest as typed {type, content/data_url} parts,
    # converting image_url -> image with the full data URL.
    data = {"output": [{"type": "message", "content": "done"}]}
    resp = MagicMock()
    resp.status_code = 200
    resp.text = json.dumps(data)
    resp.json.return_value = data
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": "Describe this."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ABC"}},
        ]},
    ]
    with patch("requests.get", return_value=MagicMock(status_code=404)), \
         patch("requests.post", return_value=resp) as post:
        out = lm_http.chat_completion(LOCAL_V1, "", "m", messages, 0.7, 100)
    assert out == "done"
    body = post.call_args_list[0][1]["json"]
    assert body["system_prompt"] == "You are a helpful assistant."
    assert body["store"] is False
    assert "messages" not in body
    parts = body["input"]
    assert parts[0] == {"type": "text", "content": "Describe this."}
    assert parts[1] == {"type": "image", "data_url": "data:image/jpeg;base64,ABC"}


def test_v1_chat_omits_reasoning_when_model_has_no_capability():
    # A model that exposes no reasoning configuration must NOT send the `reasoning` param;
    # sending it yields a 400 ("does not expose reasoning configuration").
    data = {"output": [{"type": "message", "content": "ok"}]}
    resp = MagicMock()
    resp.status_code = 200
    resp.text = json.dumps(data)
    resp.json.return_value = data
    # model_supports_reasoning probes /api/v1/models -> return a model with no reasoning key.
    models = _ok_response(status=200, text=json.dumps(
        {"data": [{"key": "m", "capabilities": {"vision": False}}]}))
    with patch("requests.get", return_value=models), \
         patch("requests.post", return_value=resp) as post:
        out = lm_http.chat_completion(LOCAL_V1, "", "m",
                                      [{"role": "user", "content": "hi"}], 0.7, 100,
                                      reasoning="on")
    assert out == "ok"
    body = post.call_args_list[0][1]["json"]
    assert "reasoning" not in body


def test_v1_chat_retries_without_reasoning_on_rejection():
    # Server rejects `reasoning` for this model -> the client must retry once without it.
    reject = _ok_response(
        status=400,
        text=json.dumps({"error": {"message":
            "Model does not expose reasoning configuration"}}))
    ok = MagicMock()
    ok.status_code = 200
    ok.text = json.dumps({"output": [{"type": "message", "content": "retried"}]})
    ok.json.return_value = {"output": [{"type": "message", "content": "retried"}]}
    # model_supports_reasoning: report that reasoning IS allowed (so it tries to send it).
    models = _ok_response(status=200, text=json.dumps(
        {"data": [{"key": "m", "capabilities": {"reasoning": {"allowed_options": ["off", "on"]}}}]}))
    with patch("requests.get", return_value=models), \
         patch("requests.post", side_effect=[reject, ok]) as post:
        out = lm_http.chat_completion(LOCAL_V1, "", "m",
                                      [{"role": "user", "content": "hi"}], 0.7, 100,
                                      reasoning="on")
    assert out == "retried"
    assert post.call_count == 2
    assert "reasoning" not in post.call_args_list[1][1]["json"]


def test_v1_chat_maps_reasoning_level_to_allowed_value():
    # gemma allows only off/on -> "high" must be mapped to "on" (strongest allowed).
    assert lm_http.map_reasoning_level("high", ["off", "on"]) == "on"
    assert lm_http.map_reasoning_level("off", ["off", "on"]) == "off"
    # No reasoning capability reported -> omit the param entirely.
    assert lm_http.map_reasoning_level("on", None) is None
    assert lm_http.map_reasoning_level("on", []) is None


def test_v1_chat_retries_without_reasoning_on_empty_message():
    # A thinking model with reasoning=on can return an empty `message` (all text in the
    # `reasoning` blob). The client must retry once without reasoning and surface the answer.
    lm_http._reasoning_cap_cache.clear()  # avoid cross-test pollution of the probe cache
    empty = {"output": [{"type": "reasoning", "content": "hmm thinking..."},
                        {"type": "message", "content": ""}]}
    ok = {"output": [{"type": "message", "content": "THE ANSWER"}]}
    empty_resp = MagicMock()
    empty_resp.status_code = 200
    empty_resp.text = json.dumps(empty)
    empty_resp.json.return_value = empty
    ok_resp = MagicMock()
    ok_resp.status_code = 200
    ok_resp.text = json.dumps(ok)
    ok_resp.json.return_value = ok
    models = _ok_response(status=200, text=json.dumps(
        {"data": [{"key": "m", "capabilities": {"reasoning": {"allowed_options": ["off", "on"]}}}]}))
    with patch("requests.get", return_value=models), \
         patch("requests.post", side_effect=[empty_resp, ok_resp]) as post:
        out = lm_http.chat_completion(LOCAL_V1, "", "m",
                                     [{"role": "user", "content": "hi"}], 0.7, 100,
                                     reasoning="on")
    assert out == "THE ANSWER"
    assert post.call_count == 2
    # The retry request must omit the reasoning param entirely.
    assert "reasoning" not in post.call_args_list[1][1]["json"]


def test_v1_chat_no_reasoning_retry_when_message_present():
    # When the message is non-empty there must be NO extra retry (single POST).
    ok = {"output": [{"type": "message", "content": "fine"}]}
    ok_resp = MagicMock()
    ok_resp.status_code = 200
    ok_resp.text = json.dumps(ok)
    ok_resp.json.return_value = ok
    models = _ok_response(status=200, text=json.dumps(
        {"data": [{"key": "m", "capabilities": {"reasoning": {"allowed_options": ["off", "on"]}}}]}))
    with patch("requests.get", return_value=models), \
         patch("requests.post", return_value=ok_resp) as post:
        out = lm_http.chat_completion(LOCAL_V1, "", "m",
                                     [{"role": "user", "content": "hi"}], 0.7, 100,
                                     reasoning="on")
    assert out == "fine"
    assert post.call_count == 1

