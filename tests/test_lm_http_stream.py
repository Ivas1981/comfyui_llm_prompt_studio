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
