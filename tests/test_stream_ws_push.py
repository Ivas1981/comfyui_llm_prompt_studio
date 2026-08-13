import os
import sys
import types

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PKG)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from comfyui_llm_prompt_studio import stream_push  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402


def _fake_server(send_json):
    mod = types.ModuleType("server")
    mod.PromptServer = types.SimpleNamespace()
    mod.PromptServer.instance = types.SimpleNamespace()
    mod.PromptServer.instance.send_json = send_json
    return mod


def test_push_stream_chunk_sends_event():
    fake = MagicMock()
    with patch.dict(sys.modules, {"server": _fake_server(fake)}):
        stream_push.push_stream_chunk("42", "hello")
    fake.assert_called_once()
    payload = fake.call_args[0][0]
    assert payload["type"] == "llm_prompt_studio.stream"
    assert payload["node_id"] == "42"
    assert payload["chunk"] == "hello"


def test_push_stream_chunk_empty_is_noop():
    fake = MagicMock()
    with patch.dict(sys.modules, {"server": _fake_server(fake)}):
        stream_push.push_stream_chunk("42", "")
    fake.assert_not_called()


def test_push_stream_chunk_without_server_is_safe():
    # When ComfyUI's `server` module is unavailable (e.g. unit tests), push is a no-op.
    with patch.dict(sys.modules, {"server": None}):
        stream_push.push_stream_chunk("42", "hi")  # should not raise
