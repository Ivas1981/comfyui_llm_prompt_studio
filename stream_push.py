"""Push streaming generation chunks to the ComfyUI frontend over the websocket.

The server-side node calls :func:`push_stream_chunk` while tokens arrive; the JS bridge
(``web/js/llm_prompt_studio_bridge.js``) listens for the ``llm_prompt_studio.stream`` event
and appends the chunk to the node's ``generation_view`` widget. Importing this module never
fails outside ComfyUI (e.g. in unit tests) — the ``server`` import is deferred and failures
are swallowed so streaming degrades to a no-op push rather than an error.
"""
from typing import Any


def push_stream_chunk(node_id: Any, chunk: str) -> None:
    if not chunk:
        return
    try:
        from server import PromptServer
        PromptServer.instance.send_json({
            "type": "llm_prompt_studio.stream",
            "node_id": node_id,
            "chunk": chunk,
        })
    except Exception:
        pass
