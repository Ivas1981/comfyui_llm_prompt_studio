import logging
import os
import sys

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PKG)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from comfyui_llm_prompt_studio import debug  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_logger():
    # The debug logger is a process-wide singleton; clear its handlers so each test gets a
    # fresh RotatingFileHandler pointed at its own tmp_path (and no leftover handlers).
    lg = logging.getLogger("llm_prompt_studio.debug")
    for h in list(lg.handlers):
        lg.removeHandler(h)
    debug._logger = None
    yield
    for h in list(lg.handlers):
        lg.removeHandler(h)
    debug._logger = None


def _set_output(tmp_path, monkeypatch, level):
    import folder_paths
    folder_paths.get_output_directory.return_value = str(tmp_path)
    monkeypatch.setattr(debug, "DEBUG_LEVEL", level)
    debug._logger = None


def test_mask_api_key():
    assert debug.mask_api_key("sk-1234567890abcdef") == "***...cdef"
    assert debug.mask_api_key("ab") == "***"
    assert debug.mask_api_key("") == ""
    assert debug.mask_api_key(None) == ""


def test_debug_off_no_file(tmp_path, monkeypatch):
    _set_output(tmp_path, monkeypatch, "OFF")
    debug.log("INFO", "NODE_ENTER", "1", {"x": 1})
    debug.log_node_enter("Writer", "1", {})
    assert debug._logger is None
    assert not (tmp_path / "llm_prompt_studio.log").exists()


def test_debug_minimal_only_nodes(tmp_path, monkeypatch):
    _set_output(tmp_path, monkeypatch, "MINIMAL")
    debug.log_node_enter("Writer", "1", {"idea": "cat"})
    debug.log_http_request("POST", "http://x/v1/chat", {}, {})  # filtered out in MINIMAL
    content = (tmp_path / "llm_prompt_studio.log").read_text(encoding="utf-8")
    assert "NODE_ENTER" in content
    assert "HTTP_REQUEST" not in content


def test_debug_full_logs_http_and_masks_key(tmp_path, monkeypatch):
    _set_output(tmp_path, monkeypatch, "FULL")
    debug.log_http_request("POST", "http://x/api/v1/chat",
                           {"Authorization": "Bearer secret1234"}, {"model": "m"})
    debug.log_http_response(200, 12.3, 456)
    content = (tmp_path / "llm_prompt_studio.log").read_text(encoding="utf-8")
    assert "HTTP_REQUEST" in content
    assert "HTTP_RESPONSE" in content
    assert "secret1234" not in content  # must be masked
    assert "***...1234" in content


def test_debug_redacts_base64(tmp_path, monkeypatch):
    _set_output(tmp_path, monkeypatch, "FULL")
    big = "A" * 200
    debug.log_node_enter("Critic", "1", {"image": big})
    content = (tmp_path / "llm_prompt_studio.log").read_text(encoding="utf-8")
    assert "<base64 200 chars>" in content


def test_log_type_mismatch(tmp_path, monkeypatch):
    _set_output(tmp_path, monkeypatch, "FULL")
    debug.log_type_mismatch("7", "sampler_name", ("base", "euler"), ["base", "euler"],
                            note="combo differs")
    content = (tmp_path / "llm_prompt_studio.log").read_text(encoding="utf-8")
    assert "TYPE_MISMATCH" in content
    assert "sampler_name" in content


def test_ksample_node_span_delegates_to_debug(tmp_path, monkeypatch):
    # The sampler nodes (KSampler Hires Fix, Face Detailer) import node_span from
    # _ksample, which must delegate to the real debug span (not a no-op).
    from comfyui_llm_prompt_studio.nodes import _ksample

    _set_output(tmp_path, monkeypatch, "FULL")
    with _ksample.node_span("SamplerProbe", "42"):
        pass
    content = (tmp_path / "llm_prompt_studio.log").read_text(encoding="utf-8")
    assert "NODE_ENTER" in content
    assert "SamplerProbe" in content
    assert "42" in content


def test_ksample_node_span_off_is_noop(tmp_path, monkeypatch):
    from comfyui_llm_prompt_studio.nodes import _ksample

    _set_output(tmp_path, monkeypatch, "OFF")
    with _ksample.node_span("SamplerProbe", "42"):
        pass
    assert not (tmp_path / "llm_prompt_studio.log").exists()
