import comfyui_llm_prompt_studio.debug as debug
from comfyui_llm_prompt_studio import nodes
from comfyui_llm_prompt_studio.nodes import _contracts


def test_current_nodes_satisfy_contract():
    mismatches = _contracts.check_combo_contracts(nodes.NODE_CLASS_MAPPINGS)
    assert mismatches == [], "current nodes must satisfy the sampler/scheduler combo contract"


def test_wiring_logs_mismatch_when_present(tmp_path, monkeypatch):
    import folder_paths

    folder_paths.get_output_directory.return_value = str(tmp_path)
    monkeypatch.setattr(debug, "DEBUG_LEVEL", "FULL")
    debug._logger = None

    fake = [{"node": "X", "input": "sampler_name", "expected": ("base", "euler"),
             "actual": ["base", "euler"]}]
    monkeypatch.setattr(_contracts, "check_combo_contracts", lambda *_: fake)

    nodes._verify_combo_contracts()

    log_path = tmp_path / "llm_prompt_studio.log"
    assert log_path.exists(), "debug log should be created when DEBUG_LEVEL=FULL"
    content = log_path.read_text(encoding="utf-8")
    assert "TYPE_MISMATCH" in content
    assert "sampler_name" in content
