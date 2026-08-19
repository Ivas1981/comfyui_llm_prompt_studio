"""Contract test: Smart Loader must expose a `ckpt_name` output so it can be wired
into Smart Parameters' optional `ckpt_name` input (used for fallback family detection).
"""
from comfyui_llm_prompt_studio.nodes.smart_loader import LLMPromptStudioSmartLoader
from comfyui_llm_prompt_studio.nodes.smart_parameters import LLMPromptStudioSmartParameters


def test_loader_exposes_ckpt_name_output():
    names = LLMPromptStudioSmartLoader.RETURN_NAMES
    types = LLMPromptStudioSmartLoader.RETURN_TYPES
    assert len(names) == len(types)
    idx = names.index("ckpt_name")
    assert types[idx] == "STRING"


def test_loader_ckpt_name_wires_into_smart_parameters():
    params_inputs = LLMPromptStudioSmartParameters.INPUT_TYPES()["optional"]
    assert params_inputs["ckpt_name"][0] == "STRING"
    assert "ckpt_name" in LLMPromptStudioSmartLoader.RETURN_NAMES
