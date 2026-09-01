"""Headless tests for the LLM Smart Multi-Clip node (architecture-aware conditioning).

These run WITHOUT comfy. A lightweight MockCLIP stands in for a real CLIP object so we can
exercise the token safety net and the per-architecture branching without loading weights.
"""
import logging
import os
import sys

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PKG)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from comfyui_llm_prompt_studio.nodes import multi_clip  # noqa: E402


class MockCLIP:
    """Minimal CLIP stand-in. `keys` lets a test choose what tokenize() returns."""

    def __init__(self, keys=("g", "l")):
        self._keys = keys

    def tokenize(self, text):
        # Return one entry per declared encoder key (value is a throwaway token list).
        return {k: [hash((text, k)) % 1000] for k in self._keys}

    def encode_from_tokens(self, tokens, return_pooled=False):
        if return_pooled:
            return ("COND", "POOLED")
        return "COND"


def test_single_encoder_clip_encodes_without_keyerror():
    # A single-encoder CLIP (SD1.5) returns only "l"; the safety net must prevent a KeyError
    # and the node must still encode successfully.
    clip = MockCLIP(keys=("l",))
    cond = multi_clip.LLMPromptStudioSmartMultiClip._encode_pair(
        clip, "a cat", "", 1024, 1024, 0, 0, 1024, 1024, arch="sd15")
    assert cond[0][0] == "COND"
    assert cond[0][1]["pooled_output"] == "POOLED"
    # width/height metadata is attached (harmless for SD1.5).
    assert cond[0][1]["width"] == 1024


def test_sdxl_dual_encoder_path():
    clip = MockCLIP(keys=("g", "l"))
    cond = multi_clip.LLMPromptStudioSmartMultiClip._encode_pair(
        clip, "a cat", "a cat", 1024, 1024, 0, 0, 1024, 1024, arch="sdxl")
    assert cond[0][0] == "COND"


def test_arch_empty_unwired_defaults_to_clip_shape():
    # No architecture -> behavior is driven by the CLIP's own encoder shape (dual here).
    clip = MockCLIP(keys=("g", "l"))
    cond = multi_clip.LLMPromptStudioSmartMultiClip._encode_pair(
        clip, "a cat", "a cat", 1024, 1024, 0, 0, 1024, 1024, arch="")
    assert cond[0][0] == "COND"


def test_flux_arch_warns_and_encodes(caplog):
    clip = MockCLIP(keys=("l",))
    with caplog.at_level(logging.WARNING, logger="llm_prompt_studio"):
        cond = multi_clip.LLMPromptStudioSmartMultiClip._encode_pair(
            clip, "a cat", "", 1024, 1024, 0, 0, 1024, 1024, arch="flux")
    assert cond[0][0] == "COND"
    assert any("Flux/SD3" in r.message for r in caplog.records)


def test_sd3_arch_warns_and_encodes(caplog):
    clip = MockCLIP(keys=("l",))
    with caplog.at_level(logging.WARNING, logger="llm_prompt_studio"):
        cond = multi_clip.LLMPromptStudioSmartMultiClip._encode_pair(
            clip, "a cat", "", 1024, 1024, 0, 0, 1024, 1024, arch="sd3")
    assert cond[0][0] == "COND"
    assert any("Flux/SD3" in r.message for r in caplog.records)


def test_encode_full_node_runs_with_architecture():
    # The public encode() entry point must accept `architecture` and pass it through.
    clip = MockCLIP(keys=("g", "l"))
    out = multi_clip.LLMPromptStudioSmartMultiClip().encode(
        clip, 1024, 1024, 0, 0, architecture="sdxl",
        positive1_g="a cat", positive1_l="a cat",
        negative1_g="bad", negative1_l="bad")
    # Returns (clip, positive1, negative1, positive2, negative2)
    assert out[0] is clip
    assert out[1][0][0] == "COND"
    assert out[3][0][0] == "COND"  # positive2 empty -> still encoded


def test_class_aliased_for_backward_compat_import():
    # nodes/__init__.py imports the old name; it must resolve to the renamed class.
    assert multi_clip.LLMPromptStudioMultiClipSDXL is multi_clip.LLMPromptStudioSmartMultiClip
