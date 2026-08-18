"""Headless tests for the distilled sampler-parameter presets.

These must run WITHOUT comfy (pure Python) so ``python -m pytest
comfyui_llm_prompt_studio/tests/`` succeeds from the parent dir.
"""
from comfyui_llm_prompt_studio.nodes import _distilled_presets as dp

# The sampler/scheduler values used in the table, validated against the live ComfyUI
# combo lists in smart_parameters.py. Mirrored here so the table can be tested without
# importing comfy.
_STANDARD_SCHEDULERS = (
    "normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform",
    "beta", "linear_quadratic", "kl_optimal",
)
_EFF_SCHEDULERS = _STANDARD_SCHEDULERS + ("AYS SD1", "AYS SDXL", "AYS SVD", "GITS")
_SAMPLERS = (
    "euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms",
    "dpmpp_2s_ancestral", "dpmpp_2m", "dpmpp_sde", "dpmpp_2m_sde", "dpmpp_3m_sde",
    "ddpm", "lcm", "ddim", "uni_pc", "uni_pc_bh2",
)


def test_families_and_presets():
    assert "base" in dp.FAMILIES
    assert dp.DISTILLED_FAMILIES == frozenset(dp.FAMILIES) - {"base"}
    for p in ("balanced", "speed", "quality"):
        assert p in dp.PRESETS


def test_table_covers_all_families():
    for fam in dp.FAMILIES:
        assert fam in dp.RECOMMENDATIONS
        for preset in dp.PRESETS:
            assert preset in dp.RECOMMENDATIONS[fam]


def test_standard_target_uses_standard_scheduler_and_valid_sampler():
    for fam in dp.FAMILIES:
        for preset in dp.PRESETS:
            rec = dp.recommend(fam, preset, "", target="standard")
            assert rec["scheduler"] in _STANDARD_SCHEDULERS, (fam, preset, rec)
            assert rec["sampler"] in _SAMPLERS, (fam, preset, rec)
            assert rec["steps"] > 0
            assert rec["cfg"] >= 0


def test_efficient_distilled_balanced_speed_uses_ays_sdxl():
    for fam in dp.DISTILLED_FAMILIES:
        for preset in ("balanced", "speed"):
            rec = dp.recommend(fam, preset, "", target="efficient")
            assert rec["scheduler"] == "AYS SDXL"
            assert rec["scheduler"] in _EFF_SCHEDULERS


def test_efficient_quality_keeps_standard_scheduler():
    for fam in dp.DISTILLED_FAMILIES:
        rec = dp.recommend(fam, "quality", "", target="efficient")
        assert rec["scheduler"] in _EFF_SCHEDULERS
        assert rec["scheduler"] != "AYS SDXL"


def test_ckpt_name_step_override():
    rec = dp.recommend("lightning", "balanced", "SDXL-Lightning_4step.safetensors",
                       target="standard")
    assert rec["steps"] == 4


def test_base_quality_preset():
    rec = dp.recommend("base", "quality", "", target="standard")
    assert rec["steps"] == 40
    assert rec["sampler"] == "dpmpp_2m_sde"


def test_unknown_family_falls_back_to_base():
    rec = dp.recommend("does-not-exist", "balanced", "", target="standard")
    assert rec["family"] == "base"
    assert rec["steps"] == 6


def test_empty_family_defaults_to_base():
    rec = dp.recommend("", "balanced", "", target="standard")
    assert rec["family"] == "base"


def test_target_default_is_standard():
    rec = dp.recommend("lightning", "balanced", "", target="")
    assert rec["target"] == "standard"
    assert rec["scheduler"] in _STANDARD_SCHEDULERS


# ---------------------------------------------------------------------------
# Node-level checks (stub comfy.samplers so smart_parameters imports headless).
# ---------------------------------------------------------------------------
def test_node_apply_parameters_and_return_types():
    from comfyui_llm_prompt_studio.nodes import smart_parameters as sp
    from comfyui_llm_prompt_studio.nodes.smart_parameters import (
        SAMPLERS_COMBO, STANDARD_SCHEDULERS, EFF_SCHEDULERS,
    )

    # RETURN_TYPES: sampler/scheduler must be the whole option-TUPLE as a single
    # element (COMBO typing), not spread.
    assert sp.LLMPromptStudioSmartParameters.RETURN_TYPES[2] is SAMPLERS_COMBO
    assert sp.LLMPromptStudioSmartParameters.RETURN_TYPES[3] is STANDARD_SCHEDULERS
    assert sp.LLMPromptStudioSmartParametersEfficient.RETURN_TYPES[3] == EFF_SCHEDULERS

    # Sentinel behavior: defaults fall through to the recommendation.
    out = sp.apply_parameters(
        "standard", "auto", "balanced", 0, -1.0, "auto", "auto",
        detected_family="lightning", ckpt_name="SDXL-Lightning_4step.safetensors")
    assert out[0] == 4          # steps from ckpt name
    assert out[1] == 1.0        # cfg from lightning balanced
    assert out[2] == "euler"    # sampler from table
    assert out[3] == "sgm_uniform"  # scheduler standard (no AYS)

    # User-edited widgets win over the recommendation.
    out2 = sp.apply_parameters(
        "efficient", "auto", "balanced", 25, 5.0, "dpmpp_2m", "karras",
        detected_family="lcm", ckpt_name="")
    assert out2[0] == 25
    assert out2[1] == 5.0
    assert out2[2] == "dpmpp_2m"
    assert out2[3] == "karras"

    # Efficient distilled balanced uses AYS SDXL.
    out3 = sp.apply_parameters(
        "efficient", "auto", "balanced", 0, -1.0, "auto", "auto",
        detected_family="lcm", ckpt_name="")
    assert out3[3] == "AYS SDXL"


# ---------------------------------------------------------------------------
# Architecture-aware recommendations (base family + known architecture).
# ---------------------------------------------------------------------------
def test_architecture_override_flux_cfg():
    # A base Flux checkpoint must NOT get SDXL defaults; cfg -> 1.0.
    rec = dp.recommend("base", "balanced", "", target="standard", architecture="flux")
    assert rec["family"] == "base"
    assert rec["cfg"] == 1.0
    assert rec["sampler"] == "euler"
    assert rec["scheduler"] == "simple"
    assert rec["steps"] == 24


def test_architecture_override_sd3_low_cfg():
    rec = dp.recommend("base", "balanced", "", target="standard", architecture="sd3")
    assert rec["cfg"] == 4.5
    assert rec["steps"] == 40
    assert rec["sampler"] == "euler"
    assert rec["scheduler"] == "simple"


def test_architecture_override_sd15_cfg():
    rec = dp.recommend("base", "balanced", "", target="standard", architecture="sd15")
    assert rec["cfg"] == 7.0
    assert rec["sampler"] is None or rec["sampler"] == "dpmpp_2m"
    # sampler must be a valid standard sampler
    assert rec["sampler"] in _SAMPLERS
    assert rec["scheduler"] in _STANDARD_SCHEDULERS


def test_architecture_override_pony():
    rec = dp.recommend("base", "balanced", "", target="standard", architecture="pony")
    assert rec["cfg"] == 6.0
    assert rec["sampler"] == "dpmpp_2m"


def test_architecture_unknown_falls_back_to_base():
    rec = dp.recommend("base", "balanced", "", target="standard", architecture="unknown")
    assert rec["cfg"] == 6.0
    assert rec["steps"] == 30


def test_distilled_family_still_wins_over_architecture():
    # Even with architecture="flux", a distilled family (e.g. lightning) keeps its own params.
    rec = dp.recommend("lightning", "balanced", "", target="standard", architecture="flux")
    assert rec["family"] == "lightning"
    assert rec["cfg"] == 1.0
    assert rec["sampler"] == "euler"
    assert rec["scheduler"] == "sgm_uniform"


def test_node_passes_architecture_through():
    from comfyui_llm_prompt_studio.nodes import smart_parameters as sp
    out = sp.apply_parameters(
        "standard", "auto", "balanced", 0, -1.0, "auto", "auto",
        detected_family="base", ckpt_name="", architecture="flux")
    # flux base -> cfg 1.0, euler/simple
    assert out[1] == 1.0
    assert out[2] == "euler"
    assert out[3] == "simple"
    assert "arch: flux" in out[4]


def test_architecture_tables_have_all_presets():
    for arch in dp.ARCHITECTURE_RECOMMENDATIONS:
        for preset in dp.PRESETS:
            assert preset in dp.ARCHITECTURE_RECOMMENDATIONS[arch]
            row = dp.ARCHITECTURE_RECOMMENDATIONS[arch][preset]
            assert len(row) == 4
            assert row[2] in _SAMPLERS
            assert row[3] in _STANDARD_SCHEDULERS
