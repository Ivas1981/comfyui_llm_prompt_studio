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

# Presets that actually have table rows (the "user" preset is a pass-through, not a row).
_TABLE_PRESETS = ("balanced", "speed", "quality")


def test_families_and_presets():
    assert "base" in dp.FAMILIES
    assert dp.DISTILLED_FAMILIES == frozenset(dp.FAMILIES) - {"base"}
    for p in ("user", "balanced", "speed", "quality"):
        assert p in dp.PRESETS


def test_table_covers_all_families():
    for fam in dp.FAMILIES:
        assert fam in dp.RECOMMENDATIONS
        for preset in _TABLE_PRESETS:
            assert preset in dp.RECOMMENDATIONS[fam]


def test_base_uses_standard_scheduler_and_valid_sampler():
    for fam in dp.FAMILIES:
        if fam in dp.DISTILLED_FAMILIES:
            continue
        for preset in _TABLE_PRESETS:
            rec = dp.recommend(fam, preset, "")
            assert rec["scheduler"] in _STANDARD_SCHEDULERS, (fam, preset, rec)
            assert rec["sampler"] in _SAMPLERS, (fam, preset, rec)
            assert rec["steps"] > 0
            assert rec["cfg"] >= 0


def test_distilled_balanced_speed_uses_ays_sdxl():
    for fam in dp.DISTILLED_FAMILIES:
        for preset in ("balanced", "speed"):
            rec = dp.recommend(fam, preset, "")
            assert rec["scheduler"] == "AYS SDXL"
            assert rec["scheduler"] in _EFF_SCHEDULERS


def test_distilled_quality_keeps_standard_scheduler():
    for fam in dp.DISTILLED_FAMILIES:
        rec = dp.recommend(fam, "quality", "")
        assert rec["scheduler"] in _EFF_SCHEDULERS
        assert rec["scheduler"] != "AYS SDXL"


def test_recommend_has_no_target_key():
    rec = dp.recommend("lightning", "balanced", "")
    assert "target" not in rec


def test_ckpt_name_step_override():
    rec = dp.recommend("lightning", "balanced", "SDXL-Lightning_4step.safetensors")
    assert rec["steps"] == 4


def test_base_quality_preset():
    rec = dp.recommend("base", "quality", "")
    assert rec["steps"] == 40
    assert rec["sampler"] == "dpmpp_2m_sde"


def test_unknown_family_falls_back_to_base():
    rec = dp.recommend("does-not-exist", "balanced", "")
    assert rec["family"] == "base"
    assert rec["steps"] == 6


def test_empty_family_defaults_to_base():
    rec = dp.recommend("", "balanced", "")
    assert rec["family"] == "base"


def test_unknown_distilled_family_uses_generic_default(monkeypatch):
    # A family present in FAMILIES / DISTILLED_FAMILIES but with no hand-written
    # RECOMMENDATIONS row must fall back to GENERIC_DISTILLED, not to base.
    monkeypatch.setattr(dp, "FAMILIES", dp.FAMILIES + ("novafam",))
    monkeypatch.setattr(dp, "DISTILLED_FAMILIES", frozenset(dp.FAMILIES) - {"base"})
    rec = dp.recommend("novafam", "balanced", "")
    # Generically distilled: low steps, cfg ~1, euler, and the balanced/speed AYS override.
    assert rec["family"] == "novafam"
    assert rec["steps"] == 4
    assert rec["cfg"] == 1.0
    assert rec["sampler"] == "euler"
    assert rec["scheduler"] == "AYS SDXL"


def test_node_auto_detects_family_from_ckpt(monkeypatch):
    from comfyui_llm_prompt_studio.nodes import smart_parameters as sp

    # No detected_family / family_override -> node self-detects from the ckpt filename.
    monkeypatch.setattr(
        sp.model_meta, "detect_checkpoint_family", lambda ckpt_name: "turbo"
    )
    out = sp.apply_parameters(
        preset="balanced", family_override="auto", steps=0, cfg=-1.0,
        sampler_name="auto", scheduler="auto",
        detected_family="", ckpt_name="some_generic.safetensors")
    # turbo distilled at balanced -> AYS SDXL scheduler.
    assert out[3] == "AYS SDXL"
    assert out[2] == "euler"
    assert "turbo" in out[4]


# ---------------------------------------------------------------------------
# Node-level checks (stub comfy.samplers so smart_parameters imports headless).
# ---------------------------------------------------------------------------
def test_node_apply_parameters_and_return_types():
    from comfyui_llm_prompt_studio.nodes import smart_parameters as sp
    from comfyui_llm_prompt_studio.nodes.smart_parameters import (
        SAMPLERS_COMBO, FULL_SCHEDULERS,
    )

    # RETURN_TYPES: sampler/scheduler must be the whole option-TUPLE as a single
    # element (COMBO typing), not spread.
    assert sp.LLMPromptStudioSmartParameters.RETURN_TYPES[2] is SAMPLERS_COMBO
    assert sp.LLMPromptStudioSmartParameters.RETURN_TYPES[3] is FULL_SCHEDULERS

    # Sentinel behavior: defaults fall through to the recommendation.
    out = sp.apply_parameters(
        preset="balanced", family_override="auto", steps=0, cfg=-1.0,
        sampler_name="auto", scheduler="auto",
        detected_family="lightning", ckpt_name="SDXL-Lightning_4step.safetensors")
    assert out[0] == 4          # steps from ckpt name
    assert out[1] == 1.0        # cfg from lightning balanced
    assert out[2] == "euler"    # sampler from table
    assert out[3] == "AYS SDXL"  # distilled lightning + balanced -> AYS SDXL

    # User-edited widgets win over the recommendation.
    out2 = sp.apply_parameters(
        preset="balanced", family_override="auto", steps=25, cfg=5.0,
        sampler_name="dpmpp_2m", scheduler="karras",
        detected_family="lcm", ckpt_name="")
    assert out2[0] == 25
    assert out2[1] == 5.0
    assert out2[2] == "dpmpp_2m"
    assert out2[3] == "karras"

    # Distilled balanced uses AYS SDXL (single full scheduler list).
    out3 = sp.apply_parameters(
        preset="balanced", family_override="auto", steps=0, cfg=-1.0,
        sampler_name="auto", scheduler="auto", detected_family="lcm")
    assert out3[3] == "AYS SDXL"


def test_user_preset_passthrough():
    from comfyui_llm_prompt_studio.nodes import smart_parameters as sp
    # Explicit user values are passed through unchanged.
    out = sp.apply_parameters(
        preset="user", family_override="auto", steps=25, cfg=5.0,
        sampler_name="dpmpp_2m", scheduler="karras")
    assert out[0] == 25
    assert out[1] == 5.0
    assert out[2] == "dpmpp_2m"
    assert out[3] == "karras"
    # Sentinels fall back to safe defaults.
    out2 = sp.apply_parameters(
        preset="user", family_override="auto", steps=0, cfg=-1.0,
        sampler_name="auto", scheduler="auto")
    assert out2[0] == 20
    assert out2[1] == 7.0
    assert out2[2] == "dpmpp_2m"
    assert out2[3] == "karras"


# ---------------------------------------------------------------------------
# Architecture-aware recommendations (base family + known architecture).
# ---------------------------------------------------------------------------
def test_architecture_override_flux_cfg():
    # A base Flux checkpoint must NOT get SDXL defaults; cfg -> 1.0.
    rec = dp.recommend("base", "balanced", "", architecture="flux")
    assert rec["family"] == "base"
    assert rec["cfg"] == 1.0
    assert rec["sampler"] == "euler"
    assert rec["scheduler"] == "simple"
    assert rec["steps"] == 24


def test_architecture_override_sd3_low_cfg():
    rec = dp.recommend("base", "balanced", "", architecture="sd3")
    assert rec["cfg"] == 4.5
    assert rec["steps"] == 40
    assert rec["sampler"] == "euler"
    assert rec["scheduler"] == "simple"


def test_architecture_override_sd15_cfg():
    rec = dp.recommend("base", "balanced", "", architecture="sd15")
    assert rec["cfg"] == 7.0
    assert rec["sampler"] in _SAMPLERS
    assert rec["scheduler"] in _STANDARD_SCHEDULERS


def test_architecture_override_pony():
    rec = dp.recommend("base", "balanced", "", architecture="pony")
    assert rec["cfg"] == 6.0
    assert rec["sampler"] == "dpmpp_2m"


def test_architecture_unknown_falls_back_to_base():
    rec = dp.recommend("base", "balanced", "", architecture="unknown")
    assert rec["cfg"] == 6.0
    assert rec["steps"] == 30


def test_distilled_family_still_wins_over_architecture():
    # Even with architecture="flux", a distilled family (e.g. lightning) keeps its own params.
    rec = dp.recommend("lightning", "balanced", "", architecture="flux")
    assert rec["family"] == "lightning"
    assert rec["cfg"] == 1.0
    assert rec["sampler"] == "euler"
    assert rec["scheduler"] == "AYS SDXL"


def test_node_passes_architecture_through():
    from comfyui_llm_prompt_studio.nodes import smart_parameters as sp
    out = sp.apply_parameters(
        preset="balanced", family_override="auto", steps=0, cfg=-1.0,
        sampler_name="auto", scheduler="auto",
        detected_family="base", ckpt_name="", architecture="flux")
    # flux base -> cfg 1.0, euler/simple
    assert out[1] == 1.0
    assert out[2] == "euler"
    assert out[3] == "simple"
    assert "arch: flux" in out[4]


def test_architecture_tables_have_all_presets():
    for arch in dp.ARCHITECTURE_RECOMMENDATIONS:
        for preset in _TABLE_PRESETS:
            assert preset in dp.ARCHITECTURE_RECOMMENDATIONS[arch]
            row = dp.ARCHITECTURE_RECOMMENDATIONS[arch][preset]
            assert len(row) == 4
            assert row[2] in _SAMPLERS
            assert row[3] in _STANDARD_SCHEDULERS
