import os
import sys

# Import the module through the package stub (registered in conftest.py) so it resolves
# against `.Testing` without touching the production package.
from comfyui_llm_prompt_studio.nodes import _distilled_presets as dp  # noqa: E402


def test_recommend_lightning_flux_keeps_own_scheduler():
    # A distilled Flux/SD3 must NOT get the SDXL-family AYS scheduler.
    rec = dp.recommend("lightning", "balanced",
                       "sd-xl-lightning-4step.safetensors", architecture="flux")
    assert rec["scheduler"] != "AYS SDXL"
    assert rec["scheduler"] == "sgm_uniform"


def test_recommend_lightning_sd15_uses_ays_sd1():
    rec = dp.recommend("lightning", "balanced",
                       "sd15-lightning-4step.safetensors", architecture="sd15")
    assert rec["scheduler"] == "AYS SD1"


def test_recommend_lightning_sdxl_uses_ays_sdxl():
    rec = dp.recommend("lightning", "balanced",
                       "sd-xl-lightning-4step.safetensors", architecture="sdxl")
    assert rec["scheduler"] == "AYS SDXL"


def test_recommend_quality_preset_never_gets_ays():
    # Only balanced/speed distilled get AYS; quality keeps its own scheduler.
    rec = dp.recommend("lightning", "quality",
                       "sd-xl-lightning-4step.safetensors", architecture="sdxl")
    assert rec["scheduler"] != "AYS SDXL"
