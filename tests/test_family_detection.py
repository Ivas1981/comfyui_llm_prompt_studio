import os
import sys
import tempfile

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PKG)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from comfyui_llm_prompt_studio import model_meta  # noqa: E402
from unittest.mock import patch  # noqa: E402


def _detect(name, metadata, tmp_path):
    # Point folder_paths at a real (empty) file so the isfile check passes, and inject the
    # safetensors metadata directly without touching disk.
    fake = tmp_path / "ckpt.safetensors"
    fake.write_bytes(b"")
    with patch.object(model_meta, "read_safetensors_metadata", return_value=metadata), \
         patch.object(model_meta.folder_paths, "get_full_path", return_value=str(fake)):
        return model_meta.detect_checkpoint_family(name)


def test_detect_family_from_parent_folder(tmp_path):
    # A generically-named file inside a family-named folder is still recognized because
    # detect_checkpoint_family() scans the immediate parent folder name.
    folder = tmp_path / "Lightning"
    folder.mkdir()
    fake = folder / "model.safetensors"
    fake.write_bytes(b"")
    with patch.object(model_meta, "read_safetensors_metadata", return_value={}), \
         patch.object(model_meta.folder_paths, "get_full_path", return_value=str(fake)):
        assert model_meta.detect_checkpoint_family("model.safetensors") == "lightning"


def test_detect_family_from_parent_folder(tmp_path):
    # A generically-named file inside a family-named folder is still recognized because
    # detect_checkpoint_family() scans the immediate parent folder name.
    folder = tmp_path / "Lightning"
    folder.mkdir()
    fake = folder / "model.safetensors"
    fake.write_bytes(b"")
    with patch.object(model_meta, "read_safetensors_metadata", return_value={}), \
         patch.object(model_meta.folder_paths, "get_full_path", return_value=str(fake)):
        assert model_meta.detect_checkpoint_family("model.safetensors") == "lightning"


def test_flash_resolves_when_flash_attention_in_metadata(tmp_path):
    # B2 regression: a real Flash checkpoint whose metadata mentions flash_attention must
    # still resolve to "flash". The old code blanket-skipped every flash match whenever
    # flash_attention appeared anywhere in the scanned text.
    assert _detect("FlashSDXL_10steps.safetensors",
                   {"model_type": "flash_attention_enabled"}, tmp_path) == "flash"


def test_flash_attention_alone_is_not_flash(tmp_path):
    # flash_attention inside a structured metadata value must NOT be detected as the Flash
    # family; the only marker ("flash") is the attention-mechanism substring and is skipped.
    assert _detect("BaseModel.safetensors",
                   {"arch": "flash_attention"}, tmp_path) == "base"


def test_flash_attention_in_filename_only_is_not_flash(tmp_path):
    # A checkpoint whose NAME embeds flash_attention (but no real Flash marker) stays base.
    assert _detect("MyFlash_attentionModel.safetensors",
                   {}, tmp_path) == "base"


def test_flash_filename_resolves_flash(tmp_path):
    # An ordinary Flash checkpoint with no flash_attention noise resolves to flash.
    assert _detect("SDXLFlashTurboMix_4step.safetensors",
                   {}, tmp_path) == "flash"


def test_guess_architecture_from_name():
    from comfyui_llm_prompt_studio import model_meta
    assert model_meta.guess_architecture_from_name("sdxlBase_v9.safetensors") == "sdxl"
    assert model_meta.guess_architecture_from_name("sd_xl_base_1.0.safetensors") == "sdxl"
    assert model_meta.guess_architecture_from_name("ponyDiffusionV6.safetensors") == "pony"
    assert model_meta.guess_architecture_from_name("illustriousXL.safetensors") == "illustrious"
    assert model_meta.guess_architecture_from_name("flux1D_schnell.safetensors") == "flux"
    assert model_meta.guess_architecture_from_name("stable-diffusion-3-medium.safetensors") == "sd3"
    assert model_meta.guess_architecture_from_name("sd3_medium.safetensors") == "sd3"
    assert model_meta.guess_architecture_from_name("v1-5-pruned.safetensors") == "sd15"
    assert model_meta.guess_architecture_from_name("sd15_v1.safetensors") == "sd15"
    # A custom model whose filename omits the architecture keyword cannot be guessed.
    assert model_meta.guess_architecture_from_name("juggernaut_v9.safetensors") == "unknown"
    assert model_meta.guess_architecture_from_name("someRandomModel.safetensors") == "unknown"


def test_is_no_negative_architecture():
    from comfyui_llm_prompt_studio import model_meta
    assert model_meta.is_no_negative_architecture("flux") is True
    assert model_meta.is_no_negative_architecture("sd3") is True
    assert model_meta.is_no_negative_architecture("sdxl") is False
    assert model_meta.is_no_negative_architecture("sd15") is False
    assert model_meta.is_no_negative_architecture("") is False
    assert model_meta.is_no_negative_architecture("PONY") is False


def test_architecture_from_class_name():
    from comfyui_llm_prompt_studio import model_meta
    assert model_meta.architecture_from_class_name("SDXL") == "sdxl"
    assert model_meta.architecture_from_class_name("Flux") == "flux"
    assert model_meta.architecture_from_class_name("Pony") == "pony"
    assert model_meta.architecture_from_class_name("SD15") == "sd15"
    assert model_meta.architecture_from_class_name("StableDiffusion3") == "sd3"
    assert model_meta.architecture_from_class_name("SomethingElse") is None


def test_resolve_architecture_pony_from_filename():
    from comfyui_llm_prompt_studio import model_meta
    # SDXL object, but a Pony checkpoint filename -> recovered as pony via refinement.
    assert model_meta.resolve_architecture("SDXL", "SDXL", "ponyDiffusionV6_XL.safetensors") == \
        ("pony", "filename")


def test_resolve_architecture_illustrious_from_filename():
    from comfyui_llm_prompt_studio import model_meta
    assert model_meta.resolve_architecture("SDXL", "SDXL", "illustriousXL_v10.safetensors") == \
        ("illustrious", "filename")


def test_resolve_architecture_real_sdxl_keeps_sdxl():
    from comfyui_llm_prompt_studio import model_meta
    # A genuine SDXL object keeps sdxl even though the filename is generic.
    arch, source = model_meta.resolve_architecture("SDXL", "SDXL", "sd_xl_base_1.0.safetensors")
    assert arch == "sdxl"
    # source stays "object" (the object detection wins, not the filename).
    assert source == "object"


def test_resolve_architecture_flux_object_not_overwritten():
    from comfyui_llm_prompt_studio import model_meta
    # A real Flux object must NOT be overwritten by a coincidental filename keyword.
    assert model_meta.resolve_architecture("Flux", "Flux", "my_pony_named_flux.safetensors") == \
        ("flux", "object")


def test_resolve_architecture_sd3_object_not_overwritten():
    from comfyui_llm_prompt_studio import model_meta
    assert model_meta.resolve_architecture("StableDiffusion3", "StableDiffusion3", "") == \
        ("sd3", "object")


def test_resolve_architecture_sd15_object():
    from comfyui_llm_prompt_studio import model_meta
    assert model_meta.resolve_architecture("SD15", "SD15", "") == ("sd15", "object")


def test_resolve_architecture_empty_falls_back_to_filename():
    from comfyui_llm_prompt_studio import model_meta
    # No object/config name -> filename heuristic drives the result.
    assert model_meta.resolve_architecture("", "", "sd15_pruned.safetensors") == \
        ("sd15", "filename")
    assert model_meta.resolve_architecture("", "", "flux1-dev.safetensors") == \
        ("flux", "filename")
    assert model_meta.resolve_architecture("", "", "totally_unknown.safetensors") == \
        ("unknown", "unknown")


def test_boundary_ok_rejects_head_of_longer_word():
    # "Flash" / "Hyper" at the head of a longer lowercase word must be rejected so
    # "Flashback" / "Hyperion" do not falsely match the Flash / Hyper families.
    assert model_meta._boundary_ok("Flashback", 0, 5) is False
    assert model_meta._boundary_ok("Hyperion", 0, 5) is False


def test_boundary_ok_accepts_camelcase_continuation():
    # "Lightning" inside "SDXLLightning" is a CamelCase continuation -> accepted.
    assert model_meta._boundary_ok("SDXLLightning", 4, 13) is True
    # "Flash" followed by an uppercase letter -> accepted (CamelCase head).
    assert model_meta._boundary_ok("FlashSDXL", 0, 5) is True


def test_boundary_ok_rejects_lowercase_glue():
    # "hyper" glued to a following lowercase letter (inside "hypernetwork") -> rejected.
    assert model_meta._boundary_ok("hypernetwork", 0, 5) is False
