import json
import os
import sys

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PKG)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from comfyui_llm_prompt_studio.nodes import writer, scene_builder  # noqa: E402
from comfyui_llm_prompt_studio.model_meta import (  # noqa: E402
    is_no_negative_family, detect_checkpoint_family, detect_checkpoint_family_info)
from comfyui_llm_prompt_studio.parsing import (  # noqa: E402
    parse_critic_json, find_missing_fields)
from unittest.mock import patch  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_writer_cache():
    writer._prompt_cache.clear()
    yield


def _json(**fields):
    return json.dumps(fields)


def _run_writer(chat_side_effect, prompt_mode="auto", family="", generate_face_prompts=False,
                max_field_retries=2, face_prompt_instruction="", revision_notes="",
                unique_id="1", architecture=""):
    with patch.object(writer, "ensure_model_loaded", lambda *a, **k: None), \
         patch.object(writer, "chat_completion", side_effect=chat_side_effect):
        result = writer.LLMPromptStudioWriter().execute(
            server_url="http://localhost:1234/v1", api_key="", model="m",
            context_length=8192, gpu_offload=1.0, system_prompt="SYS", idea="a cat",
            revision_notes=revision_notes, temperature=0.7, max_tokens=512, seed=0,
            reuse_last_prompt=False, generate_face_prompts=generate_face_prompts,
            max_field_retries=max_field_retries, face_prompt_instruction=face_prompt_instruction,
            prompt_mode=prompt_mode, family=family, unique_id=unique_id, architecture=architecture)
    return result


def _run_scene(chat_side_effect, prompt_mode="auto", family="", description="a cat sitting",
               architecture=""):
    with patch.object(scene_builder, "ensure_model_loaded", lambda *a, **k: None), \
         patch.object(scene_builder, "chat_completion", side_effect=chat_side_effect):
        return scene_builder.LLMPromptStudioSceneBuilder().execute(
            stage="2 - compose", image=None, server_url="http://localhost:1234/v1", api_key="",
            model="m", context_length=16384, gpu_offload=1.0, describe_prompt="D",
            composer_prompt="C", user_changes="", image_max_size=1024, temperature=0.7,
            max_tokens=1024, max_field_retries=2, vision_check=False, description_view=description,
            prompt_mode=prompt_mode, family=family, architecture=architecture)


# --- mode selection ---------------------------------------------------------
def test_standard_keeps_negative():
    result = _run_writer([_json(positive="a cat", negative="blurry", scene_name="cat")],
                         prompt_mode="standard")
    assert result[1] == "blurry"


def test_no_negative_forces_empty_negative():
    result = _run_writer([_json(positive="a cat", negative="blurry", scene_name="cat")],
                         prompt_mode="no_negative")
    assert result[1] == ""


def test_auto_dmd_no_negative():
    result = _run_writer([_json(positive="a cat", negative="blurry", scene_name="cat")],
                         prompt_mode="auto", family="dmd")
    assert result[1] == ""


def test_auto_base_standard():
    result = _run_writer([_json(positive="a cat", negative="blurry", scene_name="cat")],
                         prompt_mode="auto", family="base")
    assert result[1] == "blurry"


def test_auto_empty_family_standard():
    result = _run_writer([_json(positive="a cat", negative="blurry", scene_name="cat")],
                         prompt_mode="auto", family="")
    assert result[1] == "blurry"


def test_auto_lcm_no_negative():
    result = _run_writer([_json(positive="a cat", negative="blurry", scene_name="cat")],
                         prompt_mode="auto", family="lcm")
    assert result[1] == ""


def test_no_negative_face_negative_empty():
    result = _run_writer(
        [_json(positive="a cat", negative="x", scene_name="cat",
               face_positive="fp", face_negative="fn")],
        prompt_mode="no_negative", generate_face_prompts=True)
    assert result[4] == "fp"
    assert result[5] == ""


def test_generate_face_prompts_standard_returns_face_fields():
    result = _run_writer(
        [_json(positive="a cat", negative="blurry", scene_name="cat",
               face_positive="fp", face_negative="fn")],
        prompt_mode="standard", generate_face_prompts=True)
    assert result[4] == "fp"
    assert result[5] == "fn"


def test_generate_face_prompts_fallback_when_model_omits():
    # Empty face fields must fall back to the main positive/negative prompts.
    result = _run_writer(
        [_json(positive="a cat", negative="blurry", scene_name="cat",
               face_positive="", face_negative="")],
        prompt_mode="standard", generate_face_prompts=True)
    assert result[4] == "a cat"
    assert result[5] == "blurry"


def test_no_negative_runtime_error_only_on_empty_positive():
    with pytest.raises(RuntimeError):
        # empty positive -> retries then raises; provide enough responses for the retries
        _run_writer([_json(negative="x", scene_name="cat")] * 3, prompt_mode="no_negative")
    # empty negative alone must NOT raise in no-negative mode
    result = _run_writer([_json(positive="a cat", negative="", scene_name="cat")],
                         prompt_mode="no_negative")
    assert result[0] == "a cat"


def test_cache_key_distinguishes_modes():
    _run_writer([_json(positive="a cat", negative="blurry", scene_name="cat")],
                prompt_mode="standard", unique_id="cache-z")
    with patch.object(writer, "ensure_model_loaded", lambda *a, **k: None), \
         patch.object(writer, "chat_completion",
                      side_effect=lambda *a, **k: _json(positive="a cat", negative="blurry",
                                                        scene_name="cat")) as call:
        writer.LLMPromptStudioWriter().execute(
            server_url="http://localhost:1234/v1", api_key="", model="m",
            context_length=8192, gpu_offload=1.0, system_prompt="SYS", idea="a cat",
            revision_notes="", temperature=0.7, max_tokens=512, seed=0,
            reuse_last_prompt=True, generate_face_prompts=False,
            max_field_retries=2, face_prompt_instruction="",
            prompt_mode="no_negative", family="", unique_id="cache-z")
    # cache key differs (mode changed) -> LLM must be called again
    assert call.call_count == 1


def test_cache_key_survives_checkpoint_family_change():
    # First pass generates and caches the prompt for one checkpoint family.
    _run_writer([_json(positive="a cat", negative="blurry", scene_name="cat")],
                prompt_mode="standard", family="base", unique_id="cache-f")
    # Swapping to a different checkpoint (different detected family) with reuse on must NOT
    # regenerate: the same prompts are carried over to the new checkpoint.
    with patch.object(writer, "ensure_model_loaded", lambda *a, **k: None), \
         patch.object(writer, "chat_completion",
                      side_effect=lambda *a, **k: _json(positive="WRONG", negative="WRONG",
                                                        scene_name="WRONG")) as call:
        result = writer.LLMPromptStudioWriter().execute(
            server_url="http://localhost:1234/v1", api_key="", model="m",
            context_length=8192, gpu_offload=1.0, system_prompt="SYS", idea="a cat",
            revision_notes="", temperature=0.7, max_tokens=512, seed=0,
            reuse_last_prompt=True, generate_face_prompts=False,
            max_field_retries=2, face_prompt_instruction="",
            prompt_mode="standard", family="turbo", unique_id="cache-f")
    # LLM was NOT called again -> the previously generated prompts are reused verbatim.
    assert call.call_count == 0
    assert result[0] == "a cat"
    assert result[1] == "blurry"


def test_auto_flux_architecture_forces_no_negative():
    # A base family with a Flux architecture must force the empty negative (force_no_negative).
    result = _run_writer([_json(positive="a cat", negative="blurry", scene_name="cat")],
                         prompt_mode="auto", family="base", architecture="flux")
    assert result[1] == ""


def test_auto_sd3_architecture_forces_no_negative():
    result = _run_writer([_json(positive="a cat", negative="blurry", scene_name="cat")],
                         prompt_mode="auto", family="base", architecture="sd3")
    assert result[1] == ""


def test_auto_sdxl_architecture_keeps_negative():
    # SDXL architecture must NOT force no-negative in auto mode with a base family.
    result = _run_writer([_json(positive="a cat", negative="blurry", scene_name="cat")],
                         prompt_mode="auto", family="base", architecture="sdxl")
    assert result[1] == "blurry"


def test_scene_flux_architecture_forces_no_negative():
    result = _run_scene([_json(positive="a cat", negative="blurry", scene_name="cat")],
                        prompt_mode="auto", architecture="flux")
    assert result["result"][1] == ""


def test_preset_no_negative_uses_no_negative_variant():
    # Regression: selecting a style preset (that is not disabled in no-negative mode) together
    # with no_negative mode must use the preset's no-negative system prompt, which requires an
    # empty negative, instead of its standard variant (which demanded a non-empty negative).
    captured = []
    def spy(*a, **k):
        captured.append(a[3])  # the messages list
        return _json(positive="a cat", negative="IGNORED", scene_name="cat")
    with patch.object(writer, "ensure_model_loaded", lambda *a, **k: None), \
         patch.object(writer, "chat_completion", side_effect=spy):
        result = writer.LLMPromptStudioWriter().execute(
            server_url="http://localhost:1234/v1", api_key="", model="m",
            context_length=8192, gpu_offload=1.0, system_prompt=writer.DEFAULT_SYSTEM,
            idea="a cat", revision_notes="", temperature=0.7, max_tokens=512, seed=0,
            reuse_last_prompt=False, generate_face_prompts=False,
            max_field_retries=2, face_prompt_instruction="",
            prompt_mode="no_negative", family="", style_preset="Anime / Manga",
            unique_id="preset-nn")
    sys_msg = captured[0][0]["content"]
    assert '"negative": ""' in sys_msg
    assert "negative prompt (always required)" not in sys_msg
    # the node still forces the negative empty regardless of the model's reply
    assert result[1] == ""


def test_scene_standard_uses_standard_composer_when_default():
    # Regression: in standard mode with the composer widget left at its default, the node
    # must NOT switch to the no-negative composer (that produced an empty negative, which
    # then failed require_negative validation). It should keep the plain standard composer.
    captured = []
    def spy(*a, **k):
        captured.append(a[3])  # the messages list
        return _json(positive="a cat", negative="blurry", scene_name="cat")
    with patch.object(scene_builder, "ensure_model_loaded", lambda *a, **k: None), \
         patch.object(scene_builder, "chat_completion", side_effect=spy):
        scene_builder.LLMPromptStudioSceneBuilder().execute(
            stage="2 - compose", image=None, server_url="http://localhost:1234/v1", api_key="",
            model="m", context_length=16384, gpu_offload=1.0, describe_prompt="D",
            composer_prompt=scene_builder.DEFAULT_COMPOSER, user_changes="",
            image_max_size=1024, temperature=0.7, max_tokens=1024, max_field_retries=2,
            vision_check=False, description_view="a cat sitting",
            prompt_mode="standard", family="")
    sys_msg = captured[0][0]["content"]
    assert sys_msg == scene_builder.DEFAULT_COMPOSER
    assert sys_msg != scene_builder.DEFAULT_COMPOSER_NO_NEGATIVE


# --- helpers ----------------------------------------------------------------
def test_is_no_negative_family():
    assert is_no_negative_family("dmd") is True
    assert is_no_negative_family("base") is False
    assert is_no_negative_family("") is False
    assert is_no_negative_family("DMD") is True
    assert is_no_negative_family("unknown") is False


def test_detect_family_false_positives():
    assert detect_checkpoint_family("model_flash_attention.safetensors") == "base"
    assert detect_checkpoint_family("model_hypernetwork.safetensors") == "base"
    assert detect_checkpoint_family("model_flash_xl.safetensors") == "flash"
    assert detect_checkpoint_family("model_hyper_xl.safetensors") == "hyper"
    assert detect_checkpoint_family("model_lcm.safetensors") == "lcm"


def test_detect_dmd2_variant_as_dmd():
    # Versioned distill variants (e.g. DMD2 1-4 step LoRAs) must map to the dmd family
    # so no-negative mode auto-activates for them. dmd + trailing digit, not "dmd" glued
    # to a letter (which would be a different token).
    assert detect_checkpoint_family("dmd2_sdxl_4step_lora.safetensors") == "dmd"
    assert detect_checkpoint_family("model_dmd2.safetensors") == "dmd"
    assert is_no_negative_family("dmd2") is False  # detection returns "dmd", not "dmd2"
    # "dmd2x" (letter after the digits) must NOT match.
    assert detect_checkpoint_family("model_dmd2x.safetensors") == "base"


def test_detect_glued_token_families():
    # Distilled families are often concatenated to XL/SD with no separator (CamelCase),
    # which the case-aware boundary must still detect (previously missed -> "base").
    assert detect_checkpoint_family("HyperSDXL_10steps.safetensors") == "hyper"
    assert detect_checkpoint_family("SDXLLightning_4step.safetensors") == "lightning"
    assert detect_checkpoint_family("TurboSDXL.safetensors") == "turbo"
    assert detect_checkpoint_family("LCMXL.safetensors") == "lcm"
    assert detect_checkpoint_family("HyperSD_1.step.safetensors") == "hyper"


def test_detect_excludes_freetext_metadata(monkeypatch):
    # A base model whose free-text description mentions distilled families must NOT be
    # falsely flagged, while a structured field carrying a marker still is detected.
    meta = {
        "modelspec.architecture": "stableDiffusionXL_v1",
        "description": "a hyper-realistic, lightning-fast, turbo-charged base model",
    }
    monkeypatch.setattr(
        "comfyui_llm_prompt_studio.model_meta.read_safetensors_metadata",
        lambda *a, **k: meta)
    monkeypatch.setattr(
        "comfyui_llm_prompt_studio.model_meta.os.path.isfile", lambda *a, **k: True)
    import folder_paths as _fp
    monkeypatch.setattr(_fp, "get_full_path", lambda *a, **k: "/x.safetensors")
    assert detect_checkpoint_family("my_base_xl.safetensors") == "base"
    # Structured title still carries a marker.
    meta2 = {"modelspec.title": "Juggernaut XL Turbo Edition"}
    monkeypatch.setattr(
        "comfyui_llm_prompt_studio.model_meta.read_safetensors_metadata",
        lambda *a, **k: meta2)
    assert detect_checkpoint_family("my_base_xl.safetensors") == "turbo"


def test_parse_critic_json_float_score():
    score, _, _ = parse_critic_json(
        json.dumps({"score": "8.5", "verdict": "v", "revision_notes": "n"}))
    assert score == 8  # round(8.5) -> 8 (banker's rounding)
    score, _, _ = parse_critic_json(
        json.dumps({"score": 7, "verdict": "v", "revision_notes": "n"}))
    assert score == 7
    score, _, _ = parse_critic_json(
        json.dumps({"score": "8", "verdict": "v", "revision_notes": "n"}))
    assert score == 8


def test_find_missing_fields_require_negative_false():
    parsed = ("pos", "", "scene", "fp", "fn")
    assert find_missing_fields(parsed, require_negative=False) == []
    # default behaviour still requires the negative
    assert find_missing_fields(parsed) == ["negative"]


def test_smart_loader_reports_distilled_family_from_lora(monkeypatch):
    # A DMD (distillation) LoRA applied to a base checkpoint makes the effective model
    # distilled, so the reported family must reflect the LoRA, letting the Writer switch to
    # no-negative mode automatically.
    import sys
    import types as _t

    import comfyui_llm_prompt_studio.nodes.smart_loader as sl

    fake_sd = _t.ModuleType("comfy.sd")
    fake_sd.load_checkpoint_guess_config = lambda *a, **k: (object(), object(), object())
    fake_sd.load_lora_for_models = lambda m, c, l, s, z: (m, c)
    fake_utils = _t.ModuleType("comfy.utils")
    fake_utils.load_torch_file = lambda *a, **k: object()
    comfy_mod = sys.modules.get("comfy") or _t.ModuleType("comfy")
    comfy_mod.__path__ = getattr(comfy_mod, "__path__", [])
    comfy_mod.sd = fake_sd
    comfy_mod.utils = fake_utils
    sys.modules["comfy"] = comfy_mod
    sys.modules["comfy.sd"] = fake_sd
    sys.modules["comfy.utils"] = fake_utils
    monkeypatch.setattr(sl.folder_paths, "get_full_path", lambda category, name: "")

    res = sl.LLMPromptStudioSmartLoader().load(
        ckpt_name="sd_xl_base_1.0.safetensors", family_override="auto",
        lora_name="dmd_style_lora.safetensors", apply_lora="auto",
        strength_model=1.0, vae_user="[none]", unique_id="x")
    assert res["result"][4] == "dmd"
    assert res["ui"]["family"] == ["dmd"]

    # A non-distillation style LoRA must NOT change the (base) family.
    res2 = sl.LLMPromptStudioSmartLoader().load(
        ckpt_name="sd_xl_base_1.0.safetensors", family_override="auto",
        lora_name="cinematic_style.safetensors", apply_lora="auto",
        strength_model=1.0, vae_user="[none]", unique_id="y")
    assert res2["result"][4] == "base"


def test_detect_excludes_ss_tag_frequency(monkeypatch):
    # sd-scripts writes a training-tag histogram under `ss_tag_frequency`; if a base
    # model happens to carry a "lightning" (or any distilled) tag there, it must not be
    # mistaken for a distilled checkpoint. The key contains "tag", so it is excluded.
    import json as _json
    meta = {
        "modelspec.architecture": "stableDiffusionXL_v1",
        "ss_tag_frequency": _json.dumps({
            "artist": {"someartist": 1.0},
            "lightning": {"a": 1.0},  # would falsely flag without the "tag" exclusion
            "copyright": {"b": 1.0},
        }),
    }
    monkeypatch.setattr(
        "comfyui_llm_prompt_studio.model_meta.read_safetensors_metadata",
        lambda *a, **k: meta)
    monkeypatch.setattr(
        "comfyui_llm_prompt_studio.model_meta.os.path.isfile", lambda *a, **k: True)
    import folder_paths as _fp
    monkeypatch.setattr(_fp, "get_full_path", lambda *a, **k: "/x.safetensors")
    assert detect_checkpoint_family("my_base_xl.safetensors") == "base"
    # A genuinely structured field still wins.
    meta2 = {"modelspec.title": "Foo Lightning Bar"}
    monkeypatch.setattr(
        "comfyui_llm_prompt_studio.model_meta.read_safetensors_metadata",
        lambda *a, **k: meta2)
    assert detect_checkpoint_family("my_base_xl.safetensors") == "lightning"


def test_flash_attention_does_not_mask_other_families(monkeypatch):
    # 'flash_attention' is an attention mechanism, not the Flash family. A model whose
    # name also carries a real distilled marker (here 'lcm') must still be detected.
    assert detect_checkpoint_family("flash_attention_lcm_model.safetensors") == "lcm"
    # And a name that is ONLY flash_attention must remain 'base'.
    assert detect_checkpoint_family("model_flash_attention.safetensors") == "base"


def test_detect_camelcase_after_lowercase_word():
    # A capitalized marker following a lowercase word (no separator) is a valid
    # CamelCase boundary and must be detected — previously the lowercase letter
    # immediately before the marker rejected the match.
    assert detect_checkpoint_family("photoLightning.safetensors") == "lightning"
    assert detect_checkpoint_family("myLcmModel.safetensors") == "lcm"
    assert detect_checkpoint_family("v21Turbo.safetensors") == "turbo"
    assert detect_checkpoint_family("V50_v50Lightning.safetensors") == "lightning"


def test_detect_position_picks_earliest_marker():
    # When several markers are present, the one occurring earliest wins.
    assert detect_checkpoint_family("sd_xl_lcm_lightning_4step.safetensors") == "lcm"
    assert detect_checkpoint_family("sd_xl_lightning_lcm.safetensors") == "lightning"


def test_detect_schnell_tcd_pcm():
    assert detect_checkpoint_family("sdxl_schnell.safetensors") == "schnell"
    assert detect_checkpoint_family("sd_xl_tcd.safetensors") == "tcd"
    assert detect_checkpoint_family("pcm_sdxl.safetensors") == "pcm"
    assert is_no_negative_family("schnell") is True
    assert is_no_negative_family("tcd") is True
    assert is_no_negative_family("pcm") is True


def test_detect_family_info_source(monkeypatch):
    # A marker in the file name is reported with source "filename".
    assert detect_checkpoint_family_info("SDXLLightning_4step.safetensors") == ("lightning", "filename")
    # No marker anywhere -> "base" with source "base".
    assert detect_checkpoint_family_info("my_base_xl.safetensors") == ("base", "base")
    # A marker that exists ONLY in structured metadata is reported with source "metadata".
    meta = {"modelspec.title": "Foo Turbo Bar"}
    monkeypatch.setattr(
        "comfyui_llm_prompt_studio.model_meta.read_safetensors_metadata",
        lambda *a, **k: meta)
    monkeypatch.setattr(
        "comfyui_llm_prompt_studio.model_meta.os.path.isfile", lambda *a, **k: True)
    import folder_paths as _fp
    monkeypatch.setattr(_fp, "get_full_path", lambda *a, **k: "/x.safetensors")
    assert detect_checkpoint_family_info("my_base_xl.safetensors") == ("turbo", "metadata")

