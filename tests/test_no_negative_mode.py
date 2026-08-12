import json
import os
import sys

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PKG)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from comfyui_llm_prompt_studio.nodes import writer, scene_builder  # noqa: E402
from comfyui_llm_prompt_studio.model_meta import (  # noqa: E402
    is_no_negative_family, detect_checkpoint_family)
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
                unique_id="1"):
    with patch.object(writer, "ensure_model_loaded", lambda *a, **k: None), \
         patch.object(writer, "chat_completion", side_effect=chat_side_effect):
        result = writer.LLMPromptStudioWriter().execute(
            server_url="http://localhost:1234/v1", api_key="", model="m",
            context_length=8192, gpu_offload=1.0, system_prompt="SYS", idea="a cat",
            revision_notes=revision_notes, temperature=0.7, max_tokens=512, seed=0,
            reuse_last_prompt=False, generate_face_prompts=generate_face_prompts,
            max_field_retries=max_field_retries, face_prompt_instruction=face_prompt_instruction,
            prompt_mode=prompt_mode, family=family, unique_id=unique_id)
    return result


def _run_scene(chat_side_effect, prompt_mode="auto", family="", description="a cat sitting"):
    with patch.object(scene_builder, "ensure_model_loaded", lambda *a, **k: None), \
         patch.object(scene_builder, "chat_completion", side_effect=chat_side_effect):
        return scene_builder.LLMPromptStudioSceneBuilder().execute(
            stage="2 - compose", image=None, server_url="http://localhost:1234/v1", api_key="",
            model="m", context_length=16384, gpu_offload=1.0, describe_prompt="D",
            composer_prompt="C", user_changes="", image_max_size=1024, temperature=0.7,
            max_tokens=1024, max_field_retries=2, vision_check=False, description_view=description,
            prompt_mode=prompt_mode, family=family)


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


def test_scene_no_negative_forces_empty():
    result = _run_scene([_json(positive="a cat", negative="blurry", scene_name="cat")],
                         prompt_mode="no_negative")
    assert result["result"][1] == ""


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
