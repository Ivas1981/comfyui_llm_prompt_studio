import os
import sys

# Import the Writer node through its package (it uses package-relative imports).
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PKG)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from comfyui_llm_prompt_studio.nodes import writer  # noqa: E402
from comfyui_llm_prompt_studio.parsing import slugify  # noqa: E402
from unittest.mock import patch  # noqa: E402

import json  # noqa: E402
import pytest  # noqa: E402


def _json(**fields):
    return json.dumps(fields)


def _run(chat_side_effect, generate_face_prompts=False, max_field_retries=2,
         face_prompt_instruction="", prompt_mode="auto", family=""):
    with patch.object(writer, "ensure_model_loaded", lambda *a, **k: True), \
         patch.object(writer, "chat_completion", side_effect=chat_side_effect) as call:
        result = writer.LLMPromptStudioWriter().execute(
            server_url="http://localhost:1234/v1", api_key="", model="m",
            context_length=8192, gpu_offload=1.0, system_prompt="SYS", idea="a cat",
            revision_notes="", temperature=0.7, max_tokens=512, seed=0,
            reuse_last_prompt=False, generate_face_prompts=generate_face_prompts,
            max_field_retries=max_field_retries, face_prompt_instruction=face_prompt_instruction,
            prompt_mode=prompt_mode, family=family, unique_id="1")
    return result, call


def test_no_retry_when_complete():
    result, call = _run([_json(positive="a cat", negative="b", scene_name="cat_scene",
                               face_positive="fp", face_negative="fn")])
    assert call.call_count == 1
    assert result[0] == "a cat"
    assert "[FIELD RETRY" not in result[2]


def test_retry_on_missing_scene_name():
    seq = [
        _json(positive="a cat", negative="b", scene_name=""),
        _json(positive="a cat", negative="b", scene_name="cat_scene"),
    ]
    result, call = _run(seq)
    assert call.call_count == 2
    # The retry request mentions the missing field.
    retry_msgs = call.call_args_list[1].args[3]
    assert any("scene_name" in m.get("content", "") for m in retry_msgs
               if m.get("role") == "user")
    assert result[3] == "cat_scene"
    assert "[FIELD RETRY" in result[2]


def test_retry_on_missing_face_fields_when_requested():
    # Empty face fields are intentionally NOT treated as missing (the prompts allow empty
    # face_positive/face_negative when no face is present), so the node no longer retries on
    # them — it falls back to the main prompts instead.
    seq = [
        _json(positive="a cat", negative="b", scene_name="s"),
        _json(positive="a cat", negative="b", scene_name="s",
              face_positive="fp", face_negative="fn"),
    ]
    result, call = _run(seq, generate_face_prompts=True)
    assert call.call_count == 1
    assert result[4] == "a cat"  # face_positive falls back to positive
    assert result[5] == "b"      # face_negative falls back to negative


def test_no_retry_on_face_fields_when_not_requested():
    # Face fields empty and face prompts not requested -> no retry, and the node emits
    # empty face outputs (Q5) so FaceDetailer falls back to the main prompts.
    result, call = _run([_json(positive="a cat", negative="b", scene_name="s")])
    assert call.call_count == 1
    assert result[4] == ""  # face_positive blanked when face is off
    assert result[5] == ""  # face_negative blanked when face is off


def test_scene_name_fallback_via_slugify():
    # Model keeps omitting scene_name across all attempts.
    seq = [_json(positive="a red cat", negative="b", scene_name="")] * 3
    result, call = _run(seq, max_field_retries=2)
    assert call.call_count == 3  # initial + 2 retries
    assert result[3] == slugify("a red cat")
    assert "[FIELD RETRY" in result[2]


def test_runtime_error_when_positive_empty():
    seq = [_json(negative="b", scene_name="s")] * 3
    with pytest.raises(RuntimeError):
        _run(seq, max_field_retries=2)


def test_no_negative_mode_keeps_face_negative_empty():
    # Reported bug: face_negative was still generated in no-negative mode. The node must
    # force it (and the negative) empty regardless of what the model returned, because the
    # negative is inert at CFG~1.
    result, _ = _run([_json(positive="a cat", negative="b", scene_name="s",
                            face_positive="fp", face_negative="fn")],
                     generate_face_prompts=True, prompt_mode="no_negative")
    assert result[5] == ""   # face_negative forced empty
    assert result[1] == ""   # negative forced empty
    assert result[4] == "fp"  # face_positive preserved


def test_auto_mode_distilled_family_keeps_face_negative_empty():
    # In auto mode a distilled checkpoint family (e.g. turbo) must drive no_negative on,
    # so face_negative is empty even without an explicit prompt_mode override. This is the
    # path that broke when family detection returned "base" for names like *_Turbo_*.
    result, _ = _run([_json(positive="a cat", negative="b", scene_name="s",
                            face_positive="fp", face_negative="fn")],
                     generate_face_prompts=True, prompt_mode="auto", family="turbo")
    assert result[5] == ""
    assert result[1] == ""


def test_raw_marker_on_retry():
    seq = [
        _json(positive="a cat", negative="b", scene_name=""),
        _json(positive="a cat", negative="b", scene_name="cat_scene"),
    ]
    result, _ = _run(seq, max_field_retries=2)
    assert result[2].startswith("[FIELD RETRY 1/2: missing scene_name]")


def test_reuse_cache_key_excludes_family_but_includes_inputs():
    # B1 regression: the reuse cache key must widen beyond (unique_id, prompt_mode) so a
    # change to style_preset/system_prompt/idea/etc. regenerates, while `family` stays
    # excluded (it is driven by the loaded checkpoint and should carry across swaps).
    writer._prompt_cache.clear()
    with patch.object(writer, "ensure_model_loaded", lambda *a, **k: True), \
         patch.object(writer, "chat_completion",
                      return_value=_json(positive="p", negative="n", scene_name="s")) as call:
        base = dict(server_url="http://localhost:1234/v1", api_key="", model="m",
                    context_length=8192, gpu_offload=1.0, system_prompt="SYS",
                    revision_notes="", temperature=0.7, max_tokens=512, seed=0,
                    reuse_last_prompt=True, generate_face_prompts=False, max_field_retries=2,
                    face_prompt_instruction="", prompt_mode="auto", unique_id="B1")

        writer.LLMPromptStudioWriter().execute(family="turbo", idea="cat",
                                               style_preset="— none —", **base)
        assert call.call_count == 1

        # Same inputs, different family -> still cached (family is NOT part of the key).
        writer.LLMPromptStudioWriter().execute(family="base", idea="cat",
                                               style_preset="— none —", **base)
        assert call.call_count == 1

        # Different idea -> must regenerate (idea IS part of the key).
        writer.LLMPromptStudioWriter().execute(family="turbo", idea="dog",
                                               style_preset="— none —", **base)
        assert call.call_count == 2

        # Different style_preset -> must regenerate (style_preset IS part of the key).
        writer.LLMPromptStudioWriter().execute(family="turbo", idea="dog",
                                               style_preset="Anime / Manga", **base)
        assert call.call_count == 3


def test_reuse_cache_key_includes_architecture():
    # A1 regression: the reuse cache key must include `architecture` (it changes token
    # style, negatives and the no-negative path for Flux/SD3), so two architectures never
    # share a cached prompt. The previously-cached result must NOT be reused across a
    # different architecture.
    writer._prompt_cache.clear()
    with patch.object(writer, "ensure_model_loaded", lambda *a, **k: True), \
         patch.object(writer, "chat_completion",
                      return_value=_json(positive="p", negative="n", scene_name="s")) as call:
        base = dict(server_url="http://localhost:1234/v1", api_key="", model="m",
                    context_length=8192, gpu_offload=1.0, system_prompt="SYS",
                    revision_notes="", temperature=0.7, max_tokens=512, seed=0,
                    reuse_last_prompt=True, generate_face_prompts=False, max_field_retries=2,
                    face_prompt_instruction="", prompt_mode="auto", unique_id="A1")

        writer.LLMPromptStudioWriter().execute(family="", idea="cat",
                                               style_preset="— none —",
                                               architecture="sdxl", **base)
        assert call.call_count == 1

        # Same inputs but a different architecture -> must regenerate (architecture IS a key).
        writer.LLMPromptStudioWriter().execute(family="", idea="cat",
                                               style_preset="— none —",
                                               architecture="flux", **base)
        assert call.call_count == 2

        # Identical architecture again -> reused (no regenerate).
        writer.LLMPromptStudioWriter().execute(family="", idea="cat",
                                               style_preset="— none —",
                                               architecture="flux", **base)
        assert call.call_count == 2
