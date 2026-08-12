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
         face_prompt_instruction=""):
    with patch.object(writer, "ensure_model_loaded", lambda *a, **k: None), \
         patch.object(writer, "chat_completion", side_effect=chat_side_effect) as call:
        result = writer.LLMPromptStudioWriter().execute(
            server_url="http://localhost:1234/v1", api_key="", model="m",
            context_length=8192, gpu_offload=1.0, system_prompt="SYS", idea="a cat",
            revision_notes="", temperature=0.7, max_tokens=512, seed=0,
            reuse_last_prompt=False, generate_face_prompts=generate_face_prompts,
            max_field_retries=max_field_retries, face_prompt_instruction=face_prompt_instruction,
            unique_id="1")
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
    # Face fields empty but not required -> no retry; existing fallback copies them.
    result, call = _run([_json(positive="a cat", negative="b", scene_name="s")])
    assert call.call_count == 1
    assert result[4] == "a cat"  # face_positive falls back to positive


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


def test_raw_marker_on_retry():
    seq = [
        _json(positive="a cat", negative="b", scene_name=""),
        _json(positive="a cat", negative="b", scene_name="cat_scene"),
    ]
    result, _ = _run(seq, max_field_retries=2)
    assert result[2].startswith("[FIELD RETRY 1/2: missing scene_name]")
