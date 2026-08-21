import os
import sys

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PKG)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from comfyui_llm_prompt_studio.nodes import writer as writer_node  # noqa: E402
from comfyui_llm_prompt_studio.nodes import critic as critic_node  # noqa: E402
from comfyui_llm_prompt_studio.nodes import scene_builder as scene_node  # noqa: E402
from unittest.mock import patch  # noqa: E402
import json as _json_mod  # noqa: E402

LOCAL_V1 = "http://localhost:1234/v1"


def _writer_json(**fields):
    return _json_mod.dumps(fields)


def _run_writer(chat_side_effect, **extra):
    kw = dict(server_url=LOCAL_V1, api_key="", model="m", context_length=8192,
              gpu_offload=1.0, system_prompt="SYS", idea="a cat", revision_notes="",
              temperature=0.7, max_tokens=512, seed=0, reuse_last_prompt=False,
              generate_face_prompts=False, max_field_retries=0,
              face_prompt_instruction="", prompt_mode="standard", family="",
              unique_id="1", style_preset="— none —")
    kw.update(extra)
    with patch.object(writer_node, "ensure_model_loaded", lambda *a, **k: None), \
         patch.object(writer_node, "chat_completion", side_effect=chat_side_effect), \
         patch.object(writer_node, "release_after_llm") as rel, \
         patch.object(writer_node, "mark_keep_loaded") as keep, \
         patch.object(writer_node, "release_enabled", return_value=True):
        result = writer_node.LLMPromptStudioWriter().execute(**kw)
    return result, rel, keep


def test_writer_releases_once_on_success():
    result, rel, keep = _run_writer(
        [_writer_json(positive="p", negative="n", scene_name="s",
                      face_positive="fp", face_negative="fn")])
    assert result[0] == "p"
    rel.assert_called_once()
    keep.assert_called_once_with(LOCAL_V1, False)


def test_writer_releases_on_chat_error():
    def _boom(*a, **k):
        raise RuntimeError("llm down")
    result, prep, rel, keep = _run_writer(_boom)
    # The error propagates, but the finally still released the model.
    rel.assert_called_once()
    keep.assert_called_once_with(LOCAL_V1, False)


def test_writer_reuse_cache_hit_no_release():
    writer_node._prompt_cache.clear()
    kw = dict(server_url=LOCAL_V1, api_key="", model="m", context_length=8192,
              gpu_offload=1.0, system_prompt="SYS", idea="a cat", revision_notes="",
              temperature=0.7, max_tokens=512, seed=0, reuse_last_prompt=True,
              generate_face_prompts=False, max_field_retries=0,
              face_prompt_instruction="", prompt_mode="standard", family="",
              unique_id="R1", style_preset="— none —")
    with patch.object(writer_node, "ensure_model_loaded", lambda *a, **k: None), \
         patch.object(writer_node, "chat_completion",
                      return_value=_writer_json(positive="p", negative="n",
                                                 scene_name="s", face_positive="fp",
                                                 face_negative="fn")) as call, \
         patch.object(writer_node, "release_after_llm") as rel, \
         patch.object(writer_node, "release_enabled", return_value=True):
        writer_node.LLMPromptStudioWriter().execute(**kw)
        # Second run hits the cache: no chat, no release.
        writer_node.LLMPromptStudioWriter().execute(**kw)
    assert call.call_count == 1
    rel.assert_not_called()


def test_writer_no_release_when_flag_false():
    result, rel, keep = _run_writer(
        [_writer_json(positive="p", negative="n", scene_name="s",
                      face_positive="fp", face_negative="fn")],
        release_vram_after_run=False)
    assert result[0] == "p"
    rel.assert_not_called()              # no release...
    keep.assert_called_once_with(LOCAL_V1, True)  # ...but pinned keep-loaded


def _run_scene(stage, chat_side_effect, **extra):
    kw = dict(stage=stage, image="IMG", server_url=LOCAL_V1, api_key="", model="m",
              context_length=16384, gpu_offload=1.0, describe_prompt="D",
              composer_prompt="C", user_changes="", image_max_size=1024,
              temperature=0.7, max_tokens=1024, max_field_retries=0,
              vision_check=False, description_view="", prompt_mode="standard",
              family="", unique_id="2", style_preset="— none —")
    kw.update(extra)
    with patch.object(scene_node, "ensure_model_loaded", lambda *a, **k: None), \
         patch.object(scene_node, "chat_completion", side_effect=chat_side_effect), \
         patch.object(scene_node, "image_to_base64", return_value="b64img"), \
         patch.object(scene_node, "release_after_llm") as rel, \
         patch.object(scene_node, "mark_keep_loaded") as keep, \
         patch.object(scene_node, "release_enabled", return_value=True):
        result = scene_node.LLMPromptStudioSceneBuilder().execute(**kw)
    return result, rel, keep


def test_scene_stage1_releases_on_early_return():
    # Stage 1 returns BEFORE any JSON parsing; the finally must still fire.
    result, rel, keep = _run_scene(
        "1 - describe",
        ["A long description of the image."])
    rel.assert_called_once()
    keep.assert_called_once_with(LOCAL_V1, False)


def test_scene_stage2_releases_on_success():
    result, rel, keep = _run_scene(
        "2 - compose",
        [_writer_json(positive="p", negative="n", scene_name="s")],
        description_view="a scene")
    assert result[0] == "p"
    rel.assert_called_once()
    keep.assert_called_once_with(LOCAL_V1, False)


def test_critic_releases_once_on_success():
    with patch.object(critic_node, "ensure_model_loaded", lambda *a, **k: None), \
         patch.object(critic_node, "resolve_vision", return_value=True), \
         patch.object(critic_node, "image_to_base64", return_value="b64img"), \
         patch.object(critic_node, "chat_completion",
                      return_value='{"score": 8, "verdict": "ok", "notes": ""}'), \
         patch.object(critic_node, "release_after_llm") as rel, \
         patch.object(critic_node, "mark_keep_loaded") as keep, \
         patch.object(critic_node, "release_enabled", return_value=True):
        critic_node.LLMPromptStudioCritic().execute(
            image="IMG", prompt="a prompt", server_url=LOCAL_V1, api_key="",
            model="m", context_length=16384, gpu_offload=1.0, critic_prompt="C",
            threshold=7, image_max_size=1024, temperature=0.3, max_tokens=1024,
            clear_notes_on_approve=True, auto_loop=False, max_retries=3,
            vision_check=False, unique_id="3", load_model_profile="auto")
    rel.assert_called_once()
    keep.assert_called_once_with(LOCAL_V1, False)
