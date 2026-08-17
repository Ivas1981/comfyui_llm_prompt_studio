import os
import sys

# Import the module through the package stub (registered in conftest.py) so it resolves
# against `.Testing` without touching the production package.
from comfyui_llm_prompt_studio.nodes import model_recommendations as mr  # noqa: E402


def test_parse_size():
    assert mr._parse_size("qwen2.5-14b-instruct") == 14.0
    assert mr._parse_size("nvidia/nemotron-3-nano-4b") == 4.0
    assert mr._parse_size("google_gemma-3-12b-it") == 12.0
    assert mr._parse_size("unknown-model") is None
    assert mr._parse_size("") is None


def test_parse_size_takes_last_match():
    # When several size hints appear, the LAST (most specific, full model) wins, not the
    # first fragment — e.g. a MoE id "llama-3.2-8x3b-…-18.4b" must yield 18.4, not 3.0.
    assert mr._parse_size("llama-3.2-8x3b-instruct-18.4b") == 18.4
    assert mr._parse_size("mixtral-8x7b-math-22b") == 22.0


def test_recommend_for_baseline_structured_on_large():
    rec = mr.recommend_for("qwen2.5-14b-instruct", "writer")
    assert rec == {"profile": "baseline", "structured": True}


def test_recommend_for_strict_on_small():
    rec = mr.recommend_for("nvidia/nemotron-3-nano-4b", "writer")
    assert rec == {"profile": "strict", "structured": False}


def test_recommend_for_unknown_model_is_baseline():
    rec = mr.recommend_for("unknown-model", "writer")
    assert rec == {"profile": "baseline", "structured": False}


def test_recommend_for_describe_is_baseline_no_structured():
    rec = mr.recommend_for("qwen2.5-14b-instruct", "describe")
    assert rec == {"profile": "baseline", "structured": False}


def test_resolve_profile_auto_writer_no_image():
    res = mr.resolve_profile("auto", "qwen2.5-14b-instruct", "writer", has_image=False)
    assert res["profile"] == "baseline"
    assert res["structured"] is True
    assert res["response_format"] is mr.WRITER_RESPONSE_SCHEMA
    assert res["params"]["temperature"] == 0.7


def test_resolve_profile_auto_small_model_no_structured():
    res = mr.resolve_profile("auto", "nvidia/nemotron-3-nano-4b", "writer", has_image=False)
    assert res["profile"] == "strict"
    assert res["structured"] is False
    assert res["response_format"] is None


def test_resolve_profile_auto_describe_with_image_never_structured():
    res = mr.resolve_profile("auto", "qwen2.5-14b-instruct", "describe", has_image=True)
    assert res["profile"] == "baseline"
    assert res["structured"] is False
    assert res["response_format"] is None


def test_resolve_profile_named_creative():
    res = mr.resolve_profile("creative", "whatever", "writer", has_image=False)
    assert res["profile"] == "creative"
    assert res["params"]["temperature"] == 1.1
    assert res["params"]["structured"] is False
    assert res["response_format"] is None


def test_resolve_profile_structured_named_sends_schema():
    res = mr.resolve_profile("structured", "whatever", "writer", has_image=False)
    assert res["profile"] == "structured"
    assert res["structured"] is True
    assert res["response_format"] is mr.WRITER_RESPONSE_SCHEMA


def test_resolve_profile_custom_returns_none_params():
    res = mr.resolve_profile("custom", "whatever", "writer", has_image=False)
    assert res["profile"] == "custom"
    assert res["params"] is None
    assert res["response_format"] is None


def test_resolve_profile_structured_with_image_omits_schema():
    # Critic always has an image, so even a structured choice never sends response_format.
    res = mr.resolve_profile("structured", "whatever", "critic", has_image=True)
    assert res["structured"] is True
    assert res["response_format"] is None


def test_schema_for_kind():
    assert mr.schema_for_kind("writer") is mr.WRITER_RESPONSE_SCHEMA
    assert mr.schema_for_kind("compose") is mr.WRITER_RESPONSE_SCHEMA
    assert mr.schema_for_kind("critic") is mr.CRITIC_RESPONSE_SCHEMA
    assert mr.schema_for_kind("describe") is None


def test_profiles_have_reasoning_off():
    for name, p in mr.PROFILES.items():
        assert p["reasoning"] == "off"
        assert "temperature" in p
        assert "top_p" in p
        assert "top_k" in p
        assert "repeat_penalty" in p
        assert "presence_penalty" in p
        assert "min_p" in p
