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


def test_profiles_follow_modern_min_p_consensus():
    # 2024-2026 local-inference consensus: min_p is the primary truncation sampler,
    # top_k disabled (0), top_p only a generous fallback, penalties near neutral.
    # Structured (JSON) output is near-greedy for the highest schema parse rate.
    assert mr.PROFILES["baseline"]["min_p"] == 0.05
    assert mr.PROFILES["baseline"]["top_k"] == 0
    assert mr.PROFILES["baseline"]["top_p"] == 0.95
    assert mr.PROFILES["baseline"]["repeat_penalty"] == 1.05

    assert mr.PROFILES["structured"]["temperature"] == 0.1
    assert mr.PROFILES["structured"]["min_p"] == 0.05
    assert mr.PROFILES["structured"]["top_k"] == 0
    assert mr.PROFILES["structured"]["structured"] is True

    # creative must not stack repeat + presence penalties (incoherent loops).
    assert mr.PROFILES["creative"]["repeat_penalty"] == 1.05
    assert mr.PROFILES["creative"]["presence_penalty"] == 0.0
    assert mr.PROFILES["creative"]["top_k"] == 0

    assert mr.PROFILES["strict"]["temperature"] == 0.3
    assert mr.PROFILES["strict"]["min_p"] == 0.02
    assert mr.PROFILES["strict"]["top_k"] == 0


def test_resolve_profile_neutral_values_normalized_to_none():
    # B2: profile "neutral" sampling values (top_k 0, top_p 1.0, min_p 0.0,
    # presence_penalty 0.0, repeat_penalty 1.0) must normalize to None so they are
    # omitted on the wire. The "structured" profile carries all of them.
    res = mr.resolve_profile("structured", "whatever", "writer", has_image=False)
    p = res["params"]
    assert p["top_k"] is None
    assert p["top_p"] is None
    assert p["presence_penalty"] is None
    assert p["repeat_penalty"] is None
    # Genuine values are kept.
    assert p["min_p"] == 0.05
    assert p["temperature"] == 0.1


def test_resolve_profile_gemma_architecture_override():
    # B3/B4: a Gemma architecture (gemma3/gemma4) gets top_k=64 on top of the profile,
    # and PROFILES is never mutated.
    res = mr.resolve_profile("auto", "qwen2.5-14b-instruct", "writer",
                             has_image=False, architecture="gemma3")
    assert res["params"]["top_k"] == 64
    # Unrelated architecture is unaffected.
    res2 = mr.resolve_profile("auto", "qwen2.5-14b-instruct", "writer",
                              has_image=False, architecture="llama")
    assert res2["params"]["top_k"] is None
    # PROFILES must remain pristine.
    assert mr.PROFILES["baseline"]["top_k"] == 0


def test_resolve_profile_accepts_optional_params_gracefully():
    # Headless / server_routes callers that omit architecture/param_count must work.
    res = mr.resolve_profile("auto", "qwen2.5-14b-instruct", "writer", has_image=False)
    assert res["profile"] == "baseline"


def test_recommend_for_uses_param_count_over_name():
    # B4.4: when the API reports an active param count, it overrides the name heuristic.
    # A 4B model by name would be "strict", but a 70B MoE reported via param_count is
    # "baseline" (the real strength is what matters).
    assert mr.recommend_for("tiny-model", "writer", param_count=70.0) == \
        {"profile": "baseline", "structured": True}
    assert mr.recommend_for("huge-model", "writer", param_count=4.0) == \
        {"profile": "strict", "structured": False}
    # Without param_count the name heuristic is used as the fallback.
    assert mr.recommend_for("qwen2.5-14b-instruct", "writer", param_count=None) == \
        {"profile": "baseline", "structured": True}
