import json

from parsing import (_iter_brace_objects, _salvage_partial_prompt, find_missing_fields,
                       parse_critic_json, parse_prompt_json, slugify)
from model_meta import detect_checkpoint_family


def test_parse_prompt_json_valid():
    text = json.dumps({
        "positive": "a cat",
        "negative": "blurry",
        "scene_name": "cat_scene",
        "face_positive": "cat face",
        "face_negative": "bad face",
    })
    pos, neg, scene, fp, fn = parse_prompt_json(text)
    assert pos == "a cat"
    assert neg == "blurry"
    assert scene == "cat_scene"
    assert fp == "cat face"
    assert fn == "bad face"


def test_parse_prompt_json_code_fence():
    text = '```json\n{"positive": "a dog", "negative": "ugly", "scene_name": "dog"}\n```'
    pos, neg, scene, fp, fn = parse_prompt_json(text)
    assert pos == "a dog"
    assert neg == "ugly"
    assert scene == "dog"


def test_parse_prompt_json_with_prefix_text():
    text = 'Sure, here is the prompt:\n{"positive": "x", "negative": "y", "scene_name": "z"}'
    pos, neg, scene, _, _ = parse_prompt_json(text)
    assert pos == "x"
    assert neg == "y"


def test_parse_prompt_json_truncated():
    text = '{"positive": "a truncated prompt that goes on", "negative": "neg'
    pos, neg, scene, fp, fn = parse_prompt_json(text)
    assert pos.startswith("a truncated prompt")
    assert neg == "neg"


def test_iter_brace_objects_ignores_braces_in_strings():
    # A prompt value containing literal braces must not break object scanning.
    text = ('Here is the prompt:\n'
            '{"positive": "a cat sitting {like this}", "negative": "blurry", '
            '"scene_name": "cat"}')
    objs = list(_iter_brace_objects(text))
    assert len(objs) == 1
    parsed = json.loads(objs[0])
    assert parsed["positive"] == "a cat sitting {like this}"
    assert parsed["scene_name"] == "cat"


def test_iter_brace_objects_handles_escaped_quotes():
    text = '{"positive": "say \\"hello\\" {bold}"}'
    objs = list(_iter_brace_objects(text))
    assert len(objs) == 1
    assert json.loads(objs[0])["positive"] == 'say "hello" {bold}'


def test_parse_prompt_json_empty():
    pos, neg, scene, fp, fn = parse_prompt_json("")
    assert pos == ""
    assert neg == ""
    assert scene == ""


def test_parse_critic_json_valid():
    text = json.dumps({"score": 8, "verdict": "good", "revision_notes": "fix hands"})
    score, verdict, notes = parse_critic_json(text)
    assert score == 8
    assert verdict == "good"
    assert notes == "fix hands"


def test_parse_critic_json_garbage():
    score, verdict, notes = parse_critic_json("total nonsense, no json here")
    assert score == -1
    assert verdict == ""
    assert notes == "total nonsense, no json here"


def test_slugify():
    assert slugify("Red Dress Beach Sunset!") == "red_dress_beach_sunset"
    assert slugify("") == "scene"


def test_salvage_escaped_quotes():
    text = '{"positive": "a \\"quoted\\" thing", "negative": "n"}'
    salv = _salvage_partial_prompt(text)
    assert salv is not None
    pos, neg, scene, fp, fn = salv
    assert pos == 'a "quoted" thing'
    assert neg == "n"


def test_detect_family_from_name():
    assert detect_checkpoint_family("model_dmd_v2.safetensors") == "dmd"
    assert detect_checkpoint_family("model_lcm.safetensors") == "lcm"
    assert detect_checkpoint_family("model_turbo_xl.safetensors") == "turbo"
    assert detect_checkpoint_family("plain_model.safetensors") == "base"


def test_find_missing_fields_all_present():
    parsed = ("a", "b", "scene", "fp", "fn")
    assert find_missing_fields(parsed) == []
    assert find_missing_fields(parsed, require_face=True) == []


def test_find_missing_fields_empty_scene_name():
    parsed = ("a", "b", "", "fp", "fn")
    assert find_missing_fields(parsed) == ["scene_name"]


def test_find_missing_fields_face_only_when_required():
    parsed = ("a", "b", "scene", "", "")
    assert find_missing_fields(parsed) == []
    assert find_missing_fields(parsed, require_face=True) == ["face_positive", "face_negative"]