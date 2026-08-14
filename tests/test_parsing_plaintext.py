import os
import sys

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PKG)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from comfyui_llm_prompt_studio.parsing import parse_prompt_json  # noqa: E402


def test_default_allows_plain_text():
    # By default a non-JSON, non-salvageable reply is accepted as the positive prompt.
    pos, neg, scene, fp, fn = parse_prompt_json("Just write a cat")
    assert pos == "Just write a cat"
    assert neg == ""


def test_plain_text_fallback_disabled_returns_empty():
    # The Writer disables the fallback so a model refusal becomes an explicit (empty) error
    # rather than a silent bad prompt.
    pos, neg, scene, fp, fn = parse_prompt_json(
        "I cannot help with that", allow_plain_text_fallback=False)
    assert (pos, neg, scene, fp, fn) == ("", "", "", "", "")


def test_json_still_parsed_when_fallback_disabled():
    import json
    text = json.dumps({"positive": "a dog", "negative": "x", "scene_name": "d"})
    pos, neg, scene, _, _ = parse_prompt_json(text, allow_plain_text_fallback=False)
    assert pos == "a dog"
    assert scene == "d"


def test_empty_text_with_fallback_off():
    assert parse_prompt_json("", allow_plain_text_fallback=False) == ("", "", "", "", "")
