import json

from comfyui_llm_prompt_studio.parsing import parse_prompt_json, _extract_json_dict  # noqa: E402

pytest = __import__("pytest")
pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st  # noqa: E402


@given(st.text(min_size=10, max_size=1000))
def test_parse_prompt_json_never_crashes(text):
    result = parse_prompt_json(text)
    assert result is None or isinstance(result, tuple)
    if result:
        assert len(result) == 5  # (positive, negative, scene_name, face_+, face_-)


@given(st.dictionaries(
    keys=st.sampled_from(["positive", "negative", "scene_name", "face_positive", "face_negative"]),
    values=st.text()))
def test_extract_json_dict_handles_valid_json(data):
    json_str = json.dumps(data)
    result = _extract_json_dict(json_str, "positive")
    assert result is not None
