import json
import os
import sys

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PKG)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from comfyui_llm_prompt_studio import presets  # noqa: E402
import pytest  # noqa: E402


def _user_path(tmp_path):
    import folder_paths
    folder_paths.get_output_directory.return_value = str(tmp_path)
    return tmp_path / "llm_prompt_studio_presets.json"


def test_presets_first_run_creates_file(tmp_path, monkeypatch):
    p = _user_path(tmp_path)
    if p.exists():
        p.unlink()
    data = presets.load_presets()
    assert p.exists()
    assert len(data["presets"]) == 14


def test_presets_existing_file_not_overwritten(tmp_path, monkeypatch):
    p = _user_path(tmp_path)
    p.write_text(json.dumps({"schema_version": "1.0.0", "presets": [
        {"id": "x", "name": "Custom", "system_prompt": "S",
         "style_tags_positive": ["t"], "style_tags_negative": [],
         "disabled_in_no_negative_mode": False}]}))
    data = presets.load_presets()
    assert [pr["name"] for pr in data["presets"]] == ["Custom"]


def test_presets_invalid_json_fallback(tmp_path, monkeypatch):
    p = _user_path(tmp_path)
    p.write_text("{ this is not valid json")
    data = presets.load_presets()
    assert len(data["presets"]) == 14  # fell back to shipped defaults


def test_presets_migration(tmp_path, monkeypatch):
    p = _user_path(tmp_path)
    # Old schema: no schema_version, missing per-preset fields.
    p.write_text(json.dumps({"presets": [{"name": "Old", "system_prompt": "X"}]}))
    data = presets.load_presets()
    pr = data["presets"][0]
    assert pr["id"] == "old"
    assert pr["style_tags_positive"] == []
    assert pr["disabled_in_no_negative_mode"] is False
    assert data["schema_version"] == presets.CURRENT_SCHEMA_VERSION


def test_presets_reset_to_defaults(tmp_path, monkeypatch):
    p = _user_path(tmp_path)
    p.write_text(json.dumps({"schema_version": "1.0.0", "presets": [
        {"id": "x", "name": "Custom", "system_prompt": "S",
         "style_tags_positive": ["t"], "style_tags_negative": [],
         "disabled_in_no_negative_mode": False}]}))
    presets.reset_to_defaults()
    data = presets.load_presets()
    assert len(data["presets"]) == 14


def test_presets_apply_style_tags():
    preset = {"style_tags_positive": ["tag1", "tag2"], "style_tags_negative": ["ntag"]}
    pos, neg = presets.apply_preset_to_prompts(preset, "a cat", "blurry", no_negative=False)
    assert "tag1, tag2" in pos
    assert "ntag" in neg
    # In no-negative mode the negative tags must be skipped.
    pos2, neg2 = presets.apply_preset_to_prompts(preset, "a cat", "blurry", no_negative=True)
    assert neg2 == "blurry"


def test_presets_get_by_name(tmp_path, monkeypatch):
    _user_path(tmp_path)
    preset = presets.get_preset_by_name("Photorealism")
    assert preset is not None
    assert preset["id"] == "photorealism"
    assert presets.get_preset_by_name("does-not-exist") is None
