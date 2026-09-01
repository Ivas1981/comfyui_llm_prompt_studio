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
    # 14 original presets + the categorized library added by the Testing branch.
    assert len(data["presets"]) >= 49
    # Every preset now carries a category (original 14 + new ones).
    assert all(pr.get("category") for pr in data["presets"])


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
    assert len(data["presets"]) >= 49  # fell back to shipped defaults


def test_presets_migration(tmp_path, monkeypatch):
    p = _user_path(tmp_path)
    # Old schema: no schema_version, missing per-preset fields (including category).
    p.write_text(json.dumps({"presets": [{"name": "Old", "system_prompt": "X"}]}))
    data = presets.load_presets()
    pr = data["presets"][0]
    assert pr["id"] == "old"
    assert pr["style_tags_positive"] == []
    assert pr["disabled_in_no_negative_mode"] is False
    assert pr["category"] == ""  # backfilled, empty for unknown old preset
    assert data["schema_version"] == presets.CURRENT_SCHEMA_VERSION


def test_presets_reset_to_defaults(tmp_path, monkeypatch):
    p = _user_path(tmp_path)
    p.write_text(json.dumps({"schema_version": "1.0.0", "presets": [
        {"id": "x", "name": "Custom", "system_prompt": "S",
         "style_tags_positive": ["t"], "style_tags_negative": [],
         "disabled_in_no_negative_mode": False}]}))
    presets.reset_to_defaults()
    data = presets.load_presets()
    assert len(data["presets"]) >= 49


def test_presets_apply_style_tags():
    preset = {"style_tags_positive": ["tag1", "tag2"], "style_tags_negative": ["ntag"]}
    pos, neg = presets.apply_preset_to_prompts(preset, "a cat", "blurry", no_negative=False)
    assert "tag1, tag2" in pos
    assert "ntag" in neg
    # In no-negative mode the negative tags must be skipped.
    pos2, neg2 = presets.apply_preset_to_prompts(preset, "a cat", "blurry", no_negative=True)
    assert neg2 == "blurry"


def test_apply_preset_style_dedupes_positive_tags():
    # Audit7/1.2: a style tag already present in the prompt (or face prompt) must not be
    # appended again; new tags are still added. Mirrors the negative-tag dedupe behavior.
    preset = {"style_tags_positive": ["sharp focus", "cinematic"],
              "style_tags_negative": ["blurry"]}
    pos, neg, fpos, fneg = presets.apply_preset_style(
        preset, "masterpiece, sharp focus", "text, blurry",
        face_positive="face, sharp focus", face_negative="")
    assert pos.count("sharp focus") == 1
    assert "cinematic" in pos
    assert fpos.count("sharp focus") == 1
    assert "cinematic" in fpos
    # negative already had "blurry" and must not be doubled
    assert neg.count("blurry") == 1



def test_presets_get_by_name_and_label(tmp_path, monkeypatch):
    _user_path(tmp_path)
    # Bare name (old saved workflows) still resolves.
    preset = presets.get_preset_by_name("Photorealism")
    assert preset is not None
    assert preset["id"] == "photorealism"
    # Categorized label produced by get_preset_names() also resolves.
    names = presets.get_preset_names()
    label = next((n for n in names if n.endswith("Photorealism")), "")
    assert label
    assert "> " in label
    assert presets.get_preset_by_name(label) is not None
    assert presets.get_preset_by_name("does-not-exist") is None


def test_presets_get_names_are_labeled(tmp_path, monkeypatch):
    _user_path(tmp_path)
    names = presets.get_preset_names()
    assert len(names) >= 49
    # All names are "Category > Name" labels.
    assert all(">" in n for n in names)
