import json
import os
import sys

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PKG)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from comfyui_llm_prompt_studio import styles  # noqa: E402


def test_load_all_populates_presets_and_index():
    data = styles._load_all()
    assert data["presets"], "no presets loaded from Styles/"
    assert data["by_id"], "by_id index empty"
    assert len(data["by_id"]) == len(data["presets"])
    for pid, p in data["by_id"].items():
        assert p["id"] == pid


def test_preset_count_and_labels_consistent():
    labels = styles.get_style_labels()
    assert labels[0] == "— none —"
    # 1 none entry + one per preset.
    assert len(labels) == len(styles.get_all_styles()) + 1
    # The catalog should be a rich set across many domains.
    assert len(styles.get_all_styles()) >= 120


def test_resolve_style_inheritance_merges_and_unions_tags():
    # Find a preset that declares `extends`.
    child = next((p for p in styles.get_all_styles() if p.get("extends")), None)
    if not child:
        return  # No inheritance in the shipped catalog; nothing to verify.
    resolved = styles.resolve_style(child["id"])
    parent = styles.resolve_style(child["extends"])
    assert resolved is not None
    # Child system_prompt is appended to the parent's.
    assert parent["system_prompt"] in resolved["system_prompt"]
    assert child.get("system_prompt", "") in resolved["system_prompt"]
    # Tags are unioned (parent order preserved, child extras appended).
    p_pos = set(t.lower() for t in parent.get("style_tags_positive", []))
    for t in child.get("style_tags_positive", []):
        if t.lower() not in p_pos:
            assert t in resolved["style_tags_positive"]


def test_build_system_prompt_negative_on_vs_off():
    on = styles.build_system_prompt(None, negative_prompt=True)
    off = styles.build_system_prompt(None, negative_prompt=False)
    assert "NEGATIVE — required" in on
    assert "NEGATIVE — off (self-contained" in off
    assert "NEGATIVE — required" not in off


def test_build_system_prompt_nsfw_toggle():
    safe = styles.build_system_prompt(None, nsfw=False)
    explicit = styles.build_system_prompt(None, nsfw=True)
    assert "SAFETY — safe-for-work" in safe
    assert "SAFETY — nsfw permitted" in explicit


def test_build_system_prompt_format_fragments():
    natural = styles.build_system_prompt(None, prompt_format="natural")
    tags = styles.build_system_prompt(None, prompt_format="tags")
    assert "FORMAT — natural" in natural
    assert "FORMAT — tags" in tags


def test_build_system_prompt_face_toggle():
    off = styles.build_system_prompt(None, face_prompt=False)
    on = styles.build_system_prompt(None, face_prompt=True)
    assert "FACE — off" in off
    assert "FACE — on" in on


def test_build_system_prompt_uses_no_negative_variant_for_preset():
    # A preset that ships a no_negative variant must use it when negative_prompt is off.
    preset = next((p for p in styles.get_all_styles()
                   if p.get("system_prompt_no_negative")), None)
    if not preset:
        return
    with_neg = styles.build_system_prompt(preset, negative_prompt=True)
    without_neg = styles.build_system_prompt(preset, negative_prompt=False)
    assert preset["system_prompt"] in with_neg
    assert preset["system_prompt_no_negative"] in without_neg
    assert preset["system_prompt"] not in without_neg


def test_resolve_style_token_accepts_label_or_id():
    preset = styles.get_all_styles()[0]
    by_id = styles.resolve_style_token(preset["id"])
    assert by_id is not None
    label = f"{preset.get('category', '')} / {preset['name']}" if preset.get("category") else preset["name"]
    if label != preset["name"]:
        by_label = styles.resolve_style_token(label)
        assert by_label is not None
        assert by_label["id"] == preset["id"]


def test_architecture_addendum_in_build():
    flux = styles.build_system_prompt(None, architecture="flux")
    assert "Flux model" in flux
    sd15 = styles.build_system_prompt(None, architecture="sd15")
    assert "SD1.5 model" in sd15


EXPECTED_FORMATS = ["natural", "tags", "weighted", "structured", "midjourney", "booru"]


def test_build_system_prompt_covers_all_six_formats():
    # Every prompt_format must produce its own distinct fragment (no silent
    # fallback to the natural variant when the key is missing).
    outputs = {}
    for fmt in EXPECTED_FORMATS:
        out = styles.build_system_prompt(None, prompt_format=fmt)
        assert f"FORMAT — {fmt}" in out, f"missing fragment for format '{fmt}'"
        outputs[fmt] = out
    # Each format yields a recognizably different system prompt.
    assert len(set(outputs.values())) == len(EXPECTED_FORMATS)


def test_build_system_prompt_format_keys_match_expected():
    sp = styles.get_system_prompts()
    assert set(sp.get("prompt_format", {}).keys()) == set(EXPECTED_FORMATS)


def test_migration_preserves_all_legacy_presets():
    # The 51 legacy presets from presets_default.json must all be carried over
    # into the new Styles/ catalog (matched by id and by name).
    pkg_dir = os.path.dirname(styles.__file__)
    legacy_path = os.path.join(pkg_dir, "presets_default.json")
    with open(legacy_path, encoding="utf-8") as fh:
        legacy = json.load(fh)["presets"]
    assert len(legacy) >= 51
    legacy_ids = {p["id"] for p in legacy}
    legacy_names = {p["name"].lower() for p in legacy}

    new = styles.get_all_styles()
    new_ids = {p["id"] for p in new}
    new_names = {p["name"].lower() for p in new}

    missing_ids = legacy_ids - new_ids
    missing_names = legacy_names - new_names
    assert not missing_ids, f"legacy preset ids missing from catalog: {missing_ids}"
    assert not missing_names, f"legacy preset names missing from catalog: {missing_names}"
