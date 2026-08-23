import os
import sys

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PKG)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from comfyui_llm_prompt_studio.nodes.smart_save import LLMPromptStudioSmartSave  # noqa: E402


def test_atomic_reservation_distinct_under_collision(tmp_path, monkeypatch):
    # Force every colliding name to look taken so _next_path must bump the counter, while
    # tracking the names it actually "reserves" so they stay distinct.
    taken = set()

    def fake_open(path, *a, **k):
        if path in taken:
            raise FileExistsError()
        taken.add(path)
        return 999

    monkeypatch.setattr(os, "open", fake_open)
    monkeypatch.setattr(os, "close", lambda fd: None)

    p1 = LLMPromptStudioSmartSave._next_path(str(tmp_path), "scene")
    p2 = LLMPromptStudioSmartSave._next_path(str(tmp_path), "scene")
    assert p1 != p2
    assert p1.endswith("scene_00001.jpg")
    assert p2.endswith("scene_00002.jpg")


def test_atomic_reservation_picks_next_index(tmp_path):
    # Pre-existing files set the floor; the reserved name is max+1 and is actually created.
    for n in (3, 5):
        open(os.path.join(str(tmp_path), f"scene_{n:05d}.jpg"), "w").close()
    path = LLMPromptStudioSmartSave._next_path(str(tmp_path), "scene")
    assert path.endswith("scene_00006.jpg")
    assert os.path.exists(path)  # the exclusive-create actually reserved it


def _save_one(tmp_path, monkeypatch, save_metadata_to_exif):
    import torch
    monkeypatch.setattr(
        "comfyui_llm_prompt_studio.nodes.smart_save.folder_paths.get_output_directory",
        lambda: str(tmp_path))
    img = torch.zeros((1, 8, 8, 3))
    node = LLMPromptStudioSmartSave()
    out = node.save(img, True, "", "", jpeg_quality=95,
                    auto_save_to_library=False,
                    library_path="llm_prompt_studio_library.json",
                    save_metadata_to_exif=save_metadata_to_exif)
    assert out["ui"]["saved"] == [True]


def test_save_exif_off_does_not_crash(tmp_path, monkeypatch):
    # save_metadata_to_exif=False leaves exif=None; img.save(..., exif=None) must not
    # crash (it previously did via exif.tobytes() on None).
    _save_one(tmp_path, monkeypatch, save_metadata_to_exif=False)


def test_save_exif_empty_lines_does_not_crash(tmp_path, monkeypatch):
    # No prompts -> empty `lines` -> exif=None; must not crash.
    _save_one(tmp_path, monkeypatch, save_metadata_to_exif=True)
