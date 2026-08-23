import os
import sys
import json
import tempfile
import threading

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PKG)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from comfyui_llm_prompt_studio import library  # noqa: E402


def _lib():
    # Allow an absolute temp path (tests don't run inside ComfyUI's output dir).
    library.RESTRICT_PATHS_TO_OUTPUT = False
    d = tempfile.mkdtemp()
    return os.path.join(d, "lib.json")


def test_duplicate_by_positive_and_negative_skipped():
    p = _lib()
    _, added1 = library.save_prompt_to_library(p, "s", "a cat", "blurry")
    assert added1 is True
    _, added2 = library.save_prompt_to_library(p, "s", "a cat", "blurry")
    assert added2 is False


def test_duplicate_includes_face_prompts():
    p = _lib()
    # Same positive/negative but DIFFERENT face prompts must NOT be treated as a duplicate.
    _, added1 = library.save_prompt_to_library(p, "s", "a cat", "blurry",
                                               face_positive="face A", face_negative="fn A")
    assert added1 is True
    _, added2 = library.save_prompt_to_library(p, "s", "a cat", "blurry",
                                               face_positive="face B", face_negative="fn B")
    assert added2 is True
    # Identical face prompts -> duplicate.
    _, added3 = library.save_prompt_to_library(p, "s", "a cat", "blurry",
                                               face_positive="face A", face_negative="fn A")
    assert added3 is False
    entries = library.load_library(p)
    assert len(entries) == 2


def test_threaded_saves_keep_all_entries():
    p = _lib()

    def worker(i):
        library.save_prompt_to_library(p, f"scene{i}", f"prompt number {i}", f"neg {i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(25)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    entries = library.load_library(p)
    # No entries lost to a read-modify-write race under the per-path lock.
    assert len(entries) == 25
    prompts = {e["prompt"] for e in entries}
    assert all(f"prompt number {i}" in prompts for i in range(25))


def test_backup_file_created():
    p = _lib()
    # The backup is written just before overwriting an existing library file, so it appears
    # from the second save onward (holding the previous version).
    library.save_prompt_to_library(p, "s", "first", "n")
    library.save_prompt_to_library(p, "s2", "second", "n2")
    assert os.path.exists(p + ".bak")
    # Backup holds the previous (first) version.
    with open(p + ".bak", "r", encoding="utf-8") as f:
        bak = json.load(f)
    assert bak[0]["prompt"] == "first"
