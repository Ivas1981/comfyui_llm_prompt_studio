import os
import sys

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PKG)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from comfyui_llm_prompt_studio.parsing import slugify  # noqa: E402
from comfyui_llm_prompt_studio.nodes.smart_save import _slug_part  # noqa: E402


def test_slugify_ascii():
    assert slugify("Red Dress Beach Sunset!") == "red_dress_beach_sunset"
    assert slugify("") == "scene"


def test_slugify_cyrillic():
    # Cyrillic letters are preserved (not stripped) by the Unicode-aware pattern.
    assert slugify("Красное платье на пляже") == "красное_платье_на_пляже"


def test_slugify_cjk():
    # CJK has no word boundaries, so it collapses to a single token (acceptable per design).
    out = slugify("红色连衣裙")
    assert out == "红色连衣裙"


def test_slugify_emoji():
    # Emoji are non-word chars and are dropped, leaving a meaningful ASCII part.
    out = slugify("A cat 🐱 on the beach")
    assert out == "a_cat_on_the_beach"


def test_slugify_mixed():
    out = slugify("Cat Кот 猫 sunset")
    # Mixed scripts all kept, joined by underscores.
    assert "cat" in out and "кот" in out and "sunset" in out


def test_slug_part_cyrillic():
    # Checkpoint / lora names with Cyrillic stay valid filename parts (not stripped to "").
    assert _slug_part("Модель_Версия.safetensors") == "Модель_Версия"


def test_slug_part_strips_separators():
    assert _slug_part("my model!!.safetensors") == "my_model"
    assert _slug_part("") == ""
