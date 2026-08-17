import os
import sys
import tempfile

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PKG)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from comfyui_llm_prompt_studio import model_meta  # noqa: E402
from unittest.mock import patch  # noqa: E402


def _detect(name, metadata, tmp_path):
    # Point folder_paths at a real (empty) file so the isfile check passes, and inject the
    # safetensors metadata directly without touching disk.
    fake = tmp_path / "ckpt.safetensors"
    fake.write_bytes(b"")
    with patch.object(model_meta, "read_safetensors_metadata", return_value=metadata), \
         patch.object(model_meta.folder_paths, "get_full_path", return_value=str(fake)):
        return model_meta.detect_checkpoint_family(name)


def test_flash_resolves_when_flash_attention_in_metadata(tmp_path):
    # B2 regression: a real Flash checkpoint whose metadata mentions flash_attention must
    # still resolve to "flash". The old code blanket-skipped every flash match whenever
    # flash_attention appeared anywhere in the scanned text.
    assert _detect("FlashSDXL_10steps.safetensors",
                   {"model_type": "flash_attention_enabled"}, tmp_path) == "flash"


def test_flash_attention_alone_is_not_flash(tmp_path):
    # flash_attention inside a structured metadata value must NOT be detected as the Flash
    # family; the only marker ("flash") is the attention-mechanism substring and is skipped.
    assert _detect("BaseModel.safetensors",
                   {"arch": "flash_attention"}, tmp_path) == "base"


def test_flash_attention_in_filename_only_is_not_flash(tmp_path):
    # A checkpoint whose NAME embeds flash_attention (but no real Flash marker) stays base.
    assert _detect("MyFlash_attentionModel.safetensors",
                   {}, tmp_path) == "base"


def test_flash_filename_resolves_flash(tmp_path):
    # An ordinary Flash checkpoint with no flash_attention noise resolves to flash.
    assert _detect("SDXLFlashTurboMix_4step.safetensors",
                   {}, tmp_path) == "flash"
