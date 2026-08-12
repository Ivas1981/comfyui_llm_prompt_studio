import os
import sys
from unittest.mock import MagicMock

# Make the package root importable so `import parsing` etc. work in tests.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

# folder_paths is provided by ComfyUI at runtime; stub it for standalone tests.
if "folder_paths" not in sys.modules:
    _fp = MagicMock()
    _fp.get_full_path.return_value = None            # no real model files in tests
    _fp.get_output_directory.return_value = os.path.join(_PKG_ROOT, "tests", "_output")
    sys.modules["folder_paths"] = _fp

# The on-disk model cache lives inside the package directory and is read by ComfyUI at
# node-load time. Tests that exercise ensure_model_loaded() call remember_model(), which
# would otherwise persist the test model id (e.g. "m") into that file and pollute the real
# cache, causing spurious "Value not in list" errors in an actual ComfyUI session. Stub the
# write path so tests never touch the package cache file.
import comfyui_llm_prompt_studio.lm_http as _lm_http

_lm_http._persist_models = lambda *a, **k: None  # type: ignore[assignment]