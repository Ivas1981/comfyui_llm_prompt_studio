import os
import sys
import types
from unittest.mock import MagicMock

# Make the `.Testing` directory importable as the package `comfyui_llm_prompt_studio`
# so the node/helper modules (which use package-relative imports) resolve against this
# mirror — not the production package on disk. This mirrors the package stub used by
# `.Testing/research/benchmark_models.py` so `import comfyui_llm_prompt_studio...` works
# in standalone tests.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .Testing
_PKG_NAME = "comfyui_llm_prompt_studio"
if _PKG_NAME not in sys.modules or getattr(sys.modules[_PKG_NAME], "__path__", None) != [_PKG_ROOT]:
    _pkg = types.ModuleType(_PKG_NAME)
    _pkg.__path__ = [_PKG_ROOT]
    _pkg.__package__ = _PKG_NAME
    sys.modules[_PKG_NAME] = _pkg

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
