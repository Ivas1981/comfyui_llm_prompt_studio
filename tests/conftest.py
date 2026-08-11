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