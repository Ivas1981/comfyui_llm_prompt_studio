import os
import sys
import types
from unittest.mock import MagicMock

# Make the `.Testing` directory importable as the package `comfyui_llm_prompt_studio`
# so the node/helper modules (which use package-relative imports) resolve against this
# mirror - not the production package on disk. This mirrors the package stub used by
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

# comfy.samplers is needed by the Smart Parameters node at import time (it builds the
# COMBO option tuples from KSampler.SAMPLERS / KSampler.SCHEDULERS). Stub it for
# standalone tests so ``import comfyui_llm_prompt_studio.nodes`` does not require a full
# ComfyUI install. The stub exposes the real option lists (a representative subset) so
# RETURN_TYPES / INPUT_TYPES can be validated without a GPU.
if "comfy" not in sys.modules or "comfy.samplers" not in sys.modules:
    _comfy = sys.modules.get("comfy", types.ModuleType("comfy"))
    _samplers = types.ModuleType("comfy.samplers")

    class _KSampler:
        # Real ComfyUI returns LISTS here (not tuples). Mirror that so `list + tuple`
        # type errors (e.g. `STANDARD_SCHEDULERS + ("auto",)`) are caught by tests.
        SAMPLERS = [
            "euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms",
            "dpmpp_2s_ancestral", "dpmpp_2m", "dpmpp_sde", "dpmpp_2m_sde",
            "dpmpp_3m_sde", "ddpm", "lcm", "ddim", "uni_pc", "uni_pc_bh2",
        ]
        SCHEDULERS = [
            "normal", "karras", "exponential", "sgm_uniform", "simple",
            "ddim_uniform", "beta", "linear_quadratic", "kl_optimal",
        ]

    _samplers.KSampler = _KSampler
    _comfy.samplers = _samplers
    sys.modules["comfy"] = _comfy
    sys.modules["comfy.samplers"] = _samplers

# comfy.utils is used by the hires-fix / face-detailer nodes (common_upscale,
# tiled_scale). Stub the bits we need for headless tests.
if "comfy.utils" not in sys.modules:
    _cu = types.ModuleType("comfy.utils")

    def common_upscale(samples, width, height, mode, crop):
        import torch.nn.functional as _F
        return _F.interpolate(samples, size=(height, width), mode="bilinear",
                              align_corners=False)

    def tiled_scale(*a, **k):
        raise NotImplementedError("tiled_scale is not exercised in headless tests")

    _cu.common_upscale = common_upscale
    _cu.tiled_scale = tiled_scale
    sys.modules["comfy.utils"] = _cu

# In ComfyUI `nodes` is a top-level module providing KSampler / LatentUpscale / VAE*
# etc. Our node code does ``from nodes import ...`` lazily. Stub a top-level ``nodes``
# module so those imports resolve in standalone tests (the project's own `nodes`
# subpackage is reached via the `comfyui_llm_prompt_studio.nodes` package, not here).
if "nodes" not in sys.modules:
    _nodes = types.ModuleType("nodes")

    class _Placeholder:
        def sample(self, *a, **k):
            raise NotImplementedError("stub nodes.KSampler")
        def upscale(self, *a, **k):
            raise NotImplementedError("stub nodes.LatentUpscale")
        def decode(self, *a, **k):
            raise NotImplementedError("stub nodes.VAEDecode")
        def encode(self, *a, **k):
            raise NotImplementedError("stub nodes.VAEEncode")

    _nodes.KSampler = _Placeholder
    _nodes.LatentUpscale = _Placeholder
    _nodes.VAEDecode = _Placeholder
    _nodes.KSampler = _Placeholder
    _nodes.LatentUpscale = _Placeholder
    _nodes.VAEDecode = _Placeholder
    _nodes.VAEEncode = _Placeholder
    _nodes.InpaintModelConditioning = _Placeholder
    sys.modules["nodes"] = _nodes

# The on-disk model cache lives inside the package directory and is read by ComfyUI at
# node-load time. Tests that exercise ensure_model_loaded() call remember_model(), which
# would otherwise persist the test model id (e.g. "m") into that file and pollute the real
# cache, causing spurious "Value not in list" errors in an actual ComfyUI session. Stub the
# write path so tests never touch the package cache file.
import comfyui_llm_prompt_studio.lm_http as _lm_http

_lm_http._persist_models = lambda *a, **k: None  # type: ignore[assignment]
