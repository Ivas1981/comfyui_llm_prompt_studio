"""Tests for the model-based latent upscaler.

The ttl-nn ``.pt`` weights ship with the project, so we can validate that the
reconstructed ``_TtlLatentResizer`` architecture actually loads them (proving the
key names match) and that a forward pass scales the latent by the requested factor.
"""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PT = os.path.join(ROOT, "models", "upscale_models", "sdxl_resizer.pt")

from comfyui_llm_prompt_studio.nodes import _latent_upscaler as lu


def test_ttl_loads_real_weights_and_scales():
    if not os.path.isfile(PT):
        import pytest
        pytest.skip("sdxl_resizer.pt not present")
    model = lu._TtlLatentResizer.load_model(PT, torch.device("cpu"), torch.float32)
    latent = torch.zeros((1, 4, 32, 32))
    with torch.no_grad():
        out = model(0.13025 * latent, scale=2.0) / 0.13025
    assert out.shape == (1, 4, 64, 64)


def test_ttl_state_dict_keys_match():
    if not os.path.isfile(PT):
        import pytest
        pytest.skip("sdxl_resizer.pt not present")
    sd = torch.load(PT, map_location="cpu", weights_only=True)
    # from_state_dict builds the exact architecture and load_state_dict(strict=True)
    model = lu._TtlLatentResizer.from_state_dict(sd)
    assert model is not None


def test_project_local_helpers():
    ups = lu.project_local_upscale_models()
    assert any(f.endswith(".safetensors") for f in ups)
    assert any(f.endswith(".pt") for f in ups)
    bbox = lu.project_local_ultralytics_bbox()
    assert any(f.endswith(".pt") for f in bbox)


def test_city96_net_scales_by_factor():
    net = lu._LatentUpscalerNet(2)
    x = torch.zeros((1, 4, 16, 16))
    with torch.no_grad():
        out = net(x)
    assert out.shape == (1, 4, 32, 32)
