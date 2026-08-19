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


def test_model_native_scale_parses_filename():
    # Baked scale must win over the requested factor (the bug that produced "каша").
    assert lu.model_native_scale("latent-upscaler-v2.1_SDxl-x1.5.safetensors", 2.0) == 1.5
    assert lu.model_native_scale("latent-upscaler-v2.1_SDv1-x2.0.safetensors", 2.0) == 2.0
    assert lu.model_native_scale("latent-upscaler-v2.1_SDxl-x1.25.safetensors", 2.0) == 1.25
    # No scale in the name (e.g. ttl .pt) falls back to the requested factor.
    assert lu.model_native_scale("sdxl_resizer.pt", 2.0) == 2.0
    assert lu.model_native_scale("some-model.safetensors", 2.0) == 2.0


def test_latent_upscale_with_model_uses_native_scale(tmp_path, monkeypatch):
    # A 1.5x checkpoint must upscale 1.5x even when the caller requests factor=2.0.
    net = lu._LatentUpscalerNet(1.5)
    import safetensors.torch as sf
    path = tmp_path / "fake-x1.5.safetensors"
    sf.save_file({k: v.clone() for k, v in net.state_dict().items()}, str(path))

    monkeypatch.setattr(lu, "_resolve_path", lambda name: str(path))
    latent = {"samples": torch.zeros((1, 4, 32, 32))}
    out = lu.latent_upscale_with_model(latent, "fake-x1.5.safetensors", 2.0)
    # 32 * 1.5 = 48, not 64 (which a 2.0 factor would have produced).
    assert out["samples"].shape[2] == 48
    assert out["samples"].shape[3] == 48

