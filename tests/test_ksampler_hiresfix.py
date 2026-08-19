"""Headless tests for the KSampler (Hires Fix) node and the latent upscaler.

``nodes`` and ``comfy.samplers`` are stubbed by conftest, and the KSampler /
LatentUpscale / VAE* classes are replaced with lightweight fakes so the node logic
(target size, cfg2/sampler2 resolution, AYS-aware sampling) can be exercised without
a GPU.
"""

import torch

import comfyui_llm_prompt_studio.nodes.ksampler_hiresfix as kh
from comfyui_llm_prompt_studio.nodes import _latent_upscaler as lu


class _FakeKSampler:
    last = {}

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive,
               negative, latent_image, denoise=1.0):
        _FakeKSampler.last = dict(seed=seed, steps=steps, cfg=cfg,
                                  sampler_name=sampler_name, scheduler=scheduler,
                                  denoise=denoise)
        return [{"samples": latent_image["samples"].clone()}]


class _FakeLatentUpscale:
    def upscale(self, samples, method, w, h, crop):
        return [{"samples": torch.zeros((samples["samples"].shape[0], 4, h, w))}]


class _FakeVAEDecode:
    def decode(self, vae, latent):
        s = latent["samples"]
        return [torch.zeros((s.shape[0], s.shape[2] * 8, s.shape[3] * 8, 3))]


def _install_mocks(monkeypatch):
    import nodes
    monkeypatch.setattr(nodes, "KSampler", _FakeKSampler)
    monkeypatch.setattr(nodes, "LatentUpscale", _FakeLatentUpscale)
    monkeypatch.setattr(nodes, "VAEDecode", _FakeVAEDecode)
    _FakeKSampler.last = {}


def _base_args(**over):
    args = dict(
        model=object(), positive=object(), negative=object(),
        latent_image={"samples": torch.zeros((1, 4, 32, 32))},
        seed=1, steps=20, cfg=7.0, sampler_name="euler", scheduler="karras",
        denoise=1.0, hires_enabled=True, hires_upscale_type="latent (model)",
        hires_upscale_method="bilinear", hires_latent_upscale_model="none",
        hires_latent_upscale_factor=2.0, hires_upscale_iterations=1,
        hires_steps=20, hires_cfg=-1.0, hires_denoise=0.5,
        hires_sampler_name="base", hires_scheduler="base",
        hires_use_same_seed=True, hires_seed=0, vae_decode=False,
        preview_method="none",
    )
    args.update(over)
    return args


def test_hires_upscales_to_target_size(monkeypatch):
    _install_mocks(monkeypatch)
    node = __import__("comfyui_llm_prompt_studio.nodes.ksampler_hiresfix",
                      fromlist=["LLMPromptStudioKSamplerHiresFix"]).LLMPromptStudioKSamplerHiresFix()
    final, image = node.sample(**_base_args())
    # base 256px (32 latent) -> 512px target (64 latent)
    assert final["samples"].shape[2] == 64
    assert final["samples"].shape[3] == 64


def test_no_hires_when_size_already_matches(monkeypatch):
    _install_mocks(monkeypatch)
    node = kh.LLMPromptStudioKSamplerHiresFix()
    # factor 1.0 x 1 iteration => target == base => no hires pass
    final, image = node.sample(**_base_args(hires_latent_upscale_factor=1.0,
                                            hires_enabled=True))
    assert final["samples"].shape[2] == 32
    assert final["samples"].shape[3] == 32


def test_hires_iterations_scale_output(monkeypatch):
    _install_mocks(monkeypatch)
    node = kh.LLMPromptStudioKSamplerHiresFix()
    # base 256px (32 latent); factor 2.0 x 2 iterations => 1024px (128 latent)
    final, image = node.sample(**_base_args(hires_upscale_iterations=2))
    assert final["samples"].shape[2] == 128
    assert final["samples"].shape[3] == 128


def test_hires_disabled_keeps_base(monkeypatch):
    _install_mocks(monkeypatch)
    node = kh.LLMPromptStudioKSamplerHiresFix()
    final, image = node.sample(**_base_args(hires_enabled=False))
    assert final["samples"].shape[2] == 32


def test_cfg2_and_sampler2_resolution(monkeypatch):
    _install_mocks(monkeypatch)
    node = kh.LLMPromptStudioKSamplerHiresFix()
    node.sample(**_base_args(hires_cfg=9.0, hires_sampler_name="dpmpp_2m",
                             hires_scheduler="karras"))
    assert _FakeKSampler.last["cfg"] == 9.0
    assert _FakeKSampler.last["sampler_name"] == "dpmpp_2m"
    assert _FakeKSampler.last["scheduler"] == "karras"


def test_cfg2_defaults_to_base_cfg(monkeypatch):
    _install_mocks(monkeypatch)
    node = kh.LLMPromptStudioKSamplerHiresFix()
    node.sample(**_base_args(hires_cfg=-1.0, hires_sampler_name="base",
                             hires_scheduler="base"))
    assert _FakeKSampler.last["cfg"] == 7.0
    assert _FakeKSampler.last["sampler_name"] == "euler"
    assert _FakeKSampler.last["scheduler"] == "karras"


def test_vae_decode_emits_image(monkeypatch):
    _install_mocks(monkeypatch)
    node = kh.LLMPromptStudioKSamplerHiresFix()
    final, image = node.sample(**_base_args(vae_decode=True, preview_method="vae",
                                            optional_vae=object()))
    assert image.shape[0] == 1
    assert image.shape[3] == 3


def test_latent_upscale_with_model_scales_by_factor(tmp_path, monkeypatch):
    # Build a real City96-shaped net, dump its weights, and load via the public helper.
    net = lu._LatentUpscalerNet(2)
    import safetensors.torch as sf
    sd = {k: v.clone() for k, v in net.state_dict().items()}
    path = tmp_path / "fake.safetensors"
    sf.save_file(sd, str(path))

    monkeypatch.setattr(lu, "_resolve_path", lambda name: str(path))
    latent = {"samples": torch.zeros((1, 4, 16, 16))}
    out = lu.latent_upscale_with_model(latent, "fake.safetensors-ish.safetensors", 2.0)
    assert out["samples"].shape[2] == 32
    assert out["samples"].shape[3] == 32
    # non-sample keys are preserved
    assert set(out.keys()) == {"samples"}
    assert "other" not in out


def test_hires_target_follows_model_native_scale(tmp_path, monkeypatch):
    # A 1.5x latent upscale model must upscale 1.5x (not the UI's hires_latent_upscale_factor=2.0).
    net = lu._LatentUpscalerNet(1.5)
    import safetensors.torch as sf
    path = tmp_path / "fake-x1.5.safetensors"
    sf.save_file({k: v.clone() for k, v in net.state_dict().items()}, str(path))
    monkeypatch.setattr(lu, "_resolve_path", lambda name: str(path))

    _install_mocks(monkeypatch)
    node = kh.LLMPromptStudioKSamplerHiresFix()
    # base 256px (32 latent) -> 1.5x -> 384px (48 latent), not 512px (64 latent)
    final, image = node.sample(**_base_args(
        hires_upscale_type="latent (model)",
        hires_latent_upscale_model="fake-x1.5.safetensors",
        hires_latent_upscale_factor=2.0,
    ))
    assert final["samples"].shape[2] == 48
    assert final["samples"].shape[3] == 48



# -- new hires_upscale_type / preview_method features ------------------------
def test_hires_upscale_type_options_present():
    types = kh.LLMPromptStudioKSamplerHiresFix.INPUT_TYPES()
    req = types["required"]
    assert "hires_upscale_type" in req
    assert list(req["hires_upscale_type"][0]) == kh.HIRES_UPSCALE_TYPES
    assert "preview_method" in req
    assert list(req["preview_method"][0]) == kh.PREVIEW_METHODS
    # legacy boolean selector was replaced by the explicit enum
    assert "hires_pixel_upscale" not in req


def test_pixel_model_upscale_requires_connected_model(monkeypatch):
    _install_mocks(monkeypatch)
    node = kh.LLMPromptStudioKSamplerHiresFix()
    try:
        node.sample(**_base_args(hires_enabled=True, hires_upscale_type="pixel (model)"))
        assert False, "expected ValueError for missing UPSCALE_MODEL"
    except ValueError as e:
        assert "pixel (model)" in str(e)


def test_preview_method_none_emits_placeholder(monkeypatch):
    _install_mocks(monkeypatch)
    node = kh.LLMPromptStudioKSamplerHiresFix()
    final, image = node.sample(**_base_args(vae_decode=True, preview_method="none"))
    # "none" -> 1x1x3 placeholder regardless of size
    assert image.shape == (1, 1, 1, 3)


def test_latent_to_rgb_uses_samples():
    samples = torch.randn(1, 4, 16, 16)
    out = kh.LLMPromptStudioKSamplerHiresFix._latent_to_rgb(samples)
    assert out.shape == (1, 16, 16, 3)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_resolve_hires_sampler_base_aliases():
    sam, sched = kh.LLMPromptStudioKSamplerHiresFix._resolve_hires_sampler(
        "euler", "karras", "base", "base")
    assert sam == "euler" and sched == "karras"
    sam2, sched2 = kh.LLMPromptStudioKSamplerHiresFix._resolve_hires_sampler(
        "euler", "karras", "dpmpp_2m", "simple")
    assert sam2 == "dpmpp_2m" and sched2 == "simple"


# -- hires_latent_upscale_tile (memory guard for the latent upscale model) ----
def test_hires_latent_upscale_tile_widget_defaults_to_auto():
    req = kh.LLMPromptStudioKSamplerHiresFix.INPUT_TYPES()["required"]
    assert "hires_latent_upscale_tile" in req
    spec = req["hires_latent_upscale_tile"]
    assert spec[0] == "INT"
    assert spec[1]["default"] == 0          # 0 = auto (whole latent while it fits)
    assert spec[1]["min"] == 0


def test_hires_latent_upscale_tile_reaches_the_upscaler(monkeypatch):
    _install_mocks(monkeypatch)
    seen = {}

    def fake_upscale(latent, model_name, factor, iterations=1, tile=0):
        seen.update(model=model_name, factor=factor, iterations=iterations, tile=tile)
        s = latent["samples"]
        return {"samples": torch.zeros((s.shape[0], 4, s.shape[2] * 2, s.shape[3] * 2))}

    monkeypatch.setattr(kh, "latent_upscale_with_model", fake_upscale)
    node = kh.LLMPromptStudioKSamplerHiresFix()
    node.sample(**_base_args(hires_latent_upscale_model="whatever-x2.0.safetensors",
                             hires_latent_upscale_tile=64))
    assert seen["tile"] == 64
    # Default (widget absent / 0) stays on the auto path.
    seen.clear()
    node.sample(**_base_args(hires_latent_upscale_model="whatever-x2.0.safetensors"))
    assert seen["tile"] == 0
