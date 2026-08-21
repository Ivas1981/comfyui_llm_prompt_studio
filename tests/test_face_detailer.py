"""Headless tests for the Face Detailer node and image/latent helpers.

``nodes`` is stubbed by conftest; KSampler / VAE classes are replaced with fakes so
the detection, upscale math, latent-crop (//8) and no-VAEEncode behavior can be
checked without a GPU.
"""

import torch

import comfyui_llm_prompt_studio.nodes.face_detailer as fd
from comfyui_llm_prompt_studio.nodes import _imgutils as iu


# -- _imgutils ----------------------------------------------------------------
def test_tensor_resize_keeps_batch_and_channels():
    t = torch.zeros((2, 8, 8, 3))
    r = iu.tensor_resize(t, 16, 16)
    assert r.shape == (2, 16, 16, 3)


def test_tensor_paste_blends_with_mask():
    dst = torch.zeros((1, 4, 4, 3))
    src = torch.ones((1, 4, 4, 3))
    mask = torch.ones((1, 4, 4, 1))
    out = iu.tensor_paste(dst, src, mask)
    assert torch.allclose(out, src)
    mask0 = torch.zeros((1, 4, 4, 1))
    out0 = iu.tensor_paste(dst, src, mask0)
    assert torch.allclose(out0, dst)


def test_gaussian_blur_mask_identity_when_no_feather():
    m = torch.ones((1, 4, 4, 1))
    assert torch.allclose(iu.tensor_gaussian_blur_mask(m, 0), m)


def test_to_pil_to_tensor_roundtrip():
    t = torch.rand((1, 8, 8, 3))
    img = iu.to_pil(t)
    back = iu.to_tensor(img)
    assert back.shape == (1, 8, 8, 3)


# -- Face Detailer -----------------------------------------------------------
class _FakeKSampler:
    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive,
               negative, latent_image, denoise=1.0):
        return [{"samples": latent_image["samples"].clone()}]


class _FakeVAE:
    def __init__(self):
        self.decode_calls = 0
        self.encode_calls = 0

    def decode(self, latent):
        # Face Detailer passes `latent["samples"]` (a tensor) to `vae.decode`.
        self.decode_calls += 1
        assert isinstance(latent, torch.Tensor), "vae.decode must receive samples, not a dict"
        s = latent
        return torch.zeros((s.shape[0], s.shape[2] * 8, s.shape[3] * 8, 3))

    def encode(self, pixels):
        self.encode_calls += 1
        # Production ComfyUI `vae.encode` returns the samples TENSOR (not a dict);
        # VAEEncode wraps it as {"samples": ...}. to_latent_image expects the tensor.
        return torch.zeros((pixels.shape[0], 4,
                            pixels.shape[1] // 8, pixels.shape[2] // 8))


def _install(monkeypatch):
    import nodes
    monkeypatch.setattr(nodes, "KSampler", _FakeKSampler)


def test_refine_skipped_when_upscale_le_1():
    node = fd.LLMPromptStudioFaceDetailer()
    out = node._refine_face(
        model=object(), vae=None,
        img=torch.zeros((1, 256, 256, 3)), i=0, x1f=10, y1f=10, x2f=210, y2f=210,
        positive=object(), negative=object(), face_positive=None, face_negative=None,
        seed=0, steps=20, cfg=7.0, sampler_name="euler", scheduler="karras",
        denoise=0.5, guide_size=128, max_size=1024, crop_factor=1.5, feather=5,
        inpaint_model=False)
    assert out is None


def test_latent_path_refines(monkeypatch):
    _install(monkeypatch)
    node = fd.LLMPromptStudioFaceDetailer()
    vae = _FakeVAE()
    # patch detection to a fixed box in a 64px image (latent 8x8)
    monkeypatch.setattr(node, "_detect", lambda *a, **k: [(8, 8, 24, 24)])
    latent = {"samples": torch.zeros((1, 4, 8, 8))}
    out = node.detect_and_detail(
        model=object(), vae=vae, positive=object(), negative=object(),
        seed=0, steps=20, cfg=7.0, sampler_name="euler", scheduler="karras",
        denoise=0.5, guide_size=512, max_size=1024, crop_factor=1.5,
        detection_method="haar", yolo_model_name="face_yolov8s.pt",
        detection_threshold=0.5, feather=5, inpaint_model=False, latent=latent)
    assert out[0].shape == (1, 64, 64, 3)
    # latent path decodes the whole latent for detection, then encodes the refined crop
    assert vae.encode_calls == 1
    assert vae.decode_calls >= 1


def test_pixel_path_encodes(monkeypatch):
    _install(monkeypatch)
    node = fd.LLMPromptStudioFaceDetailer()
    vae = _FakeVAE()
    monkeypatch.setattr(node, "_detect", lambda *a, **k: [(8, 8, 24, 24)])
    image = torch.zeros((1, 64, 64, 3))
    out = node.detect_and_detail(
        model=object(), vae=vae, positive=object(), negative=object(),
        seed=0, steps=20, cfg=7.0, sampler_name="euler", scheduler="karras",
        denoise=0.5, guide_size=512, max_size=1024, crop_factor=1.5,
        detection_method="haar", yolo_model_name="face_yolov8s.pt",
        detection_threshold=0.5, feather=5, inpaint_model=False, image=image)
    assert out[0].shape == (1, 64, 64, 3)
    # pixel path encodes the refined crop
    assert vae.encode_calls == 1
    assert vae.decode_calls >= 1


def test_no_faces_returns_original(monkeypatch):
    _install(monkeypatch)
    node = fd.LLMPromptStudioFaceDetailer()
    vae = _FakeVAE()
    monkeypatch.setattr(node, "_detect", lambda *a, **k: [])
    latent = {"samples": torch.zeros((1, 4, 8, 8))}
    out = node.detect_and_detail(
        model=object(), vae=vae, positive=object(), negative=object(),
        seed=0, steps=20, cfg=7.0, sampler_name="euler", scheduler="karras",
        denoise=0.5, guide_size=512, max_size=1024, crop_factor=1.5,
        detection_method="haar", yolo_model_name="face_yolov8s.pt",
        detection_threshold=0.5, feather=5, inpaint_model=False, latent=latent)
    assert out[0].shape == (1, 64, 64, 3)


# -- tiled VAE decode helper -----------------------------------------------
class _PlainVAE:
    def __init__(self):
        self.decoded = []

    def decode(self, samples):
        self.decoded.append(samples)
        return samples * 2


class _KwTiledVAE(_PlainVAE):
    def __init__(self):
        super().__init__()
        self.tiled = []

    def tiled_decode(self, samples, tile_x=0, tile_y=0):
        self.tiled.append((tile_x, tile_y))
        return samples * 3


class _PosTiledVAE(_PlainVAE):
    def __init__(self):
        super().__init__()
        self.tiled = []

    def tiled_decode(self, samples, tile, overlap):
        self.tiled.append((tile, overlap))
        return samples * 4


class _BrokenTiledVAE(_PlainVAE):
    def tiled_decode(self, *a, **k):
        raise RuntimeError("no tiled decode")


def test_decode_latent_uses_plain_when_tile_disabled():
    vae = _PlainVAE()
    s = torch.ones(1)
    out = fd._decode_latent(vae, s, 0)
    assert vae.decoded == [s] and torch.allclose(out, s * 2)


def test_decode_latent_uses_keyword_tiled_when_present():
    vae = _KwTiledVAE()
    s = torch.ones(1)
    out = fd._decode_latent(vae, s, 512)
    assert vae.tiled == [(512, 512)] and torch.allclose(out, s * 3) and vae.decoded == []


def test_decode_latent_uses_positional_tiled_fallback():
    vae = _PosTiledVAE()
    s = torch.ones(1)
    out = fd._decode_latent(vae, s, 256)
    assert vae.tiled == [(256, 16)] and torch.allclose(out, s * 4) and vae.decoded == []


def test_decode_latent_falls_back_without_tiled_decode():
    vae = _PlainVAE()
    s = torch.ones(1)
    out = fd._decode_latent(vae, s, 512)
    assert vae.decoded == [s] and torch.allclose(out, s * 2)


def test_decode_latent_falls_back_when_tiled_raises():
    vae = _BrokenTiledVAE()
    s = torch.ones(1)
    out = fd._decode_latent(vae, s, 512)
    assert vae.decoded == [s] and torch.allclose(out, s * 2)


def test_seed_widget_exposes_control_after_generate():
    req = fd.LLMPromptStudioFaceDetailer.INPUT_TYPES()["required"]
    assert req["seed"][1].get("control_after_generate") is True

