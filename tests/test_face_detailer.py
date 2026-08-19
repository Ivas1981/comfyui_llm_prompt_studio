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
        return [torch.zeros((s.shape[0], s.shape[2] * 8, s.shape[3] * 8, 3))]

    def encode(self, pixels):
        self.encode_calls += 1
        return {"samples": torch.zeros((pixels.shape[0], 4,
                                        pixels.shape[1] // 8, pixels.shape[2] // 8))}


def _install(monkeypatch):
    import nodes
    monkeypatch.setattr(nodes, "KSampler", _FakeKSampler)


def test_refine_skipped_when_upscale_le_1():
    node = fd.LLMPromptStudioFaceDetailer()
    out = node._refine_face(
        model=object(), vae=None, lat_mode=False, latent=None,
        img=torch.zeros((1, 256, 256, 3)), i=0, x1=10, y1=10, x2=210, y2=210,
        positive=object(), negative=object(), face_positive=None, face_negative=None,
        seed=0, steps=20, cfg=7.0, sampler_name="euler", scheduler="karras",
        denoise=0.5, guide_size=128, max_size=1024, feather=5, inpaint_model=False)
    assert out is None


def test_latent_path_needs_no_vae_encode(monkeypatch):
    _install(monkeypatch)
    node = fd.LLMPromptStudioFaceDetailer()
    vae = _FakeVAE()
    # patch detection to a fixed box in a 64px image (latent 8x8)
    monkeypatch.setattr(node, "_detect", lambda *a, **k: [(8, 8, 24, 24)])
    latent = {"samples": torch.zeros((1, 4, 8, 8))}
    out = node.detect_and_detail(
        model=object(), vae=vae, positive=object(), negative=object(),
        seed=0, steps=20, cfg=7.0, sampler_name="euler", scheduler="karras",
        denoise=0.5, guide_size=512, max_size=1024, detection_method="haar",
        yolo_model_name="face_yolov8s.pt", detection_threshold=0.5, feather=5,
        inpaint_model=False, latent=latent)
    assert out[0].shape == (1, 64, 64, 3)
    # lat_mode: NO per-face VAEEncode (the optimized path).
    assert vae.encode_calls == 0
    # at least the detection decode happened
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
        denoise=0.5, guide_size=512, max_size=1024, detection_method="haar",
        yolo_model_name="face_yolov8s.pt", detection_threshold=0.5, feather=5,
        inpaint_model=False, image=image)
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
        denoise=0.5, guide_size=512, max_size=1024, detection_method="haar",
        yolo_model_name="face_yolov8s.pt", detection_threshold=0.5, feather=5,
        inpaint_model=False, latent=latent)
    assert out[0].shape == (1, 64, 64, 3)

