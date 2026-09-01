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
    monkeypatch.setattr(node, "_detect", lambda *a, **k: [(8, 8, 24, 24, None)])
    latent = {"samples": torch.zeros((1, 4, 8, 8))}
    out = node.detect_and_detail(
        model=object(), vae=vae, positive=object(), negative=object(),
        seed=0, steps=20, cfg=7.0, sampler_name="euler", scheduler="karras",
        denoise=0.5, guide_size=512, max_size=1024, crop_factor=1.5,
        detection_method="haar", yolo_model_name="face_yolov8s.pt",
        yolo_seg_model_name="(none)",
        detection_threshold=0.5,         feather=5, mask_shape="square", bbox_scale=1.0, iterations=1,
        inpaint_model=False, latent=latent)
    assert out[0].shape == (1, 64, 64, 3)
    # latent path decodes the whole latent for detection, then encodes the refined crop
    assert vae.encode_calls == 1
    assert vae.decode_calls >= 1


def test_pixel_path_encodes(monkeypatch):
    _install(monkeypatch)
    node = fd.LLMPromptStudioFaceDetailer()
    vae = _FakeVAE()
    monkeypatch.setattr(node, "_detect", lambda *a, **k: [(8, 8, 24, 24, None)])
    image = torch.zeros((1, 64, 64, 3))
    out = node.detect_and_detail(
        model=object(), vae=vae, positive=object(), negative=object(),
        seed=0, steps=20, cfg=7.0, sampler_name="euler", scheduler="karras",
        denoise=0.5, guide_size=512, max_size=1024, crop_factor=1.5,
        detection_method="haar", yolo_model_name="face_yolov8s.pt",
        yolo_seg_model_name="(none)",
        detection_threshold=0.5,         feather=5, mask_shape="square", bbox_scale=1.0, iterations=1,
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
        denoise=0.5, guide_size=512, max_size=1024, crop_factor=1.5,
        detection_method="haar", yolo_model_name="face_yolov8s.pt",
        yolo_seg_model_name="(none)",
        detection_threshold=0.5,         feather=5, mask_shape="square", bbox_scale=1.0, iterations=1,
        inpaint_model=False, latent=latent)
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


# -- mask_resize -------------------------------------------------------------
def test_mask_resize_returns_3d_and_preserves_shape():
    m = torch.zeros((16, 16))
    m[4:12, 4:12] = 1.0
    r = iu.mask_resize(m, 8, 8)                       # downscale
    assert r.shape == (1, 8, 8)
    assert r.dtype.is_floating_point
    assert r[0, 3:5, 3:5].mean() > 0.5               # central block still set
    up = iu.mask_resize(m, 32, 32)                   # upscale
    assert up.shape == (1, 32, 32)
    # alternate binarize threshold still yields a 0/1 mask
    b = iu.mask_resize(m, 32, 32, binarize=0.9)
    assert set(b.unique().tolist()).issubset({0.0, 1.0})


# -- yolo_seg detection ------------------------------------------------------
class _Box:
    def __init__(self, xyxy):
        self.xyxy = torch.tensor([xyxy])     # [1, 4] like ultralytics
        self.conf = torch.tensor([0.9])       # [1]
        self.cls = torch.tensor([0])          # [1] face class 0


class _FakeMasks:
    def __init__(self, data):
        self.data = data                              # torch tensor [N, H, W]


class _FakeResult:
    def __init__(self, boxes, masks):
        self.boxes = boxes
        self.masks = masks


class _FakeYOLO:
    def __init__(self, path):
        self.path = path

    def __call__(self, arr, verbose=False, retina_masks=False, conf=0.25):
        H, W = arr.shape[0], arr.shape[1]
        boxes = [_Box([8.0, 8.0, 56.0, 56.0])]
        # circular/elliptical mask, not a rectangle -> proves shape is used
        ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                                torch.arange(W, dtype=torch.float32), indexing="ij")
        cy, cx = H / 2.0, W / 2.0
        m = ((xs - cx) ** 2 / (20.0 ** 2) + (ys - cy) ** 2 / (20.0 ** 2)) <= 1.0
        data = m.unsqueeze(0).to(torch.float32)
        return [_FakeResult(boxes, _FakeMasks(data))]


def _install_ultralytics(monkeypatch):
    import sys, types
    fake = types.ModuleType("ultralytics")

    def YOLO(path):
        return _FakeYOLO(path)
    fake.YOLO = YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)


def test_detect_yolo_seg_returns_shape_mask(monkeypatch):
    _install_ultralytics(monkeypatch)
    node = fd.LLMPromptStudioFaceDetailer()
    monkeypatch.setattr(node, "_resolve_yolo_seg", lambda name: "/fake/seg.pt")
    img = torch.zeros((1, 64, 64, 3))
    boxes = node._detect_yolo_seg(img, "seg.pt", 0.5)
    assert len(boxes) == 1
    x1, y1, x2, y2, seg = boxes[0]
    assert (x1, y1, x2, y2) == (8, 8, 56, 56)
    assert seg is not None
    assert seg.shape == (64, 64)
    assert seg.dim() == 2
    # ellipse mask: corners are outside, center is inside
    assert seg[0, 0].item() == 0
    assert seg[32, 32].item() == 1


def test_detect_yolo_seg_falls_back_to_rectangle_when_no_masks(monkeypatch):
    _install_ultralytics(monkeypatch)

    class _FlatResult(_FakeResult):
        def __init__(self, boxes):
            self.boxes = boxes
            self.masks = None
    orig = _FakeYOLO.__call__
    _FakeYOLO.__call__ = lambda self, arr, **k: [_FlatResult([_Box([1.0, 1.0, 20.0, 20.0])])]
    try:
        node = fd.LLMPromptStudioFaceDetailer()
        monkeypatch.setattr(node, "_resolve_yolo_seg", lambda name: "/fake/seg.pt")
        boxes = node._detect_yolo_seg(torch.zeros((1, 64, 64, 3)), "seg.pt", 0.5)
        assert len(boxes) == 1
        assert boxes[0][4] is None                     # no seg mask -> rectangle path
    finally:
        _FakeYOLO.__call__ = orig


# -- conf forwarding (ultralytics default 0.25 pre-filter) -------------------
class _WeakBox:
    def __init__(self, xyxy, conf, cls=0):
        self.xyxy = torch.tensor([xyxy])
        self.conf = torch.tensor([conf])
        self.cls = torch.tensor([cls])


class _BoxesOnly:
    def __init__(self, boxes):
        self.boxes = boxes


def _install_ultralytics_weak(monkeypatch, face_conf):
    """Stub ultralytics like the real one: it drops boxes below the `conf` it is
    called with (default 0.25). This reproduces the pre-filter that previously
    made weak faces (<0.25) undetectable regardless of the node's threshold."""
    import sys, types
    import comfyui_llm_prompt_studio.tests.test_face_detailer as selfmod

    class _WeakSegYOLO:
        def __init__(self, path):
            self.path = path

        def __call__(self, arr, verbose=False, retina_masks=False, conf=0.25):
            H, W = arr.shape[0], arr.shape[1]
            boxes = [selfmod._WeakBox([8.0, 8.0, 56.0, 56.0], face_conf)]
            boxes = [b for b in boxes if float(b.conf[0]) >= conf]
            ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                                    torch.arange(W, dtype=torch.float32), indexing="ij")
            m = ((xs - W / 2.0) ** 2 / (20.0 ** 2) + (ys - H / 2.0) ** 2 / (20.0 ** 2)) <= 1.0
            return [selfmod._FakeResult(boxes, selfmod._FakeMasks(m.unsqueeze(0).to(torch.float32)))]

    class _WeakBboxYOLO:
        def __init__(self, path):
            self.path = path

        def __call__(self, arr, verbose=False, conf=0.25):
            boxes = [selfmod._WeakBox([8.0, 8.0, 56.0, 56.0], face_conf)]
            boxes = [b for b in boxes if float(b.conf[0]) >= conf]
            return [selfmod._BoxesOnly(boxes)]

    fake = types.ModuleType("ultralytics")
    fake.YOLO = lambda p: _WeakSegYOLO(p)
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    return _WeakBboxYOLO


def test_detect_yolo_seg_forwards_conf_to_recover_weak_faces(monkeypatch):
    # face detected at conf 0.10; without forwarding the threshold the model's
    # default 0.25 filter would drop it -> 0 boxes. Forwarding 0.1 recovers it.
    _install_ultralytics_weak(monkeypatch, face_conf=0.10)
    node = fd.LLMPromptStudioFaceDetailer()
    monkeypatch.setattr(node, "_resolve_yolo_seg", lambda name: "/fake/seg.pt")
    boxes = node._detect_yolo_seg(torch.zeros((1, 64, 64, 3)), "seg.pt", 0.1)
    assert len(boxes) == 1, "detection_threshold must be forwarded to ultralytics"


def test_detect_yolo_forwards_conf_to_recover_weak_faces(monkeypatch):
    import sys, types
    import comfyui_llm_prompt_studio.tests.test_face_detailer as selfmod

    class _WeakBboxYOLO:
        def __init__(self, path):
            self.path = path

        def __call__(self, arr, verbose=False, conf=0.25):
            boxes = [selfmod._WeakBox([8.0, 8.0, 56.0, 56.0], 0.10)]
            boxes = [b for b in boxes if float(b.conf[0]) >= conf]
            return [selfmod._BoxesOnly(boxes)]

    fake = types.ModuleType("ultralytics")
    fake.YOLO = lambda p: _WeakBboxYOLO(p)
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    node = fd.LLMPromptStudioFaceDetailer()
    monkeypatch.setattr(node, "_resolve_yolo", lambda name: "/fake/bbox.pt")
    boxes = node._detect_yolo(torch.zeros((1, 64, 64, 3)), "bbox.pt", 0.1)
    assert len(boxes) == 1, "detection_threshold must be forwarded to ultralytics"


# -- seg-mask refinement -----------------------------------------------------
def test_refine_face_seg_mask_smaller_than_rectangle(monkeypatch):
    _install(monkeypatch)
    node = fd.LLMPromptStudioFaceDetailer()
    vae = _FakeVAE()
    base = dict(model=object(), vae=vae, img=torch.zeros((1, 64, 64, 3)), i=0,
                x1f=8, y1f=8, x2f=56, y2f=56, positive=object(), negative=object(),
                face_positive=None, face_negative=None, seed=0, steps=20, cfg=7.0,
                sampler_name="euler", scheduler="karras", denoise=0.5,
                guide_size=512, max_size=1024, crop_factor=1.5, feather=0,
                inpaint_model=False)
    seg = torch.zeros((64, 64))
    seg[24:40, 24:40] = 1.0                           # true shape inside bbox
    rect_out = node._refine_face(seg_mask=None, **base)
    seg_out = node._refine_face(seg_mask=seg, **base)
    assert rect_out is not None and seg_out is not None
    _, _, _, _, _, rect_mask = rect_out
    _, _, _, _, _, seg_mask = seg_out
    # seg mask covers only the true shape -> strictly smaller than the rectangle
    assert seg_mask.sum() < rect_mask.sum()
    assert seg_mask.sum() > 0


def test_refine_face_seg_empty_shape_skips(monkeypatch):
    _install(monkeypatch)
    node = fd.LLMPromptStudioFaceDetailer()
    vae = _FakeVAE()
    seg = torch.zeros((64, 64))                       # nothing detected
    out = node._refine_face(
        model=object(), vae=vae, img=torch.zeros((1, 64, 64, 3)), i=0,
        x1f=8, y1f=8, x2f=56, y2f=56, positive=object(), negative=object(),
        face_positive=None, face_negative=None, seed=0, steps=20, cfg=7.0,
        sampler_name="euler", scheduler="karras", denoise=0.5,
        guide_size=512, max_size=1024, crop_factor=1.5, feather=0,
        inpaint_model=False, seg_mask=seg)
    assert out is None


# -- mask_shape (square / oval) --------------------------------------------
def test_build_shape_mask_oval_inscribed_in_rect():
    m_sq = fd.LLMPromptStudioFaceDetailer._build_shape_mask("square", 64, 64, 8, 56, 8, 56)
    m_ov = fd.LLMPromptStudioFaceDetailer._build_shape_mask("oval", 64, 64, 8, 56, 8, 56)
    assert m_sq.sum().item() == 48 * 48
    assert m_ov.sum().item() < m_sq.sum().item()
    assert m_ov[0, 8, 8].item() == 0    # rectangle corner -> outside ellipse
    assert m_ov[0, 32, 32].item() == 1  # center -> inside ellipse


def test_build_shape_mask_empty_rect_is_zero():
    m = fd.LLMPromptStudioFaceDetailer._build_shape_mask("square", 64, 64, 10, 10, 20, 20)
    assert m.sum().item() == 0


def test_refine_face_oval_composite_smaller_than_square(monkeypatch):
    _install(monkeypatch)
    node = fd.LLMPromptStudioFaceDetailer()
    vae = _FakeVAE()
    base = dict(model=object(), vae=vae, img=torch.zeros((1, 64, 64, 3)), i=0,
                x1f=8, y1f=8, x2f=56, y2f=56, positive=object(), negative=object(),
                face_positive=None, face_negative=None, seed=0, steps=20, cfg=7.0,
                sampler_name="euler", scheduler="karras", denoise=0.5,
                guide_size=512, max_size=1024, crop_factor=1.5, feather=0,
                inpaint_model=False)
    sq = node._refine_face(mask_shape="square", **base)
    ov = node._refine_face(mask_shape="oval", **base)
    assert sq is not None and ov is not None
    _, _, _, _, _, sq_mask = sq
    _, _, _, _, _, ov_mask = ov
    assert ov_mask.sum() < sq_mask.sum()


# -- bbox_scale -------------------------------------------------------------
def test_refine_face_bbox_scale_expands_mask(monkeypatch):
    _install(monkeypatch)
    node = fd.LLMPromptStudioFaceDetailer()
    vae = _FakeVAE()
    base = dict(model=object(), vae=vae, img=torch.zeros((1, 64, 64, 3)), i=0,
                x1f=16, y1f=16, x2f=48, y2f=48, positive=object(), negative=object(),
                face_positive=None, face_negative=None, seed=0, steps=20, cfg=7.0,
                sampler_name="euler", scheduler="karras", denoise=0.5,
                guide_size=512, max_size=1024, crop_factor=1.0, feather=0,
                inpaint_model=False, mask_shape="square")
    base_out = node._refine_face(bbox_scale=1.0, **base)
    scaled_out = node._refine_face(bbox_scale=2.0, **base)
    assert base_out is not None and scaled_out is not None
    _, _, _, _, _, base_mask = base_out
    _, _, _, _, _, scaled_mask = scaled_out
    assert scaled_mask.sum() > base_mask.sum()


# -- iterations --------------------------------------------------------------
def test_refine_face_iterations_sample_called_n_times(monkeypatch):
    import nodes

    class _CountKSampler:
        n = 0

        def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive,
                   negative, latent_image, denoise=1.0):
            _CountKSampler.n += 1
            return [{"samples": latent_image["samples"].clone()}]
    monkeypatch.setattr(nodes, "KSampler", _CountKSampler)
    node = fd.LLMPromptStudioFaceDetailer()
    vae = _FakeVAE()
    _CountKSampler.n = 0
    out = node._refine_face(
        model=object(), vae=vae, img=torch.zeros((1, 64, 64, 3)), i=0,
        x1f=8, y1f=10, x2f=56, y2f=58, positive=object(), negative=object(),
        face_positive=None, face_negative=None, seed=0, steps=20, cfg=7.0,
        sampler_name="euler", scheduler="karras", denoise=0.5,
        guide_size=512, max_size=1024, crop_factor=1.5, feather=0,
        inpaint_model=False, iterations=3)
    assert out is not None
    assert _CountKSampler.n == 3


# -- multiple faces ----------------------------------------------------------
def test_multiple_faces_get_distinct_seeds(monkeypatch):
    import nodes

    seen = []

    class _CapKSampler:
        def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive,
                   negative, latent_image, denoise=1.0):
            seen.append(seed)
            return [{"samples": latent_image["samples"].clone()}]
    monkeypatch.setattr(nodes, "KSampler", _CapKSampler)
    node = fd.LLMPromptStudioFaceDetailer()
    vae = _FakeVAE()
    monkeypatch.setattr(node, "_detect", lambda *a, **k: [
        (8, 8, 24, 24, None), (32, 32, 48, 48, None)])
    image = torch.zeros((1, 64, 64, 3))
    node.detect_and_detail(
        model=object(), vae=vae, positive=object(), negative=object(),
        seed=0, steps=20, cfg=7.0, sampler_name="euler", scheduler="karras",
        denoise=0.5, guide_size=512, max_size=1024, crop_factor=1.5,
        detection_method="haar", yolo_model_name="face_yolov8s.pt",
        yolo_seg_model_name="(none)", detection_threshold=0.5, feather=5,
        mask_shape="square", bbox_scale=1.0, iterations=1, inpaint_model=False,
        image=image)
    assert seen == [0, 1]


# -- gender threshold --------------------------------------------------------
class _GenderProbs:
    def __init__(self, data, top1):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.top1 = top1


class _GenderResult:
    def __init__(self, probs):
        self.probs = probs
        self.boxes = None


class _GenderYOLO:
    probs = None

    def __init__(self, path):
        self.path = path

    def __call__(self, crop, verbose=False):
        return [_GenderResult(_GenderYOLO.probs)]


def _install_ultralytics_gender(monkeypatch, probs_data, top1):
    import sys, types
    _GenderYOLO.probs = _GenderProbs(probs_data, top1)
    fake = types.ModuleType("ultralytics")
    fake.YOLO = lambda p: _GenderYOLO(p)
    monkeypatch.setitem(sys.modules, "ultralytics", fake)


def _gender_boxes(monkeypatch, arr, gender_filter, female_class=0, threshold=0.5):
    node = fd.LLMPromptStudioFaceDetailer()
    monkeypatch.setattr(node, "_resolve_yolo_gender", lambda n: "/fake/gender.pt")
    return node._apply_gender_filter(
        [(0, 0, 50, 50, None)], arr, gender_filter, "(model)",
        gender_female_class=female_class, gender_threshold=threshold)


def test_gender_low_conf_kept_as_unknown_when_filtering(monkeypatch):
    # predicts male (class 1) with conf 0.55; below threshold 0.7 -> kept (unknown)
    _install_ultralytics_gender(monkeypatch, [0.45, 0.55], top1=1)
    arr = torch.zeros((100, 100, 3), dtype=torch.uint8)
    out = _gender_boxes(monkeypatch, arr, "female", female_class=0, threshold=0.7)
    assert len(out) == 1


def test_gender_high_conf_dropped_when_wrong_gender(monkeypatch):
    # predicts male (class 1) with conf 0.95; filtering female -> dropped
    _install_ultralytics_gender(monkeypatch, [0.05, 0.95], top1=1)
    arr = torch.zeros((100, 100, 3), dtype=torch.uint8)
    out = _gender_boxes(monkeypatch, arr, "female", female_class=0, threshold=0.5)
    assert len(out) == 0


def test_gender_high_conf_kept_when_matching_gender(monkeypatch):
    # predicts male (class 1) with conf 0.95; filtering male -> kept
    _install_ultralytics_gender(monkeypatch, [0.05, 0.95], top1=1)
    arr = torch.zeros((100, 100, 3), dtype=torch.uint8)
    out = _gender_boxes(monkeypatch, arr, "male", female_class=0, threshold=0.5)
    assert len(out) == 1


def test_gender_any_ignores_threshold(monkeypatch):
    # gender_filter=any keeps all faces regardless of confidence
    _install_ultralytics_gender(monkeypatch, [0.45, 0.55], top1=1)
    arr = torch.zeros((100, 100, 3), dtype=torch.uint8)
    out = _gender_boxes(monkeypatch, arr, "any", female_class=0, threshold=0.9)
    assert len(out) == 1


def test_gender_no_model_returns_all(monkeypatch):
    node = fd.LLMPromptStudioFaceDetailer()
    arr = torch.zeros((100, 100, 3), dtype=torch.uint8)
    out = node._apply_gender_filter(
        [(0, 0, 50, 50, None)], arr, "female", "(none)", gender_female_class=0)
    assert len(out) == 1

