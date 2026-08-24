"""Pure image/latent tensor helpers (no third-party sampler packs).

Everything here works on plain ``torch`` tensors in ComfyUI's image layout
``(B, H, W, C)`` float 0..1 and latent layout ``(B, 4, H, W)``. ``cv2`` is only
imported inside the blur helper so the module still imports headlessly.
"""

import torch
import numpy as np
from PIL import Image


def tensor_resize(tensor, w, h):
    """Resize a ``(B, H, W, C)`` image tensor to ``(B, h, w, C)`` via bilinear."""
    if w is None or h is None or w <= 0 or h <= 0:
        return tensor
    t = tensor.permute(0, 3, 1, 2)
    t = torch.nn.functional.interpolate(
        t, size=(int(h), int(w)), mode="bilinear", align_corners=False)
    return t.permute(0, 2, 3, 1)


def mask_resize(mask, w, h, binarize=0.5):
    """Resize a 2D ``[H, W]`` or 3D ``[B, H, W]`` mask to ``(B, h, w)`` via bilinear.

    Unlike :func:`tensor_resize` (which only handles 4D ``[B, H, W, C]`` images),
    this operates on ``[B, H, W]`` masks. The result is a float tensor; when
    ``binarize`` is not ``None`` values are thresholded at ``binarize`` to 0/1.
    """
    if not isinstance(mask, torch.Tensor):
        mask = torch.from_numpy(np.asarray(mask))
    t = mask.to(torch.float32)
    if t.dim() == 2:
        t = t[None, ...]                       # [1, H, W]
    if w is None or h is None or w <= 0 or h <= 0:
        return t
    t = t.unsqueeze(1)                         # [B, 1, H, W]
    t = torch.nn.functional.interpolate(
        t, size=(int(h), int(w)), mode="bilinear", align_corners=False)
    t = t.squeeze(1)                           # [B, H, W]
    if binarize is not None:
        t = (t >= float(binarize)).to(torch.float32)
    return t


def to_pil(tensor):
    """Convert a single-image ``(1, H, W, C)`` tensor to a ``PIL.Image``."""
    arr = (tensor[0].clamp(0.0, 1.0).cpu().numpy() * 255.0).astype("uint8")
    return Image.fromarray(arr)


def to_tensor(img):
    """Convert a ``PIL.Image`` to a ``(1, H, W, C)`` float tensor."""
    arr = np.array(img).astype("float32") / 255.0
    return torch.from_numpy(arr)[None, ...]


def to_latent_image(img, vae):
    """Encode an image tensor ``(B, H, W, C)`` into a latent dict.

    ``VAE.encode`` returns the latent samples tensor directly; ComfyUI's
    latent dict is ``{"samples": tensor}`` (see ``nodes.VAEEncode``). The
    returned dict is what callers expect (e.g. when assigning
    ``lat["noise_mask"]``).
    """
    return {"samples": vae.encode(img)}


def tensor_gaussian_blur_mask(mask, feather):
    """Blur a ``(1, H, W, 1)`` 0/1 mask by ``feather`` px (identity if <= 0)."""
    if feather is None or feather <= 0:
        return mask
    import cv2
    m = (mask[0, :, :, 0].cpu().numpy() * 255.0).astype("uint8")
    ksize = int(feather) * 2 + 1
    blurred = cv2.GaussianBlur(m, (ksize, ksize), float(feather))
    out = torch.from_numpy(blurred.astype("float32") / 255.0)
    return out[None, :, :, None]


def tensor_paste(dst, src, mask):
    """Composite ``src`` over ``dst`` where ``mask`` (``(B, h, w, 1)`` 0..1) is set."""
    return dst * (1.0 - mask) + src * mask


def crop_region_from_xywh(x1, y1, x2, y2):
    """Return slice objects selecting the ``[y1:y2, x1:x2]`` region of an image."""
    return (slice(int(y1), int(y2)), slice(int(x1), int(x2)))


def match_luminance(src, ref, weight, eps=1e-6):
    """Match ``src`` brightness/contrast to ``ref`` inside the ``weight`` region.

    Per-channel affine transfer ``src_adj = (src - mu_src) * (std_ref/std_src) + mu_ref``
    is computed only over the weighted mask (``weight`` is ``(B, H, W, 1)`` in 0..1,
    ``src``/``ref`` are ``(B, H, W, C)``). This lets a regenerated face blend into the
    original image without a brightness seam after an inpaint/detail pass. The scale is
    clamped to [0.5, 2.0] so a degenerate (flat) source region cannot blow up.
    """
    if weight is None:
        return src
    w = weight[..., 0]                                   # (B, H, W)
    w_sum = w.sum(dim=(1, 2), keepdim=True).clamp_min(eps)  # (B, 1)
    out = src.clone()
    for c in range(src.shape[-1]):
        s = src[..., c]                                  # (B, H, W)
        r = ref[..., c]
        mu_s = (w * s).sum(dim=(1, 2), keepdim=True) / w_sum
        mu_r = (w * r).sum(dim=(1, 2), keepdim=True) / w_sum
        var_s = (w * (s - mu_s) ** 2).sum(dim=(1, 2), keepdim=True) / w_sum
        var_r = (w * (r - mu_r) ** 2).sum(dim=(1, 2), keepdim=True) / w_sum
        std_s = (var_s + eps).sqrt()
        std_r = (var_r + eps).sqrt()
        scale = (std_r / std_s).clamp(0.5, 2.0)
        out[..., c] = (s - mu_s) * scale + mu_r
    return out
