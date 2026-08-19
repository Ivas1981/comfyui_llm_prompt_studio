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
