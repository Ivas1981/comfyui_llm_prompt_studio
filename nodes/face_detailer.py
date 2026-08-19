"""LLM Prompt Studio Face Detailer.

Refines detected faces *in place* (the way Impact-Pack's FaceDetailer does it):
it crops a region a little larger than the detected face (context margin),
upscales it, then re-samples only the face area while preserving the
surrounding context. The original face is used as the denoising base (via a
face mask / inpaint mask), so the result *refines* the face instead of
replacing it with a freshly generated one that is then pasted on top.

Detection: haar (cv2) or yolo (ultralytics, optional).
"""

import os

import torch

from ._ksample import sample_latent, node_span
from ._imgutils import (
    tensor_resize, to_latent_image, tensor_gaussian_blur_mask, tensor_paste,
)
from .smart_parameters import SAMPLERS_WITH_BASE, SCHEDULERS_WITH_BASE
from ._latent_upscaler import project_local_ultralytics_bbox


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _decode_latent(vae, samples, tile_size):
    """Decode ``samples`` with the VAE, optionally using ComfyUI's tiled decode.

    Tiled decode bounds VRAM on large hires latents (a plain decode materialises the
    whole image at once, which is what makes a 1600x1200 hires frame expensive). It
    falls back to a plain decode when tiling is disabled, or when the VAE does not
    expose ``tiled_decode`` / uses a different signature than the one we call.
    """
    if not tile_size or int(tile_size) <= 0:
        return vae.decode(samples)
    t = int(tile_size)
    td = getattr(vae, "tiled_decode", None)
    if td is None:
        return vae.decode(samples)
    try:
        return td(samples, tile_x=t, tile_y=t)          # current ComfyUI keyword form
    except TypeError:
        try:
            return td(samples, t, 16)                   # older positional (tile, overlap)
        except TypeError:
            return vae.decode(samples)
    except Exception:
        return vae.decode(samples)


class LLMPromptStudioFaceDetailer:
    CATEGORY = "LLM Prompt Studio"
    FUNCTION = "detect_and_detail"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths
            yolo_list = folder_paths.get_filename_list("ultralytics_bbox") or []
        except Exception:
            yolo_list = []
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (SAMPLERS_WITH_BASE, {"default": "dpmpp_2m"}),
                "scheduler": (SCHEDULERS_WITH_BASE, {"default": "karras"}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                             "tooltip": "How much the face is changed. Lower = closer to the "
                                        "original face (refine), higher = more regeneration."}),
                "guide_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8,
                             "tooltip": "Target size (px) of the smaller face dimension after upscale."}),
                "max_size": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8,
                            "tooltip": "Max size of the upscaled crop (used to clamp very large faces)."}),
                "crop_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05,
                                "tooltip": "How much margin to include around the detected face for "
                                           "context. >1.0 keeps surrounding pixels so the refined "
                                           "face blends seamlessly with the rest of the image."}),
                "detection_method": (["haar", "yolo"], {"default": "haar"}),
                "yolo_model_name": (
                    ["face_yolov8s.pt"] + project_local_ultralytics_bbox() + list(yolo_list),
                    {"default": "face_yolov8s.pt"}),
                "detection_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1,
                          "tooltip": "Feather (blur) radius, in px, of the face mask used when "
                                     "compositing the refined face back into the image."}),
                "inpaint_model": ("BOOLEAN", {"default": False,
                                  "tooltip": "Use ComfyUI's InpaintModelConditioning so the model "
                                             "sees the original face structure (concat latent) while "
                                             "regenerating the face area. Off = differential noise "
                                             "mask (refines in place)."}),
                "vae_tile_size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16,
                                  "tooltip": "Tile size (px) for the VAE decode of the incoming "
                                             "latent. 0 = decode the whole frame at once (best "
                                             "quality). A positive value (e.g. 512) tiles the decode "
                                             "to cap VRAM on large hires latents."}),
            },
            "optional": {
                "image": ("IMAGE", {"forceInput": True}),
                "latent": ("LATENT", {"forceInput": True}),
                "face_positive": ("CONDITIONING", {"forceInput": True}),
                "face_negative": ("CONDITIONING", {"forceInput": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    # -- detection ---------------------------------------------------------
    def _detect(self, img, method, yolo_name, threshold, vae):
        if method == "haar":
            return self._detect_haar(img)
        return self._detect_yolo(img, yolo_name, threshold)

    def _detect_haar(self, img):
        import cv2
        arr = (img[0].clamp(0.0, 1.0).cpu().numpy() * 255.0).astype("uint8")
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        detected = classifier.detectMultiScale(gray, 1.1, 3)
        boxes = []
        for (x, y, bw, bh) in detected:
            boxes.append((int(x), int(y), int(x + bw), int(y + bh)))
        return boxes

    def _detect_yolo(self, img, yolo_name, threshold):
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError(
                "YOLO detection requires the optional `ultralytics` package") from e
        path = self._resolve_yolo(yolo_name)
        if not path:
            raise FileNotFoundError("YOLO model not found: %s" % yolo_name)
        model = YOLO(path)
        arr = (img[0].clamp(0.0, 1.0).cpu().numpy() * 255.0).astype("uint8")
        res = model(arr, verbose=False)[0]
        boxes = []
        for box in res.boxes:
            if float(box.conf[0]) < threshold:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return boxes

    def _resolve_yolo(self, name):
        from ._latent_upscaler import PROJECT_ROOT
        local = os.path.join(PROJECT_ROOT, "models", "ultralytics", "bbox", name)
        if os.path.isfile(local):
            return local
        try:
            import folder_paths
            for cat in ("ultralytics_bbox", "ultralytics"):
                p = folder_paths.get_full_path(cat, name)
                if p and os.path.isfile(p):
                    return p
        except Exception:
            pass
        return None

    # -- refinement --------------------------------------------------------
    @staticmethod
    def _inpaint_encode(pos, neg, pixels, vae, mask):
        from nodes import InpaintModelConditioning
        return InpaintModelConditioning().encode(pos, neg, pixels, vae, mask, noise_mask=True)

    def _refine_face(self, model, vae, img, i, x1f, y1f, x2f, y2f,
                     positive, negative, face_positive, face_negative,
                     seed, steps, cfg, sampler_name, scheduler, denoise,
                     guide_size, max_size, crop_factor, feather, inpaint_model):
        fh = y2f - y1f
        fw = x2f - x1f
        upscale = guide_size / float(min(fw, fh))
        if upscale <= 1.0:
            return None

        pos = face_positive if face_positive is not None else positive
        neg = face_negative if face_negative is not None else negative

        H, W = img.shape[1], img.shape[2]
        # Context margin around the face so the refined face blends with its
        # surroundings (the margin is never re-sampled, only the face is).
        half = (max(crop_factor, 1.0) - 1.0) / 2.0
        dx = int(round(fw * half))
        dy = int(round(fh * half))
        x1 = max(0, x1f - dx)
        y1 = max(0, y1f - dy)
        x2 = min(W, x2f + dx)
        y2 = min(H, y2f + dy)
        cw = x2 - x1
        ch = y2 - y1
        if cw <= 0 or ch <= 0:
            return None

        # Face rectangle expressed inside the (margin-padded) crop.
        fx1, fy1 = x1f - x1, y1f - y1
        fx2, fy2 = x2f - x1, y2f - y1

        crop = img[i:i + 1, y1:y2, x1:x2, :]

        new_w = int(round(cw * upscale))
        new_h = int(round(ch * upscale))
        if new_w > max_size or new_h > max_size:
            s = max_size / float(max(new_w, new_h))
            new_w = int(round(new_w * s))
            new_h = int(round(new_h * s))
        new_w = _clamp(new_w, 1, max_size)
        new_h = _clamp(new_h, 1, max_size)

        up_crop = tensor_resize(crop, new_w, new_h)

        # Face mask inside the upscaled crop: 3D [B, H, W], 1 = refine, 0 = keep.
        mask = torch.zeros((1, new_h, new_w), dtype=up_crop.dtype, device=up_crop.device)
        sx, sy = new_w / float(cw), new_h / float(ch)
        mx1 = _clamp(int(round(fx1 * sx)), 0, new_w)
        my1 = _clamp(int(round(fy1 * sy)), 0, new_h)
        mx2 = _clamp(int(round(fx2 * sx)), 0, new_w)
        my2 = _clamp(int(round(fy2 * sy)), 0, new_h)
        if mx2 > mx1 and my2 > my1:
            mask[:, my1:my2, mx1:mx2] = 1.0

        if inpaint_model:
            # InpaintModelConditioning keeps the (grayed) original as concat context
            # and only regenerates within the face mask.
            pos, neg, lat = self._inpaint_encode(pos, neg, up_crop, vae, mask)
        else:
            # Differential noise mask: the VAE-encoded upscaled crop is the base,
            # and sampling only touches the face area (margin is preserved).
            lat = to_latent_image(up_crop, vae)
            lat["noise_mask"] = mask

        refined = sample_latent(model, seed, steps, cfg, sampler_name, scheduler,
                                pos, neg, lat, denoise)
        ref_crop = vae.decode(refined["samples"])
        ref_crop = tensor_resize(ref_crop, cw, ch)  # back to crop resolution

        # Composite only the face region; the margin stays untouched (mask = 0).
        face_mask = torch.zeros((1, ch, cw, 1), dtype=ref_crop.dtype, device=ref_crop.device)
        fy1c = _clamp(fy1, 0, ch)
        fy2c = _clamp(fy2, 0, ch)
        fx1c = _clamp(fx1, 0, cw)
        fx2c = _clamp(fx2, 0, cw)
        if fx2c > fx1c and fy2c > fy1c:
            face_mask[:, fy1c:fy2c, fx1c:fx2c, :] = 1.0
        face_mask = tensor_gaussian_blur_mask(face_mask, feather).to(ref_crop.device)

        return (y1, y2, x1, x2, ref_crop, face_mask)

    def detect_and_detail(self, model, vae, positive, negative, seed, steps, cfg,
                          sampler_name, scheduler, denoise, guide_size, max_size,
                          crop_factor, detection_method, yolo_model_name,
                          detection_threshold, feather, inpaint_model,
                          vae_tile_size=0,
                          image=None, latent=None,
                          face_positive=None, face_negative=None, unique_id=None):
        with node_span("LLMPromptStudioFaceDetailer", unique_id):
            if latent is not None:
                img = _decode_latent(vae, latent["samples"], vae_tile_size)
            elif image is not None:
                img = image
            else:
                raise ValueError("Face Detailer requires `latent` or `image` input")

            boxes = self._detect(img, detection_method, yolo_model_name,
                                 detection_threshold, vae)
            if not boxes:
                return (img,)

            result = img.clone()
            for i in range(img.shape[0]):
                for (x1f, y1f, x2f, y2f) in boxes:
                    out = self._refine_face(
                        model, vae, img, i, x1f, y1f, x2f, y2f,
                        positive, negative, face_positive, face_negative,
                        seed + i, steps, cfg, sampler_name, scheduler, denoise,
                        guide_size, max_size, crop_factor, feather, inpaint_model)
                    if out is None:
                        continue
                    y1, y2, x1, x2, ref_crop, face_mask = out
                    region = result[i:i + 1, y1:y2, x1:x2, :]
                    result[i:i + 1, y1:y2, x1:x2, :] = tensor_paste(region, ref_crop, face_mask)
            return (result,)
