"""LLM Prompt Studio Face Detailer.

Refines detected faces *in place* (the way Impact-Pack's FaceDetailer does it):
it crops a region a little larger than the detected face (context margin),
upscales it, then re-samples only the face area while preserving the
surrounding context. The original face is used as the denoising base (via a
face mask / inpaint mask), so the result *refines* the face instead of
replacing it with a freshly generated one that is then pasted on top.

Detection: haar (cv2) or yolo (ultralytics, optional).
"""

import logging
import os

import torch

logger = logging.getLogger("llm_prompt_studio")

from ._ksample import sample_latent, node_span
from ._imgutils import (
    tensor_resize, mask_resize, to_latent_image,
    tensor_gaussian_blur_mask, tensor_paste,
)
from .smart_parameters import SAMPLERS_WITH_BASE, SCHEDULERS_WITH_BASE
from ._latent_upscaler import (
    project_local_ultralytics_bbox, project_local_ultralytics_seg,
)
from ..vram import release_before_sample


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
        try:
            import folder_paths
            seg_list = folder_paths.get_filename_list("ultralytics_seg") or []
        except Exception:
            seg_list = []
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                "control_after_generate": True,
                                "tooltip": "Randomize / increment / decrement / fixed after "
                                           "each Generate (the control is shown next to the field). "
                                           "Per-detected-face the seed is incremented."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (SAMPLERS_WITH_BASE, {"default": "dpmpp_2m"}),
                "scheduler": (SCHEDULERS_WITH_BASE, {"default": "karras"}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                              "tooltip": "How much the face is changed. Lower = closer to the "
                                         "original face (refine), higher = more regeneration."}),
                "guide_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8,
                              "tooltip": "Minimum size (px) of the shorter face side. Faces "
                                         "smaller than this are upscaled so their shorter side "
                                         "reaches guide_size before refinement; faces already "
                                         "larger are skipped."}),
                "max_size": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8,
                             "tooltip": "Max size of the upscaled crop (used to clamp very large faces)."}),
                "crop_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05,
                                 "tooltip": "How much margin to include around the detected face for "
                                            "context. >1.0 keeps surrounding pixels so the refined "
                                            "face blends seamlessly with the rest of the image."}),
                "detection_method": (["haar", "yolo", "yolo_seg"], {"default": "haar"}),
                "yolo_model_name": (
                    ["face_yolov8s.pt"] + project_local_ultralytics_bbox() + list(yolo_list),
                    {"default": "face_yolov8s.pt"}),
                "yolo_seg_model_name": (
                    ["(none)"] + project_local_ultralytics_seg() + list(seg_list),
                    {"default": "(none)",
                     "tooltip": "Segmentation model for `yolo_seg`. Only used when "
                                "detection_method = yolo_seg. The per-detection mask "
                                "(real shape) replaces the rectangular face mask."}),
                "detection_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1,
                          "tooltip": "Feather (blur) radius, in px, of the face mask used when "
                                     "compositing the refined face back into the image."}),
                "mask_shape": (["square", "oval"], {"default": "square",
                              "tooltip": "Shape of the inpaint mask used for bbox detection "
                                         "(haar / yolo). `square` = rectangle, `oval` = ellipse "
                                         "inscribed in the detected face. Ignored for yolo_seg "
                                         "(the seg mask defines the shape)."}),
                "bbox_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05,
                               "tooltip": "Scale the detected face bbox around its center before "
                                          "refining. >1.0 captures more (e.g. jaw/neck), <1.0 keeps "
                                          "only the face center. 1.0 = exact detection."}),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1,
                               "tooltip": "Number of refinement passes over the upscaled crop. 1 = "
                                          "single pass (default). >1 re-samples the result of the "
                                          "previous pass for further refinement."}),
                "inpaint_model": ("BOOLEAN", {"default": False,
                                  "tooltip": "Use ComfyUI's InpaintModelConditioning so the model "
                                             "sees the original face structure (concat latent) while "
                                             "regenerating the face area. Off = differential noise "
                                             "mask (refines in place)."}),
            },
            "optional": {
                "image": ("IMAGE", {"forceInput": True}),
                "latent": ("LATENT", {"forceInput": True}),
                "face_positive": ("CONDITIONING", {"forceInput": True}),
                "face_negative": ("CONDITIONING", {"forceInput": True}),
                "vae_tile_size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16,
                                  "tooltip": "Tile size (px) for the VAE decode of the incoming "
                                             "latent. 0 = decode the whole frame at once (best "
                                             "quality). A positive value (e.g. 512) tiles the decode "
                                             "to cap VRAM on large hires latents. Can be wired from "
                                             "KSamplerHiresFix's VAE_TILE_SIZE output."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    # -- detection ---------------------------------------------------------
    def _detect(self, img, method, yolo_name, yolo_seg_name, threshold):
        if method == "haar":
            return self._detect_haar(img)
        if method == "yolo_seg":
            return self._detect_yolo_seg(img, yolo_seg_name, threshold)
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
            boxes.append((int(x), int(y), int(x + bw), int(y + bh), None))
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
            boxes.append((int(x1), int(y1), int(x2), int(y2), None))
        return boxes

    def _detect_yolo_seg(self, img, yolo_name, threshold):
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError(
                "YOLO segmentation requires the optional `ultralytics` package") from e
        path = self._resolve_yolo_seg(yolo_name)
        if not path:
            raise FileNotFoundError("YOLO seg model not found: %s" % yolo_name)
        model = YOLO(path)
        arr = (img[0].clamp(0.0, 1.0).cpu().numpy() * 255.0).astype("uint8")
        res = model(arr, verbose=False, retina_masks=True)[0]
        H, W = arr.shape[0], arr.shape[1]
        masks = getattr(res, "masks", None)
        boxes = []
        for idx, box in enumerate(res.boxes):
            if float(box.conf[0]) < threshold:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            seg = None
            if masks is not None and getattr(masks, "data", None) is not None \
                    and idx < len(masks.data):
                seg = (masks.data[idx] > 0.5).to(torch.float32)  # [H, W], 0/1
                seg = seg.to(torch.device("cpu"))
            boxes.append((int(x1), int(y1), int(x2), int(y2), seg))
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

    def _resolve_yolo_seg(self, name):
        from ._latent_upscaler import PROJECT_ROOT
        if name == "(none)":
            return None
        local = os.path.join(PROJECT_ROOT, "models", "ultralytics", "seg", name)
        if os.path.isfile(local):
            return local
        try:
            import folder_paths
            for cat in ("ultralytics_seg", "ultralytics"):
                p = folder_paths.get_full_path(cat, name)
                if p and os.path.isfile(p):
                    return p
        except Exception:
            pass
        return None

    # -- refinement --------------------------------------------------------
    @staticmethod
    def _build_shape_mask(mask_shape, h, w, r1, r2, c1, c2):
        """3D ``[1, h, w]`` mask filling ``rect (r1:r2, c1:c2)``.

        ``square`` fills the whole rectangle; ``oval`` inscribes an ellipse in it.
        Returns a float mask (values 0/1) on CPU; caller moves device/dtype.
        """
        m = torch.zeros((1, h, w), dtype=torch.float32)
        if not (c2 > c1 and r2 > r1):
            return m
        if mask_shape == "oval":
            cy = (r1 + r2) / 2.0
            cx = (c1 + c2) / 2.0
            ry = (r2 - r1) / 2.0
            rx = (c2 - c1) / 2.0
            yy, xx = torch.meshgrid(
                torch.arange(h, dtype=torch.float32),
                torch.arange(w, dtype=torch.float32), indexing="ij")
            m[0, ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0] = 1.0
        else:
            m[:, r1:r2, c1:c2] = 1.0
        return m

    @staticmethod
    def _inpaint_encode(pos, neg, pixels, vae, mask):
        from nodes import InpaintModelConditioning
        return InpaintModelConditioning().encode(pos, neg, pixels, vae, mask, noise_mask=True)

    def _refine_face(self, model, vae, img, i, x1f, y1f, x2f, y2f,
                     positive, negative, face_positive, face_negative,
                     seed, steps, cfg, sampler_name, scheduler, denoise,
                     guide_size, max_size, crop_factor, feather, inpaint_model,
                     seg_mask=None, mask_shape="square", bbox_scale=1.0,
                     iterations=1, vae_tile_size=0):
        # Optional scaling of the detected face bbox around its center. >1 captures
        # more of the region (jaw/neck), <1 keeps only the face center. 1.0 = exact.
        if bbox_scale is not None and float(bbox_scale) != 1.0:
            cx, cy = (x1f + x2f) / 2.0, (y1f + y2f) / 2.0
            sf = float(bbox_scale)
            hw = (x2f - x1f) / 2.0 * sf
            hh = (y2f - y1f) / 2.0 * sf
            x1f = int(round(cx - hw))
            x2f = int(round(cx + hw))
            y1f = int(round(cy - hh))
            y2f = int(round(cy + hh))
        fh = y2f - y1f
        fw = x2f - x1f
        upscale = guide_size / float(min(fw, fh))
        if upscale <= 1.0:
            logger.info("Face skipped: shorter side (%d) already >= guide_size (%d)",
                        min(fw, fh), guide_size)
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
        # When a seg mask is supplied, the *true shape* drives refinement instead
        # of a rectangle; the bbox still defines the crop geometry / context.
        seg_crop = None
        if seg_mask is not None:
            # seg_mask: 2D [H_img, W_img] (0/1) aligned to the full input image.
            seg_crop = seg_mask[y1:y2, x1:x2].to(torch.float32)
            mask = mask_resize(seg_crop, new_w, new_h)        # 3D [1, new_h, new_w]
            mask = mask.to(device=up_crop.device, dtype=up_crop.dtype)
            if mask.sum() <= 0:
                return None                                   # empty shape -> skip
        else:
            sx, sy = new_w / float(cw), new_h / float(ch)
            mx1 = _clamp(int(round(fx1 * sx)), 0, new_w)
            my1 = _clamp(int(round(fy1 * sy)), 0, new_h)
            mx2 = _clamp(int(round(fx2 * sx)), 0, new_w)
            my2 = _clamp(int(round(fy2 * sy)), 0, new_h)
            mask = self._build_shape_mask(
                mask_shape, new_h, new_w, my1, my2, mx1, mx2
            ).to(device=up_crop.device, dtype=up_crop.dtype)

        if inpaint_model:
            # InpaintModelConditioning keeps the (grayed) original as concat context
            # and only regenerates within the face mask.
            pos, neg, lat = self._inpaint_encode(pos, neg, up_crop, vae, mask)
        else:
            # Differential noise mask: the VAE-encoded upscaled crop is the base,
            # and sampling only touches the face area (margin is preserved).
            lat = to_latent_image(up_crop, vae)
            lat["noise_mask"] = mask

        # Iterations: re-sample the refined result of the previous pass (the first
        # pass keeps the original seed, subsequent passes vary it). For inpaint the
        # concat context is only used on the first pass; later passes refine in place.
        samples = lat["samples"]
        n_iter = max(1, int(iterations))
        for k in range(n_iter):
            lat_k = dict(lat)
            lat_k["samples"] = samples
            refined = sample_latent(model, seed + k, steps, cfg, sampler_name,
                                    scheduler, pos, neg, lat_k, denoise)
            samples = refined["samples"]
            ref_crop = _decode_latent(vae, samples, vae_tile_size)
        ref_crop = tensor_resize(ref_crop, cw, ch)  # back to crop resolution

        # Composite only the face region; the margin stays untouched (mask = 0).
        if seg_mask is not None and seg_crop is not None:
            # Paste back only the true shape; the margin and the area inside the
            # bbox but outside the shape stay untouched.
            cm = mask_resize(seg_crop, cw, ch)                    # 3D [1, ch, cw]
            face_mask = cm.unsqueeze(-1).to(device=ref_crop.device, dtype=ref_crop.dtype)
        else:
            fy1c = _clamp(fy1, 0, ch)
            fy2c = _clamp(fy2, 0, ch)
            fx1c = _clamp(fx1, 0, cw)
            fx2c = _clamp(fx2, 0, cw)
            fm = self._build_shape_mask(
                mask_shape, ch, cw, fy1c, fy2c, fx1c, fx2c
            ).unsqueeze(-1).to(device=ref_crop.device, dtype=ref_crop.dtype)
            face_mask = fm
        face_mask = tensor_gaussian_blur_mask(face_mask, feather).to(ref_crop.device)

        return (y1, y2, x1, x2, ref_crop, face_mask)

    def detect_and_detail(self, model, vae, positive, negative, seed, steps, cfg,
                           sampler_name, scheduler, denoise, guide_size, max_size,
                           crop_factor, detection_method, yolo_model_name,
                           yolo_seg_model_name, detection_threshold, feather,
                           mask_shape, bbox_scale, iterations,
                           inpaint_model, vae_tile_size=0,
                           image=None, latent=None,
                           face_positive=None, face_negative=None, unique_id=None):
        with node_span("LLMPromptStudioFaceDetailer", unique_id):
            release_before_sample()
            if latent is not None:
                img = _decode_latent(vae, latent["samples"], vae_tile_size)
            elif image is not None:
                img = image
            else:
                raise ValueError("Face Detailer requires `latent` or `image` input")

            result = img.clone()
            face_idx = 0
            for i in range(img.shape[0]):
                # Detect faces per image in the batch: the seg masks returned by
                # _detect_yolo_seg are sized to the image passed in, so detection must
                # run on the individual crop `img[i:i+1]` to keep mask coordinates
                # aligned to that frame (previously detection ran once on the whole
                # batch and reused the same boxes for every image).
                boxes = self._detect(img[i:i + 1], detection_method, yolo_model_name,
                                     yolo_seg_model_name, detection_threshold)
                if not boxes:
                    continue
                for (x1f, y1f, x2f, y2f, seg_mask) in boxes:
                    out = self._refine_face(
                        model, vae, img, i, x1f, y1f, x2f, y2f,
                        positive, negative, face_positive, face_negative,
                        seed + face_idx, steps, cfg, sampler_name, scheduler, denoise,
                        guide_size, max_size, crop_factor, feather, inpaint_model,
                        seg_mask=seg_mask, mask_shape=mask_shape,
                        bbox_scale=bbox_scale, iterations=iterations,
                        vae_tile_size=vae_tile_size)
                    face_idx += 1
                    if out is None:
                        continue
                    y1, y2, x1, x2, ref_crop, face_mask = out
                    region = result[i:i + 1, y1:y2, x1:x2, :]
                    result[i:i + 1, y1:y2, x1:x2, :] = tensor_paste(region, ref_crop, face_mask)
            return (result,)
