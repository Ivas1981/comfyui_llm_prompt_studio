"""LLM Prompt Studio Face Detailer.

Refines detected faces. Accepts either a ``latent`` (optimized path - one VAE
decode for detection, crops straight from the latent, no per-face VAEEncode) or
an ``image`` (legacy pixel path). Detection: haar (cv2) or yolo (ultralytics,
optional).
"""

import os

import torch

from ._ksample import sample_latent, node_span
from ._imgutils import tensor_resize, to_latent_image, tensor_gaussian_blur_mask, tensor_paste
from .smart_parameters import SAMPLERS_WITH_BASE, SCHEDULERS_WITH_BASE
from ._latent_upscaler import project_local_ultralytics_bbox


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


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
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "guide_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "max_size": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "detection_method": (["haar", "yolo"], {"default": "haar"}),
                "yolo_model_name": (
                    ["face_yolov8s.pt"] + project_local_ultralytics_bbox() + list(yolo_list),
                    {"default": "face_yolov8s.pt"}),
                "detection_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                "inpaint_model": ("BOOLEAN", {"default": False}),
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

    def _refine_face(self, model, vae, lat_mode, latent, img, i, x1, y1, x2, y2,
                     positive, negative, face_positive, face_negative,
                     seed, steps, cfg, sampler_name, scheduler, denoise,
                     guide_size, max_size, feather, inpaint_model):
        bh = y2 - y1
        bw = x2 - x1
        upscale = guide_size / float(min(bw, bh))
        if upscale <= 1.0:
            return None
        pos = face_positive if face_positive is not None else positive
        neg = face_negative if face_negative is not None else negative

        if lat_mode:
            ly1, ly2 = y1 // 8, y2 // 8
            lx1, lx2 = x1 // 8, x2 // 8
            lc = latent["samples"][i:i + 1, :, ly1:ly2, lx1:lx2]
            nlw = _clamp(round((lx2 - lx1) * upscale), 1, max_size // 8)
            nlh = _clamp(round((ly2 - ly1) * upscale), 1, max_size // 8)
            up_lat = torch.nn.functional.interpolate(
                lc, size=(nlh, nlw), mode="bilinear", align_corners=False)
            up_pixels = vae.decode(up_lat)[0]
            if inpaint_model:
                inpaint_mask = torch.ones_like(up_pixels)[:, :, :, :1]
                pos, neg, lat = self._inpaint_encode(pos, neg, up_pixels, vae, inpaint_mask)
            else:
                lat = {"samples": up_lat}
        else:
            crop = img[i:i + 1, y1:y2, x1:x2, :]
            nw = _clamp(round(bw * upscale), 1, max_size)
            nh = _clamp(round(bh * upscale), 1, max_size)
            up = tensor_resize(crop, nw, nh)
            if inpaint_model:
                inpaint_mask = torch.ones_like(up)[:, :, :, :1]
                pos, neg, lat = self._inpaint_encode(pos, neg, up, vae, inpaint_mask)
            else:
                lat = to_latent_image(up, vae)

        refined = sample_latent(model, seed, steps, cfg, sampler_name, scheduler,
                                 pos, neg, lat, denoise)
        ref = vae.decode(refined["samples"])[0]
        ref = tensor_resize(ref, bw, bh)
        composite_mask = tensor_gaussian_blur_mask(torch.ones((1, bh, bw, 1)), feather)
        return ref, composite_mask

    def detect_and_detail(self, model, vae, positive, negative, seed, steps, cfg,
                          sampler_name, scheduler, denoise, guide_size, max_size,
                          detection_method, yolo_model_name, detection_threshold,
                          feather, inpaint_model, image=None, latent=None,
                          face_positive=None, face_negative=None, unique_id=None):
        with node_span("LLMPromptStudioFaceDetailer", unique_id):
            if latent is not None:
                img = vae.decode(latent["samples"])
            elif image is not None:
                img = image
            else:
                raise ValueError("Face Detailer requires `latent` or `image` input")
            lat_mode = latent is not None

            boxes = self._detect(img, detection_method, yolo_model_name,
                                 detection_threshold, vae)
            if not boxes:
                return (img,)

            result = img.clone()
            for i in range(img.shape[0]):
                for (x1, y1, x2, y2) in boxes:
                    out = self._refine_face(
                        model, vae, lat_mode, latent, img, i, x1, y1, x2, y2,
                        positive, negative, face_positive, face_negative,
                        seed + i, steps, cfg, sampler_name, scheduler, denoise,
                        guide_size, max_size, feather, inpaint_model)
                    if out is None:
                        continue
                    ref, mask = out
                    region = result[i:i + 1, y1:y2, x1:x2, :]
                    result[i:i + 1, y1:y2, x1:x2, :] = tensor_paste(region, ref, mask)
            return (result,)

