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
    tensor_gaussian_blur_mask, tensor_paste, match_luminance,
)
from .smart_parameters import SAMPLERS_WITH_BASE, SCHEDULERS_WITH_BASE
from ._latent_upscaler import (
    project_local_ultralytics_bbox, project_local_ultralytics_seg,
    project_local_ultralytics_gender,
)
from ..vram import release_before_sample


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _resolve_class_index(value, model):
    """Map a user-supplied class selector (int index OR class name) to an int index.

    ``value`` may be an int (used directly) or a string. Strings are accepted as
    either a decimal index or a class name (case-insensitive match against the
    model's ``names``). Falls back to ``0`` with a warning when nothing matches,
    so a mistyped combo value never silently crashes the node."""
    names = getattr(getattr(model, "model", None), "names", None) or {}
    if isinstance(value, int):
        return value
    sval = str(value).strip()
    try:
        return int(sval)
    except (TypeError, ValueError):
        pass
    lowered = {str(n).lower(): i for i, n in names.items()}
    if sval.lower() in lowered:
        return lowered[sval.lower()]
    logger.warning("[FaceDetailer] class '%s' not found in model names %s; "
                   "defaulting to index 0", value, names)
    return 0


def _bbox_iou(a, b):
    """IoU of two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


# Module-level cache so a YOLO model is loaded once and reused across every image
# in a batch (and across runs), instead of being re-initialised per detection call.
_YOLO_CACHE = {}


def _get_yolo_model(path):
    """Return a cached ``ultralytics.YOLO`` instance for ``path`` (loaded on first use)."""
    model = _YOLO_CACHE.get(path)
    if model is None:
        from ultralytics import YOLO
        model = YOLO(path)
        _YOLO_CACHE[path] = model
    return model


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
                "detection_method": (["haar", "yolo", "yolo_seg", "yolo_bbox_seg"],
                                     {"default": "haar",
                                      "tooltip": "How faces are found and masked. "
                                      "haar/yolo: bbox detection, rectangular/oval mask. "
                                      "yolo_seg: one seg model does detection AND mask "
                                      "(needs a low threshold, weak on 3/4 turns). "
                                      "yolo_bbox_seg: a strong bbox model (yolo_model_name) "
                                      "locates the face; the seg model (yolo_seg_model_name) "
                                      "only provides the mask shape inside that crop."}),
                "yolo_model_name": (
                    ["face_yolov8s.pt"] + project_local_ultralytics_bbox() + list(yolo_list),
                    {"default": "face_yolov8s.pt"}),
                "yolo_seg_model_name": (
                    ["(none)"] + project_local_ultralytics_seg() + list(seg_list),
                    {"default": "(none)",
                     "tooltip": "Segmentation model for `yolo_seg`. Only used when "
                                "detection_method = yolo_seg. The per-detection mask "
                                "(real shape) replaces the rectangular face mask."}),
                "yolo_seg_class": (["face", "hair", "skin"], {"default": "face",
                                  "tooltip": "Класс сег-модели, который считать лицом. Список "
                                             "заполняется автоматически из выбранной модели "
                                             "(yolo_seg_model_name): при выборе модели в комбобоксе "
                                             "появляются реальные имена её классов (напр. face / hair "
                                             "/ skin). Детекции других классов игнорируются — это "
                                             "убирает ложные срабатывания. Если модель не выбрана "
                                             "(none), действует значение по умолчанию 'face'."}),
                "gender_filter": (["any", "female", "male"], {"default": "any",
                                  "tooltip": "Пол лиц для обработки. `any` — все найденные "
                                             "лица; `female` / `male` — обрабатывать только "
                                             "лица выбранного пола (остальные пропускаются). "
                                             "Требует выбранной gender-модели в "
                                             "gender_model_name, иначе фильтр отключается с "
                                             "предупреждением."}),
                "gender_model_name": (
                    ["(none)"] + project_local_ultralytics_gender(),
                    {"default": "(none)",
                     "tooltip": "Ultralytics-модель классификации пола (кроп лица -> класс). "
                                "Поместите .pt в models/ultralytics/gender/. Используется "
                                "только когда gender_filter != any. Класс, соответствующий "
                                "женщине, задаётся в gender_model_female_class."}),
                "gender_model_female_class": (["female", "male"], {"default": "female",
                                  "tooltip": "Класс, который gender-модель выдаёт для женского "
                                             "лица. Список заполняется автоматически из выбранной "
                                             "модели (gender_model_name): при выборе модели в "
                                             "комбобоксе появляются реальные имена её классов "
                                             "(напр. female / male). Лица с другим классом считаются "
                                             "мужскими. Используется при gender_filter != any."}),
                "gender_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                  "tooltip": "Минимальная уверенность классификатора пола. "
                                             "Лица, для которых gender-модель выдаёт "
                                             "уверенность ниже порога, считаются 'unknown' и "
                                             "НЕ отбрасываются гендер-фильтром (сохраняются). "
                                             "Работает только при gender_filter != any."}),
                "detection_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                  "tooltip": "Порог уверенности детекции bbox-моделью (режимы haar / "
                                             "yolo / yolo_bbox_seg). Для yolo_bbox_seg это порог "
                                             "нахождения прямоугольника лица; сег-модель его не "
                                             "использует (см. seg_threshold)."}),
                "seg_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                                  "tooltip": "Порог уверенности сег-модели. Используется, когда "
                                             "сег-модель применяется: в режиме yolo_seg — для "
                                             "детекции лица, в режиме yolo_bbox_seg — только для "
                                             "получения формы маски внутри уже найденного bbox-"
                                             "моделью прямоугольника. Отделён от detection_threshold, "
                                             "потому что сег-модели выдают лица с гораздо меньшей "
                                             "уверенностью (~0.01-0.1), чем bbox-модели (~0.5-0.95). "
                                             "Низкий порог даёт больше масок, но и больше ложных "
                                             "срабатываний."}),
                "drop_size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1,
                             "tooltip": "Минимальный размер лица (px) по короткой стороне. "
                                        "Лица меньше этого размера пропускаются и не "
                                        "обрабатываются (полезно, чтобы не тратить время на "
                                        "крошечные/фоновые лица). 0 = не отсекать никакие "
                                        "лица (обрабатывать все найденные)."}),
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
    def _detect(self, img, method, yolo_name, yolo_seg_name, threshold, seg_class=0,
                seg_threshold=0.1):
        if method == "haar":
            return self._detect_haar(img)
        if method == "yolo_seg":
            return self._detect_yolo_seg(img, yolo_seg_name, seg_threshold, seg_class)
        if method == "yolo_bbox_seg":
            return self._detect_yolo_bbox_seg(img, yolo_name, yolo_seg_name, threshold,
                                              seg_threshold, seg_class)
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
            from ultralytics import YOLO  # noqa: F401  (ensures import error surfaces early)
        except Exception as e:
            raise RuntimeError(
                "YOLO detection requires the optional `ultralytics` package") from e
        path = self._resolve_yolo(yolo_name)
        if not path:
            raise FileNotFoundError("YOLO model not found: %s" % yolo_name)
        model = _get_yolo_model(path)
        arr = (img[0].clamp(0.0, 1.0).cpu().numpy() * 255.0).astype("uint8")
        # Forward `conf=threshold`: ultralytics applies its own default conf=0.25
        # otherwise, which would silently drop every detection below 0.25 before our
        # `detection_threshold` is ever evaluated (making the slider ineffective for
        # weak faces). Passing it makes the node's threshold authoritative.
        res = model(arr, verbose=False, conf=threshold)[0]
        boxes = []
        dropped_low = 0
        for box in res.boxes:
            conf = float(box.conf[0])
            if conf < threshold:
                dropped_low += 1
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append((int(x1), int(y1), int(x2), int(y2), None))
        if dropped_low:
            logger.info("[FaceDetailer] yolo detection: kept %d face(s), dropped %d "
                        "below confidence threshold (threshold=%.2f)", len(boxes),
                        dropped_low, threshold)
        return boxes

    def _detect_yolo_seg(self, img, yolo_name, threshold, seg_class=0):
        try:
            from ultralytics import YOLO  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "YOLO segmentation requires the optional `ultralytics` package") from e
        path = self._resolve_yolo_seg(yolo_name)
        if not path:
            raise FileNotFoundError("YOLO seg model not found: %s" % yolo_name)
        model = _get_yolo_model(path)
        # Resolve the (now name-based) class selector against this model's labels.
        seg_class = _resolve_class_index(seg_class, model)
        arr = (img[0].clamp(0.0, 1.0).cpu().numpy() * 255.0).astype("uint8")
        # Forward `conf=threshold` (see _detect_yolo for why this matters).
        res = model(arr, verbose=False, retina_masks=True, conf=threshold)[0]
        H, W = arr.shape[0], arr.shape[1]
        masks = getattr(res, "masks", None)
        boxes = []
        dropped_class = 0
        dropped_low = 0
        for idx, box in enumerate(res.boxes):
            # Only accept the class the user declared as "face". Without this filter a
            # generic seg model emits person/car/... masks too, which previously forced
            # the threshold down to 0.1-0.2 and produced false positives.
            if seg_class is not None and int(box.cls[0]) != int(seg_class):
                dropped_class += 1
                continue
            conf = float(box.conf[0])
            if conf < threshold:
                dropped_low += 1
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            seg = None
            if masks is not None and getattr(masks, "data", None) is not None \
                    and idx < len(masks.data):
                seg = (masks.data[idx] > 0.5).to(torch.float32)  # [H, W], 0/1
                seg = seg.to(torch.device("cpu"))
            boxes.append((int(x1), int(y1), int(x2), int(y2), seg))
        if dropped_class or dropped_low:
            logger.info("[FaceDetailer] yolo_seg detection: kept %d face(s); dropped %d "
                        "(%d wrong class != %s, %d below confidence threshold %.2f)",
                        len(boxes), dropped_class + dropped_low, dropped_class,
                        seg_class, dropped_low, threshold)
        return boxes

    def _detect_yolo_bbox_seg(self, img, yolo_name, yolo_seg_name, threshold, seg_threshold=0.1,
                              seg_class=0):
        """Locate faces with a strong bbox model, shape the mask with a seg model.

        The bbox model (user-chosen ``yolo_model_name``) detects faces at the normal
        ``threshold`` and defines the crop rectangle (margin from ``crop_factor``). For
        each detected face the seg model (``yolo_seg_model_name``) is queried at
        ``seg_threshold`` so its mask is used only as the shape inside that rectangle; if
        no seg mask overlaps the box, the face falls back to the rectangular/oval mask.
        """
        bbox_boxes = self._detect_yolo(img, yolo_name, threshold)
        if not bbox_boxes:
            return []
        seg_dets = []
        if yolo_seg_name and yolo_seg_name != "(none)":
            try:
                # Low conf: we only want the mask shape; the bbox already located the face.
                seg_dets = self._detect_yolo_seg(img, yolo_seg_name, seg_threshold, seg_class)
            except Exception as e:
                logger.warning("[FaceDetailer] yolo_bbox_seg: seg mask unavailable (%s); "
                              "falling back to rectangle mask", e)
        out = []
        for (x1, y1, x2, y2, _) in bbox_boxes:
            best_mask = None
            best_iou = 0.0
            for (sx1, sy1, sx2, sy2, smask) in seg_dets:
                if smask is None:
                    continue
                iou = _bbox_iou((x1, y1, x2, y2), (sx1, sy1, sx2, sy2))
                if iou > best_iou:
                    best_iou = iou
                    best_mask = smask
            out.append((x1, y1, x2, y2, best_mask))
        return out

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

    def _resolve_yolo_gender(self, name):
        from ._latent_upscaler import PROJECT_ROOT
        if not name or name == "(none)":
            return None
        local = os.path.join(PROJECT_ROOT, "models", "ultralytics", "gender", name)
        if os.path.isfile(local):
            return local
        try:
            import folder_paths
            for cat in ("ultralytics_gender", "ultralytics"):
                p = folder_paths.get_full_path(cat, name)
                if p and os.path.isfile(p):
                    return p
        except Exception:
            pass
        return None

    @staticmethod
    def _classify_gender(crop, gender_model, female_class):
        """Return ``(cls, conf)`` for a cropped face, or ``(None, 0.0)``.

        Supports ultralytics classification models (``res.probs``) and, as a
        fallback, detection models (highest-confidence box class). ``female_class`` is
        the index the model uses for the female class. ``conf`` is the classifier's
        confidence in its top prediction (used by the gender threshold).
        """
        try:
            res = gender_model(crop, verbose=False)[0]
        except Exception as e:  # transient inference error -> keep the face
            logger.warning("[FaceDetailer] gender classification error: %s — face kept", e)
            return (None, 0.0)
        probs = getattr(res, "probs", None)
        if probs is not None:
            top1 = int(probs.top1)
            return (top1, float(probs.data[top1]))
        boxes = getattr(res, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            return (int(boxes.cls[0]), float(boxes.conf[0]))
        return (None, 0.0)

    def _apply_gender_filter(self, boxes, arr, gender_filter,
                             gender_model_name, gender_female_class,
                             gender_threshold=0.5):
        """Classify each face's gender and drop faces whose gender does not match
        ``gender_filter``. A prediction whose confidence is below ``gender_threshold``
        is treated as "unknown" and kept (never dropped) so an uncertain classifier
        cannot wrongly discard faces. Gender counts are logged whenever a gender model
        is selected."""
        if not gender_model_name or gender_model_name == "(none)":
            return boxes
        path = self._resolve_yolo_gender(gender_model_name)
        if not path:
            logger.warning("[FaceDetailer] gender model not found: %s — gender detection "
                           "skipped (all faces kept)", gender_model_name)
            return boxes
        model = _get_yolo_model(path)
        # Resolve the (now name-based) class selector against this model's labels.
        gender_female_class = _resolve_class_index(gender_female_class, model)
        H, W = arr.shape[0], arr.shape[1]
        want_female = (gender_filter == "female")
        active = gender_filter not in (None, "any")
        kept = []
        n_female = 0
        n_male = 0
        n_unknown = 0
        n_dropped = 0
        for (x1, y1, x2, y2, seg) in boxes:
            cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            cls, conf = self._classify_gender(arr[cy1:cy2, cx1:cx2], model, gender_female_class)
            if cls is None:
                n_unknown += 1
                kept.append((x1, y1, x2, y2, seg))  # unknown -> keep
                continue
            is_female = (int(cls) == int(gender_female_class))
            if not active:
                # No filtering requested: keep everything, just tally the stats.
                if is_female:
                    n_female += 1
                else:
                    n_male += 1
                kept.append((x1, y1, x2, y2, seg))
                continue
            # Active filter: a low-confidence prediction is kept as "unknown" rather
            # than risk a wrong drop.
            if conf < float(gender_threshold):
                n_unknown += 1
                kept.append((x1, y1, x2, y2, seg))
                logger.info("[FaceDetailer] gender filter '%s': keeping face as unknown — "
                            "predicted %s (class %s) conf %.2f < gender_threshold %.2f",
                            gender_filter, "female" if is_female else "male", cls,
                            conf, float(gender_threshold))
                continue
            if is_female:
                n_female += 1
            else:
                n_male += 1
            if is_female == want_female:
                kept.append((x1, y1, x2, y2, seg))
            else:
                n_dropped += 1
                logger.info("[FaceDetailer] gender filter '%s': dropping face — predicted "
                            "class %s (%s, conf %.2f), wanted %s", gender_filter, cls,
                            "female" if is_female else "male", conf, gender_filter)
        total = n_female + n_male + n_unknown
        if active:
            logger.info("[FaceDetailer] gender detection: %d female, %d male, %d unknown "
                        "of %d face(s); filter '%s' dropped %d",
                        n_female, n_male, n_unknown, total, gender_filter, n_dropped)
        else:
            logger.info("[FaceDetailer] gender detection: %d female, %d male, %d unknown "
                        "of %d face(s)", n_female, n_male, n_unknown, total)
        return kept

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
        short = min(fw, fh)
        upscale = guide_size / float(short)
        if upscale <= 1.0:
            logger.info("[FaceDetailer] face skipped: short side %d px already >= "
                        "guide_size (%d) — upscaling not needed", short, guide_size)
            return None
        logger.info("[FaceDetailer] processing face: short side %d px -> "
                    "upscaling x%.2f to guide_size %dpx", short, upscale, guide_size)

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
            logger.info("[FaceDetailer] crop clamped to max_size (%d px): upscale factor "
                        "reduced to x%.2f", max_size, upscale * s)
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
                            inpaint_model, seg_threshold=0.1, vae_tile_size=0, yolo_seg_class="face",
                            drop_size=0, gender_filter="any",
                            gender_model_name="(none)", gender_model_female_class="female",
                           gender_threshold=0.5,
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
            total = img.shape[0]
            logger.info("[FaceDetailer] start: frames=%d, method=%s, guide_size=%d, "
                        "max_size=%d, drop_size=%d, yolo_seg_class=%s",
                        total, detection_method, guide_size, max_size, drop_size, yolo_seg_class)
            for i in range(img.shape[0]):
                # Detect faces per image in the batch: the seg masks returned by
                # _detect_yolo_seg are sized to the image passed in, so detection must
                # run on the individual crop `img[i:i+1]` to keep mask coordinates
                # aligned to that frame (previously detection ran once on the whole
                # batch and reused the same boxes for every image).
                arr = (img[i].clamp(0.0, 1.0).cpu().numpy() * 255.0).astype("uint8")
                boxes = self._detect(img[i:i + 1], detection_method, yolo_model_name,
                                     yolo_seg_model_name, detection_threshold,
                                     seg_class=yolo_seg_class, seg_threshold=seg_threshold)
                if not boxes:
                    continue
                boxes = self._apply_gender_filter(
                    boxes, arr, gender_filter, gender_model_name,
                    gender_female_class=gender_model_female_class,
                    gender_threshold=gender_threshold)
                if not boxes:
                    continue
                for (x1f, y1f, x2f, y2f, seg_mask) in boxes:
                    fw = x2f - x1f
                    fh = y2f - y1f
                    short = min(fw, fh)
                    logger.info("[FaceDetailer] face detected: size %dx%d px "
                                "(short side %d px)%s", fw, fh, short,
                                ", seg mask" if seg_mask is not None else "")
                    if drop_size and short < int(drop_size):
                        logger.info("[FaceDetailer] face skipped: short side %d px "
                                    "< drop_size (%d) — too small", short, drop_size)
                        continue
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
                    # Match the refined face's brightness/contrast to the surrounding
                    # original (within the feathered mask) so the inpaint does not leave
                    # a brighter seam. The margin is masked out at composite time, so
                    # adjusting the whole crop by masked-region stats is safe.
                    ref_crop = match_luminance(ref_crop, region, face_mask)
                    result[i:i + 1, y1:y2, x1:x2, :] = tensor_paste(region, ref_crop, face_mask)
            logger.info("[FaceDetailer] done: processed %d face(s) across %d frame(s)",
                        face_idx, total)
            return (result,)
