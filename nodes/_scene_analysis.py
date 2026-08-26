"""Lightweight computer-vision scene/lighting classification for image inputs.

Mirrors the ideas behind ComfyUI-EasyColorCorrector (edge density + saturation +
face cues) but relies only on OpenCV/numpy so it works without scikit-learn or
skimage. It gives the LLM Critic/Writer a cheap, deterministic content hint
(scene type, lighting, face count) without spending a vision-model call.
"""

import logging

import numpy as np

logger = logging.getLogger("llm_prompt_studio")

try:
    import cv2
    _CV2 = True
except Exception:  # pragma: no cover - optional at runtime
    _CV2 = False


def _face_count(rgb):
    try:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))
        return 0 if faces is None else int(len(faces))
    except Exception:
        return 0


def analyze_scene(image):
    """Return ``{"scene_type", "lighting", "faces"}`` for a ``(B,H,W,C)`` tensor.

    The tensor is float 0..1 (ComfyUI image layout). Falls back to neutral values
    when OpenCV is unavailable or analysis fails, so callers can always use it.
    """
    if not _CV2:
        return {"scene_type": "unknown", "lighting": "auto", "faces": 0}
    try:
        arr = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
        img = arr[0] if arr.ndim == 4 else arr
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)

        h, w = img.shape[:2]
        max_side = max(h, w)
        if max_side > 512:
            scale = 512.0 / max_side
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        edge_density = float(np.mean(np.abs(edges))) / 255.0  # normalize to ~0..1

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        avg_sat = float(np.mean(hsv[:, :, 1]))
        sat_std = float(np.std(hsv[:, :, 1]))
        tex_contrast = float(np.std(gray))
        faces = _face_count(img)

        if edge_density < 0.08 and avg_sat > 120 and sat_std > 40:
            scene = "anime"
        elif edge_density < 0.12 and avg_sat > 80 and tex_contrast < 35:
            scene = "stylized_art"
        elif avg_sat > 100 and sat_std > 50 and tex_contrast > 40:
            scene = "concept_art"
        elif edge_density > 0.25 and tex_contrast > 65 and avg_sat > 90:
            scene = "detailed_illustration"
        elif faces > 0 and edge_density < 0.15:
            scene = "portrait"
        elif edge_density > 0.15 and avg_sat < 80:
            scene = "realistic_photo"
        else:
            scene = "general"

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l = lab[:, :, 0]
        brightness = float(np.mean(l))
        contrast = float(np.std(l))
        if brightness < 85:
            lighting = "low_light"
        elif brightness > 170:
            lighting = "bright"
        elif contrast < 20:
            lighting = "flat"
        else:
            lighting = "good"

        return {"scene_type": scene, "lighting": lighting, "faces": faces}
    except Exception as e:  # noqa: BLE001 - never break the node on CV failure
        logger.warning("scene analysis failed: %s", e)
        return {"scene_type": "unknown", "lighting": "auto", "faces": 0}


def describe_scene(scene):
    """One-line, model-friendly summary of :func:`analyze_scene` output.

    Returns an empty string when the classification is inconclusive so callers
    can skip injecting a hint.
    """
    if not scene:
        return ""
    st = scene.get("scene_type")
    if st in (None, "unknown", "general"):
        return ""
    lt = scene.get("lighting")
    faces = int(scene.get("faces", 0) or 0)
    parts = [f"scene_type={st}"]
    if lt and lt != "auto":
        parts.append(f"lighting={lt}")
    if faces > 0:
        parts.append(f"faces={faces}")
    return "CV pre-analysis: " + ", ".join(parts) + "."
