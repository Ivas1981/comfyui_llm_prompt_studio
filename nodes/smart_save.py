import os
import re
import logging

import folder_paths
import numpy as np
from PIL import Image

from ..library import resolve_library_path, safe_path_in_output, save_prompt_to_library
from ..model_meta import collect_generation_meta
from ..debug import node_span

logger = logging.getLogger("llm_prompt_studio")


def _slug_part(name, max_len=40):
    base = os.path.splitext(os.path.basename(str(name)))[0]
    # Unicode-aware: keep non-ASCII letters/digits (e.g. Cyrillic checkpoint names) instead
    # of stripping them; replace any run of non-word chars with a single underscore.
    base = re.sub(r"[^\w]+", "_", base, flags=re.UNICODE).strip("_")
    return base[:max_len].rstrip("_")


class LLMPromptStudioSmartSave:
    """JPEG q95 + optional EXIF (prompts, face prompts, model, LoRA, params) + preview.
    Filename: [prefix_]checkpoint[_lora1_lora2...]_NNNNN.jpg. Prompt to library via button.
    Set `save_metadata_to_exif=False` to write JPEGs without metadata (smaller files,
    better privacy)."""
    CATEGORY = "LLM Prompt Studio"
    FUNCTION = "save"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "approved": ("BOOLEAN", {"default": False}),
                "filename_prefix": ("STRING", {"default": ""}),
                "save_dir": ("STRING", {"default": ""}),
                "jpeg_quality": ("INT", {"default": 95, "min": 1, "max": 100}),
                "auto_save_to_library": ("BOOLEAN", {"default": False}),
                "library_path": ("STRING", {"default": "llm_prompt_studio_library.json"}),
                "save_metadata_to_exif": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "positive": ("STRING", {"forceInput": True}),
                "negative": ("STRING", {"forceInput": True}),
                "scene_name": ("STRING", {"forceInput": True}),
                "face_positive": ("STRING", {"forceInput": True}),
                "face_negative": ("STRING", {"forceInput": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"},
        }

    def save(self, image, approved, filename_prefix, save_dir, jpeg_quality,
              auto_save_to_library, library_path, save_metadata_to_exif=True,
              positive="", negative="", scene_name="",
              face_positive="", face_negative="",
              unique_id=None, prompt=None):
        with node_span("Smart Save", unique_id,
                       {"approved": approved, "filename_prefix": filename_prefix,
                        "auto_save_to_library": auto_save_to_library}):
            if not approved:
                return {"ui": {"saved": [False]}}

            out_dir = folder_paths.get_output_directory()
            if save_dir.strip():
                folder = safe_path_in_output(save_dir.strip())
            else:
                folder = out_dir
            try:
                os.makedirs(folder, exist_ok=True)
            except OSError as e:
                raise RuntimeError(f"Could not create save folder {folder}: {e}")

            ckpt, loras, gen = "", [], {}
            if isinstance(prompt, dict) and unique_id is not None:
                ckpt, loras, gen = collect_generation_meta(prompt, unique_id)

            lines = []
            if positive:
                lines.append(f"Positive prompt: {positive}")
            if negative:
                lines.append(f"Negative prompt: {negative}")
            if face_positive:
                lines.append(f"Face positive: {face_positive}")
            if face_negative:
                lines.append(f"Face negative: {face_negative}")
            if ckpt:
                lines.append(f"Model: {ckpt}")
            if loras:
                lines.append("LoRA: " + "; ".join(loras))
            if gen:
                lines.append("Params: " + ", ".join(f"{k}={v}" for k, v in gen.items()))
            exif = None
            if save_metadata_to_exif and lines:
                text = "\n".join(lines)
                exif = Image.Exif()
                exif[0x010E] = text.encode("ascii", "replace").decode("ascii")
                exif[0x9286] = b"UNICODE\0" + text.encode("utf-16-be")

            parts = []
            if filename_prefix.strip():
                parts.append(_slug_part(filename_prefix))
            if ckpt:
                parts.append(_slug_part(ckpt))
            for lora in loras:
                parts.append(_slug_part(lora))
            base = "_".join(p for p in parts if p) or "llm_prompt_studio"

            results = []
            images_ui = []
            for t in image:
                arr = (t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(arr).convert("RGB")
                path = self._next_path(folder, base)
                save_kwargs = {"quality": jpeg_quality}
                if exif is not None:
                    try:
                        save_kwargs["exif"] = exif.tobytes()
                    except Exception as e:  # noqa: BLE001 — a malformed EXIF/UserComment
                        # must not abort the whole batch; save without metadata instead.
                        logger.warning("Smart Save could not serialize EXIF for %s: %s", path, e)
                        save_kwargs.pop("exif", None)
                try:
                    img.save(path, "JPEG", **save_kwargs)
                except Exception as e:  # noqa: BLE001 — keep partial results on a save error
                    if save_kwargs.get("exif") is not None:
                        logger.warning("Smart Save failed with EXIF for %s (%s); retrying without "
                                       "metadata.", path, e)
                        save_kwargs.pop("exif", None)
                        img.save(path, "JPEG", **save_kwargs)
                    else:
                        raise
                results.append(path)
                rel = os.path.relpath(folder, out_dir)
                if not rel.startswith(".."):
                    images_ui.append({
                        "filename": os.path.basename(path),
                        "subfolder": "" if rel == "." else rel.replace(os.sep, "/"),
                        "type": "output",
                    })

            library_name = ""
            if auto_save_to_library and positive:
                name, added = self._save_to_library(library_path, scene_name,
                                                    positive, negative,
                                                    face_positive, face_negative)
                if added:
                    library_name = name

            return {"ui": {"saved": [True], "images": images_ui, "paths": results,
                           "last_positive": [positive],
                           "last_negative": [negative],
                           "last_scene_name": [scene_name],
                           "last_face_positive": [face_positive],
                           "last_face_negative": [face_negative],
                           "library": [library_name] if library_name else []}}

    @staticmethod
    def _save_to_library(library_path, scene_name, positive, negative,
                         face_positive="", face_negative=""):
        lib = resolve_library_path(library_path)
        try:
            return save_prompt_to_library(lib, scene_name, positive, negative,
                                          face_positive, face_negative)
        except OSError as e:
            raise RuntimeError(f"Could not save prompt library {lib}: {e}")

    @staticmethod
    def _next_path(folder, base):
        pattern = re.compile(re.escape(base) + r"_(\d+)\.jpg$", re.I)
        # Seed the counter from the highest existing index, then reserve the next name
        # atomically with an exclusive-create (O_EXCL) so two concurrent saves can never
        # pick the same number and clobber each other. On a collision we bump and retry.
        max_c = 0
        try:
            for f in os.listdir(folder):
                m = pattern.match(f)
                if m:
                    max_c = max(max_c, int(m.group(1)))
        except OSError:
            pass
        n = max_c + 1
        for _ in range(10000):
            path = os.path.join(folder, f"{base}_{n:05d}.jpg")
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return path
            except FileExistsError:
                n += 1
        # Practically unreachable: fall back to a timestamp-suffixed unique name.
        import time as _t
        return os.path.join(folder, f"{base}_{int(_t.time()):010d}.jpg")