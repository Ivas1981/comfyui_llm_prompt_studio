import logging

MAX_RESOLUTION = 16384

from ..debug import node_span

logger = logging.getLogger("llm_prompt_studio")


class LLMPromptStudioSmartMultiClip:
    """Encodes up to four prompt pairs (positive1/negative1/positive2/negative2) with a
    single CLIP and shared size settings. *_l falls back to *_g when empty, target_* falls
    back to width/height when empty. Passes CLIP through.

    Architecture-aware: accepts the Smart Loader's ``detected_architecture`` so non-SDXL
    checkpoints are conditioned correctly. SDXL / Pony / Illustrious use the dual g/l path;
    a single-encoder CLIP (SD1.5) is encoded through its single encoder; Flux/SD3 emit
    best-effort conditioning (with a warning recommending the core CLIPTextEncodeFlux/SD3)."""
    CATEGORY = "LLM Prompt Studio"
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "width": ("INT", {"default": 1024, "min": 0, "max": MAX_RESOLUTION}),
                "height": ("INT", {"default": 1024, "min": 0, "max": MAX_RESOLUTION}),
                "crop_w": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                "crop_h": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            },
            "optional": {
                "target_width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION,
                                         "forceInput": True}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION,
                                          "forceInput": True}),
                "positive1_g": ("STRING", {"default": "", "forceInput": True}),
                "positive1_l": ("STRING", {"default": "", "forceInput": True}),
                "negative1_g": ("STRING", {"default": "", "forceInput": True}),
                "negative1_l": ("STRING", {"default": "", "forceInput": True}),
                "positive2_g": ("STRING", {"default": "", "forceInput": True}),
                "positive2_l": ("STRING", {"default": "", "forceInput": True}),
                "negative2_g": ("STRING", {"default": "", "forceInput": True}),
                "negative2_l": ("STRING", {"default": "", "forceInput": True}),
                "architecture": ("STRING", {"default": "",
                                            "tooltip": "Base architecture from Smart Loader's "
                                                       "detected_architecture; selects correct CLIP "
                                                       "conditioning for non-SDXL checkpoints"}),
            }
        }

    RETURN_TYPES = ("CLIP", "CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("clip", "positive1", "negative1", "positive2", "negative2")

    @staticmethod
    def _encode_pair(clip, text_g, text_l, width, height, crop_w, crop_h, tw, th, arch=""):
        g = text_g if text_g else ""
        l = text_l if text_l else g  # *_l falls back to *_g
        tokens = clip.tokenize(g)
        # Capture whether the CLIP genuinely exposes both encoders BEFORE the safety net.
        dual_clip = "g" in tokens and "l" in tokens
        # Safety net: a single-encoder CLIP (e.g. SD1.5) returns only one key. Mirror it so
        # the g/l equalization below never KeyErrors on a missing key (and the single path
        # can still fall back to whichever key exists).
        if "g" not in tokens and "l" in tokens:
            tokens["g"] = tokens["l"]
        if "l" not in tokens and "g" in tokens:
            tokens["l"] = tokens["g"]

        arch = (arch or "").strip().lower()
        is_sdxl_like = arch in ("sdxl", "pony", "illustrious")
        is_flux_sd3 = arch in ("flux", "sd3")

        if is_flux_sd3:
            logger.warning(
                "LLM Smart Multi-Clip: Flux/SD3 native-quality conditioning requires ComfyUI "
                "core CLIPTextEncodeFlux/CLIPTextEncodeSD3; emitting best-effort conditioning.")

        use_dual = is_sdxl_like or dual_clip
        if use_dual:
            if len(tokens["l"]) != len(tokens["g"]):
                empty = clip.tokenize("")
                while len(tokens["l"]) < len(tokens["g"]):
                    tokens["l"] += empty["l"]
                while len(tokens["l"]) > len(tokens["g"]):
                    tokens["g"] += empty["g"]
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        else:
            # Single-encoder CLIP (SD1.5) or unknown arch: encode the available key only.
            key = "l" if "l" in tokens else "g"
            try:
                cond, pooled = clip.encode_from_tokens({key: tokens[key]}, return_pooled=True)
            except Exception as e:
                logger.warning("LLM Smart Multi-Clip: single-encoder encode failed (%s); "
                               "attempting full dict.", e)
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        return [[cond, {"pooled_output": pooled,
                        "width": width, "height": height,
                        "crop_w": crop_w, "crop_h": crop_h,
                        "target_width": tw, "target_height": th}]]

    def encode(self, clip, width, height, crop_w, crop_h,
                target_width=None, target_height=None, unique_id=None, architecture="",
                positive1_g="", positive1_l="",
                negative1_g="", negative1_l="",
                positive2_g="", positive2_l="",
                negative2_g="", negative2_l=""):
        with node_span("Smart Multi-Clip", unique_id, {"width": width, "height": height}):
            tw = target_width if target_width else width
            th = target_height if target_height else height

            positive1 = self._encode_pair(clip, positive1_g, positive1_l,
                                          width, height, crop_w, crop_h, tw, th, architecture)
            negative1 = self._encode_pair(clip, negative1_g, negative1_l,
                                          width, height, crop_w, crop_h, tw, th, architecture)
            positive2 = self._encode_pair(clip, positive2_g, positive2_l,
                                          width, height, crop_w, crop_h, tw, th, architecture)
            negative2 = self._encode_pair(clip, negative2_g, negative2_l,
                                          width, height, crop_w, crop_h, tw, th, architecture)

            return (clip, positive1, negative1, positive2, negative2)


# Backwards-compatible alias: existing saved graphs reference the registered key
# "LLMPromptStudioMultiClipSDXL" (unchanged in nodes/__init__.py), and the import there
# pulls this name in. Keeping the alias means the class can be renamed to
# LLMPromptStudioSmartMultiClip without touching the import line or the registration key.
LLMPromptStudioMultiClipSDXL = LLMPromptStudioSmartMultiClip
