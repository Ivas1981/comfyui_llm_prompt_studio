MAX_RESOLUTION = 16384

from ..debug import node_span


class LLMPromptStudioMultiClipSDXL:
    """Encodes up to four SDXL prompt pairs (positive1/negative1/positive2/negative2)
    with a single CLIP and shared size settings. *_l falls back to *_g when empty,
    target_* falls back to width/height when empty. Passes CLIP through."""
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
            }
        }

    RETURN_TYPES = ("CLIP", "CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("clip", "positive1", "negative1", "positive2", "negative2")

    @staticmethod
    def _encode_pair(clip, text_g, text_l, width, height, crop_w, crop_h, tw, th):
        g = text_g if text_g else ""
        l = text_l if text_l else g  # *_l falls back to *_g
        tokens = clip.tokenize(g)
        tokens["l"] = clip.tokenize(l)["l"]
        if len(tokens["l"]) != len(tokens["g"]):
            empty = clip.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty["g"]
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled,
                        "width": width, "height": height,
                        "crop_w": crop_w, "crop_h": crop_h,
                        "target_width": tw, "target_height": th}]]

    def encode(self, clip, width, height, crop_w, crop_h,
                target_width=None, target_height=None, unique_id=None,
                positive1_g="", positive1_l="",
                negative1_g="", negative1_l="",
                positive2_g="", positive2_l="",
                negative2_g="", negative2_l=""):
        with node_span("Multi-CLIP SDXL", unique_id, {"width": width, "height": height}):
            tw = target_width if target_width else width
            th = target_height if target_height else height

            positive1 = self._encode_pair(clip, positive1_g, positive1_l,
                                           width, height, crop_w, crop_h, tw, th)
            negative1 = self._encode_pair(clip, negative1_g, negative1_l,
                                           width, height, crop_w, crop_h, tw, th)
            positive2 = self._encode_pair(clip, positive2_g, positive2_l,
                                           width, height, crop_w, crop_h, tw, th)
            negative2 = self._encode_pair(clip, negative2_g, negative2_l,
                                           width, height, crop_w, crop_h, tw, th)

            return (clip, positive1, negative1, positive2, negative2)