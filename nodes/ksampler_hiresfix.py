"""LLM Prompt Studio KSampler (Hires Fix).

Single node (no efficient variant) that encodes a base pass and an optional
hires pass with a latent- or pixel-model upscale. Scheduler list == FULL_SCHEDULERS
(imported from smart_parameters - the single source of truth).
"""

import torch

from ._ksample import sample_latent, node_span
from ._latent_upscaler import (
    latent_upscale_with_model, project_local_upscale_models,
)
from .smart_parameters import FULL_SCHEDULERS, SAMPLERS_COMBO


_LATENT_UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]


def _round8(v):
    return max(8, int(round(v / 8.0)) * 8)


class LLMPromptStudioKSamplerHiresFix:
    CATEGORY = "LLM Prompt Studio"
    FUNCTION = "sample"
    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("LATENT", "IMAGE")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths
            model_list = folder_paths.get_filename_list("upscale_models") or []
        except Exception:
            model_list = []
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (SAMPLERS_COMBO,),
                "scheduler": (FULL_SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "hires_enabled": ("BOOLEAN", {"default": True}),
                "hires_upscale_method": (_LATENT_UPSCALE_METHODS, {"default": "nearest-exact"}),
                "hires_latent_upscale_model": (
                    ["none"] + project_local_upscale_models() + list(model_list),
                    {"default": "none"}),
                "hires_latent_upscale_factor": (
                    "FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05}),
                "hires_pixel_upscale": ("BOOLEAN", {"default": False}),
                "hires_width": ("INT", {"default": 1024, "min": 8, "max": 16384, "step": 8}),
                "hires_height": ("INT", {"default": 1024, "min": 8, "max": 16384, "step": 8}),
                "hires_steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1}),
                "hires_cfg": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 100.0, "step": 0.1}),
                "hires_denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "hires_sampler_name": (["base"] + list(SAMPLERS_COMBO), {"default": "base"}),
                "hires_scheduler": (["base"] + list(FULL_SCHEDULERS), {"default": "base"}),
                "hires_use_same_seed": ("BOOLEAN", {"default": True}),
                "hires_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "vae_decode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "hires_upscale_model": ("UPSCALE_MODEL", {"forceInput": True}),
                "hires_positive": ("CONDITIONING", {"forceInput": True}),
                "hires_negative": ("CONDITIONING", {"forceInput": True}),
                "optional_vae": ("VAE", {"forceInput": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @staticmethod
    def _latent_hw(latent):
        s = latent["samples"]
        return s.shape[2], s.shape[3]

    @staticmethod
    def _sample(model, seed, steps, cfg, sampler_name, scheduler, positive,
               negative, latent_image, denoise):
        return sample_latent(model, seed, steps, cfg, sampler_name, scheduler,
                             positive, negative, latent_image, denoise)

    @staticmethod
    def _latent_interp(latent, method, w, h):
        from nodes import LatentUpscale
        return LatentUpscale().upscale(latent, method, w, h, "disabled")[0]

    def _ensure_latent_size(self, latent, method, w, h):
        w, h = _round8(w), _round8(h)
        cur_h, cur_w = self._latent_hw(latent)
        if cur_w == w and cur_h == h:
            return latent
        return self._latent_interp(latent, method, w, h)

    def _hires_upscale(self, base, method, model_name, factor, width, height):
        target_w = _round8(width) // 8
        target_h = _round8(height) // 8
        if model_name not in (None, "none", ""):
            upscaled = latent_upscale_with_model(base, model_name, factor)
        else:
            upscaled = self._latent_interp(base, method, target_w, target_h)
        return self._ensure_latent_size(upscaled, method, target_w, target_h)

    def _pixel_stage(self, latent, upscale_model, method, width, height, vae):
        from nodes import VAEDecode, VAEEncode
        from comfy.utils import tiled_scale
        pixels = VAEDecode().decode(vae, latent)[0]
        up = tiled_scale(pixels, lambda a: upscale_model(a.float()),
                         upscale_amount=getattr(upscale_model, "scale", 2),
                         tile_x=512, tile_y=512, overlap=32)
        enc = VAEEncode().encode(vae, up)[0]
        return self._ensure_latent_size(enc, method, width // 8, height // 8)

    @staticmethod
    def _resolve_hires_sampler(base_sampler, base_sched, hires_sampler, hires_sched):
        sam = base_sampler if hires_sampler == "base" else hires_sampler
        sched = base_sched if hires_sched == "base" else hires_sched
        return sam, sched

    def _maybe_decode(self, latent, vae_decode, vae):
        if vae_decode and vae is not None:
            from nodes import VAEDecode
            return VAEDecode().decode(vae, latent)[0]
        b = latent["samples"].shape[0]
        return torch.zeros((b, 1, 1, 3), dtype=latent["samples"].dtype)

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
               sampler_name, scheduler, denoise, hires_enabled, hires_upscale_method,
               hires_latent_upscale_model, hires_latent_upscale_factor, hires_pixel_upscale,
               hires_width, hires_height, hires_steps, hires_cfg, hires_denoise,
               hires_sampler_name, hires_scheduler, hires_use_same_seed, hires_seed,
               vae_decode, hires_upscale_model=None, hires_positive=None,
               hires_negative=None, optional_vae=None, unique_id=None):
        with node_span("LLMPromptStudioKSamplerHiresFix", unique_id):
            base = self._sample(model, seed, steps, cfg, sampler_name, scheduler,
                                positive, negative, latent_image, denoise)

            b_h, b_w = self._latent_hw(base)
            target_w = _round8(hires_width) // 8
            target_h = _round8(hires_height) // 8
            need_hires = bool(hires_enabled) and (target_w != b_w or target_h != b_h)

            final = base
            if need_hires:
                upscaled = self._hires_upscale(
                    base, hires_upscale_method, hires_latent_upscale_model,
                    hires_latent_upscale_factor, hires_width, hires_height)
                if hires_pixel_upscale and hires_upscale_model is not None:
                    upscaled = self._pixel_stage(
                        upscaled, hires_upscale_model, hires_upscale_method,
                        hires_width, hires_height, optional_vae)
                sam2, sched2 = self._resolve_hires_sampler(
                    sampler_name, scheduler, hires_sampler_name, hires_scheduler)
                cfg2 = hires_cfg if hires_cfg >= 0 else cfg
                pos2 = hires_positive if hires_positive is not None else positive
                neg2 = hires_negative if hires_negative is not None else negative
                seed2 = seed if hires_use_same_seed else hires_seed
                final = self._sample(model, seed2, hires_steps, cfg2, sam2, sched2,
                                     pos2, neg2, upscaled, hires_denoise)

            image = self._maybe_decode(final, vae_decode, optional_vae)
            return (final, image)

