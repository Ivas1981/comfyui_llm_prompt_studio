"""LLM Prompt Studio KSampler (Hires Fix).

Single node (no efficient variant) that encodes a base pass and an optional
hires pass with a latent- or pixel-model upscale. Scheduler list == FULL_SCHEDULERS
(imported from smart_parameters - the single source of truth).
"""

import torch

from ._ksample import sample_latent, node_span
from ._latent_upscaler import (
    latent_upscale_with_model, project_local_upscale_models, model_native_scale,
)
from .smart_parameters import SAMPLERS_WITH_BASE, SCHEDULERS_WITH_BASE
from ..vram import release_before_sample


_LATENT_UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
# How the hires pass upscales the base latent. This makes it explicit which upscaler runs:
#  - "latent"        : bilinear/interp of the latent (hires_upscale_method), no model needed
#  - "latent (model)": a LatentUpscaleModel (hires_latent_upscale_model)
#  - "pixel (model)" : decode -> super-res UPSCALE_MODEL -> re-encode (hires_upscale_model input)
HIRES_UPSCALE_TYPES = ["latent", "latent (model)", "pixel (model)"]
# How the node's preview IMAGE is produced (mirrors Efficient KSampler's preview_method).
PREVIEW_METHODS = ["none", "vae", "latent2rgb", "taesd"]


def _round8(v):
    return max(8, int(round(v / 8.0)) * 8)


class LLMPromptStudioKSamplerHiresFix:
    CATEGORY = "LLM Prompt Studio"
    FUNCTION = "sample"
    RETURN_TYPES = ("LATENT", "IMAGE", "INT")
    RETURN_NAMES = ("LATENT", "IMAGE", "VAE_TILE_SIZE")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (SAMPLERS_WITH_BASE, {"default": "dpmpp_2m"}),
                "scheduler": (SCHEDULERS_WITH_BASE, {"default": "karras"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "hires_enabled": ("BOOLEAN", {"default": True}),
                "hires_upscale_type": (HIRES_UPSCALE_TYPES, {
                    "default": "latent (model)",
                    "tooltip": "Which upscaler the hires pass uses: 'latent' = interpolate the "
                               "latent (hires_upscale_method, no model); 'latent (model)' = a "
                               "LatentUpscaleModel (hires_latent_upscale_model); 'pixel (model)' = "
                               "decode to pixels, super-res with a UPSCALE_MODEL (hires_upscale_model "
                               "input), then re-encode."}),
                "hires_upscale_method": (_LATENT_UPSCALE_METHODS, {"default": "nearest-exact"}),
                "hires_latent_upscale_model": (
                    ["none"] + project_local_upscale_models(),
                    {"default": "none",
                     "tooltip": "LatentUpscaleModel used when hires_upscale_type = 'latent (model)'. "
                                "Only the project's latent upscalers (models/upscale_models, e.g. "
                                "ttl-nn sdxl_resizer.pt) are listed - pixel ESRGAN models are not "
                                "valid here."}),
                "hires_latent_upscale_factor": (
                    "FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05,
                              "tooltip": "Per-pass upscale scale for latent hires types "
                                         "('latent' and 'latent (model)'). The hires output "
                                         "size is base x factor^iterations; raise it for a "
                                         "bigger result."}),
                "hires_upscale_iterations": (
                    "INT", {"default": 1, "min": 1, "max": 8, "step": 1,
                            "tooltip": "Number of progressive upscale passes. The hires output "
                                        "size is base x per_pass_scale^iterations, where "
                                        "per_pass_scale is hires_latent_upscale_factor for "
                                        "latent types, or the model scale for 'pixel (model)'."}),
                "hires_latent_upscale_tile": (
                    "INT", {"default": 0, "min": 0, "max": 512, "step": 8,
                            "tooltip": "Tile size (in latent cells) for 'latent (model)'. 0 = auto: "
                                       "the whole latent is upscaled at once while it fits in "
                                       "RAM/VRAM, and only oversized latents are tiled. Set a value "
                                       "(e.g. 64) to force tiling on a low-memory machine, at the "
                                       "cost of some quality (each tile is normalized on its own)."}),
                "hires_steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1}),
                "hires_cfg": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 100.0, "step": 0.1}),
                "hires_denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "hires_sampler_name": (SAMPLERS_WITH_BASE, {"default": "base"}),
                "hires_scheduler": (SCHEDULERS_WITH_BASE, {"default": "base"}),
                "hires_use_same_seed": ("BOOLEAN", {"default": True}),
                "hires_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "vae_decode": ("BOOLEAN", {"default": True,
                                 "tooltip": "Decode the final latent into an IMAGE preview. Turn "
                                            "off to skip the VAE decode (saves VRAM)."}),
                "preview_method": (PREVIEW_METHODS, {
                    "default": "vae",
                    "tooltip": "How the preview IMAGE is generated: 'vae' = full VAE decode (most "
                                "accurate), 'latent2rgb' = fast approximate latent->RGB, 'taesd' = "
                                "TAESD preview if available (else VAE), 'none' = no preview image."}),
                "vae_tile_size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16,
                                   "tooltip": "Tile size (px) for the VAE decode of the final "
                                              "latent. 0 = decode the whole frame at once (best "
                                              "quality). A positive value (e.g. 512) tiles the decode "
                                              "to cap VRAM on large hires latents. Also emitted on the "
                                              "VAE_TILE_SIZE output so it can be wired into "
                                              "FaceDetailer's vae_tile_size."}),
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
        # w/h are LATENT cell dims (samples space). LatentUpscale treats its
        # width/height as PIXELS and divides by 8 internally, so convert back,
        # otherwise the latent gets shrunk to 1/64 of the intended area and the
        # UNet forward aborts on the degenerate tensor.
        return LatentUpscale().upscale(latent, method, w * 8, h * 8, "disabled")[0]

    def _ensure_latent_size(self, latent, method, w, h):
        w, h = _round8(w), _round8(h)
        cur_h, cur_w = self._latent_hw(latent)
        if cur_w == w and cur_h == h:
            return latent
        return self._latent_interp(latent, method, w, h)

    def _hires_upscale(self, base, upscale_type, method, model_name, factor, target_w,
                       target_h, iterations, tile=0):
        iterations = max(1, int(iterations))
        if upscale_type == "latent (model)" and model_name not in (None, "none", ""):
            upscaled = latent_upscale_with_model(base, model_name, factor,
                                                 iterations=iterations, tile=max(0, int(tile)))
        else:
            cur_h, cur_w = self._latent_hw(base)
            step_h = (target_h / float(cur_h)) ** (1.0 / iterations)
            step_w = (target_w / float(cur_w)) ** (1.0 / iterations)
            upscaled = base
            for _ in range(iterations):
                cur_h, cur_w = self._latent_hw(upscaled)
                nw = _round8(max(8, round(cur_w * step_w)))
                nh = _round8(max(8, round(cur_h * step_h)))
                upscaled = self._latent_interp(upscaled, method, nw, nh)
        return self._ensure_latent_size(upscaled, method, target_w, target_h)

    def _pixel_stage(self, latent, upscale_model, method, width, height, vae, iterations):
        from nodes import VAEDecode, VAEEncode
        from comfy.utils import tiled_scale
        iterations = max(1, int(iterations))
        pixels = VAEDecode().decode(vae, latent)[0]
        up = pixels.permute(0, 3, 1, 2)
        for _ in range(iterations):
            up = tiled_scale(up, lambda a: upscale_model(a.float()),
                             upscale_amount=getattr(upscale_model, "scale", 2),
                             tile_x=512, tile_y=512, overlap=32)
        up = up.permute(0, 2, 3, 1)
        enc = VAEEncode().encode(vae, up)[0]
        return self._ensure_latent_size(enc, method, width // 8, height // 8)

    @staticmethod
    def _resolve_hires_sampler(base_sampler, base_sched, hires_sampler, hires_sched):
        sam = base_sampler if hires_sampler == "base" else hires_sampler
        sched = base_sched if hires_sched == "base" else hires_sched
        return sam, sched

    @staticmethod
    def _latent_to_rgb(samples):
        # Cheap approximate latent -> RGB preview (no VAE). Useful when no VAE is wired.
        # ComfyUI IMAGE is [B,H,W,C], so permute from the latent [B,C,H,W] layout.
        if samples.shape[1] >= 3:
            rgb = samples[:, :3]
        else:
            rgb = samples.repeat(1, 3, 1, 1)[:, :3]
        rgb = rgb.permute(0, 2, 3, 1)
        return (rgb * 0.5 + 0.5).clamp(0.0, 1.0)

    @staticmethod
    def _decode_latent(vae, latent, tile_size):
        # VRAM-bounded VAE decode with a robust fallback chain. Mirrors the logic
        # in FaceDetailer so a single tile-size value drives both nodes.
        samples = latent["samples"]
        t = int(tile_size or 0)
        if t > 0 and vae is not None:
            td = getattr(vae, "tiled_decode", None)
            if callable(td):
                try:
                    return td(samples, tile_x=t, tile_y=t)
                except Exception:
                    try:
                        return td(samples, t, 16)
                    except Exception:
                        pass
        if vae is not None:
            from nodes import VAEDecode
            return VAEDecode().decode(vae, latent)[0]
        return None

    def _taesd_preview(self, samples, vae):
        # Best-effort TAESD preview; falls back to the caller on any failure.
        try:
            from comfy.taesd import TAESD
        except Exception:
            raise RuntimeError("taesd preview unavailable")
        if vae is None or not hasattr(vae, "taesd_decoder"):
            raise RuntimeError("no taesd decoder on this VAE")
        dec = vae.taesd_decoder
        out = dec(samples.to(dec.device))[0]
        # TAESD decoder emits channel-first [B,C,H,W]; ComfyUI IMAGE is [B,H,W,C].
        if out.dim() == 4 and out.shape[1] == 3 and out.shape[3] != 3:
            out = out.permute(0, 2, 3, 1)
        return out

    def _preview_image(self, final, preview_method, vae, vae_tile_size=0):
        samples = final["samples"]
        if preview_method == "none":
            return torch.zeros((samples.shape[0], 1, 1, 3), dtype=samples.dtype)
        if preview_method == "latent2rgb":
            return self._latent_to_rgb(samples)
        if preview_method == "taesd":
            try:
                return self._taesd_preview(samples, vae)
            except Exception:
                pass
        # default "vae"
        if vae is not None:
            decoded = self._decode_latent(vae, final, vae_tile_size)
            if decoded is not None:
                return decoded
        return self._latent_to_rgb(samples)

    def _maybe_preview(self, final, vae_decode, preview_method, vae, vae_tile_size=0):
        if not vae_decode:
            samples = final["samples"]
            return torch.zeros((samples.shape[0], 1, 1, 3), dtype=samples.dtype)
        return self._preview_image(final, preview_method, vae, vae_tile_size)

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
                sampler_name, scheduler, denoise, hires_enabled, hires_upscale_type,
                hires_upscale_method, hires_latent_upscale_model,
                hires_latent_upscale_factor, hires_upscale_iterations, hires_steps,
                hires_cfg, hires_denoise, hires_sampler_name, hires_scheduler,
                hires_use_same_seed, hires_seed, vae_decode, preview_method,
                vae_tile_size=0, hires_latent_upscale_tile=0, hires_upscale_model=None,
                hires_positive=None, hires_negative=None, optional_vae=None, unique_id=None):
        with node_span("LLMPromptStudioKSamplerHiresFix", unique_id):
            release_before_sample()
            base = self._sample(model, seed, steps, cfg, sampler_name, scheduler,
                                positive, negative, latent_image, denoise)

            b_h, b_w = self._latent_hw(base)
            # Hires output size is derived from the base size and the per-pass scale,
            # so there are no explicit width/height fields to confuse the user.
            if hires_upscale_type == "pixel (model)":
                per_pass = float(getattr(hires_upscale_model, "scale", 2))
            elif (hires_upscale_type == "latent (model)"
                  and hires_latent_upscale_model not in (None, "none", "")):
                # A LatentUpscaleModel (.safetensors) bakes its scale into the architecture;
                # use its native scale (from the filename) so target size matches the actual
                # upscale instead of hires_latent_upscale_factor.
                per_pass = model_native_scale(
                    hires_latent_upscale_model, float(hires_latent_upscale_factor))
            else:
                per_pass = float(hires_latent_upscale_factor)
            iterations = max(1, int(hires_upscale_iterations))
            target_pw = _round8(b_w * 8 * (per_pass ** iterations))
            target_ph = _round8(b_h * 8 * (per_pass ** iterations))
            target_w = max(1, target_pw // 8)
            target_h = max(1, target_ph // 8)
            need_hires = bool(hires_enabled) and (target_w != b_w or target_h != b_h)

            final = base
            if need_hires:
                if hires_upscale_type == "pixel (model)":
                    if hires_upscale_model is None:
                        raise ValueError(
                            "hires_upscale_type 'pixel (model)' requires a connected "
                            "UPSCALE_MODEL (hires_upscale_model input)")
                    upscaled = self._pixel_stage(
                        base, hires_upscale_model, hires_upscale_method,
                        target_pw, target_ph, optional_vae, hires_upscale_iterations)
                else:
                    upscaled = self._hires_upscale(
                        base, hires_upscale_type, hires_upscale_method,
                        hires_latent_upscale_model, hires_latent_upscale_factor,
                        target_w, target_h, hires_upscale_iterations,
                        tile=hires_latent_upscale_tile)
                sam2, sched2 = self._resolve_hires_sampler(
                    sampler_name, scheduler, hires_sampler_name, hires_scheduler)
                cfg2 = hires_cfg if hires_cfg >= 0 else cfg
                pos2 = hires_positive if hires_positive is not None else positive
                neg2 = hires_negative if hires_negative is not None else negative
                seed2 = seed if hires_use_same_seed else hires_seed
                final = self._sample(model, seed2, hires_steps, cfg2, sam2, sched2,
                                     pos2, neg2, upscaled, hires_denoise)

            image = self._maybe_preview(final, vae_decode, preview_method, optional_vae,
                                        vae_tile_size)
            return (final, image, vae_tile_size)

