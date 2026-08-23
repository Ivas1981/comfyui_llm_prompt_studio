import folder_paths

from ..combos import combo_checkpoints, combo_loras, combo_vae
from ..model_meta import (detect_checkpoint_family, detect_checkpoint_family_info,
                           resolve_architecture)
from ..debug import node_span


class LLMPromptStudioSmartLoader:
    """Loads a checkpoint, detects its family and architecture, and optionally applies a
    chosen LoRA. In ``auto`` mode the LoRA is applied only to a base (non-distilled)
    checkpoint; the LoRA itself may be any LoRA (distillation or otherwise). Exposes both
    the built-in VAE and a user-selected VAE."""
    CATEGORY = "LLM Prompt Studio"
    FUNCTION = "load"

    FAMILY_OVERRIDES = ["auto", "base", "dmd", "lcm", "turbo", "hyper",
                         "lightning", "flash", "schnell", "tcd", "pcm"]
    APPLY_MODES = ["auto", "always", "never"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (combo_checkpoints(),),
                "family_override": (cls.FAMILY_OVERRIDES,),
                "lora_name": (combo_loras(),),
                "apply_lora": (cls.APPLY_MODES,
                               {"tooltip": "auto = apply the chosen LoRA only to a "
                                           "base (non-distilled) checkpoint; always = "
                                           "apply it regardless; never = don't apply"}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": 0.0,
                                             "max": 2.0, "step": 0.01}),
                "vae_user": (combo_vae(),),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "VAE", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE_MODEL", "VAE_USER",
                    "detected_family", "detected_family_info",
                    "detected_architecture", "detected_architecture_info", "ckpt_name")

    def load(self, ckpt_name, family_override, lora_name, apply_lora,
              strength_model, vae_user, unique_id=None):
        with node_span("Smart Loader", unique_id, {"ckpt_name": ckpt_name,
                      "family_override": family_override, "lora_name": lora_name}):
            import comfy.sd
            import comfy.utils

            ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path, output_vae=True, output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"))
            model, clip, vae_model = out[:3]

            # Checkpoint family: manual override wins, otherwise detection by name + metadata.
            # This drives the LoRA auto-apply decision below, so a distillation LoRA is still
            # only auto-added to a non-distilled (base) checkpoint.
            if family_override != "auto":
                ckpt_family = family_override
                ckpt_source = "override"
            else:
                ckpt_family, ckpt_source = detect_checkpoint_family_info(ckpt_name)

            # LoRA is applied when forced, or in auto mode for non-distilled (base) models
            should_apply = (apply_lora == "always") or \
                           (apply_lora == "auto" and ckpt_family == "base")
            applied_distilled_lora = False
            lora_family = "base"
            if should_apply and lora_name and lora_name != "[none]":
                lora_path = folder_paths.get_full_path("loras", lora_name)
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                model, clip = comfy.sd.load_lora_for_models(
                    model, clip, lora, strength_model, strength_model)
                # A distillation LoRA (dmd/lcm/turbo/hyper/lightning/flash) applied on top of a
                # base checkpoint makes the whole pipeline distilled, so it drives no-negative
                # mode downstream.
                lora_family = detect_checkpoint_family(lora_name)
                if lora_family != "base":
                    applied_distilled_lora = True

            # Effective family (with the distillation LoRA folded in) drives downstream
            # no-negative auto mode. When a base checkpoint gets a distilled LoRA, the effective
            # family becomes the LoRA's family; otherwise it stays the checkpoint's own family.
            detected = lora_family if (should_apply and applied_distilled_lora) else ckpt_family
            source = ckpt_source

            # The visible widget shows the ORIGINAL checkpoint family plus a notification when a
            # distillation LoRA was applied (so the user always sees the true checkpoint family,
            # not the LoRA's), while `detected_family` above already carries the effective family.
            lora_note = ""
            if should_apply and applied_distilled_lora:
                lora_note = " | LoRA applied: %s (distilled: %s)" % (lora_name, lora_family)

            # User VAE falls back to the built-in one when nothing is selected
            if vae_user and vae_user != "[none]":
                vae_path = folder_paths.get_full_path("vae", vae_user)
                vae_user_obj = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
            else:
                vae_user_obj = vae_model

            # Human-readable provenance for the UI widget: original checkpoint family, detection
            # source, and — when relevant — a note that a distillation LoRA was applied.
            family_info = "family: %s | source: %s%s" % (ckpt_family, source, lora_note)

            # Base architecture detection (a separate axis from the distillation family above).
            # Prefer inspecting the live comfy model object; fall back to a filename heuristic
            # when the object is unavailable (e.g. tests). SDXL stays canonical 'sdxl', and
            # Pony/Illustrious (SDXL finetunes) are recovered from the checkpoint filename via
            # the refinement inside resolve_architecture.
            obj_name = ""
            try:
                obj_name = type(model.model).__name__
            except Exception:
                obj_name = ""
            cfg_name = ""
            try:
                cfg_name = model.model_config.__class__.__name__
            except Exception:
                cfg_name = ""
            arch, arch_source = resolve_architecture(obj_name, cfg_name, ckpt_name)
            architecture_info = "architecture: %s | source: %s" % (arch, arch_source)

            return {"ui": {"family": [detected], "family_info": [family_info],
                           "architecture": [arch], "architecture_info": [architecture_info]},
                    "result": (model, clip, vae_model, vae_user_obj, detected, family_info,
                               arch, architecture_info, ckpt_name)}