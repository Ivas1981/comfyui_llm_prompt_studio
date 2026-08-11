import folder_paths

from ..combos import combo_checkpoints, combo_loras, combo_vae
from ..model_meta import detect_checkpoint_family


class LLMPromptStudioSmartLoader:
    """Loads a checkpoint, detects its distillation family and optionally applies
    a distillation LoRA. Exposes both the built-in VAE and a user-selected VAE."""
    CATEGORY = "LLM Prompt Studio"
    FUNCTION = "load"

    FAMILY_OVERRIDES = ["auto", "base", "dmd", "lcm", "turbo", "hyper", "lightning", "flash"]
    APPLY_MODES = ["auto", "always", "never"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (combo_checkpoints(),),
                "family_override": (cls.FAMILY_OVERRIDES,),
                "lora_name": (combo_loras(),),
                "apply_lora": (cls.APPLY_MODES,),
                "strength_model": ("FLOAT", {"default": 1.0, "min": 0.0,
                                             "max": 2.0, "step": 0.01}),
                "vae_user": (combo_vae(),),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "VAE", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE_MODEL", "VAE_USER", "detected_family")

    def load(self, ckpt_name, family_override, lora_name, apply_lora,
             strength_model, vae_user):
        import comfy.sd
        import comfy.utils

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path, output_vae=True, output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"))
        model, clip, vae_model = out[:3]

        # Family: manual override wins, otherwise detection by name + metadata
        family = family_override if family_override != "auto" \
            else detect_checkpoint_family(ckpt_name)

        # LoRA is applied when forced, or in auto mode for non-distilled (base) models
        should_apply = (apply_lora == "always") or \
                       (apply_lora == "auto" and family == "base")
        if should_apply and lora_name and lora_name != "[none]":
            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model, _ = comfy.sd.load_lora_for_models(
                model, clip, lora, strength_model, 0)

        # User VAE falls back to the built-in one when nothing is selected
        if vae_user and vae_user != "[none]":
            vae_path = folder_paths.get_full_path("vae", vae_user)
            vae_user_obj = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
        else:
            vae_user_obj = vae_model

        return {"ui": {"family": [family]},
                "result": (model, clip, vae_model, vae_user_obj, family)}