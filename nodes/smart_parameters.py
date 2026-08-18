"""LLM Prompt Studio Smart Parameters nodes.

Two sibling nodes that recommend sampler parameters (steps / cfg / sampler / scheduler)
for a checkpoint family and emit them as COMBO outputs that wire directly into a KSampler:

* ``LLMPromptStudioSmartParameters``       → standard ``KSampler`` (scheduler list WITHOUT
  AYS/GITS; full backward compatibility).
* ``LLMPromptStudioSmartParametersEfficient`` → Efficient KSampler (jags111), whose
  scheduler list adds ``AYS SD1/SDXL/SVD`` and ``GITS``. For distilled families the
  efficient node recommends ``AYS SDXL``.

ComfyUI validates COMBO outputs by EXACT option-list equality at prompt-validation time,
so a single node cannot switch its output list at runtime. That is why there are two nodes
instead of one with a ``target`` widget.
"""

import comfy.samplers

SAMPLERS_COMBO = comfy.samplers.KSampler.SAMPLERS
STANDARD_SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS
EFF_SCHEDULERS = STANDARD_SCHEDULERS + ("AYS SD1", "AYS SDXL", "AYS SVD", "GITS")

from ._distilled_presets import recommend, PRESETS, FAMILIES  # noqa: E402


def _build_input_types(sched_combo):
    return {
        "required": {
            "family_override": (["auto"] + list(FAMILIES), {"default": "auto"}),
            "preset": (list(PRESETS), {"default": "balanced"}),
            "steps": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            "cfg": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 100.0, "step": 0.1}),
            "sampler_name": (SAMPLERS_COMBO + ("auto",), {"default": "auto"}),
            "scheduler": (sched_combo + ("auto",), {"default": "auto"}),
        },
        "optional": {
            "detected_family": ("STRING", {"default": ""}),
            "ckpt_name": ("STRING", {"default": ""}),
        },
        "hidden": {"unique_id": "UNIQUE_ID"},
    }


def apply_parameters(target, family_override, preset, steps, cfg, sampler_name, scheduler,
                     detected_family="", ckpt_name="", unique_id=None):
    """Resolve effective params, applying recommendations for any sentinel value.

    Sentinels: ``steps <= 0``, ``cfg < 0``, ``sampler_name == "auto"``,
    ``scheduler == "auto"`` mean "use the recommendation". The backend does not clamp
    the user-editable widgets; only the sentinel defaults fall through to the table.
    """
    eff = (detected_family or "").strip() or (family_override if family_override != "auto" else "base")
    rec = recommend(eff, preset, ckpt_name or "", target=target)
    out_steps = steps if steps and steps > 0 else rec["steps"]
    out_cfg = cfg if cfg is not None and cfg >= 0 else rec["cfg"]
    out_sampler = sampler_name if sampler_name and sampler_name != "auto" else rec["sampler"]
    out_sched = scheduler if scheduler and scheduler != "auto" else rec["scheduler"]
    info = "family: %s | preset: %s | target: %s | %s" % (
        eff, preset, target, rec.get("note", ""))
    return (out_steps, out_cfg, out_sampler, out_sched, info)


class LLMPromptStudioSmartParameters:
    CATEGORY = "LLM Prompt Studio"
    FUNCTION = "apply"
    RETURN_TYPES = ("INT", "FLOAT", SAMPLERS_COMBO, STANDARD_SCHEDULERS, "STRING")
    RETURN_NAMES = ("steps", "cfg", "sampler_name", "scheduler", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return _build_input_types(STANDARD_SCHEDULERS)

    def apply(self, *a, **k):
        return apply_parameters("standard", *a, **k)


class LLMPromptStudioSmartParametersEfficient:
    CATEGORY = "LLM Prompt Studio"
    FUNCTION = "apply"
    RETURN_TYPES = ("INT", "FLOAT", SAMPLERS_COMBO, EFF_SCHEDULERS, "STRING")
    RETURN_NAMES = ("steps", "cfg", "sampler_name", "scheduler", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return _build_input_types(EFF_SCHEDULERS)

    def apply(self, *a, **k):
        return apply_parameters("efficient", *a, **k)
