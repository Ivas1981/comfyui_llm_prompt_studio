"""LLM Prompt Studio Smart Parameters node.

A single node that recommends sampler parameters (steps / cfg / sampler /
scheduler) for a checkpoint family and emits them as COMBO outputs that wire
directly into a KSampler. The scheduler list is ``FULL_SCHEDULERS`` (the standard
scheduler list plus AYS SD1/SDXL/SVD and GITS) - our own KSampler node supports
the full list via a ``calculate_sigmas`` patch, so one node (and one scheduler
list) is enough; no ``target`` / "efficient" bifurcation is needed.
"""

import comfy.samplers

SAMPLERS_COMBO = comfy.samplers.KSampler.SAMPLERS
STANDARD_SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS
FULL_SCHEDULERS = list(STANDARD_SCHEDULERS) + ["AYS SD1", "AYS SDXL", "AYS SVD", "GITS"]

# Combo types that also expose a "base" option. ComfyUI validates a link by comparing the
# source node's output TYPE tuple against the destination input TYPE tuple for equality, so
# every node that emits or consumes a sampler/scheduler combo must share the *exact same*
# tuple object. The "base" entry lets the hires pass reuse the base pass sampler/scheduler.
SAMPLERS_WITH_BASE = ("base",) + tuple(SAMPLERS_COMBO)
SCHEDULERS_WITH_BASE = ("base",) + tuple(FULL_SCHEDULERS)

from ._distilled_presets import recommend, PRESETS, FAMILIES  # noqa: E402
from .. import model_meta  # noqa: E402  (comfy-free family/architecture detection)


def _build_input_types(sched_combo):
    return {
        "required": {
            "family_override": (["auto"] + list(FAMILIES), {
                "default": "auto",
                "tooltip": "Force a checkpoint family (lightning/hyper/turbo/lcm/tcd/pcm/flash/"
                           "schnell) or leave 'auto'. Detection order: detected_family (from "
                           "Smart Loader) -> family_override -> auto-detect from the checkpoint "
                           "filename, its safetensors metadata, and its parent folder name.",
            }),
            "preset": (list(PRESETS), {
                "default": "user",
                "tooltip": "user = pass your widgets through; balanced = recommended steps/cfg/"
                           "sampler; speed = lowest step count; quality = highest step count.",
            }),
            "steps": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            "cfg": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 100.0, "step": 0.1}),
            "sampler_name": (SAMPLERS_COMBO + ["auto"], {"default": "auto"}),
            "scheduler": (sched_combo + ["auto"], {"default": "auto"}),
        },
        "optional": {
            "detected_family": ("STRING", {
                "default": "",
                "tooltip": "Optional family from Smart Loader; overrides filename/folder "
                           "auto-detection when set.",
            }),
            "ckpt_name": ("STRING", {
                "default": "",
                "tooltip": "Checkpoint filename; used to auto-detect family (filename, metadata, "
                           "and parent folder).",
            }),
            "architecture": ("STRING", {"default": "",
                                        "tooltip": "Base architecture from Smart Loader's "
                                                   "detected_architecture; selects architecture-"
                                                   "appropriate sampler defaults for base models"}),
        },
        "hidden": {"unique_id": "UNIQUE_ID"},
    }


def apply_parameters(preset, family_override, steps, cfg, sampler_name, scheduler,
                     detected_family="", ckpt_name="", architecture="", unique_id=None):
    """Resolve effective params, applying recommendations for any sentinel value.

    Sentinels: ``steps <= 0``, ``cfg < 0``, ``sampler_name == "auto"``,
    ``scheduler == "auto"`` mean "use the recommendation". The backend does not clamp
    the user-editable widgets; only the sentinel defaults fall through to the table.

    For the ``user`` preset the widget values are passed through unchanged (with safe
    defaults when a sentinel is still present), so manual edits are never overwritten.
    """
    if preset == "user":
        out_steps = steps if steps and steps > 0 else 20
        out_cfg = cfg if cfg is not None and cfg >= 0 else 7.0
        out_sampler = sampler_name if sampler_name and sampler_name != "auto" else "dpmpp_2m"
        out_sched = scheduler if scheduler and scheduler != "auto" else "karras"
        info = "preset: user | arch: %s" % (architecture or "")
        return (out_steps, out_cfg, out_sampler, out_sched, info)

    # Resolve the effective family: Smart Loader's detected_family wins, then an explicit
    # family_override, then auto-detect from the checkpoint filename / metadata / folder.
    eff = (detected_family or "").strip()
    if not eff or eff == "auto":
        eff = family_override if (family_override and family_override != "auto") else ""
    if not eff:
        eff = model_meta.detect_checkpoint_family(ckpt_name or "")
    rec = recommend(eff or "base", preset, ckpt_name or "", architecture=architecture or "")
    out_steps = steps if steps and steps > 0 else rec["steps"]
    out_cfg = cfg if cfg is not None and cfg >= 0 else rec["cfg"]
    out_sampler = sampler_name if sampler_name and sampler_name != "auto" else rec["sampler"]
    out_sched = scheduler if scheduler and scheduler != "auto" else rec["scheduler"]
    info = "family: %s | preset: %s | arch: %s | %s" % (
        eff, preset, architecture or "", rec.get("note", ""))
    return (out_steps, out_cfg, out_sampler, out_sched, info)


class LLMPromptStudioSmartParameters:
    DESCRIPTION = (
        "Recommends sampler parameters (steps/cfg/sampler/scheduler) for a checkpoint "
        "family and emits them as COMBO outputs that wire directly into a KSampler."
    )
    CATEGORY = "LLM Prompt Studio"
    FUNCTION = "apply"
    RETURN_TYPES = ("INT", "FLOAT", SAMPLERS_WITH_BASE, SCHEDULERS_WITH_BASE, "STRING")
    RETURN_NAMES = ("steps", "cfg", "sampler_name", "scheduler", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return _build_input_types(FULL_SCHEDULERS)

    def apply(self, *a, **k):
        return apply_parameters(*a, **k)
