"""Distilled-checkpoint sampler-parameter presets (pure Python, NO comfy import).

This module is intentionally free of any ``import comfy`` so it can be unit-tested
headless (``python -m pytest comfyui_llm_prompt_studio/tests/``) and reused both by
the ``smart_parameters`` nodes and by the ``/llm_prompt_studio/sampler_params`` route.

``FAMILIES`` mirrors ``model_meta.FAMILY_MARKERS`` (dmd, lcm, turbo, hyper, lightning,
flash, schnell, tcd, pcm) plus ``"base"``. If ``model_meta.FAMILY_MARKERS`` ever gains
a key, add it here too — the two definitions must stay in sync.
"""

import re

__all__ = [
    "PRESETS",
    "FAMILIES",
    "DISTILLED_FAMILIES",
    "RECOMMENDATIONS",
    "recommend",
]

PRESETS = ("balanced", "speed", "quality")

# Keep in sync with model_meta.FAMILY_MARKERS (plus the synthetic "base").
FAMILIES = (
    "base",
    "lightning",
    "hyper",
    "dmd",
    "turbo",
    "lcm",
    "tcd",
    "pcm",
    "flash",
    "schnell",
)

DISTILLED_FAMILIES = frozenset(FAMILIES) - {"base"}

# (steps, cfg, sampler, scheduler, note) per family per preset.
# Samplers belong to comfy.samplers.KSampler.SAMPLERS; schedulers are the STANDARD
# ComfyUI scheduler list (no AYS/GITS) so they validate against BOTH the standard and
# the efficient KSampler. The efficient node swaps in "AYS SDXL" for distilled families
# at balanced/speed (see recommend()).
RECOMMENDATIONS = {
    "base": {
        "balanced": (30, 6.0, "dpmpp_2m", "karras", ""),
        "speed":    (15, 5.0, "dpmpp_2m", "karras", ""),
        "quality":  (40, 7.0, "dpmpp_2m_sde", "karras", ""),
    },
    "lightning": {
        "balanced": (4, 1.0, "euler", "sgm_uniform", ""),
        "speed":    (1, 1.0, "euler", "normal", ""),
        "quality":  (8, 2.0, "dpmpp_sde", "karras", ""),
    },
    "hyper": {
        "balanced": (4, 1.5, "euler", "sgm_uniform", ""),
        "speed":    (1, 1.0, "euler", "normal", ""),
        "quality":  (8, 2.0, "dpmpp_sde", "sgm_uniform", ""),
    },
    "dmd": {
        "balanced": (6, 1.4, "lcm", "normal", ""),
        "speed":    (4, 1.0, "lcm", "simple", ""),
        "quality":  (8, 1.8, "lcm", "normal", ""),
    },
    "turbo": {
        "balanced": (4, 1.0, "euler", "sgm_uniform", ""),
        "speed":    (1, 1.0, "euler", "normal", ""),
        "quality":  (4, 2.0, "dpmpp_sde", "karras", ""),
    },
    "lcm": {
        "balanced": (4, 1.5, "lcm", "sgm_uniform", ""),
        "speed":    (2, 1.0, "lcm", "sgm_uniform", ""),
        "quality":  (8, 2.0, "lcm", "sgm_uniform", ""),
    },
    "tcd": {
        "balanced": (8, 1.0, "euler_ancestral", "sgm_uniform",
                     "For best results use the custom TCDScheduler (ComfyUI-TCD)."),
        "speed":    (4, 0.7, "euler_ancestral", "sgm_uniform",
                     "For best results use the custom TCDScheduler (ComfyUI-TCD)."),
        "quality":  (12, 1.5, "dpmpp_sde", "sgm_uniform",
                     "For best results use the custom TCDScheduler (ComfyUI-TCD)."),
    },
    "pcm": {
        "balanced": (4, 1.5, "euler", "sgm_uniform", "Do NOT use the lcm sampler."),
        "speed":    (2, 1.0, "euler", "normal", "Do NOT use the lcm sampler."),
        "quality":  (8, 4.0, "euler", "karras", "Do NOT use the lcm sampler."),
    },
    "flash": {
        "balanced": (4, 1.0, "euler", "sgm_uniform", ""),
        "speed":    (1, 1.0, "euler", "normal", ""),
        "quality":  (4, 2.0, "dpmpp_sde", "karras", ""),
    },
    "schnell": {
        "balanced": (4, 1.0, "euler", "sgm_uniform", ""),
        "speed":    (1, 1.0, "euler", "normal", ""),
        "quality":  (4, 2.0, "dpmpp_sde", "karras", ""),
    },
}

# Matches "4step" / "4 steps" / "8-Step" in a checkpoint filename.
_STEP_RE = re.compile(r"(\d+)\s*step", re.IGNORECASE)


def _family_key(family: str):
    """Return ``(resolved_family, is_known)``.

    ``resolved_family`` is ``"base"`` for empty/unknown input; ``is_known`` is False
    when the input was not one of :data:`FAMILIES` (so the caller can apply the
    "unknown family -> fallback 6 steps" rule without colliding with the real ``base``).
    """
    fam = (family or "").strip().lower()
    if not fam or fam not in FAMILIES:
        return "base", False
    return fam, True


def _steps_from_ckpt(ckpt_name: str, distilled: bool) -> int:
    """Extract the step count from a filename like ``SDXL-Lightning_4step``.

    Returns 0 when no ``Nstep`` token is present. The caller decides whether to use it.
    """
    if not ckpt_name:
        return 0
    name = ckpt_name
    if "." in name:
        name = name.rsplit(".", 1)[0]
    m = _STEP_RE.search(name)
    if not m:
        return 0
    n = int(m.group(1))
    if distilled:
        return max(1, min(12, n))
    return max(1, min(100, n))


def recommend(family="", preset="balanced", ckpt_name="", target="standard") -> dict:
    """Return recommended sampler params for a checkpoint family.

    Returns ``{"family", "preset", "target", "steps", "cfg", "sampler",
    "scheduler", "note"}``. ``target`` is ``"standard"`` (broader default) or
    ``"efficient"`` (Efficient KSampler, which adds AYS/GITS schedulers).
    """
    fam, is_known = _family_key(family)
    if preset not in PRESETS:
        preset = "balanced"
    distilled = fam in DISTILLED_FAMILIES

    table = RECOMMENDATIONS.get(fam, RECOMMENDATIONS["base"])
    steps_t, cfg, sampler, scheduler, note = table.get(
        preset, RECOMMENDATIONS["base"][preset]
    )

    # Unknown family falls back to base params but reports a safe step count.
    if not is_known:
        steps_t = 6

    # Checkpoint filename Nstep override (matches the official Lightning/Hyper naming).
    name_steps = _steps_from_ckpt(ckpt_name, distilled)
    steps = name_steps if name_steps else steps_t

    # Efficient KSampler exposes AYS; for distilled families use AYS SDXL at the
    # low-step presets (balancing speed). quality keeps karras/sgm_uniform, which
    # exist in both scheduler lists.
    if target == "efficient" and distilled and preset in ("balanced", "speed"):
        scheduler = "AYS SDXL"

    return {
        "family": fam,
        "preset": preset,
        "target": target or "standard",
        "steps": steps,
        "cfg": cfg,
        "sampler": sampler,
        "scheduler": scheduler,
        "note": note or "",
    }
