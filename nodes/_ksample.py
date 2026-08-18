"""Shared KSampler invocation with AYS/GITS scheduler support.

ComfyUI's ``comfy.samplers.calculate_sigmas`` does not know the AYS/GITS
schedulers, so we temporarily patch it (restored in ``finally``) so that
``nodes.KSampler().sample(...)`` accepts any ``scheduler`` from ``FULL_SCHEDULERS``.
"""

import contextlib

import comfy.samplers

_ORIGINAL_CALC = None


@contextlib.contextmanager
def node_span(name, unique_id=None):
    """Lightweight wrapper around a node's heavy work.

    Defined here so every studio node can use the same helper without pulling in
    an optional debug module. A no-op by default (keeps node code clean).
    """
    yield


def _patched_calculate_sigmas(model, scheduler, steps):
    if scheduler in ("AYS SD1", "AYS SDXL", "AYS SVD"):
        name = scheduler.split(" ", 1)[1]
        sched = comfy.samplers.NODE_CLASS_MAPPINGS["AlignYourStepsScheduler"]()
        return sched.get_sigmas(name, steps, denoise=1.0)[0]
    if scheduler == "GITS":
        gits = comfy.samplers.NODE_CLASS_MAPPINGS["GITSScheduler"]()
        return gits.execute(1.20, steps, denoise=1.0)[0]
    return _ORIGINAL_CALC(model, scheduler, steps)


def sample_latent(model, seed, steps, cfg, sampler_name, scheduler, positive,
                  negative, latent, denoise=1.0):
    """Run ``nodes.KSampler().sample`` with AYS/GITS-awareness.

    ``from nodes import ...`` is lazy so this module imports headlessly in tests.
    """
    global _ORIGINAL_CALC
    if _ORIGINAL_CALC is None:
        _ORIGINAL_CALC = getattr(comfy.samplers, "calculate_sigmas", None)

    from nodes import KSampler  # noqa: F401  (LatentUpscale/VAE* used by callers)

    saved = getattr(comfy.samplers, "calculate_sigmas", None)
    comfy.samplers.calculate_sigmas = _patched_calculate_sigmas
    try:
        out = KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler,
                                positive, negative, latent, denoise=denoise)
        return out[0]
    finally:
        comfy.samplers.calculate_sigmas = saved

