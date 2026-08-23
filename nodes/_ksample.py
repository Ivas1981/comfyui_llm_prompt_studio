"""Shared KSampler invocation with AYS/GITS scheduler support.

ComfyUI's ``comfy.samplers.calculate_sigmas`` does not know the AYS/GITS
schedulers, so we temporarily patch it (restored in ``finally``) so that
``nodes.KSampler().sample(...)`` accepts any ``scheduler`` from ``FULL_SCHEDULERS``.

Thread-safety: the patch is global, so every sampler call is serialized with a
module-level lock to avoid one thread restoring the patch mid-call of another.
"""

import contextlib
import logging
import threading

import comfy.samplers

logger = logging.getLogger("llm_prompt_studio")

_ORIGINAL_CALC = None
_SIG_LOCK = threading.Lock()


try:
    # Delegate to the real debug span when the studio debug module is available so the
    # sampler nodes (KSampler Hires Fix, Face Detailer) get the same enter/exit/error
    # coverage as the other studio nodes. Falls back to a no-op if debug is unavailable.
    from ..debug import node_span as _real_node_span  # type: ignore
except Exception:  # pragma: no cover - only when the package layout is unexpected
    @contextlib.contextmanager
    def _real_node_span(name, unique_id=None):
        yield


def node_span(name, unique_id=None):
    """Wrap a node's heavy work with debug enter/exit/error logging.

    Delegates to ``debug.node_span`` (a no-op when ``DEBUG_LEVEL == "OFF"``), so this
    is cheap and safe to call unconditionally from every studio node.
    """
    return _real_node_span(name, unique_id)


def _patched_calculate_sigmas(model, scheduler, steps):
    if scheduler in ("AYS SD1", "AYS SDXL", "AYS SVD"):
        # The AYS schedulers come from an external custom node (ComfyUI-AlignYourSteps).
        # If it isn't installed, fall back to a standard scheduler so the studio keeps
        # working out-of-the-box instead of crashing.
        calc = _ORIGINAL_CALC
        try:
            name = scheduler.split(" ", 1)[1]
            sched = comfy.samplers.NODE_CLASS_MAPPINGS["AlignYourStepsScheduler"]()
            return sched.get_sigmas(name, steps, denoise=1.0)[0]
        except KeyError:
            logger.warning(
                "AYS scheduler '%s' requested but the AlignYourSteps custom node is not "
                "installed; falling back to the 'karras' scheduler.", scheduler)
            if calc is None:
                raise
            return calc(model, "karras", steps)
    if scheduler == "GITS":
        # GITS also comes from an external custom node (KJNodes). Fall back to 'simple'.
        calc = _ORIGINAL_CALC
        try:
            gits = comfy.samplers.NODE_CLASS_MAPPINGS["GITSScheduler"]()
            return gits.execute(1.20, steps, denoise=1.0)[0]
        except KeyError:
            logger.warning(
                "GITS scheduler requested but the GITS custom node is not installed; "
                "falling back to the 'simple' scheduler.")
            if calc is None:
                raise
            return calc(model, "simple", steps)
    return _ORIGINAL_CALC(model, scheduler, steps)


def sample_latent(model, seed, steps, cfg, sampler_name, scheduler, positive,
                  negative, latent, denoise=1.0):
    """Run ``nodes.KSampler().sample`` with AYS/GITS-awareness.

    ``from nodes import ...`` is lazy so this module imports headlessly in tests.
    The global ``comfy.samplers.calculate_sigmas`` patch is serialized with a lock so
    concurrent sampler calls don't race on the patch/restore.
    """
    global _ORIGINAL_CALC
    if _ORIGINAL_CALC is None:
        _ORIGINAL_CALC = getattr(comfy.samplers, "calculate_sigmas", None)

    from nodes import KSampler  # noqa: F401  (LatentUpscale/VAE* used by callers)

    with _SIG_LOCK:
        saved = getattr(comfy.samplers, "calculate_sigmas", None)
        comfy.samplers.calculate_sigmas = _patched_calculate_sigmas
        try:
            out = KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler,
                                    positive, negative, latent, denoise=denoise)
        finally:
            comfy.samplers.calculate_sigmas = saved
    return out[0]
