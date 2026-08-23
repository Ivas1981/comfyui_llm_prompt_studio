"""Studio node combo type-contract verification (debug aid).

ComfyUI validates a prompt link by comparing the source node's output TYPE descriptor
against the destination node's input TYPE descriptor for exact equality (``!=``). For
sampler/scheduler COMBOs the descriptor is the options sequence itself, so the producer
and the consumer must expose *the same* sequence. If they differ (a different list, a
missing ``"base"`` entry, or a rebuilt tuple), the whole prompt fails to validate with
``Return type mismatch`` and nothing downstream ever runs.

This module catches exactly that class of bug: it finds the combo producer
(``LLMPromptStudioSmartParameters``, whose ``RETURN_TYPES`` carry ``SAMPLERS_WITH_BASE``
/ ``SCHEDULERS_WITH_BASE``) and checks that every consumer node declares the matching
shared constant for its sampler/scheduler inputs.
"""
from .smart_parameters import SAMPLERS_WITH_BASE, SCHEDULERS_WITH_BASE

# Logical sampler/scheduler input field -> the shared constant it must expose.
_COMBO_FIELDS = {
    "sampler_name": SAMPLERS_WITH_BASE,
    "scheduler": SCHEDULERS_WITH_BASE,
    "hires_sampler_name": SAMPLERS_WITH_BASE,
    "hires_scheduler": SCHEDULERS_WITH_BASE,
}


def _find_producer(node_classes):
    """Return (name, class) of the node that emits the shared sampler combo, or None."""
    for name, cls in node_classes.items():
        rt = getattr(cls, "RETURN_TYPES", None)
        if isinstance(rt, (list, tuple)) and SAMPLERS_WITH_BASE in tuple(rt):
            return name, cls
    return None


def check_combo_contracts(node_classes):
    """Return a list of mismatch dicts for sampler/scheduler combo inputs.

    Each entry: ``{"node", "input", "expected", "actual"}``. A mismatch is reported when
    a consumer's combo options differ in content from the studio's shared constant (the
    condition that makes ComfyUI reject the prompt link).
    """
    mismatches = []
    producer = _find_producer(node_classes)
    if producer is None:
        return mismatches
    prod_name, _ = producer
    for name, cls in node_classes.items():
        if name == prod_name:
            continue
        it = getattr(cls, "INPUT_TYPES", None)
        if not callable(it):
            continue
        try:
            inputs = it()
        except Exception:
            continue
        required = (inputs or {}).get("required", {})
        for field, expected in _COMBO_FIELDS.items():
            spec = required.get(field)
            if not spec:
                continue
            actual = spec[0] if isinstance(spec, (list, tuple)) and len(spec) > 0 else None
            if actual is None or actual != expected:
                mismatches.append({
                    "node": name,
                    "input": field,
                    "expected": expected,
                    "actual": actual,
                })
    return mismatches
