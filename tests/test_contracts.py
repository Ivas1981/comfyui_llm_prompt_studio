import os
import sys

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PKG)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from comfyui_llm_prompt_studio.nodes import _contracts  # noqa: E402
from comfyui_llm_prompt_studio.nodes.smart_parameters import (  # noqa: E402
    SAMPLERS_WITH_BASE,
    SCHEDULERS_WITH_BASE,
)


class _Producer:
    RETURN_TYPES = ("INT", "FLOAT", SAMPLERS_WITH_BASE, SCHEDULERS_WITH_BASE, "STRING")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}


class _GoodConsumer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sampler_name": (SAMPLERS_WITH_BASE, {}),
            "scheduler": (SCHEDULERS_WITH_BASE, {}),
            "hires_sampler_name": (SAMPLERS_WITH_BASE, {}),
            "hires_scheduler": (SCHEDULERS_WITH_BASE, {}),
        }}


class _BadConsumer:
    # Original bug shape: a freshly built LIST (not the shared tuple) for the combo.
    # ComfyUI's `!=` validation rejects list-vs-tuple even with equal contents, which
    # surfaces as "Return type mismatch" and makes the whole prompt fail to validate.
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sampler_name": (list(SAMPLERS_WITH_BASE), {}),
            "scheduler": (SCHEDULERS_WITH_BASE, {}),
            "hires_sampler_name": (SAMPLERS_WITH_BASE, {}),
            "hires_scheduler": (SCHEDULERS_WITH_BASE, {}),
        }}


def test_contract_ok():
    assert _contracts.check_combo_contracts({"Prod": _Producer, "Good": _GoodConsumer}) == []


def test_contract_flags_list_mismatch():
    ms = _contracts.check_combo_contracts({"Prod": _Producer, "Bad": _BadConsumer})
    assert len(ms) == 1
    assert ms[0]["node"] == "Bad"
    assert ms[0]["input"] == "sampler_name"


def test_contract_no_producer_is_noop():
    assert _contracts.check_combo_contracts({"Good": _GoodConsumer}) == []
