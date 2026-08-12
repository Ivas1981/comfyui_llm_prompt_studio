"""Dropdown (combo) value builders for node widgets."""
import folder_paths

from .lm_http import cached_model_list, get_cached_models

__all__ = [
    "combo_models",
    "combo_checkpoints",
    "combo_loras",
    "combo_vae",
]


def combo_models():
    # Prefer the disk-persisted model list (populated by "Refresh models" and on the first
    # successful fetch). Reading the file directly each call avoids any dependence on
    # import-time seeding, so a saved workflow whose model is present in the cache always
    # validates against a real list.
    cached = cached_model_list()
    if cached:
        return cached
    # Cache empty: do a best-effort fetch (short timeout) so a running LM Studio populates
    # the combo at node-build time. This lets a saved workflow validate without requiring a
    # manual Refresh first. When the server is down the call fails fast and we fall back to
    # the placeholder; the failure is negative-cached so we don't block on every build.
    return get_cached_models(allow_fetch=True, timeout=2.0)


def combo_checkpoints():
    return folder_paths.get_filename_list("checkpoints")


def combo_loras():
    return ["[none]"] + folder_paths.get_filename_list("loras")


def combo_vae():
    return ["[none]"] + folder_paths.get_filename_list("vae")