"""Dropdown (combo) value builders for node widgets."""
import folder_paths

from .constants import PLACEHOLDER
from .lm_http import DEFAULT_SERVER, cached_model_list

__all__ = [
    "combo_models",
    "combo_checkpoints",
    "combo_loras",
    "combo_vae",
]


def combo_models(server_url: str = DEFAULT_SERVER):
    # Return the disk-persisted model list for THIS server (populated by "Refresh models"
    # and on the first successful fetch). Reading the file directly avoids any dependence
    # on import-time seeding, so a saved workflow whose model is present in the cache always
    # validates against a real list — scoped to its own server_url (no cross-server mix-up).
    # INPUT_TYPES must NOT hit the network: the manual Refresh is the source of truth and the
    # nodeCreated handler refreshes shortly after the node appears.
    cached = cached_model_list(server_url)
    if cached:
        return cached
    return [PLACEHOLDER]


def combo_checkpoints():
    return folder_paths.get_filename_list("checkpoints")


def combo_loras():
    return ["[none]"] + folder_paths.get_filename_list("loras")


def combo_vae():
    return ["[none]"] + folder_paths.get_filename_list("vae")