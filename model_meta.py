"""Generation metadata traversal, safetensors header reading and family detection."""
import json
import os
import re
import struct

import folder_paths

# distillation markers for Smart Loader
FAMILY_MARKERS = {
    "dmd":       ("dmd",),
    "lcm":       ("lcm",),
    "turbo":     ("turbo",),
    "hyper":     ("hyper",),
    "lightning": ("lightning",),
    "flash":     ("flash",),
}

# Families for which the negative prompt is inert (they sample at CFG ~1). Derived from
# FAMILY_MARKERS so the two definitions never drift apart; "base" is never included.
NO_NEGATIVE_FAMILIES = set(FAMILY_MARKERS) - {"base"}

# Whole-token match for each family marker: a marker must not be glued to other
# alphanumerics, so "hyper" does not match "hypernetwork" and "lcm" does not match
# "calcium". This replaces the old substring scan that mis-fired on tokens such as
# "flash_attention".
_FAMILY_TOKEN_RE = {
    fam: re.compile(r"(?<![a-z0-9])" + re.escape(fam) + r"(?![a-z0-9])")
    for fam in FAMILY_MARKERS
}


def is_no_negative_family(family):
    """True for distilled families (dmd/lcm/turbo/hyper/lightning/flash) that ignore the
    negative prompt at CFG~1. Empty string, 'base' and any unknown family return False."""
    return str(family or "").strip().lower() in NO_NEGATIVE_FAMILIES


__all__ = [
    "FAMILY_MARKERS",
    "NO_NEGATIVE_FAMILIES",
    "is_no_negative_family",
    "collect_generation_meta",
    "read_safetensors_metadata",
    "detect_checkpoint_family",
]


def collect_generation_meta(prompt, start_id):
    ckpt = ""
    loras = []
    gen = {}
    visited = set()
    stack = [str(start_id)]
    while stack:
        nid = stack.pop()
        if nid in visited:
            continue
        visited.add(nid)
        node = prompt.get(nid) if isinstance(prompt, dict) else None
        if not isinstance(node, dict):
            continue
        cls = str(node.get("class_type", "")).lower()
        inputs = node.get("inputs", {})
        for key, val in inputs.items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[1], int):
                stack.append(str(val[0]))
                continue
            if key in ("ckpt_name", "checkpoint_name", "unet_name") \
                    and isinstance(val, str) and not ckpt:
                ckpt = val
            elif key == "lora_name" and isinstance(val, str):
                loras.append(val)
        if not gen and "sampler" in cls:
            for key in ("seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"):
                if key in inputs:
                    gen[key] = inputs[key]
    loras.reverse()
    return ckpt, loras, gen


def read_safetensors_metadata(path: str, max_header: int = 10 * 1024 * 1024) -> dict:
    """Reads __metadata__ from the safetensors header without loading the weights."""
    try:
        with open(path, "rb") as f:
            raw_len = f.read(8)
            if len(raw_len) < 8:
                return {}
            header_len = struct.unpack("<Q", raw_len)[0]
            if header_len <= 0 or header_len > max_header:
                return {}
            header = json.loads(f.read(header_len).decode("utf-8", "ignore"))
        meta = header.get("__metadata__", {})
        return meta if isinstance(meta, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def detect_checkpoint_family(ckpt_name: str) -> str:
    """Detects the checkpoint family from filename + safetensors metadata.
    Returns 'base' if no distillation marker is found."""
    haystack = str(ckpt_name).lower()
    full_path = folder_paths.get_full_path("checkpoints", ckpt_name)
    if full_path and os.path.isfile(full_path):
        meta = read_safetensors_metadata(full_path)
        if meta:
            haystack += " " + json.dumps(meta, ensure_ascii=False).lower()
    for family, rx in _FAMILY_TOKEN_RE.items():
        if rx.search(haystack):
            # Guard: 'flash' routinely surfaces inside the token 'flash_attention', which is
            # an attention mechanism, not a distilled Flash model family. Skip the match
            # whenever that token is present (in either the filename or the metadata).
            if family == "flash" and "flash_attention" in haystack:
                continue
            return family
    return "base"