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

# Family markers are matched case-insensitively in the checkpoint name (and a curated
# subset of the safetensors metadata). Boundaries are checked on the ORIGINAL case so we
# can tell a CamelCase continuation ("HyperSDXL", "SDXLLightning", "LCMXL") apart from a
# marker glued into a longer lowercase word ("hypernetwork" -> hyper, "calcium" -> lcm,
# "flash_attention" -> flash). An optional trailing digit is allowed so versioned variants
# such as "dmd2" (DeepMind's 1-4 step distillation) still map to the "dmd" distilled family.
_FAMILY_TOKEN_RE = {
    fam: re.compile(re.escape(fam) + r"\d*", re.IGNORECASE)
    for fam in FAMILY_MARKERS
}

# Metadata keys whose values are free-text (descriptions, comments, prompts, …). Scanning
# them for family markers produced false positives (e.g. a "hyper-realistic" description
# flagging a base model as HyperSD), so they are excluded from detection.
_FREE_TEXT_HINTS = ("description", "comment", "note", "prompt", "license", "artist", "tags")


def _meta_text(meta: dict) -> str:
    """Curated, structured-only text from safetensors metadata for family detection."""
    if not isinstance(meta, dict):
        return ""
    chunks = []
    for key, value in meta.items():
        if any(h in str(key).lower() for h in _FREE_TEXT_HINTS):
            continue
        chunks.append(str(value))
    return " ".join(chunks)


def _boundary_ok(text: str, start: int, end: int) -> bool:
    """True when the matched family token is a standalone word or a CamelCase/version
    continuation, i.e. not glued into a longer lowercase word on either side."""
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    if before and (before.islower() or before.isdigit()):
        return False
    if after and (after.islower() or after.isdigit()):
        return False
    return True


def is_no_negative_family(family):
    """True for distilled families (dmd/lcm/turbo/hyper/lightning/flash, including
    versioned variants like dmd2) that ignore the negative prompt at CFG~1. Empty
    string, 'base' and any unknown family return False."""
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
    """Detects the checkpoint family from filename + (structured) safetensors metadata.
    Returns 'base' if no distillation marker is found.

    Family words may be concatenated to other words without a separator (e.g.
    "HyperSDXL", "SDXLLightning", "LCMXL"); those are caught via case-aware boundaries.
    Markers glued into a longer lowercase word (e.g. "hypernetwork", "calcium") are
    ignored, and free-text metadata fields (descriptions, comments, …) are excluded so
    phrasing like "hyper-realistic" cannot falsely flag a base model as distilled."""
    text = str(ckpt_name)
    full_path = folder_paths.get_full_path("checkpoints", ckpt_name)
    if full_path and os.path.isfile(full_path):
        meta = read_safetensors_metadata(full_path)
        if meta:
            text += " " + _meta_text(meta)
    for family, rx in _FAMILY_TOKEN_RE.items():
        for m in rx.finditer(text):
            if not _boundary_ok(text, m.start(), m.end()):
                continue
            # Guard: 'flash' routinely surfaces inside the token 'flash_attention', which is
            # an attention mechanism, not a distilled Flash model family. Skip the family
            # whenever that token is present (in either the filename or the metadata).
            if family == "flash" and "flash_attention" in text.lower():
                break
            return family
    return "base"