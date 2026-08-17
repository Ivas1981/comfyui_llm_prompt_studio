"""Model profile recommendations for the LLM Prompt Studio nodes.

Provides the four sampler profiles deduced from the benchmark results, the strict
JSON-schema response contracts, and a UNIVERSAL model recommendation that needs no
hard-coded model list: it infers the model size from its id (``(\\d+(?:\\.\\d+)?)b``)
and maps it to a profile + whether structured output is safe. This keeps the pack
working with any model on any machine, including when no benchmark data is present.

No dependency on ``research/`` — this module is the single source of truth the nodes
consume.
"""

import re

# ---------------------------------------------------------------------------
# Sampler profiles (mirror research/benchmark_models.SAMPLER_PROFILES).
# Every profile uses reasoning="off" — the benchmark showed reasoning_on does not
# improve these tasks.
# ---------------------------------------------------------------------------
PROFILES = {
    "baseline": {
        "temperature": 0.7, "top_p": 0.9, "top_k": 40,
        "repeat_penalty": 1.1, "presence_penalty": 0.0, "min_p": 0.0,
        "reasoning": "off", "structured": False,
    },
    "structured": {
        "temperature": 0.7, "top_p": 0.9, "top_k": 40,
        "repeat_penalty": 1.1, "presence_penalty": 0.0, "min_p": 0.0,
        "reasoning": "off", "structured": True,
    },
    "creative": {
        "temperature": 1.1, "top_p": 0.95, "top_k": 60,
        "repeat_penalty": 1.15, "presence_penalty": 0.3, "min_p": 0.05,
        "reasoning": "off", "structured": False,
    },
    "strict": {
        "temperature": 0.3, "top_p": 0.85, "top_k": 20,
        "repeat_penalty": 1.05, "presence_penalty": 0.0, "min_p": 0.0,
        "reasoning": "off", "structured": False,
    },
}

# Strict JSON-schema response contracts (only for writer/critic text output).
WRITER_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "writer",
        "strict": True,
        "schema": {
            "type": "object",
            "required": ["positive", "negative", "scene_name"],
            "properties": {
                "positive": {"type": "string"},
                "negative": {"type": "string"},
                "scene_name": {"type": "string"},
                "face_positive": {"type": "string"},
                "face_negative": {"type": "string"},
            },
        },
    },
}

CRITIC_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "critic",
        "strict": True,
        "schema": {
            "type": "object",
            "required": ["score", "verdict", "revision_notes"],
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 10},
                "verdict": {"type": "string"},
                "revision_notes": {"type": "string"},
            },
        },
    },
}

_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)b", re.IGNORECASE)


def _parse_size(model_id):
    """Return the model size in billions parsed from its id, or None if not found.

    e.g. ``qwen2.5-14b-instruct`` -> 14.0, ``nvidia/nemotron-3-nano-4b`` -> 4.0,
    ``google_gemma-3-12b-it`` -> 12.0, ``unknown-model`` -> None.
    """
    if not model_id:
        return None
    matches = _SIZE_RE.findall(model_id)
    if not matches:
        return None
    return float(matches[-1])


def schema_for_kind(kind):
    """Return the strict JSON-schema for `kind`, or None when not applicable.

    Returns WRITER_RESPONSE_SCHEMA for ``writer`` / ``compose`` (stage-2),
    CRITIC_RESPONSE_SCHEMA for ``critic``; describe and unknown kinds return None
    (those are prose / vision tasks that never use a response_format).
    """
    if kind in ("writer", "compose"):
        return WRITER_RESPONSE_SCHEMA
    if kind == "critic":
        return CRITIC_RESPONSE_SCHEMA
    return None


def recommend_for(model_id, kind):
    """Universal recommendation: no hard-coded model list, only a size heuristic.

    - kind == "describe"  -> baseline, structured never (prose output, has image)
    - size is None        -> baseline, structured False (safe default)
    - size < 7.0          -> strict, structured False (small models break JSON at t=0.7)
    - size >= 7.0         -> baseline, structured True (>=7B hit ~100% JSON with structured)
    """
    if kind == "describe":
        return {"profile": "baseline", "structured": False}
    size = _parse_size(model_id)
    if size is None:
        return {"profile": "baseline", "structured": False}
    if size < 7.0:
        return {"profile": "strict", "structured": False}
    return {"profile": "baseline", "structured": True}


def resolve_profile(choice, model_id, kind, has_image=False):
    """Resolve a node's ``load_model_profile`` choice into concrete params.

    Returns a dict:
      - ``profile``: the effective profile name ("auto" resolves to the recommended one),
        or ``"custom"`` when choice == "custom".
      - ``structured``: whether structured (response_format) is active.
      - ``params``: sampling params dict from PROFILES, or None for ``custom`` (the node
        then uses its widget values).
      - ``response_format``: the strict JSON-schema to pass to chat_completion, or None.

    Structured output is NEVER combined with an image (vision) input or with ``describe``.
    """
    if choice == "auto":
        rec = recommend_for(model_id, kind)
    elif choice in PROFILES:
        rec = {"profile": choice, "structured": choice == "structured"}
    else:  # custom
        return {"profile": "custom", "structured": False, "params": None, "response_format": None}

    profile_name = rec["profile"]
    structured = rec["structured"]
    params = dict(PROFILES[profile_name])
    if structured and not has_image and kind != "describe":
        response_format = schema_for_kind(kind)
    else:
        response_format = None
    return {
        "profile": profile_name,
        "structured": structured,
        "params": params,
        "response_format": response_format,
    }
