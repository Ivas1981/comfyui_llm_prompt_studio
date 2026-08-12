import logging

from ..combos import combo_models
from ..lm_http import chat_completion, ensure_model_loaded
from ..model_meta import is_no_negative_family
from ..parsing import find_missing_fields, parse_prompt_json, slugify
from ._defaults import (DEFAULT_SYSTEM, DEFAULT_SYSTEM_NO_NEGATIVE,
                        FACE_PROMPT_INSTRUCTION, FACE_PROMPT_INSTRUCTION_NO_NEGATIVE)

logger = logging.getLogger("llm_prompt_studio")

# Cache of the last generated prompt per node instance: {unique_id: result_tuple}
_PROMPT_CACHE_MAX = 128
_prompt_cache = {}


class LLMPromptStudioWriter:
    """Generates an SDXL prompt from an idea.

    If the model returns a JSON answer missing required fields, the node re-asks it up to
    `max_field_retries` times for a complete answer. Empty `positive`/`negative` after
    retries raise an error (in no-negative mode only `positive` is required); an empty
    `scene_name` falls back to `slugify(positive)`; empty face fields fall back to the main
    prompts.

    `prompt_mode` selects the negative-handling strategy:
    - `auto` (default): switches to the no-negative prompt when the detected checkpoint
      family is a distilled one (DMD/LCM/Turbo/Hyper/Lightning/Flash). Wire the Smart
      Loader's `detected_family` output into the `family` input to enable detection.
    - `standard`: always emit a negative prompt.
    - `no_negative`: always emit an empty negative (for CFG~1 distilled sampling)."""
    CATEGORY = "LLM Prompt Studio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server_url": ("STRING", {"default": "http://localhost:1234/v1"}),
                "api_key": ("STRING", {"default": ""}),
                "model": (combo_models(),),
                "context_length": ("INT", {"default": 8192, "min": 512, "max": 131072, "step": 512}),
                "gpu_offload": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "system_prompt": ("STRING", {"multiline": True, "default": DEFAULT_SYSTEM}),
                "idea": ("STRING", {"multiline": True, "default": ""}),
                "revision_notes": ("STRING", {"multiline": True, "default": ""}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 8192}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "reuse_last_prompt": ("BOOLEAN", {"default": False}),
                "generate_face_prompts": ("BOOLEAN", {"default": False}),
                "max_field_retries": ("INT", {"default": 2, "min": 0, "max": 5}),
                "face_prompt_instruction": ("STRING", {"multiline": True,
                                                        "default": FACE_PROMPT_INSTRUCTION}),
                "prompt_mode": (["auto", "standard", "no_negative"], {"default": "auto"}),
            },
            "optional": {
                "family": ("STRING", {"default": ""}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "raw", "scene_name",
                    "face_positive", "face_negative")
    FUNCTION = "execute"

    def execute(self, server_url, api_key, model, context_length, gpu_offload, system_prompt, idea,
                 revision_notes, temperature, max_tokens, seed,
                 reuse_last_prompt=False, generate_face_prompts=False,
                 max_field_retries=2, face_prompt_instruction="",
                 prompt_mode="auto", family="", unique_id=None):
        cache_key = (unique_id, prompt_mode, family)
        # Reuse mode: return cached result without calling the LLM
        if reuse_last_prompt:
            cached = _prompt_cache.get(cache_key)
            if cached:
                return cached

        if model.startswith("—"):
            raise RuntimeError(
                "No model selected. Start the LM Studio server, load a model "
                "and press the Refresh button on the node.")

        ensure_model_loaded(f"{server_url}::writer", server_url, api_key, model,
                            context_length, gpu_offload)

        # Resolve the effective mode. 'no_negative' forces it; 'standard' forbids it;
        # 'auto' (and anything unexpected) defers to the detected checkpoint family.
        if prompt_mode == "no_negative":
            no_negative = True
        elif prompt_mode == "standard":
            no_negative = False
        else:
            no_negative = is_no_negative_family(family)
        logger.info("Writer node %s prompt mode: %s (family=%r) -> no_negative=%s",
                    unique_id, prompt_mode, family, no_negative)

        # System prompt selection. A user-edited widget value overrides the built-in
        # no-negative default; otherwise we use the dedicated no-negative prompt.
        if no_negative:
            effective_system = system_prompt if system_prompt != DEFAULT_SYSTEM \
                else DEFAULT_SYSTEM_NO_NEGATIVE
            if generate_face_prompts:
                inst = face_prompt_instruction.strip() if face_prompt_instruction else ""
                if not inst:
                    inst = FACE_PROMPT_INSTRUCTION_NO_NEGATIVE
                effective_system = effective_system + inst
        else:
            effective_system = system_prompt
            if generate_face_prompts:
                inst = face_prompt_instruction.strip() if face_prompt_instruction else ""
                if not inst:
                    inst = FACE_PROMPT_INSTRUCTION
                effective_system = effective_system + inst

        messages = [
            {"role": "system", "content": effective_system},
            {"role": "user", "content": f"Idea: {idea}"},
        ]
        if revision_notes.strip():
            messages.append({"role": "user", "content": (
                "The previous version of the prompt did not pass the critic's check. "
                f"Requested fixes: {revision_notes}\n"
                "Generate a CORRECTED prompt taking these fixes into account, "
                "in the same JSON format."
            )})

        raw = chat_completion(server_url, api_key, model, messages,
                                temperature, max_tokens, seed=seed)
        parsed = parse_prompt_json(raw)

        # Field-retry: if the model omitted required JSON fields, re-ask it (up to
        # max_field_retries times) for a complete answer before falling back. In no-negative
        # mode the negative is intentionally empty, so it is not treated as missing.
        attempt = 0
        while attempt < max_field_retries:
            missing = find_missing_fields(
                parsed, require_face=generate_face_prompts, require_negative=not no_negative)
            if not missing:
                break
            attempt += 1
            logger.info("Field retry %d/%d for node %s: missing %s",
                        attempt, max_field_retries, unique_id, ", ".join(missing))
            messages.append({"role": "user", "content":
                f"You omitted the required JSON field(s): {', '.join(missing)}. "
                f"Respond again with a COMPLETE JSON object containing ALL required fields."})
            raw_new = chat_completion(server_url, api_key, model, messages,
                                       temperature, max_tokens, seed=seed)
            raw = (f"[FIELD RETRY {attempt}/{max_field_retries}: "
                   f"missing {', '.join(missing)}]\n{raw_new}")
            parsed = parse_prompt_json(raw_new)

        positive, negative, scene_name, face_positive, face_negative = parsed

        # In no-negative mode the negative fields are forced empty regardless of what the
        # model returned (the negative is inert at CFG~1 and must stay consistent).
        if no_negative:
            negative = ""
            face_negative = ""

        if not positive.strip():
            raise RuntimeError(
                f"Model failed to produce a required positive prompt after "
                f"{max_field_retries} field-retry attempt(s). Try a model with better "
                "instruction following, or lower max_field_retries and edit manually.")
        if not no_negative and not negative.strip():
            raise RuntimeError(
                f"Model failed to produce a required negative prompt after "
                f"{max_field_retries} field-retry attempt(s). Try a model with better "
                "instruction following, or lower max_field_retries and edit manually.")
        if not scene_name.strip():
            scene_name = slugify(positive)

        # Fallback: if face prompts were not generated, the regular prompts go to their outputs
        if not face_positive:
            face_positive = positive
        if not no_negative and not face_negative.strip():
            face_negative = negative

        result = (positive, negative, raw, scene_name, face_positive, face_negative)
        _prompt_cache[cache_key] = result
        if len(_prompt_cache) > _PROMPT_CACHE_MAX:
            _prompt_cache.pop(next(iter(_prompt_cache)))  # drop oldest
        return result