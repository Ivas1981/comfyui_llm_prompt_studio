import logging

from ..combos import combo_models
from ..imaging import image_to_base64
from ..lm_http import chat_completion, ensure_model_loaded, looks_like_vision
from ..model_meta import is_no_negative_family
from ..parsing import find_missing_fields, parse_prompt_json, slugify
from ._defaults import DEFAULT_COMPOSER, DEFAULT_COMPOSER_NO_NEGATIVE, DEFAULT_DESCRIBE

logger = logging.getLogger("llm_prompt_studio")

# Local helper to append or merge user messages (keeps role sequence valid for strict servers)
def _append_or_merge_user(messages, text):
    if messages and messages[-1].get("role") == "user":
        prev = messages[-1]["content"]
        if isinstance(prev, list):
            prev.append({"type": "text", "text": text})
            messages[-1]["content"] = prev
        else:
            messages[-1]["content"] = f"{prev}\n\n{text}"
    else:
        messages.append({"role": "user", "content": text})


class LLMPromptStudioSceneBuilder:
    """Two-stage scene builder: image description, then prompt generation from it.
    Stage is selected with the stage switch: '1 - describe' or '2 - compose'.

    In stage 2, if the model returns JSON missing required fields, the node re-asks it up
    to `max_field_retries` times for a complete answer; an empty `scene_name` falls back to
    `slugify(positive)`, and empty `positive`/`negative` raise an error."""
    CATEGORY = "LLM Prompt Studio"
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage": (["1 - describe", "2 - compose"],),
                "image": ("IMAGE",),
                "server_url": ("STRING", {"default": "http://localhost:1234/v1"}),
                "api_key": ("STRING", {"default": ""}),
                "model": (combo_models(),),
                "context_length": ("INT", {"default": 16384, "min": 512, "max": 131072, "step": 512}),
                "gpu_offload": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "describe_prompt": ("STRING", {"multiline": True,
                                                "default": DEFAULT_DESCRIBE}),
                "composer_prompt": ("STRING", {"multiline": True,
                                               "default": DEFAULT_COMPOSER}),
                "user_changes": ("STRING", {"multiline": True, "default": ""}),
                "image_max_size": ("INT", {"default": 1024, "min": 256,
                                           "max": 2048, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0,
                                           "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "max_field_retries": ("INT", {"default": 2, "min": 0, "max": 5}),
                "vision_check": ("BOOLEAN", {"default": True}),
                "description_view": ("STRING", {"multiline": True, "default": ""}),
                "prompt_mode": (["auto", "standard", "no_negative"], {"default": "auto"}),
            },
            "optional": {
                "family": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "scene_name", "prompt_view", "description")

    def execute(self, stage, image, server_url, api_key, model, context_length, gpu_offload,
                 describe_prompt, composer_prompt, user_changes, image_max_size, temperature,
                 max_tokens, max_field_retries=2, vision_check=True, description_view="",
                 prompt_mode="auto", family=""):
        if model.startswith("—"):
            raise RuntimeError(
                "No model selected. Start the LM Studio server, load a model "
                "and press the Refresh button on the node.")
        if vision_check and not looks_like_vision(model):
            raise RuntimeError(
                f"Model '{model}' does not look like a vision model. For image analysis "
                "choose a vision-capable model (Qwen2.5-VL, LLaVA, Gemma-3, etc.) "
                "or disable the vision_check option.")

        ensure_model_loaded(f"{server_url}::scene", server_url, api_key, model,
                            context_length, gpu_offload)

        # Stage 1: describe the image (no JSON here, plain text from the model)
        if stage.startswith("1"):
            b64 = image_to_base64(image, image_max_size)
            messages = [
                {"role": "system", "content": describe_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]},
            ]
            raw = chat_completion(server_url, api_key, model, messages,
                                  temperature, max_tokens)
            description = raw.strip()
            return {"ui": {"description": [description]},
                    "result": ("", "", "", "", description)}

        # Stage 2: compose the prompt from the description
        if not description_view.strip():
            raise RuntimeError(
                "The description field is empty — run stage 1 (Describe) "
                "on this workflow first.")

        # Resolve the effective mode (same semantics as the Writer).
        if prompt_mode == "no_negative":
            no_negative = True
        elif prompt_mode == "standard":
            no_negative = False
        else:
            no_negative = is_no_negative_family(family)
        logger.info("Scene Builder prompt mode: %s (family=%r) -> no_negative=%s",
                    prompt_mode, family, no_negative)

        # Mirror the Writer: only switch to the no-negative composer when we are actually in
        # no-negative mode; otherwise keep the standard composer verbatim so a real negative
        # is still produced and required-field validation (require_negative) does not fail.
        if no_negative:
            effective_composer = composer_prompt if composer_prompt != DEFAULT_COMPOSER \
                else DEFAULT_COMPOSER_NO_NEGATIVE
        else:
            effective_composer = composer_prompt

        user_text = f"Scene description:\n{description_view.strip()}\n\n"
        if user_changes.strip():
            user_text += f"User's requested changes to the scene:\n{user_changes.strip()}\n\n"
        else:
            user_text += "No changes requested — carry the scene over as is.\n\n"
        user_text += ("Compose a prompt for SDXL and a scene name, "
                      "respond strictly in the required JSON format.")

        messages = [
            {"role": "system", "content": effective_composer},
            {"role": "user", "content": user_text},
        ]
        raw = chat_completion(server_url, api_key, model, messages,
                                temperature, max_tokens)
        parsed = parse_prompt_json(raw)

        # Field-retry: if the model omitted required JSON fields, re-ask it (up to
        # max_field_retries times) for a complete answer before falling back. In no-negative
        # mode the negative is intentionally empty, so it is not treated as missing.
        attempt = 0
        while attempt < max_field_retries:
            missing = find_missing_fields(
                parsed, require_face=False, require_negative=not no_negative)
            if not missing:
                break
            attempt += 1
            logger.info("Scene Builder field retry %d/%d: missing %s",
                        attempt, max_field_retries, ", ".join(missing))
            _append_or_merge_user(messages,
                f"You omitted the required JSON field(s): {', '.join(missing)}. "
                f"Respond again with a COMPLETE JSON object containing ALL required fields.")
            raw_new = chat_completion(server_url, api_key, model, messages,
                                       temperature, max_tokens)
            raw = (f"[FIELD RETRY {attempt}/{max_field_retries}: "
                   f"missing {', '.join(missing)}]\n{raw_new}")
            parsed = parse_prompt_json(raw_new)

        positive, negative, scene_name, _fp, _fn = parsed

        # In no-negative mode the negative is forced empty (inert at CFG~1).
        if no_negative:
            negative = ""

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

        return {"ui": {"prompt_view": [positive]},
                "result": (positive, negative, scene_name, positive, "")}