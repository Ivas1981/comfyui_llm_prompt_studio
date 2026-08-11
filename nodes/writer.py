from ..combos import combo_models
from ..lm_http import chat_completion, ensure_model_loaded
from ..parsing import parse_prompt_json
from ._defaults import DEFAULT_SYSTEM, FACE_PROMPT_INSTRUCTION

# Cache of the last generated prompt per node instance: {unique_id: result_tuple}
_PROMPT_CACHE_MAX = 128
_prompt_cache = {}


class LLMPromptStudioWriter:
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
                "face_prompt_instruction": ("STRING", {"multiline": True,
                                                        "default": FACE_PROMPT_INSTRUCTION}),
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
                 face_prompt_instruction="", unique_id=None):
        # Reuse mode: return cached result without calling the LLM
        if reuse_last_prompt:
            cached = _prompt_cache.get(unique_id)
            if cached:
                return cached

        if model.startswith("—"):
            raise RuntimeError(
                "No model selected. Start the LM Studio server, load a model "
                "and press the Refresh button on the node.")

        ensure_model_loaded(f"{server_url}::writer", server_url, api_key, model,
                            context_length, gpu_offload)

        # System prompt + face-fields instruction when face prompt generation is on
        effective_system = system_prompt
        if generate_face_prompts:
            inst = face_prompt_instruction.strip() if face_prompt_instruction else ""
            if not inst:
                inst = FACE_PROMPT_INSTRUCTION
            effective_system = system_prompt + inst

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
        positive, negative, scene_name, face_positive, face_negative = parse_prompt_json(raw)

        # Fallback: if face prompts were not generated, the regular prompts go to their outputs
        if not face_positive:
            face_positive = positive
        if not face_negative:
            face_negative = negative

        result = (positive, negative, raw, scene_name, face_positive, face_negative)
        _prompt_cache[unique_id] = result
        if len(_prompt_cache) > _PROMPT_CACHE_MAX:
            _prompt_cache.pop(next(iter(_prompt_cache)))  # drop oldest
        return result