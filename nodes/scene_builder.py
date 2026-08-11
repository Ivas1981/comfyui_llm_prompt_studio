from ..combos import combo_models
from ..imaging import image_to_base64
from ..lm_http import chat_completion, ensure_model_loaded, looks_like_vision
from ..parsing import parse_prompt_json
from ._defaults import DEFAULT_COMPOSER, DEFAULT_DESCRIBE


class LLMPromptStudioSceneBuilder:
    """Two-stage scene builder: image description, then prompt generation from it.
    Stage is selected with the stage switch: '1 - describe' or '2 - compose'."""
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
                "vision_check": ("BOOLEAN", {"default": True}),
                "description_view": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "scene_name", "prompt_view", "description")

    def execute(self, stage, image, server_url, api_key, model, context_length, gpu_offload,
                 describe_prompt, composer_prompt, user_changes, image_max_size, temperature,
                 max_tokens, vision_check=True, description_view=""):
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

        user_text = f"Scene description:\n{description_view.strip()}\n\n"
        if user_changes.strip():
            user_text += f"User's requested changes to the scene:\n{user_changes.strip()}\n\n"
        else:
            user_text += "No changes requested — carry the scene over as is.\n\n"
        user_text += ("Compose a prompt for SDXL and a scene name, "
                      "respond strictly in the required JSON format.")

        messages = [
            {"role": "system", "content": composer_prompt},
            {"role": "user", "content": user_text},
        ]
        raw = chat_completion(server_url, api_key, model, messages,
                              temperature, max_tokens)
        positive, negative, scene_name, _fp, _fn = parse_prompt_json(raw)

        return {"ui": {"prompt_view": [positive]},
                "result": (positive, negative, scene_name, positive, "")}