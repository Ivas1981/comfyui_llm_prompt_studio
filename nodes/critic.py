from ..combos import combo_models
from ..imaging import image_to_base64
from ..lm_http import chat_completion, ensure_model_loaded, looks_like_vision
from ..parsing import parse_critic_json
from ._defaults import DEFAULT_CRITIC


class LLMPromptStudioCritic:
    CATEGORY = "LLM Prompt Studio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"forceInput": True}),
                "server_url": ("STRING", {"default": "http://localhost:1234/v1"}),
                "api_key": ("STRING", {"default": ""}),
                "model": (combo_models(),),
                "context_length": ("INT", {"default": 16384, "min": 512, "max": 131072, "step": 512}),
                "gpu_offload": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "critic_prompt": ("STRING", {"multiline": True, "default": DEFAULT_CRITIC}),
                "threshold": ("INT", {"default": 7, "min": 0, "max": 10}),
                "image_max_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "clear_notes_on_approve": ("BOOLEAN", {"default": True}),
                "auto_loop": ("BOOLEAN", {"default": False}),
                "max_retries": ("INT", {"default": 3, "min": 1, "max": 10}),
                "vision_check": ("BOOLEAN", {"default": True}),
                "revision_view": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("approved", "score", "revision_notes", "verdict", "raw")
    FUNCTION = "execute"

    def execute(self, image, prompt, server_url, api_key, model, context_length, gpu_offload,
                 critic_prompt, threshold, image_max_size, temperature, max_tokens,
                 clear_notes_on_approve, auto_loop, max_retries,
                 vision_check=True, revision_view=""):
        if model.startswith("—"):
            raise RuntimeError(
                "No model selected. Start the LM Studio server, load a model "
                "and press the Refresh button on the node.")
        if vision_check and not looks_like_vision(model):
            raise RuntimeError(
                f"Model '{model}' does not look like a vision model. For image analysis "
                "choose a vision-capable model (Qwen2.5-VL, LLaVA, Gemma-3, etc.) "
                "or disable the vision_check option.")

        ensure_model_loaded(f"{server_url}::critic", server_url, api_key, model,
                            context_length, gpu_offload)
        b64 = image_to_base64(image, image_max_size)

        messages = [
            {"role": "system", "content": critic_prompt},
            {"role": "user", "content": [
                {"type": "text",
                 "text": f"The prompt this image was generated with: {prompt}\n"
                         f"Evaluate how well the image matches this prompt."},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]},
        ]
        raw = chat_completion(server_url, api_key, model, messages,
                              temperature, max_tokens)
        score, verdict, notes = parse_critic_json(raw)
        approved = 0 <= score and score >= threshold

        return {
            "ui": {
                "approved": [approved],
                "score": [score],
                "revision_notes": [notes],
                "auto_loop": [auto_loop],
                "max_retries": [max_retries],
                "clear_notes_on_approve": [clear_notes_on_approve],
            },
            "result": (approved, score, notes, verdict, raw),
        }