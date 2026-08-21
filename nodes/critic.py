import logging
import time
import traceback

from ..combos import combo_models
from ..imaging import image_to_base64
from ..lm_http import chat_completion, ensure_model_loaded, resolve_vision
from ..parsing import parse_critic_json
from ..debug import log_node_enter, log_node_exit, log_error
from ..vram import (release_after_llm, release_enabled,
                     mark_keep_loaded, coerce_bool_widget)
from ._defaults import DEFAULT_CRITIC
from .model_recommendations import resolve_profile

logger = logging.getLogger("llm_prompt_studio")


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
                "load_model_profile": (["auto", "baseline", "structured", "creative", "strict", "custom"],
                                       {"default": "auto",
                                         "tooltip": "auto = recommended profile from a universal model-size heuristic (no benchmark list)"}),
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
                "flash_attention": ("BOOLEAN", {"default": False,
                                    "tooltip": "Enable Flash Attention for faster generation and lower VRAM usage"}),
                "offload_kv_cache_to_gpu": ("BOOLEAN", {"default": True,
                                          "tooltip": "Store KV cache in GPU memory (faster) vs CPU RAM (lower VRAM)"}),
            },
            "optional": {
                "server_status": ("STRING", {"default": "",
                                             "multiline": False,
                                             "tooltip": "Live LM Studio server status "
                                                        "(reachable / loaded model), refreshed automatically"}),
                "release_vram_after_run": ("BOOLEAN", {"default": True,
                                             "tooltip": "Unload the LM Studio model when this node "
                                                        "finishes so the diffusion pipeline gets the "
                                                        "VRAM back. Turn off only if you have enough "
                                                        "VRAM for the LLM and the checkpoint at the "
                                                        "same time."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("BOOLEAN", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("approved", "score", "revision_notes", "verdict", "raw")
    FUNCTION = "execute"

    def execute(self, image, prompt, server_url, api_key, model, context_length, gpu_offload,
                 critic_prompt, threshold, image_max_size, temperature, max_tokens,
                 clear_notes_on_approve, auto_loop, max_retries,
                  vision_check=True, revision_view="",
                   flash_attention=None, offload_kv_cache_to_gpu=None, unique_id=None,
                   load_model_profile="auto", server_status="",
                   release_vram_after_run=True):
        _t0 = time.time()
        log_node_enter("Critic", unique_id, {
            "server_url": server_url, "model": model, "threshold": threshold,
            "image_max_size": image_max_size, "temperature": temperature,
            "max_tokens": max_tokens, "vision_check": vision_check,
        })
        try:
            return self._run(image, prompt, server_url, api_key, model, context_length,
                             gpu_offload,
                             critic_prompt, threshold, image_max_size, temperature, max_tokens,
                             clear_notes_on_approve, auto_loop, max_retries,
                              vision_check, revision_view, flash_attention,
                              offload_kv_cache_to_gpu, unique_id, _t0, load_model_profile,
                              release_vram_after_run)
        except Exception as e:
            log_error(unique_id, e, traceback.format_exc())
            raise

    def _run(self, image, prompt, server_url, api_key, model, context_length, gpu_offload,
              critic_prompt, threshold, image_max_size, temperature, max_tokens,
              clear_notes_on_approve, auto_loop, max_retries,
               vision_check=True, revision_view="",
               flash_attention=None, offload_kv_cache_to_gpu=None, unique_id=None, _t0=None,
               load_model_profile="auto", release_vram_after_run=True):
        if model.startswith("—"):
            raise RuntimeError(
                "No model selected. Start the LM Studio server, load a model "
                "and press the Refresh button on the node.")
        if vision_check and not resolve_vision(server_url, api_key, model):
            raise RuntimeError(
                f"Model '{model}' is not a vision model (the server reports it does not "
                "support image inputs). For image analysis choose a vision-capable model "
                "(Qwen2.5-VL, LLaVA, Gemma-3/4, etc.) or disable the vision_check option.")

        slot = f"{server_url}::critic"
        release = coerce_bool_widget(release_vram_after_run, True)
        loaded = False
        try:
            ensure_model_loaded(f"{server_url}::critic", server_url, api_key, model,
                                context_length, gpu_offload,
                                flash_attention=flash_attention,
                                offload_kv_cache_to_gpu=offload_kv_cache_to_gpu)
            loaded = True
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
            # Resolve the "load_model_profile" choice. The Critic is always a vision model, so
            # `has_image=True` here and `response_format` is never produced. "custom" keeps the
            # old behavior (only temperature/max_tokens are sent); any other choice expands the
            # call with the recommended profile's sampling params.
            _resolved = resolve_profile(load_model_profile, model, "critic", has_image=True)
            if _resolved["params"] is not None:
                logger.info("Critic node %s: profile '%s' overrides widget sampling params",
                            unique_id, _resolved["profile"])
                p = _resolved["params"]
                raw = chat_completion(server_url, api_key, model, messages,
                                       p["temperature"], max_tokens,
                                       repeat_penalty=p["repeat_penalty"], top_k=p["top_k"],
                                       top_p=p["top_p"], min_p=p["min_p"],
                                       presence_penalty=p["presence_penalty"],
                                       response_format=_resolved["response_format"])
            else:
                raw = chat_completion(server_url, api_key, model, messages,
                                       temperature, max_tokens)
            score, verdict, notes = parse_critic_json(raw)
            approved = score >= threshold
    
            log_node_exit("Critic", unique_id, {"score": score, "approved": approved,
                           "notes_len": len(notes)}, (time.time() - _t0) * 1000)
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
        finally:
            if loaded and release and release_enabled():
                mark_keep_loaded(server_url, False)
                release_after_llm(slot, server_url, api_key, "critic")
            elif loaded:
                mark_keep_loaded(server_url, True)