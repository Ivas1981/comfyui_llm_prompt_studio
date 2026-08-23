import logging
import time
import traceback

from ..combos import combo_models
from ..imaging import image_to_base64
from ..lm_http import (chat_completion, ensure_model_loaded, resolve_vision,
                       model_architecture, model_param_count)
from ..model_meta import is_no_negative_family, is_no_negative_architecture
from ..parsing import find_missing_fields, parse_prompt_json, slugify
from ..presets import get_architecture_guidance, append_negative_tags
from ..debug import log_node_enter, log_node_exit, log_error
from ..vram import (release_after_llm, release_enabled,
                     mark_keep_loaded, coerce_bool_widget)
from ._defaults import DEFAULT_COMPOSER, DEFAULT_COMPOSER_NO_NEGATIVE, DEFAULT_DESCRIBE
from .model_recommendations import resolve_profile

logger = logging.getLogger("llm_prompt_studio")

# Hard ceiling for max_tokens so the field-retry growth below can never exceed the
# node widget's max (8192) and blow up the request.
_MAX_TOKENS_CAP = 8192

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
                "load_model_profile": (["auto", "baseline", "structured", "creative", "strict", "custom"],
                                       {"default": "auto",
                                         "tooltip": "auto = recommended profile from a universal model-size heuristic (no benchmark list)"}),
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
                "flash_attention": ("BOOLEAN", {"default": False,
                                    "tooltip": "Enable Flash Attention for faster generation and lower VRAM usage"}),
                "offload_kv_cache_to_gpu": ("BOOLEAN", {"default": True,
                                          "tooltip": "Store KV cache in GPU memory (faster) vs CPU RAM (lower VRAM)"}),
                "reasoning": (["off", "low", "medium", "high", "on"], {"default": "off",
                             "tooltip": "Reasoning level for thinking models. Applied only when "
                                        "load_model_profile = custom; otherwise the chosen profile "
                                        "overrides it."}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.05,
                                     "tooltip": "Penalty for repeating tokens (1.0 = off). Applied "
                                                "only when load_model_profile = custom; otherwise "
                                                "the chosen profile overrides it."}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 200, "step": 1,
                          "tooltip": "Top-k sampling (0 = disabled). Applied only when "
                                     "load_model_profile = custom; otherwise the chosen profile "
                                     "overrides it."}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                          "tooltip": "Nucleus sampling (1.0 = off). Applied only when "
                                     "load_model_profile = custom; otherwise the chosen profile "
                                     "overrides it."}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                           "tooltip": "Minimum probability floor (0.0 = off). Applied only when "
                                      "load_model_profile = custom; otherwise the chosen profile "
                                      "overrides it."}),
            },
            "optional": {
                "family": ("STRING", {"default": ""}),
                "architecture": ("STRING", {"default": "",
                                            "tooltip": "Base architecture detected by Smart "
                                                       "Loader's detected_architecture output; "
                                                       "adapts token style and negatives (stage 2)"}),
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

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "scene_name", "prompt_view", "description")

    def execute(self, stage, image, server_url, api_key, model, context_length, gpu_offload,
                  describe_prompt, composer_prompt, user_changes, image_max_size, temperature,
                  max_tokens, max_field_retries=2, vision_check=True, description_view="",
                   prompt_mode="auto", family="", unique_id=None,
                   flash_attention=None, offload_kv_cache_to_gpu=None, reasoning="off",
                    repeat_penalty=1.0, top_k=0, top_p=1.0, min_p=0.0,
                      load_model_profile="auto", server_status="", architecture="",
                   release_vram_after_run=True):
        _t0 = time.time()
        log_node_enter("Scene Builder", unique_id, {
            "stage": stage, "server_url": server_url, "model": model,
            "image_max_size": image_max_size, "temperature": temperature,
             "max_tokens": max_tokens, "prompt_mode": prompt_mode, "family": family,
             "reasoning": reasoning,
        })
        try:
            return self._run(stage, image, server_url, api_key, model, context_length,
                             gpu_offload, describe_prompt, composer_prompt, user_changes,
                             image_max_size, temperature, max_tokens, max_field_retries,
                             vision_check, description_view, prompt_mode, family, unique_id, _t0,
                              flash_attention, offload_kv_cache_to_gpu, reasoning,
                               repeat_penalty, top_k, top_p, min_p,
                                load_model_profile, architecture, release_vram_after_run)
        except Exception as e:
            log_error(unique_id, e, traceback.format_exc())
            raise

    def _run(self, stage, image, server_url, api_key, model, context_length, gpu_offload,
               describe_prompt, composer_prompt, user_changes, image_max_size, temperature,
               max_tokens, max_field_retries, vision_check, description_view, prompt_mode, family,
               unique_id, _t0, flash_attention, offload_kv_cache_to_gpu, reasoning,
                repeat_penalty, top_k, top_p, min_p,
                   load_model_profile="auto", architecture="",
                   release_vram_after_run=True):
        if model.startswith("—"):
            raise RuntimeError(
                "No model selected. Start the LM Studio server, load a model "
                "and press the Refresh button on the node.")
        # Stage 1 ("describe") sends an image; stage 2 ("compose") only consumes the
        # pre-computed text description. This drives whether the vision check applies.
        _stage_is_describe = stage.startswith("1")
        _has_image = _stage_is_describe

        # The vision check only matters for stage 1 (describe), which actually sends an
        # image. Stage 2 (compose) consumes only the pre-computed text description and never
        # touches the image, so a non-vision model must still be allowed there.
        if vision_check and _has_image and not resolve_vision(server_url, api_key, model):
            raise RuntimeError(
                f"Model '{model}' is not a vision model (the server reports it does not "
                "support image inputs). For image analysis choose a vision-capable model "
                "(Qwen2.5-VL, LLaVA, Gemma-3/4, etc.) or disable the vision_check option.")

        slot = f"{server_url}::scene"
        release = coerce_bool_widget(release_vram_after_run, True)
        loaded = False
        try:
            loaded = ensure_model_loaded(f"{server_url}::scene", server_url, api_key, model,
                                 context_length, gpu_offload,
                                 flash_attention=flash_attention,
                                 offload_kv_cache_to_gpu=offload_kv_cache_to_gpu)
            if not loaded:
                raise RuntimeError(
                    f"Model '{model}' could not be loaded into LM Studio. Start the "
                    "server, load the model, and press Refresh on the node.")
    
            # v1-native sampling params: convert "off"/default widget values to None so the
            # default call stays on the OpenAI-compatible path (backward compatible).
            top_k_v = top_k if top_k and top_k > 0 else None
            top_p_v = top_p if (top_p is not None and 0.0 < top_p < 1.0) else None
            min_p_v = min_p if (min_p is not None and min_p > 0.0) else None
            repeat_penalty_v = repeat_penalty if repeat_penalty != 1.0 else None
    
            # Resolve the "load_model_profile" choice. Stage 1 (describe) is prose with an
            # image, so its kind is "describe" (structured never applies); stage 2 (compose) is
            # the JSON writer contract. "custom" keeps the widget-derived values above.
            _kind = "describe" if _stage_is_describe else "compose"
            # B4: pull the loaded LLM model's architecture / param count from LM Studio's
            # native API to drive the auto-profile (e.g. Gemma -> top_k=64). Skipped for
            # custom mode and never raises.
            llm_architecture = ""
            llm_param_count = None
            if load_model_profile != "custom":
                try:
                    llm_architecture = model_architecture(
                        server_url, api_key, model) or ""
                    llm_param_count = model_param_count(server_url, api_key, model)
                except Exception as e:  # noqa: BLE001 — API auto-profile is best-effort
                    logger.debug("Scene Builder LLM auto-profile probe failed: %s", e)
            _resolved = resolve_profile(load_model_profile, model, _kind,
                                       has_image=_has_image, architecture=llm_architecture,
                                       param_count=llm_param_count)
            if _resolved["params"] is not None:
                logger.info("Scene Builder node %s: profile '%s' overrides widget sampling params",
                            unique_id, _resolved["profile"])
                temperature = _resolved["params"]["temperature"]
                top_p_v = _resolved["params"]["top_p"]
                top_k_v = _resolved["params"]["top_k"]
                min_p_v = _resolved["params"]["min_p"]
                repeat_penalty_v = _resolved["params"]["repeat_penalty"]
                presence_penalty_v = _resolved["params"]["presence_penalty"]
                reasoning = _resolved["params"]["reasoning"]
                response_format = _resolved["response_format"]
            else:
                presence_penalty_v = None
                response_format = None
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
                                        temperature, max_tokens,
                                        reasoning=reasoning, repeat_penalty=repeat_penalty_v,
                                        top_k=top_k_v, top_p=top_p_v, min_p=min_p_v,
                                        presence_penalty=presence_penalty_v,
                                        response_format=response_format)
                description = raw.strip()
                log_node_exit("Scene Builder", unique_id, {"stage": 1, "desc_len": len(description)},
                              (time.time() - _t0) * 1000)
                return {"ui": {"description": [description]},
                        "result": ("", "", "", description, description)}
    
            # Stage 2: compose the prompt from the description
            if not description_view.strip():
                raise RuntimeError(
                    "The description field is empty — run stage 1 (Describe) "
                    "on this workflow first.")
    
            # Architecture adaptation lookups (computed BEFORE the no-negative resolution so an
            # arch-level force_no_negative can flip the mode). SDXL guidance is empty, so SDXL
            # output is unchanged when `architecture` is empty/unwired.
            arch = (architecture or "").strip().lower()
            arch_guidance = get_architecture_guidance().get(arch, {}) if arch else {}
    
            # Resolve the effective mode (same semantics as the Writer).
            if prompt_mode == "no_negative":
                no_negative = True
            elif prompt_mode == "standard":
                no_negative = False
            else:
                no_negative = is_no_negative_family(family) or is_no_negative_architecture(architecture)
            if not no_negative and arch_guidance.get("force_no_negative"):
                no_negative = True
            logger.info("Scene Builder prompt mode: %s (family=%r, arch=%r) -> no_negative=%s",
                         prompt_mode, family, architecture, no_negative)
    
            # Mirror the Writer: only switch to the no-negative composer when we are actually in
            # no-negative mode; otherwise keep the standard composer verbatim so a real negative
            # is still produced and required-field validation (require_negative) does not fail.
            if no_negative:
                effective_composer = composer_prompt if composer_prompt != DEFAULT_COMPOSER \
                    else DEFAULT_COMPOSER_NO_NEGATIVE
            else:
                effective_composer = composer_prompt
    
            # Architecture adaptation (stage 2 only): append architecture-specific guidance to the
            # composer prompt. SDXL guidance is empty, so SDXL output is unchanged when `architecture`
            # is empty/unwired. Flux/SD3 force no-negative via is_no_negative_architecture above.
            if arch_guidance:
                addendum = arch_guidance.get("system_addendum", "")
                if addendum:
                    effective_composer = effective_composer + "\n\n" + addendum
    
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
                                     temperature, max_tokens,
                                     reasoning=reasoning, repeat_penalty=repeat_penalty_v,
                                     top_k=top_k_v, top_p=top_p_v, min_p=min_p_v,
                                     presence_penalty=presence_penalty_v,
                                      response_format=response_format)
            parsed = parse_prompt_json(raw, allow_plain_text_fallback=False)
    
            # Field-retry: if the model omitted required JSON fields, re-ask it (up to
            # max_field_retries times) for a complete answer before falling back. In no-negative
            # mode the negative is intentionally empty, so it is not treated as missing.
            attempt = 0
            cur_max_tokens = max_tokens
            while attempt < max_field_retries:
                missing = find_missing_fields(
                    parsed, require_face=False, require_negative=not no_negative)
                if not missing:
                    break
                attempt += 1
                # Grow the token budget on each retry so a prompt that was truncated
                # (and therefore missing fields) gets room to complete. Capped at the
                # widget maximum so it can never exceed what the UI allows.
                cur_max_tokens = min(int(cur_max_tokens * 1.25), _MAX_TOKENS_CAP)
                logger.info("Scene Builder field retry %d/%d: missing %s (max_tokens=%d)",
                            attempt, max_field_retries, ", ".join(missing), cur_max_tokens)
                _append_or_merge_user(messages,
                    f"You omitted the required JSON field(s): {', '.join(missing)}. "
                    f"Respond again with a COMPLETE JSON object containing ALL required fields.")
                raw_new = chat_completion(server_url, api_key, model, messages,
                                               temperature, cur_max_tokens,
                                               reasoning=reasoning, repeat_penalty=repeat_penalty_v,
                                               top_k=top_k_v, top_p=top_p_v, min_p=min_p_v,
                                               presence_penalty=presence_penalty_v,
                                               response_format=response_format)
                raw = (f"[FIELD RETRY {attempt}/{max_field_retries}: "
                        f"missing {', '.join(missing)}]\n{raw_new}")
                parsed = parse_prompt_json(raw_new, allow_plain_text_fallback=False)
    
            positive, negative, scene_name, _fp, _fn = parsed
    
            # In no-negative mode the negative is forced empty (inert at CFG~1).
            if no_negative:
                negative = ""
    
            # Architecture default negatives (standard mode only; inert when no_negative already
            # forced the negative empty above). Deduped against the generated negative.
            if arch and not no_negative and arch_guidance:
                negative = append_negative_tags(negative, arch_guidance.get("default_negative", ""))
    
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
    
            log_node_exit("Scene Builder", unique_id,
                          {"stage": 2, "positive": positive[:200], "negative": negative[:200],
                           "scene_name": scene_name[:200]}, (time.time() - _t0) * 1000)
            return {"ui": {"prompt_view": [positive]},
                    "result": (positive, negative, scene_name, positive, "")}
        finally:
            if loaded and release and release_enabled():
                mark_keep_loaded(server_url, False)
                release_after_llm(slot, server_url, api_key, "scene")
            elif loaded:
                mark_keep_loaded(server_url, True)