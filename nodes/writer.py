import logging
import time
import traceback

from ..combos import combo_models
from ..lm_http import (chat_completion, ensure_model_loaded,
                       model_architecture, model_param_count)
from ..model_meta import is_no_negative_family, is_no_negative_architecture
from ..parsing import find_missing_fields, parse_prompt_json, slugify
from ..presets import (apply_preset_to_prompts, get_preset_by_name,
                       get_architecture_guidance, append_negative_tags)
from .model_recommendations import resolve_profile
from ..debug import log_node_enter, log_node_exit, log_error
from ..vram import (release_after_llm, release_enabled,
                     mark_keep_loaded, coerce_bool_widget)
from ._defaults import (DEFAULT_SYSTEM, DEFAULT_SYSTEM_NO_NEGATIVE,
                        FACE_PROMPT_INSTRUCTION, FACE_PROMPT_INSTRUCTION_NO_NEGATIVE,
                        REASONING_HINT)

logger = logging.getLogger("llm_prompt_studio")

# Hard ceiling for max_tokens so the field-retry growth below can never exceed the
# node widget's max (8192) and blow up the request.
_MAX_TOKENS_CAP = 8192

# Helper: append a user message, or merge into the last user message if one already exists.
def _append_or_merge_user(messages, text):
    if messages and messages[-1].get("role") == "user":
        prev = messages[-1]["content"]
        # If previous content is a multimodal list, add a text part
        if isinstance(prev, list):
            prev.append({"type": "text", "text": text})
            messages[-1]["content"] = prev
        else:
            messages[-1]["content"] = f"{prev}\n\n{text}"
    else:
        messages.append({"role": "user", "content": text})

# Cache of the last generated prompt per node instance: {unique_id: result_tuple}
_PROMPT_CACHE_MAX = 128
_prompt_cache = {}


def _preset_names():
    """Combo options for the style_preset widget (with a no-op default first)."""
    try:
        from ..presets import get_preset_names
        return ["— none —"] + list(get_preset_names())
    except Exception:
        return ["— none —"]


class LLMPromptStudioWriter:
    """Generates an SDXL prompt from an idea.

    If the model returns a JSON answer missing required fields, the node re-asks it up to
    `max_field_retries` times for a complete answer. Empty `positive`/`negative` after
    retries raise an error (in no-negative mode only `positive` is required); an empty
    `scene_name` falls back to `slugify(positive)`; empty face fields fall back to the main
    prompts.

    `prompt_mode` selects the negative-handling strategy:
    - `auto` (default): switches to the no-negative prompt when the detected checkpoint
      family is a distilled one (DMD/LCM/Turbo/Hyper/Lightning/Flash/Schnell/TCD/PCM).
      Wire the Smart Loader's `detected_family` output into the `family` input to enable
      detection (the full family set is defined in `model_meta.FAMILY_MARKERS`).
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
                "load_model_profile": (["auto", "baseline", "structured", "creative", "strict", "custom"],
                                       {"default": "auto",
                                         "tooltip": "auto = recommended profile from a universal model-size heuristic (no benchmark list)"}),
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
                "style_preset": (_preset_names(), {"default": "— none —",
                                  "tooltip": "Apply a style preset's system prompt and style tags"}),
                "use_preset_system_prompt": ("BOOLEAN", {"default": True,
                                  "tooltip": "When a style preset is selected, override the "
                                             "system prompt with the preset's (unless you have "
                                             "customized it). Turn off to keep your own system "
                                             "prompt even with a preset loaded."}),
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
                                                       "adapts token style and negatives"}),
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

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "raw", "scene_name",
                    "face_positive", "face_negative")
    FUNCTION = "execute"

    def execute(self, server_url, api_key, model, context_length, gpu_offload, system_prompt, idea,
                 revision_notes, temperature, max_tokens, seed,
                 reuse_last_prompt=False, generate_face_prompts=False,
                 max_field_retries=2, face_prompt_instruction="",
                 prompt_mode="auto", family="", unique_id=None,
                 style_preset="— none —", use_preset_system_prompt=True, flash_attention=None,
                  offload_kv_cache_to_gpu=None, reasoning="off", repeat_penalty=1.0,
                   top_k=0, top_p=1.0, min_p=0.0, load_model_profile="auto",
                     server_status="", architecture="", release_vram_after_run=True):
        _t0 = time.time()
        log_node_enter("Writer", unique_id, {
            "server_url": server_url, "model": model, "idea": idea,
            "prompt_mode": prompt_mode, "family": family,
            "architecture": architecture,
            "generate_face_prompts": generate_face_prompts,
            "temperature": temperature, "max_tokens": max_tokens, "seed": seed,
             "reuse_last_prompt": reuse_last_prompt, "max_field_retries": max_field_retries,
             "style_preset": style_preset, "reasoning": reasoning,
        })
        try:
            return self._run(server_url, api_key, model, context_length,
                             gpu_offload, system_prompt,
                             idea, revision_notes, temperature, max_tokens, seed,
                             reuse_last_prompt, generate_face_prompts, max_field_retries,
                             face_prompt_instruction, prompt_mode, family, unique_id, _t0,
                              style_preset, use_preset_system_prompt, flash_attention,
                              offload_kv_cache_to_gpu, reasoning,
                repeat_penalty, top_k, top_p, min_p,
                  load_model_profile, architecture, release_vram_after_run)
        except Exception as e:
            log_error(unique_id, e, traceback.format_exc())
            raise

    def _run(self, server_url, api_key, model, context_length, gpu_offload, system_prompt, idea,
              revision_notes, temperature, max_tokens, seed,
              reuse_last_prompt, generate_face_prompts, max_field_retries,
              face_prompt_instruction, prompt_mode, family, unique_id, _t0,
              style_preset, use_preset_system_prompt=True, flash_attention=None,
              offload_kv_cache_to_gpu=None, reasoning="off",
                repeat_penalty=1.0, top_k=0, top_p=1.0, min_p=0.0,
                  load_model_profile="auto", architecture="",
                  release_vram_after_run=True):
        # Reuse mode: return the cached prompt without calling the LLM. The key is the node
        # id + the inputs that actually shape the generated prompt, so a change to any of
        # them bypasses the cache and regenerates. `family` is intentionally excluded: it is
        # driven by the loaded checkpoint, and with reuse on we want the same prompts to
        # carry over when the user swaps to a different checkpoint. `architecture` IS included
        # because it changes token style, negatives and (for Flux/SD3) the no-negative path,
        # so two architectures must never share a cached prompt.
        cache_key = (unique_id, prompt_mode, style_preset, system_prompt,
                     idea, revision_notes, generate_face_prompts, face_prompt_instruction,
                     architecture)
        # Reuse mode: return cached result without calling the LLM
        if reuse_last_prompt:
            cached = _prompt_cache.get(cache_key)
            if cached:
                log_node_exit("Writer", unique_id,
                              {"positive": cached[0][:200], "negative": cached[1][:200],
                               "scene_name": cached[3][:200]}, (time.time() - _t0) * 1000)
                return cached

        if model.startswith("—"):
            raise RuntimeError(
                "No model selected. Start the LM Studio server, load a model "
                "and press the Refresh button on the node.")

        # Resolve the selected style preset (the no-negative opt-out is applied below,
        # once the effective mode is known).
        preset = None
        if style_preset and style_preset != "— none —":
            preset = get_preset_by_name(style_preset)

        slot = f"{server_url}::writer"
        release = coerce_bool_widget(release_vram_after_run, True)
        loaded = False
        try:
            loaded = ensure_model_loaded(f"{server_url}::writer", server_url, api_key, model,
                                context_length, gpu_offload,
                                flash_attention=flash_attention,
                                offload_kv_cache_to_gpu=offload_kv_cache_to_gpu)
            if not loaded:
                raise RuntimeError(
                    f"Model '{model}' could not be loaded into LM Studio. Start the "
                    "server, load the model, and press Refresh on the node.")
    
            # Architecture adaptation lookups (computed BEFORE the no-negative resolution so an
            # arch-level ``force_no_negative`` can flip the mode; see below). SDXL guidance is
            # empty, so SDXL generation is unchanged when `architecture` is empty/unwired.
            arch = (architecture or "").strip().lower()
            arch_guidance = get_architecture_guidance().get(arch, {}) if arch else {}
    
            # Resolve the effective mode. 'no_negative' forces it; 'standard' forbids it;
            # 'auto' (and anything unexpected) defers to the detected checkpoint family.
            if prompt_mode == "no_negative":
                no_negative = True
            elif prompt_mode == "standard":
                no_negative = False
            else:
                no_negative = is_no_negative_family(family) or is_no_negative_architecture(architecture)
            # An architecture may declare force_no_negative (e.g. Flux/SD3) to force the empty
            # negative path even when the family is not itself a no-negative one.
            if not no_negative and arch_guidance.get("force_no_negative"):
                no_negative = True
            logger.info("Writer node %s prompt mode: %s (family=%r, arch=%r) -> no_negative=%s",
                         unique_id, prompt_mode, family, architecture, no_negative)
    
            # Drop a preset that opts out of no-negative mode, then optionally override the
            # system prompt with the preset's when the user hasn't customized it.
            if preset and no_negative and preset.get("disabled_in_no_negative_mode"):
                logger.info("Preset '%s' is disabled in no-negative mode; skipping.",
                            preset.get("name"))
                preset = None
    
            # System prompt selection. A user-edited widget value overrides the built-in
            # no-negative default; otherwise we use the dedicated no-negative prompt.
            if no_negative:
                effective_system = system_prompt if system_prompt != DEFAULT_SYSTEM \
                    else DEFAULT_SYSTEM_NO_NEGATIVE
            else:
                effective_system = system_prompt
            if generate_face_prompts:
                inst = face_prompt_instruction.strip() if face_prompt_instruction else ""
                if not inst:
                    inst = FACE_PROMPT_INSTRUCTION_NO_NEGATIVE if no_negative \
                        else FACE_PROMPT_INSTRUCTION
                effective_system = effective_system + inst
    
            # Override the system prompt with the preset's only when the user left it at default.
            # In no-negative mode we prefer the preset's dedicated no-negative variant so the
            # negative is correctly required to be empty; otherwise its standard variant is used.
            if preset and use_preset_system_prompt:
                if no_negative and preset.get("system_prompt_no_negative"):
                    effective_system = preset["system_prompt_no_negative"]
                else:
                    effective_system = preset.get("system_prompt") or effective_system
                # Presets embed their own short reasoning hint; if it is absent for some reason,
                # make sure the canonical reasoning hint is still present.
                if REASONING_HINT and "ALWAYS finish your reply with the complete JSON object" \
                        not in effective_system:
                    effective_system = effective_system + REASONING_HINT
    
            # Architecture adaptation: append architecture-specific guidance to the system prompt
            # (only when a detected architecture is wired in). SDXL guidance is empty, so SDXL
            # generation is unchanged when `architecture` is empty/unwired. A Flux/SD3 architecture
            # also forces no-negative via is_no_negative_architecture above.
            if arch_guidance:
                addendum = arch_guidance.get("system_addendum", "")
                if addendum:
                    effective_system = effective_system + "\n\n" + addendum
    
            # v1-native sampling params: convert "off"/default widget values to None so the
            # default call stays on the OpenAI-compatible path (backward compatible).
            top_k_v = top_k if top_k and top_k > 0 else None
            top_p_v = top_p if (top_p is not None and 0.0 < top_p < 1.0) else None
            min_p_v = min_p if (min_p is not None and min_p > 0.0) else None
            repeat_penalty_v = repeat_penalty if repeat_penalty != 1.0 else None
    
            # B4: pull the loaded LLM model's architecture / param count from LM Studio's
            # native API to drive the auto-profile (e.g. Gemma -> top_k=64). Skipped for
            # custom mode (the node uses widget values there) and never raises.
            llm_architecture = ""
            llm_param_count = None
            if load_model_profile != "custom":
                try:
                    llm_architecture = model_architecture(
                        server_url, api_key, model) or ""
                    llm_param_count = model_param_count(server_url, api_key, model)
                except Exception as e:  # noqa: BLE001 — API auto-profile is best-effort
                    logger.debug("Writer LLM auto-profile probe failed: %s", e)

            # Resolve the "load_model_profile" choice into concrete sampling params. "custom"
            # keeps the widget-derived values above (full backward compatibility); any other
            # choice overrides the individual sampling widgets with the recommended profile.
            _resolved = resolve_profile(load_model_profile, model, "writer",
                                       has_image=False, architecture=llm_architecture,
                                       param_count=llm_param_count)
            if _resolved["params"] is not None:
                logger.info("Writer node %s: profile '%s' overrides widget sampling params",
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
            logger.debug("Writer node %s effective sampling: profile=%s temperature=%s top_p=%s "
                         "top_k=%s min_p=%s repeat_penalty=%s presence_penalty=%s reasoning=%s "
                         "structured=%s",
                         unique_id, _resolved["profile"], temperature, top_p_v, top_k_v, min_p_v,
                         repeat_penalty_v, presence_penalty_v, reasoning,
                         bool(response_format))
    
            messages = [
                {"role": "system", "content": effective_system},
                {"role": "user", "content": f"Idea: {idea}"},
            ]
            if revision_notes.strip():
                _append_or_merge_user(messages, (
                    "The previous version of the prompt did not pass the critic's check. "
                    f"Requested fixes: {revision_notes}\n"
                    "Generate a CORRECTED prompt taking these fixes into account, "
                    "in the same JSON format."
                ))
    
            raw = chat_completion(server_url, api_key, model, messages,
                                    temperature, max_tokens, seed=seed,
                                    reasoning=reasoning,
                                    repeat_penalty=repeat_penalty_v, top_k=top_k_v,
                                    top_p=top_p_v, min_p=min_p_v,
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
                    parsed, require_face=generate_face_prompts, require_negative=not no_negative)
                if not missing:
                    break
                attempt += 1
                # Grow the token budget on each retry so a prompt that was truncated
                # (and therefore missing fields) gets room to complete. Capped at the
                # widget maximum so it can never exceed what the UI allows.
                cur_max_tokens = min(int(cur_max_tokens * 1.25), _MAX_TOKENS_CAP)
                logger.info("Field retry %d/%d for node %s: missing %s (max_tokens=%d)",
                            attempt, max_field_retries, unique_id, ", ".join(missing),
                            cur_max_tokens)
                _append_or_merge_user(messages,
                    f"You omitted the required JSON field(s): {', '.join(missing)}. "
                    f"Respond again with a COMPLETE JSON object containing ALL required fields.")
                raw_new = chat_completion(server_url, api_key, model, messages,
                                               temperature, cur_max_tokens, seed=seed,
                                               reasoning=reasoning, repeat_penalty=repeat_penalty_v,
                                               top_k=top_k_v, top_p=top_p_v, min_p=min_p_v,
                                               presence_penalty=presence_penalty_v,
                                               response_format=response_format)
                raw = (f"[FIELD RETRY {attempt}/{max_field_retries}: "
                       f"missing {', '.join(missing)}]\n{raw_new}")
                parsed = parse_prompt_json(raw_new, allow_plain_text_fallback=False)
    
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
    
            # Architecture default negatives (standard mode only; inert when no_negative already
            # forced the negative empty above). Deduped against the generated negative.
            if arch and not no_negative and arch_guidance:
                negative = append_negative_tags(negative, arch_guidance.get("default_negative", ""))
    
            # Append the preset's style tags to the prompts (negative tags only when not in
            # no-negative mode, where the negative is intentionally empty).
            if preset:
                positive, negative = apply_preset_to_prompts(preset, positive, negative, no_negative)
    
            result = (positive, negative, raw, scene_name, face_positive, face_negative)
            _prompt_cache[cache_key] = result
            if len(_prompt_cache) > _PROMPT_CACHE_MAX:
                _prompt_cache.pop(next(iter(_prompt_cache)))  # drop oldest
            log_node_exit("Writer", unique_id,
                          {"positive": positive[:200], "negative": negative[:200],
                           "scene_name": scene_name[:200]}, (time.time() - _t0) * 1000)
            return result
        finally:
            if loaded and release and release_enabled():
                mark_keep_loaded(server_url, False)
                release_after_llm(slot, server_url, api_key, "writer")
            elif loaded:
                mark_keep_loaded(server_url, True)