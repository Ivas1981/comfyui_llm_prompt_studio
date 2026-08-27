# QA Checklist — ComfyUI LLM Prompt Studio (v1.1.0)

Manual / smoke checks for the major features. Automated coverage lives in `pytest tests/ -v`.
Tested against ComfyUI 0.19.3 (classic).

## Phase 1 — Debug logging
- [ ] `debug.py` `DEBUG_LEVEL = "OFF"` (default): no `llm_prompt_studio.log` file is created even
      after running nodes.
- [ ] `DEBUG_LEVEL = "MINIMAL"`: log contains `NODE_ENTER` / `NODE_EXIT` / `NODE_ERROR` for each
      node `execute()`.
- [ ] `DEBUG_LEVEL = "FULL"`: log additionally contains `HTTP_REQUEST` / `HTTP_RESPONSE` /
      `PARSE_ATTEMPT`.
- [ ] An `Authorization: Bearer <token>` header in the log shows as `Bearer ***...<last4>`.
- [ ] A `base64` image blob in inputs is logged as `<base64 N chars>`, never the raw data.
- [ ] The log rotates when it reaches `LOG_MAX_SIZE_MB`.

## Phase 2 — LM Studio v1 API
- [ ] Model loads via the native `/api/v1/models/load` endpoint (flash attention / KV-cache GPU
      offload honored when supported); legacy endpoints are used as fallback.
- [ ] `reasoning` ("low".."on") is honored on a thinking model (via the v1 `/api/v1/chat` path).
- [ ] `repeat_penalty` visibly affects output when set away from 1.0.
- [ ] `instance_id` from a successful load is stored for later use.
- [ ] Backward compatible: defaults reproduce the old OpenAI `/chat/completions` behavior.

## Phase 3 — Style presets
- [ ] First node run creates `llm_prompt_studio_presets.json` from `presets_default.json`.
- [ ] 51 presets are listed in the `style_preset` combobox.
- [ ] Selecting a preset appends its style tags to `positive`/`negative`; in no-negative mode the
      negative tags are skipped.
- [ ] A preset with `disabled_in_no_negative_mode` is skipped automatically in no-negative mode.
- [ ] "Reload Styles" refreshes the `style_preset` combo after editing a `Styles/` file.
- [ ] "Reset Styles" restores the shipped Styles/; "Copy styles path" puts the folder path on the
      clipboard.
- [ ] A broken `presets` JSON file falls back to defaults without crashing.
- [ ] Selecting a preset overrides the system prompt only when the system-prompt widget is at its
      default value.

## Phase 4 — Tests
- [ ] `python -m pytest tests/ -v` is green (CI matrix py3.10 / 3.11).
- [ ] `test_debug.py`, `test_presets.py`, `test_lm_http_stream.py`, `test_parsing_property.py`
      all present and passing. `test_stream_ws_push.py` must NOT exist (streaming removed).
- [ ] `node --check` passes on `web/js/llm_prompt_studio_actions.js` and
      `web/js/llm_prompt_studio_bridge.js`.

## Phase 5 — Model profiles, Advanced settings & streaming removal
- [ ] `load_model_profile` combobox is always present (default `auto`) on Writer / Image Critic /
      Scene Builder, right under `model`.
- [ ] «⚙ Advanced settings» button is located reliably (element refs + DOM fallback) on ComfyUI
      0.19.3; clicking toggles `context_length` / `gpu_offload` / `flash_attention` /
      `offload_kv_cache_to_gpu`.
- [ ] Collapsing / expanding Advanced settings keeps widget values and does not shrink the node
      width.
- [ ] With `load_model_profile != custom`, the sampling widgets are overridden and the node logs
      `profile '<name>' overrides widget sampling params`.
- [ ] Model-size heuristic `(\d+(?:\.\d+)?)b` resolves real model ids (e.g. `qwen2.5-14b-instruct`
      → 14B); an unrecognized size falls back to `baseline` + log.
- [ ] No `stream` / `generation_view` / live-token UI remains; generation returns the full result
       once at completion.

## Phase 6 — VRAM release
- [ ] `release_vram_after_run` boolean widget (default `true`) is the **last** `optional` key on
      Writer / Image Critic / Scene Builder and appears inside ⚙ Advanced settings.
- [ ] A full run Writer → Smart Loader → Multi-Clip → KSampler Hires Fix → Face Detailer → Smart
      Save completes; the console shows LLM release free-VRAM lines (`before-llm-release` /
      `after-llm-release`) with free VRAM rising by roughly the LLM size at the release point, and
      the ComfyUI checkpoint stays loaded throughout (no `after-comfy-unload` line anymore).
- [ ] The LLM model disappears from LM Studio before the KSampler pass and reloads for the next LLM
      node (watch `nvidia-smi -l 1` or LM Studio's model panel).
- [ ] `release_vram_after_run = false` keeps the model loaded through the sampler pass (proves the
      widget and the sampler-side skip both work).
- [ ] `LLM_PROMPT_STUDIO_KEEP_MODEL_LOADED=1` (restart ComfyUI) disables all release.
- [ ] A LAN/remote `server_url` still releases the remote LM Studio model, while the ComfyUI
      checkpoint is never evicted (the plugin no longer touches diffusion VRAM).
- [ ] Stopping LM Studio mid-run so the unload POST fails → the node logs a warning and finishes, not
      raises.
- [ ] An Image Critic after the sampler loads its vision model successfully (previously OOM / CPU
      fallback).
- [ ] A saved workflow with stray trailing `widgets_values` still reads `release_vram_after_run` as
      **true** (Python coercion + JS `configure` guard), and `system_prompt` / `idea` / `seed` /
      `server_status` are unchanged.
- [ ] `python -m pytest tests/test_vram.py tests/test_vram_nodes.py -v` is green; `node --check`
      passes on both JS files.
