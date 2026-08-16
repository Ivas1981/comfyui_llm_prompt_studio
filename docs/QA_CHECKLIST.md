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
- [ ] 14 presets are listed in the `style_preset` combobox.
- [ ] Selecting a preset appends its style tags to `positive`/`negative`; in no-negative mode the
      negative tags are skipped.
- [ ] A preset with `disabled_in_no_negative_mode` is skipped automatically in no-negative mode.
- [ ] "Reload presets" refreshes the combo after editing the user file.
- [ ] "Reset to defaults" restores the shipped presets; "Copy presets path" puts the path on the
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
