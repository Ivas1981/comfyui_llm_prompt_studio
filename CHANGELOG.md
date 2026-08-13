# Changelog

All notable changes to **ComfyUI LLM Prompt Studio** are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [1.0.3] — 2026-08-13

### Fixed
- **Vision detection was wrong both ways (`vision_check` blocked good models and let
  bad ones through).** The name-substring heuristic `looks_like_vision` rejected genuine
  multimodal models like `gemma4-12b-uncensored-...` (Gemma-4 is multimodal) while
  passing names like `qwen2.5-vl-...` that the server then rejected with
  `HTTP 400 ... does not support image inputs`. Scene Builder and Critic now consult the
  server's authoritative `capabilities.vision` flag via the new `resolve_vision()` helper
  (which queries LM Studio's native `GET /api/v1/models`). The server answer is decisive
  in both directions, falling back to the name heuristic only when the probe is
  unavailable (old server, network error, 404) so behavior never regresses.
- **Model preload was rejected with `400 Unrecognized key(s): 'gpuOffload'`.** The native
  v1 `/api/v1/models/load` body used the camelCase key `gpuOffload` while every sibling
  v1 key is snake_case; it now sends `gpu_offload` (matching the other v1 options). The
  legacy load body keeps `gpuOffload` (camelCase) for genuinely old servers.
- **`load_model` falsely reported success after a v1 rejection.** When the native v1 load
  returned 400, the code still fell through to the legacy `/v1/models/{id}/load` endpoint,
  which this server answers `200 anyway` ("Unexpected endpoint"), so a failed load looked
  successful. `load_model` now returns `False` on a real client rejection (400–499) from
  the v1 route and only tries the legacy endpoints on a connection error or 404. The node
  still works via LM Studio JIT loading; the failure is logged clearly.
- **Raw HTTP 400s from the server were unhelpful.** `chat_completion` now enriches
  server errors via `_enrich_http_error()`: an image/vision rejection is explained as
  "does not support image inputs … pick a vision model or disable vision_check", and a
  model-load failure ("exited before becoming healthy") points at VRAM/the model file.

---

## [1.0.2] — 2026-08-13

### Fixed
- **Model preload used a malformed URL (`/v1/api/v1/models/load`) and silently did
  nothing.** `load_model` built the native LM Studio v1 load endpoint from
  `server_url` without stripping the trailing `/v1`, so with the default
  `http://localhost:1234/v1` the request went to `/v1/api/v1/models/load`. LM Studio
  returned 200 for that unknown route, so the node believed the model had loaded — but
  it hadn't. The model was then loaded lazily by LM Studio on the first `chat/completions`
  call, making the first generation slow, and the `flash_attention` /
  `offload_kv_cache_to_gpu` load options were never applied. `load_model` now strips the
  trailing `/v1` via the existing `_server_root()` helper (matching `_chat_v1` and
  `maybe_unload_old`), so the endpoint is correctly `/api/v1/models/load` and models
  preload before the first inference. `maybe_unload_old` now reuses `_server_root()` too.
- **Checkpoint family detection missed many distilled models (false negatives).**
  `detect_checkpoint_family` required a separator after the family token, so concatenated
  CamelCase names like `HyperSDXL`, `SDXLLightning`, `TurboSDXL` and `LCMXL` (and the
  `dmd2` LoRA variant) were reported as `base`, so `no_negative` mode never
  auto-activated for them. Detection is now case-aware: a family word is accepted as a
  standalone token or a CamelCase/version continuation (`HyperSDXL`, `LCMXL`, `dmd2`),
  while still rejecting markers glued into a longer lowercase word (`hypernetwork`,
  `calcium`, `flash_attention`). Versioned variants (`dmd2`) map to the `dmd` family.
- **Checkpoint family detection produced false positives from metadata free text.**
  The whole safetensors metadata JSON (including `description`/`comments`) was scanned,
  so a base model whose description said "hyper-realistic" / "lightning-fast" /
  "turbo-charged" was falsely flagged as distilled. The scan now uses only structured
  metadata fields and excludes free-text keys, so descriptions can no longer trigger a
  false family match (structured fields like `modelspec.title` are still honored).
- **Writer wasted tokens emitting face fields when not requested.** The base
  `writer_system` / `writer_system_no_negative` prompts mandated `face_positive` /
  `face_negative` in every response and showed them in the JSON example, so the model
  returned them even with `generate_face_prompts` off. The prompts now mark face fields
  as OPTIONAL — included only when face-prompt generation is requested, omitted
  otherwise (`parse_prompt_json` already defaults missing face fields to empty strings,
  and the node falls back to the main prompts).

---

## [1.0.1] — 2026-08-13

### Fixed
- **Runtime `TypeError` when running any node (Writer / Image Critic / Scene Builder).**
  The `generation_view` widget was declared in each node's `INPUT_TYPES` (added for the
  1.0.0 live-streaming feature) but the corresponding `execute` methods did not accept it,
  so ComfyUI raised `execute() got an unexpected keyword argument 'generation_view'` on every
  run. The parameter is now accepted on all three nodes (it is a display widget — streaming
  pushes tokens to it via the websocket bridge keyed by `unique_id`), so the nodes execute
  whether or not `stream` is enabled.
- **Package version was out of sync with the changelog.** `pyproject.toml` still declared
  `0.9.1` after the 1.0.0 release; it now tracks the changelog (`1.0.1`).

---

## [1.0.0] — 2026-08-13

### Added
- **Style presets** — 14 ready-made style presets (Photorealism, Cinematic, Anime, 3D Render,
  Digital Art, …) ship in `presets_default.json`. The Writer exposes a `style_preset` combobox;
  selecting one appends the preset's `style_tags_positive`/`style_tags_negative` to the generated
  prompt and (when the system prompt is left at default) applies the preset's `system_prompt`.
  Presets are copied to an editable user file (`llm_prompt_studio_presets.json` in the ComfyUI
  output dir) on first run; "Reload presets" refreshes the combo and "Reset to defaults" restores
  the shipped set. Presets that opt out of no-negative mode are skipped automatically there.
- **Debug logging** (`debug.py`) — opt-in, OFF by default. `DEBUG_LEVEL` = `MINIMAL` logs node
  enter/exit/error; `FULL` adds HTTP request/response and JSON-parse attempts. API keys and
  base64 blobs are masked/truncated; the rotating log file is only created when logging is active.
- **LM Studio native v1 API support** — `lm_http.load_model` now tries LM Studio's native
  `/api/v1/models/load` first (with `flash_attention`, `offload_kv_cache_to_gpu`,
  `eval_batch_size`, `num_experts`, `echo_load_config`) and falls back to the legacy
  `/v1/.../load` and `/api/v0/.../load` endpoints (which still honor `gpuOffload`) on failure.
- **Advanced sampling widgets** on Writer / Image Critic / Scene Builder: `reasoning`
  (`off`/`low`/`medium`/`high`/`on`), `flash_attention`, `offload_kv_cache_to_gpu`,
  `repeat_penalty`, `top_k`, `top_p`, `min_p`. These are v1-native and only take effect when the
  model server supports them (or when streaming is enabled); defaults keep the old behavior.
- **Live streaming** — when `stream` is enabled, tokens are pushed to the node's `generation_view`
  widget in real time over the ComfyUI websocket (`web/js/llm_prompt_studio_bridge.js` listens for
  the `llm_prompt_studio.stream` event and appends chunks with auto-scroll).

### Changed
- `lm_http.chat_completion` automatically uses LM Studio's native `/api/v1/chat` endpoint when
  `stream=True` or any v1-only parameter is set, falling back to the OpenAI `/chat/completions`
  path for multimodal (image) requests or when a streaming call fails (so a stalled/errored
  stream still returns text). Streaming uses SSE with a no-activity watchdog, server `error` events
  are surfaced as `RuntimeError`, and a single failed stream falls back to a normal completion.

---

## [0.9.1] — 2026-08-13

### Fixed
- **Pack loaded zero nodes**: `nodes/writer.py` and `nodes/scene_builder.py` had been
  committed as unified-diff text instead of Python, so the package import failed and every
  node silently disappeared. Restored the real sources and re-applied the intended
  merge-user-messages helper.
- **Scene Builder no-negative mode**: stage 2 always switched to the no-negative composer
  whenever the composer widget was left at its default, even in `standard` mode — producing
  an empty `negative` that then failed `require_negative` validation. It now mirrors the
  Writer and only uses the no-negative composer in no-negative mode.
- **Server-side "Value not in list" on queue**: `combo_models()` now fetches the model list
  when the cache is empty, and `cache_models()` always persists the refreshed list to the
  default on-disk cache that `combo_models()` reads, so a queued prompt validates without a
  manual Refresh first. (The cache could otherwise stay empty or get polluted, e.g. `['m']`.)
- **Model-cache persistence**: `cache_models()` only persisted when the refresh used the
  exact default `server_url`; a custom `server_url` left the combo empty. It now always
  persists to the default cache.
- **Tests**: `conftest.py` no longer writes the on-disk model cache (stubbed), so test runs
  can't poison the real cache file.

### Changed
- **`combo_models()`** no longer blocks the UI while building a node; when the on-disk model
  cache is empty it does a best-effort fetch (short timeout) so a running LM Studio populates
  the combo automatically, and the **🔄 Refresh models** button remains the explicit
  population path.

---

## [0.9.0] — 2026-08-12

### Added
- **No-negative mode (`prompt_mode`)** for the Writer and Scene Builder. Distilled SDXL
  families (DMD / LCM / Turbo / Hyper / Lightning / Flash) sample at CFG ~1, where the
  negative prompt is mathematically ignored, so these nodes can automatically switch to a
  positive-only system prompt that rephrases every constraint as a positive statement.
  - `prompt_mode`: `auto` (default, switches on a detected distilled family), `standard`
    (always emit a negative), `no_negative` (always empty negative).
  - Optional `family` input on both nodes: wire the Smart Loader's `detected_family` output
    into it to enable automatic detection. Unconnected → behaves as `standard`.
  - Three new system prompts in `prompts.json`: `writer_system_no_negative`,
    `composer_no_negative`, `face_instruction_no_negative`.

### Changed
- **Family detection hardened** (`model_meta.py`): whole-token matching replaces the old
  substring scan, so `flash` no longer matches `flash_attention` and `hyper` no longer
  matches `hypernetwork`. Added `is_no_negative_family()` for the new mode logic.
- **Critic score parsing** now tolerates float/string scores (`round(float(...))`) instead
  of crashing on `"8.5"`, which previously forced a score of `-1`.
- **`parsing.find_missing_fields`** gained `require_negative` / `require_face_negative`
  parameters (defaults preserve prior behavior). Face fields are no longer treated as
  "missing" (the prompts allow empty `face_positive`/`face_negative` when no face is
  present), eliminating needless field-retries.
- **Prompt library dedupe** now considers both `positive` and `negative`.
- **Writer cache key** includes `prompt_mode` + `family`, so `reuse_last_prompt` never
  returns a result from a different mode.
- **`combo_models()`** no longer blocks the UI while building a node; the **🔄 Refresh
  models** button remains the explicit population path (see 0.9.1 for the on-empty-cache
  fetch).

---

## [0.8.0] — 2026-08-12

### Added
- **Smart Save `save_metadata_to_exif` toggle** (default `True`). When off, approved
  images are saved as JPEG **without** any EXIF metadata (prompts, model, LoRA, params) —
  smaller files and better privacy when sharing images. EXIF is still written by default.
- **Field-retry mechanism for incomplete JSON answers** in the Writer and Scene Builder
  (stage 2), via a new `max_field_retries` widget (default 2). If the model omits required
  fields, the node re-asks for a complete answer; an empty `scene_name` falls back to
  `slugify(positive)`, and empty `positive`/`negative` raise a clear error.
- `parsing.find_missing_fields()` helper for detecting empty critical JSON fields.

### Changed
- **Improved `writer_system` / `composer` system prompts** to require all mandatory JSON
  fields (`positive`, `negative`, `scene_name`, and `face_positive`/`face_negative` when
  face prompts are requested), reducing the rate of incomplete answers from the model.

### Fixed
- **Vision regression in `chat_completion`**: multimodal message content (the image sent to
  Critic / Scene Builder) was being serialized into the request body, dropping the actual
  image. Messages are now sent as-is; serialization is used only for the debug log.
- **Dependency pins relaxed** in `requirements.txt` (`numpy`/`Pillow` no longer capped) so the
  pack installs on Python 3.13/3.14 and modern ComfyUI (numpy 2.x). Package version bumped to
  0.8.0 to match the changelog.

---

## [0.7.3] — 2026-08-10

### Added
- **On-demand model loading via the LM Studio API.** New `load_model()` / `ensure_model_loaded()`
  in `lm_http.py` call the server's load endpoint (`/v1/models/{id}/load`, falling back to
  `/api/v0/...`) with explicit `contextLength` and `gpuOffload`, so the model is loaded with the
  right parameters automatically — no need to pre-load it in the LM Studio GUI. The previous model
  for the same node type is unloaded when you switch (same slot logic as before).
- **`context_length` and `gpu_offload` widgets** on Writer (default 8192 / 1.0), Critic
  (default 16384 / 1.0) and Scene Builder (default 16384 / 1.0), exposing the recommended load
  parameters directly in the nodes. Writer uses 8192; the vision nodes default to 16384 to leave
  room for image tokens.

### Changed
- `VISION_NAME_HINTS` expanded so vision models are detected more reliably: added `vlm`,
  `visual`, `llama-3.2-vision`, `qwen2-vl`/`qwen-vl`/`qvq`, `gemma-3`/`gemma3`, `mistral-vl`,
  `minicpmo`, `deepseek-vl`, `cogvlm`, `idefics`, `smolvlm`, `molmo`, `aya-vision`, `xcomposer`,
  `glm-4v`, `mantis`, `ovis`, `janus`, `embo`, `florence` and others. Gemma 3 and similar VL
  models now pass `vision_check` instead of being wrongly rejected.

---

## [0.7.2] — 2026-08-10

### Fixed
- **Regression: "Value not in list" on Writer/Critic/Scene Builder when loading a saved
  workflow (second attempt).** The model list now persists to `llm_prompt_studio_models_cache.json`
  **inside the package directory** instead of the ComfyUI output directory. The previous
  location depended on `folder_paths.get_output_directory()`, whose value could differ
  between import time and request time, so the cache was not found at ComfyUI startup and
  `INPUT_TYPES` fell back to `— server unavailable —`. `combo_models()` now reads the cache
  file directly on every call (non-blocking) and also migrates an older output-directory
  cache if present. The **🔄 Refresh models** button and the first successful fetch still
  populate it, and the saved workflow validates against the real model list.
- The **🔄 Refresh models** button now also patches the cached node definitions
  (`app.nodeDefs`) for Writer/Critic/Scene Builder. ComfyUI caches combo options from
  `/object_info` at page load, so without this a loaded workflow still validated against the
  stale placeholder list even after the live widgets were updated. This makes saved workflows
  load correctly within the same session (no page reload needed).
- Fixed a latent `ReferenceError` in `refreshModels` (`llm_prompt_studio_actions.js`): `data.error`
  was referenced outside the `try` block, which would throw when the model fetch failed.

---

## [0.7.1] — 2026-08-10

### Fixed
- **RGBA/alpha crash in Critic & Scene Builder**: `imaging.image_to_base64` now converts
  the image to `RGB` before JPEG encoding, matching `smart_save.py`. Previously a tensor
  with an alpha (or single) channel would raise on `img.save(format="JPEG")`.
- **Blocking network call inside `INPUT_TYPES`**: `combos.combo_models()` no longer
  performs a synchronous HTTP request to LM Studio while a node is being built (the cache
  is now read with `allow_fetch=False`). The model dropdown is still populated by the
  **🔄 Refresh models** button, so the UI no longer hangs up to the 5s timeout when the
  server is down.
- **Cross-server model-list contamination**: the combo builder no longer merges models
  cached for *other* `server_url`/`api_key` pairs into the dropdown.
- **JSON parsing broke on braces inside string values**: `_iter_brace_objects` now tracks
  JSON string literals (and `\` escapes), so prompts containing `{`/`}` in their text parse
  correctly instead of falling back to salvage mode.

### Changed
- The model list now distinguishes **server unreachable** (`— server unavailable —`) from
  **server reachable but no models loaded** (`— no models on server —`). `lm_http.get_cached_models`
  returns the latter when the server answers with an empty list, and the JS `refreshModels`
  uses `PH_EMPTY` (`— no models on server —`) when the server is up but lists nothing. The
  previously-unused `PLACEHOLDER_EMPTY` / `PH_EMPTY` constants are now wired in.

### Docs
- README "Requirements" and "Notes & troubleshooting" plus the 0.6.0 changelog entry now
  accurately describe the reasoning fallback: when a model returns empty `content` but a
  non-empty `reasoning_content`, that reasoning text is used as the result, so thinking
  should be disabled in LM Studio to avoid it leaking into prompts/critiques.

---

## [0.7.0] — 2026-08-10

### Added
- **Auto-revision loop (now actually functional)**: the Critic's `auto_loop`,
  `max_retries` and `clear_notes_on_approve` widgets are now processed by the JS bridge.
  When enabled and the image is rejected, the critic's `revision_notes` are injected into
  the Writer's widget and the workflow is re-queued automatically, up to `max_retries`.
  On approval the retry counter resets and the Writer's revision notes are cleared.
  (Requires `reuse_last_prompt` OFF on the Writer.)
- **Scene Builder** now exposes the stage-1 `description` as a 5th STRING output, so it can
  be wired downstream (not only via the "Send to Writer" button).
- **Structured logging**: all modules use Python `logging` under the `llm_prompt_studio`
  logger with a `[LLMPromptStudio]` prefix (replaces `print()`), including LLM call timing
  and errors.
- **Security hardening**:
  - `validate_server_url()` — SSRF guard. Only `http`/`https` and, by default, local/private
    hosts (`localhost`, loopback, RFC-1918, link-local). Set `ALLOW_PUBLIC_SERVER_URLS = True`
    in `lm_http.py` to allow remote hosts.
  - `safe_path_in_output()` — path-traversal guard. Library and save paths are confined to the
    ComfyUI output directory. Set `RESTRICT_PATHS_TO_OUTPUT = False` in `library.py` to allow
    paths outside.
  - Atomic prompt-library writes (`.tmp` + `os.replace`) to prevent corruption on concurrent
    writes.
- **Bounded caches**: `_prompt_cache` (max 128), `_model_cache` (max 32) and the JS
  `lastSaveData` map (max 100) now evict their oldest entries instead of growing forever.
- **Test suite**: `tests/` with pytest golden tests for prompt/critic JSON parsing, the
  salvage logic, `slugify` and checkpoint-family detection (`python -m pytest tests/ -v`).
- **`.gitignore`** (Python caches, venvs, OS/IDE files, logs).

### Changed
- **Python module split**: the monolithic `lm_client.py` is split into focused modules —
  `constants.py`, `lm_http.py`, `combos.py`, `parsing.py`, `library.py`, `imaging.py`,
  `model_meta.py`. All node and route imports updated accordingly.
- **JS split**: the single `llm_prompt_studio_bridge.js` is split into `llm_prompt_studio_shared.js`
  (constants/helpers/state), `llm_prompt_studio_actions.js` (button handlers, `requeuePrompt`) and
  `llm_prompt_studio_bridge.js` (entry point: `registerExtension` + `executed` handler).
- **Default prompts** reworked into the layered photographic style from the SDXL guide;
  anchoring examples removed so each prompt is composed from scratch; explicit NSFW
  authorization for anatomically precise terms; negative prompts described as categories
  instead of a fixed block.
- `api_key` default changed from the misleading `"lm-studio"` placeholder to an empty string.
- Writer `seed` is now actually forwarded to the LLM request, making the seed widget
  meaningful.
- Dependencies now explicitly include `numpy` and `pillow` in `requirements.txt` and
  `pyproject.toml`.

### Fixed
- **Cache TTL bug**: `get_cached_models()` no longer returns stale data forever — an expired
  cache now triggers a re-fetch, falling back to the stale list only on network error.
- **Family consistency**: `flash` added to Smart Loader `FAMILY_OVERRIDES` to match
  `FAMILY_MARKERS` detection.
- README filename pattern clarified to `[prefix_]checkpoint[_lora1_lora2...]_NNNNN.jpg`.

### Removed
- The monolithic `lm_client.py` (replaced by the split modules above).
- The single-file `llm_prompt_studio_bridge.js` (replaced by the three-file JS split).

---

## [0.6.0] — 2026-08-09

### Added
- **LLM Prompt Studio Smart Loader** node: loads a checkpoint, auto-detects its distillation family
  (dmd / lcm / turbo / hyper / lightning / base) from the filename + safetensors metadata,
  and conditionally applies a distillation LoRA (`apply_lora = auto / always / never`).
  Exposes `MODEL`, `CLIP`, `VAE_MODEL`, `VAE_USER`, `detected_family`.
- **LLM Prompt Studio Multi-CLIP SDXL** node: encodes up to four SDXL prompt pairs with one CLIP and
  shared size settings; `*_l` falls back to `*_g`, `target_*` falls back to `width`/`height`,
  CLIP passed through. Replaces several `CLIPTextEncodeSDXL` nodes.
- **`prompts.json`**: all default system prompts moved to an external editable file.
- **`nodes/` package**: the monolithic `nodes.py` split into one module per node, plus
  `_defaults.py` (loads `prompts.json`).
- `reasoning_hint` template key (can be cleared to disable the reasoning reminder).
- Smart Loader family indicator shown in the node title.

### Changed
- All system prompts and UI messages are now English-only.
- Scene Builder image description is now produced in English.
- Default `writer_system`, `critic_system`, `composer`, `describe`, `face_instruction`
  templates reworked in English.

### Removed
  - Reasoning/thinking filter: `THINK_RE`, `REASON_MARKERS`, `strip_thinking()`,
    `_looks_like_reasoning()`. The pack no longer strips reasoning from a normal reply.
    Note: `chat_completion` falls back to a model's `reasoning_content` when `content` is
    empty, so thinking output can still surface in results — disable thinking/reasoning for
    the model in LM Studio to get clean output.
- The single-file `nodes.py` (replaced by the `nodes/` package).

---

## [0.5.0]

### Added
- **Face prompts**: Writer gained `generate_face_prompts` toggle and `face_positive` /
  `face_negative` outputs for face inpainting / FaceDetailer.
- **Face fallback**: empty `face_positive` / `face_negative` automatically fall back to the
  main `positive` / `negative`.
- **`face_prompt_instruction`** widget: editable instruction for generating face prompts,
  including carrying over age and face-specific details from the idea.
- Face fields stored in the prompt library and written to EXIF by Smart Save.
- Smart Save exposes `face_positive` / `face_negative` optional inputs.

---

## [0.4.0]

### Added
- **LLM Prompt Studio Library Loader** node: reload saved scenes from the prompt library (positive,
  negative, scene name, face prompts).
- **LLM Prompt Studio Scene Builder** node: two-stage scene construction — stage 1 describes an image
  with a vision model, stage 2 composes a new SDXL prompt from that description plus user edits.
- JS buttons: **→ Send to Writer** (Scene Builder), **🔄 Refresh scene list** (Library Loader).

---

## [0.3.0]

### Added
- **LLM Prompt Studio Smart Save** node: saves only approved images as JPEG with prompts, checkpoint,
  LoRA and sampler params written into EXIF (ASCII + Unicode tags).
- **Prompt library**: JSON storage of reusable scenes with duplicate detection.
- **💾 Save prompt to library** button and `auto_save_to_library` option.
- Auto-incremented filename `[prefix_]checkpoint_loras_number.jpg`.

---

## [0.2.0]

### Added
- Auto-revision concept: Critic's `revision_notes` feed back into the Writer to regenerate a
  corrected prompt.
- `reuse_last_prompt` option to reuse the cached prompt without re-calling the LLM.
- Automatic unloading of the previously loaded LLM when switching models.
- **🔄 Refresh models** button (Writer, Critic).
- Robust JSON parsing: code-fence stripping, brace-object scanning, salvage of truncated
  answers.

---

## [0.1.0]

### Added
- Initial release.
- **LLM Prompt Studio Writer**: generates an SDXL prompt (positive / negative / scene name)
  from a free-form idea in any language via a local OpenAI-compatible LLM.
- **LLM Prompt Studio Image Critic**: a vision model scores the rendered image against the prompt
  (0–10) and returns a verdict with revision notes.
- JS bridge for buttons, node titles and live model-list updates.