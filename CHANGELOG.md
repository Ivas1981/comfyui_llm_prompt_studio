# Changelog

All notable changes to **ComfyUI LLM Prompt Studio** are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [1.1.5] — 2026-08-18

Added the **LLM Prompt Studio Smart Parameters** nodes that recommend KSampler parameters
(steps / cfg / sampler / scheduler) from the detected checkpoint family.

### Added
- **`LLM Prompt Studio Smart Parameters`** node — emits `steps`, `cfg`, `sampler_name`,
  `scheduler` (COMBO) and `info` connectable directly into a standard `KSampler`. Full
  backward compatibility: the scheduler list contains no AYS/GITS.
- **`LLM Prompt Studio Smart Parameters (Efficient)`** node — same outputs but the scheduler
  COMBO also exposes `AYS SD1 / AYS SDXL / AYS SVD / GITS`, so it links to an Efficient
  KSampler (jags111). For distilled families it recommends `AYS SDXL` at the balanced/speed
  presets.
- Per-family recommended parameters for `base`, `lightning`, `hyper`, `dmd`, `turbo`, `lcm`,
  `tcd`, `pcm`, `flash`, `schnell` across three presets (`balanced` / `speed` / `quality`),
  in `nodes/_distilled_presets.py` (pure Python, no `comfy` import — unit-testable headless).
- **`GET /llm_prompt_studio/sampler_params`** route returning the recommendation for a given
  `family` / `preset` / `ckpt` / `target`; the web UI calls it to autofill the widgets.
- Web autofill: on creation and whenever `detected_family` / `family_override` / `preset` /
  `ckpt_name` change, the node fetches recommendations and fills the editable widgets — unless
  a widget has been manually edited (dirty-flag), which is preserved. The `auto` sentinel is
  injected into saved workflows so loading never trips "Value not in list".

### Removed
- Nothing.

---

## [1.1.4] — 2026-08-17

Correctness review of the `update3`/`update4` assessment (`plans/update3.md`,
`plans/update4.md`). All claims were verified against the source; this release
applies the valid fixes and explicitly ignores the non-bug items.

### Fixed
- **Writer reuse cache key was too narrow (B1).** `nodes/writer.py` cached by
  `(unique_id, prompt_mode)` only, so changing `style_preset`/`system_prompt`/`idea`/
  `revision_notes`/`generate_face_prompts`/`face_prompt_instruction` silently returned a
  stale prompt. The cache key now includes those inputs. `family` stays deliberately
  excluded so a checkpoint swap still carries the prompts over in reuse mode.
- **Family detection blanket-skipped every `flash` match when `flash_attention` appeared
  anywhere (B2).** In `model_meta.py` the `flash` family is now skipped only for the exact
  `flash_attention` substring occurrence, so a real Flash checkpoint (e.g. `FlashSDXL`,
  `SDXLFlash`) is still detected correctly.
- **Native v1 HTTP error ignored image context (B3).** `lm_http.py` hardcoded
  `has_images=False` into `_enrich_http_error`; `_chat_v1` now derives `has_images` from the
  request messages so the "model does not support image inputs" hint fires for vision
  rejections.
- **Field-retry silently fell back to plain text (B4).** The retry `parse_prompt_json` calls
  in `nodes/writer.py` and `nodes/scene_builder.py` now pass `allow_plain_text_fallback=False`
  to match the initial call (a malformed retry answer must not be treated as a valid prompt).
- **Scene Builder stage-1 `prompt_view` output was empty (D1).** Stage 1 now returns the
  description in the `prompt_view` slot (`("", "", "", description, description)`).
- **Duplicated NSFW sentence in the Anime preset (C1).** `presets_default.json` had the
  "Explicit and intimate content is fully authorized…" sentence twice in both the standard
  and no-negative Anime variants; the duplicate is removed.

### Changed
- **`_is_reasoning_rejection` clarified (B5).** Parentheses make the operator precedence of
  the `and`/`or` vision-rejection checks explicit (no behaviour change).

### Removed
- **Unused `require_face_negative` parameter (D2).** `parsing.find_missing_fields` no longer
  accepts the dead parameter; it had no real callers and face fields are intentionally not
  treated as missing.

### Docs
- **`combos.combo_models` comment corrected (D3).** The "scoped to its own server_url" claim
  was inaccurate: `INPUT_TYPES` can only express the default `DEFAULT_SERVER`, so the per-URL
  cache is reached with that default at build time and the node's real `server_url` is applied
  via the widget at runtime.
- **`update4` items #1–#8 were reviewed and ignored.** #3 (unconditional LoRA detection in
  Smart Loader) is FALSE — detection is already guarded by
  `if should_apply and lora_name and lora_name != "[none]"`; #1, #2, #4, #5, #6, #7, #8 are
  intentional/by-design and require no change.

## [1.1.3] — 2026-08-17

### Fixed
- **`WRITER_RESPONSE_SCHEMA` required a non-empty `negative` even in no-negative (distilled)
  mode.** The strict JSON schema at `nodes/model_recommendations.py` no longer lists `negative`
  as required, so a structured-output (≥7B `auto` / `structured`) request does not push the
  model to fabricate a negative at CFG~1. The Writer still force-blanks `negative` in
  no-negative mode, and standard mode keeps re-asking for it via field-retry
  (`require_negative=True`).
- **`load_model_profile` tooltip was misleading.** It claimed the `auto` profile came "from the
  benchmark for this model", but the recommendation is a universal size heuristic with no
  hard-coded model list. The tooltip now says so on Writer / Image Critic / Scene Builder.

### Changed
- **README now lists every Writer input and documents the full Advanced-settings collapse.**
  `load_model_profile` and `server_status` were added to the Writer's Inputs list, and the
  Advanced-settings description now states that the sampling widgets (`temperature`, `max_tokens`,
  `repeat_penalty`, `top_k`, `top_p`, `min_p`) collapse along with the load knobs (this already
  matched `web/js/llm_prompt_studio_actions.js` `ADVANCED_WIDGETS`).
- **`reasoning` is now part of the collapsed Advanced-settings block** (`ADVANCED_WIDGETS` in
  `web/js/llm_prompt_studio_actions.js`), so it hides with the other sampling knobs.

### Added
- **Explicit logging of the resolved sampling profile.** The Writer logs the effective profile
  name and final sampling parameters (temperature / top_p / top_k / min_p / repeat_penalty /
  presence_penalty / reasoning / structured flag) after `load_model_profile` resolution.

## [1.1.1] — 2026-08-17

### Fixed
- **`face_negative` was non-empty in no-negative (distilled) mode.** When `prompt_mode` was
  `auto`/`no_negative`, the Writer (`nodes/writer.py`) force-resets both `face_negative` and
  `negative` to empty once the no-negative composer is chosen (the negative is inert at CFG~1 and
  must stay consistent). Previously a stale `face_negative` leaked into the prompt even though
  `negative` was empty, contradicting CFG~1 (negative-ignored) sampling. Scene Builder
  (`nodes/scene_builder.py`) only exposes `positive`/`negative` outputs — its face fields are parsed
  but not surfaced — so it resets `negative` to empty in no-negative mode, but has no `face_negative`
  output that could leak.
- **Checkpoint family detection missed several distilled families.** `model_meta.py` now maps
  `schnell`, `tcd` and `pcm` to their base distilled families (SD1.5 `schnell`/`tcd` → `turbo`;
  `pcm` → `lcm`), and the free-text metadata hint key is corrected from `"tags"` to `"tag"`. An
  uppercase family token at/after the boundary is unconditionally accepted, and the `flash_attention`
  guard no longer suppresses a real family match.

### Added
- **Smart Loader `detected_family_info` output.** `nodes/smart_loader.py` now also returns
  `detected_family_info` (STRING) describing the provenance of the detected family, e.g.
  `family: turbo | source: filename` / `source: metadata` / `source: base` / `source: override`.
  `detected_family` reports the checkpoint's own family only (a distillation LoRA loaded on a
  base checkpoint does not change it). The `model_meta.detect_checkpoint_family_info()` helper
  returns `(family, source)` and is exported in `model_meta.__all__`.
- **LM Studio server-status indicator.** Writer / Image Critic / Scene Builder gained an optional
  `server_status` STRING widget that polls `GET /llm_prompt_studio/status` every 3 s (via
  `lm_http.server_status()` and the new `pollServerStatus` action) and shows `● Connected — <loaded
  model names>`, `● Connected (no model loaded)`, or `● Server down`. The route is added to
  `server_routes.py`; `refreshModels` no longer clobbers the user's current model selection.
- **Embedding models are filtered out of the model combo.** `lm_http.fetch_models` /
  `_parse_native_models` now skip any entry whose `"type": "embedding"` (on both the native
  `/api/v1/models` path and the OpenAI `/v1/models` fallback), so text/embedding models from LM
  Studio no longer appear as chat models.

### Changed
- **`model_recommendations._parse_size` takes the last size match.** The universal size heuristic
  now reads the trailing `(\d+(?:\.\d+)?)b` capture (e.g. `nvidia/nemotron-3-nano-4b` → 4B) instead
  of the first, so a misleading leading number in the model id no longer mis-sizes the profile.
- **`WEB_DIRECTORY` is set before the node modules are imported** in `__init__.py`, so the JS assets
  are registered even if an early node import touches web-loading code.
- **CI / requirements cleanup.** `.github/workflows/python-tests.yml` now runs on Python 3.10 and
  3.11 (was a malformed `3.1` float); `requirements.txt` no longer pins `hypothesis` (tests don't
   use it).

---

## [1.1.2] — 2026-08-17

### Changed
- **`detected_family` now folds a distilled LoRA into the effective family.** When Smart Loader
  applies a distillation LoRA (DMD / LCM / Turbo / Hyper / Lightning / Flash, or `schnell`/`tcd`/`pcm`)
  on top of a base checkpoint, `detected_family` reports the LoRA's family (the effective distilled
  family) so Writer / Scene Builder enable no-negative mode automatically. The checkpoint's own family
  no longer overrides an applied distilled LoRA.

### Added
- **Smart Loader widget shows the original checkpoint family + a distilled-LoRA notification.**
  `detected_family_info` displays the checkpoint's true family and detection source
  (`family: base | source: filename`), and appends ` | LoRA applied: <name> (distilled: <family>)`
  when a distillation LoRA was applied. The user always sees the real checkpoint family while the
  downstream `detected_family` value already carries the effective (LoRA-folded) family — no extra
  `distilled` output/input is needed; wire `detected_family` into the Writer / Scene Builder `family`
  input as before.

---

## [1.1.0] — 2026-08-16

### Removed
- **Live streaming.** The `stream` widget, the `generation_view` display widget, the SSE
  consumption loop (`_consume_sse`), the `on_delta` callback, and the
  `web/js/llm_prompt_studio_bridge.js` websocket streaming path have all been removed. Generation
  now always returns the full prompt/description at completion. The `STREAM_WATCHDOG_SEC` env var
  and `test_stream_ws_push.py` are gone; `test_lm_http_stream.py` now covers the non-streaming
  native `/api/v1/chat` path only. This removes the SSE-no-activity watchdog and the streaming
  fallback, simplifying the code paths and the websocket bridge.

### Breaking Changes
- **Live streaming is gone.** The `stream` / `generation_view` widgets, the SSE consumer
  (`_consume_sse`), the `on_delta` callback, the websocket bridge push path, the
  `STREAM_WATCHDOG_SEC` env var, and `test_stream_ws_push.py` no longer exist. Existing workflows
  that relied on the streaming view will simply regenerate the full result at completion.
- **`load_model_profile` is now always present** (default `auto`) on Writer / Image Critic / Scene
  Builder. Old workflows saved without this widget will load with the recommended `auto` profile
  applied until you explicitly choose `custom`.
- **System/style prompts now live only in `presets_default.json`.** The editable `prompts.json`
  file was deleted; the base prompts are read from `presets_default.json` via `nodes/_defaults.py`.

## [1.0.10] — 2026-08-15

### Added
- **`load_model_profile` combobox on Writer / Image Critic / Scene Builder.** A always-visible
  profile selector (right under `model`) applies a recommended sampling profile. `auto` (default)
  uses a **universal model-size heuristic that needs no hard-coded model list** — the size is read
  from the model id via the regex `(\d+(?:\.\d+)?)b` (`qwen2.5-14b-instruct` → 14B,
  `nvidia/nemotron-3-nano-4b` → 4B), then: ≥7B models get `baseline` + structured JSON output,
  <7B models get `strict`, and an unrecognized size falls back to a safe `baseline`. The other
  choices (`baseline` / `structured` / `creative` / `strict`) pick a fixed profile, and `custom`
  keeps the individually-set sampling widgets (full backward compatibility with existing workflows).
  When a profile is active its sampling params (temperature / top_p / top_k / repeat_penalty /
  presence_penalty / min_p / reasoning) **override** the individual sampling widgets (the override
  is logged). A new `nodes/model_recommendations.py` holds the four profiles, the strict
  JSON-schemas (`WRITER_RESPONSE_SCHEMA` / `CRITIC_RESPONSE_SCHEMA`), and the public
  `schema_for_kind`, `recommend_for`, `resolve_profile` helpers. Structured (JSON-schema
  `response_format`) is only ever used for `writer` / `compose` text output and **never** with an
  image (vision) input or the `describe` stage. The Critic's `chat_completion` call is expanded to
  forward the profile's sampling parameters (previously it only sent temperature / max_tokens).
- **«⚙ Advanced settings» toggle button.** `context_length`, `gpu_offload`, `flash_attention`
  and `offload_kv_cache_to_gpu` are now hidden behind a custom **«⚙ Advanced settings»** button
  (next to `load_model_profile`) on all three nodes. The four model-load widgets are collapsed by
  default and the button toggles their visibility (label flips ▸/▾). Hiding is done via the widget
  element only, so the widget **values are preserved** and the node always receives the load
  params — this works on every ComfyUI front-end, including the 0.19.3 target that does not render
  native `section` collapsibles.

### Changed
- The default `load_model_profile` is `auto`, so a freshly loaded model now runs with the
  recommended profile instead of the raw widget values. Switch the combo to `custom` to keep the
  previous per-widget behavior.

### Fixed
- **Native `/api/v1/chat` now tolerates servers that reject optional keys (e.g. `seed`).**
  `_chat_v1` drops any `unrecognized_keys` body parameter (like `load_model` already does for
  loads) and retries, so a single unsupported parameter no longer fails the whole call and forces
  a fallback to the OpenAI path. Fixes `HTTP 400 Unrecognized key(s) in object: 'seed'` seen on
  some LM Studio builds/models.

---

## [1.0.9] — 2026-08-14

### Changed
- **Model loading now unloads every resident model before loading the selected one.**
  `ensure_model_loaded` evicts **all** models currently loaded on the LM Studio server — including
  ones left resident by a different node/slot — before loading the requested model, so VRAM holds
  only the model you selected. It still skips the unload+reload when the requested model is already
  loaded with the identical config `(model, context_length, gpu_offload, flash_attention,
  offload_kv_cache_to_gpu, eval_batch_size, num_experts)`. Eviction enumerates loaded instances via
  the native `GET /api/v1/models` and unloads each by `instance_id` (`unload_all_loaded`).
- **Model list is fetched from LM Studio's native endpoint.** `fetch_models` now queries the native
  `GET /api/v1/models` first (preferring each entry's `key`, falling back to `id`), with an
  OpenAI-compatible `/v1/models` fallback for pre-0.4.0 servers. Listed identifiers now match those
  used for chat/load and the capability probes.
- **The OpenAI-compatible chat fallback forwards sampling parameters.** When the native
  `/api/v1/chat` path is unavailable, the `/chat/completions` fallback now forwards `top_p`, `top_k`
  and `repeat_penalty` (and `seed`) so sampling stays consistent with the native path; `min_p` remains
  native-v1-only.

---

## [1.0.8] — 2026-08-14

### Changed
- **System prompts now follow SDXL prompt-engineering best practices.** Every preset's
  `system_prompt` / `system_prompt_no_negative` gained an "Engineering rules" block that instructs the LLM to
  compose prompts in clear layers (subject -> appearance -> action/pose -> object relationships -> environment ->
  composition -> camera/lens -> lighting -> color palette -> mood -> style), lead with the main subject, prefer
  concrete visual descriptors over abstract booster adjectives, state spatial/object relationships explicitly,
  describe the visible result rather than the intention, blend natural-language with a few key style tags, keep the
  prompt concise, and use prompt weighting sparingly `(concept:1.1)`–`(concept:1.3)` only on the most important trait.
  Negatives are told to stay short and targeted (no giant legacy dumps). NSFW authorization (anatomically correct,
  precise terms, no censorship) is preserved and reinforced.
- **Removed the legacy `8k uhd` booster tag** from the Photorealism `style_tags_positive` list (it is appended to the
  positive prompt and the manual flags `8k` as a non-magical SD1.5-era quality dump).
- **Base system prompts in `prompts.json` received the same treatment.** `writer_system`,
  `writer_system_no_negative`, `composer` and `composer_no_negative` gained an "Additional engineering rules" block:
  explicit object/spatial relationships, added color-palette and mood/atmosphere layers, concrete-over-abstract
  descriptors, "describe the visible result not the intention", sparing `(concept:1.1)`–`(concept:1.3)` weighting,
  composition variety, and short targeted negatives. NSFW authorization is preserved.
- **Anime / Manga preset now uses Danbooru/booru tagging.** Per the SDXL manual, most SDXL anime checkpoints
  (Pony, Illustrious, Animagine, NoobAI-derived) are trained on booru vocabulary and respond to it best, so the
  preset now instructs ordered booru tags (rating → character count → identity → traits → clothing → pose →
  expression → composition → environment → lighting → style), with score tags only when the model expects them.
  `style_tags_positive` was made booru-friendly (`anime`, `cel shading`, `vibrant colors`, `detailed illustration`).
  NSFW authorization is preserved (precise booru + natural-language terms, no censorship).
- **Single source of truth for all system prompts.** Base default prompts (Writer, Scene Builder describe/composer,
  Critic, face instructions) and the 14 style presets now live in **one file**, `presets_default.json`, with two
  top-level sections: `defaults` (the base prompts used when no preset is selected) and `presets` (the style
  presets). `prompts.json` was deleted; `nodes/_defaults.py` now reads the base prompts from `presets_default.json`
  via `presets.get_defaults()`, and `presets._migrate` backfills a missing `defaults` section for older user files.
  When the Writer's `style_preset` combobox is left at "— none —" it uses `defaults.writer_system`; selecting a
  preset overrides it (and appends the preset's style tags). This matches the pre-existing behavior, now in one place.

## [1.0.7] — 2026-08-14

### Fixed
- **Style presets ignored no-negative mode.** When a preset was selected with the system prompt left
  at default, the Writer overrode the whole system prompt with the preset's standard variant, which
  demanded a non-empty `negative` — contradicting no-negative (CFG~1) mode where the negative must be
  empty. Each preset now ships a `system_prompt_no_negative` variant (empty `negative`/`face_negative`)
  and the Writer uses it automatically in `no_negative` mode; `photorealism` is still fully disabled
  there via `disabled_in_no_negative_mode`.
- **Preset system prompts lost the reasoning hint.** Presets embed their own reasoning instruction, so
  the canonical `reasoning_hint` is now guaranteed present when a preset is applied.
- **Preset face instructions lacked the "empty string if no face is visible" fallback.** The new
  `system_prompt_no_negative` variants keep `face_negative` empty (consistent with no-negative mode).
- **Preset no-negative variants did not explain positive-only composition.** Each
  `system_prompt_no_negative` now states that distilled models sample at CFG~1 (negative ignored) and
  that every stylistic/quality constraint must be phrased as a positive statement in the prompt —
  i.e. the prompt is composed so it works without a negative.
- **Native `/api/v1/chat` was sending OpenAI `messages` as `input` (HTTP 400 `Invalid
  discriminator value. Expected 'text' | 'image'`), so every text/vision/streaming/reasoning
  call failed.** `_chat_v1` now builds the correct native schema: the `system` role is extracted
  into a top-level `system_prompt`, and the remaining messages are flattened into typed `input`
  parts — `{type:"text", content:...}` and `{type:"image", data_url:<full data URL>}` (images
  keep the whole `data:image/...;base64,...` string, not just the base64). This single path now
  covers text, vision, streaming and reasoning (the OpenAI `/chat/completions` route is kept only
  as a graceful fallback, e.g. when native vision is rejected).
- **Reasoning was sent unconditionally and broke models that don't expose it** (HTTP 400 `does
  not expose reasoning configuration` / `Reasoning setting '…' is not supported`). `chat_completion`
  now detects support via `capabilities.reasoning.allowed_options` and maps the widget level
  `off/low/medium/high/on` to the nearest allowed value (omitting the param entirely when the model
  has no reasoning configuration). On a reasoning rejection it retries once without `reasoning`.
- **Native chat calls were stateful.** `store: false` is now sent on every `/api/v1/chat` request so
  LM Studio keeps no server-side conversation state (reproducible, no hidden dependency on a previous
  run).

## [1.0.6] — 2026-08-14

### Fixed
- **Model loading was completely broken on LM Studio builds that reject the `gpu_offload` key
  (HTTP 400 `Unrecognized key(s)`).** `load_model` now treats any `unrecognized_keys` rejection
  generically: it parses the rejected key names (from the `error.code`/`keys` body or the message),
  drops them, and retries — so an unsupported optional parameter (e.g. `gpu_offload`,
  `offload_kv_cache_to_gpu`) no longer hard-fails the whole load. Known-rejected keys are remembered
  per server+model so a subsequent load drops them up front.
- **Switching models did not actually unload the previous one.** `maybe_unload_old` now unloads via
  the exact v1 `instance_id` captured from a successful load (a 404 / already-gone id is treated as
  idempotent success). The legacy `/api/v0/models/unload` is only used as a fallback when the v1
  route is unreachable (connection error / 404), never on a 2xx "Unexpected endpoint".
- **`ensure_model_loaded` ignored context/gpu/flash/KV changes.** It now fingerprints the requested
  config `(model, context_length, gpu_offload, flash_attention, offload_kv_cache_to_gpu,
  eval_batch_size, num_experts)` and force-reloads when any of them change (a pre-fingerprint string
  value triggers a one-time reload).
- **Model combos mixed up servers.** The model cache is now keyed by normalized server URL (the API
  key is never persisted). `Refresh models` patches only the widgets whose `server_url` matches the
  refreshed server, so each server's combo shows only its own models. `INPUT_TYPES` no longer hits the
  network (Refresh / node-creation refresh is the source of truth).
- **Concurrent library saves could lose entries.** `save_prompt_to_library` is now serialized by a
  per-library-path lock, written atomically via `.tmp` + replace, and keeps a `.bak` of the previous
  file. The duplicate check now also compares `face_positive` / `face_negative`.
- **Smart Save could overwrite files on a numbering race.** Filename reservation now uses an exclusive
  create (`O_EXCL`) and bumps the counter on collision instead of a racy `listdir → max+1`.
- **Streaming failures left `generation_view` stale / partial.** A streaming failure now resets the
  widget and shows the full non-streaming result exactly once via `on_delta`.
- **Auto-revision loop always targeted the first Writer.** It now follows the Critic's `prompt` link
  back to the producing Writer, so multi-Writer/Critic workflows are deterministic.
- **`slugify` dropped Cyrillic/Unicode and plain-text model refusals were silently accepted as
  prompts.** `slugify` (and `smart_save._slug_part`) are now Unicode-aware; the Writer sets
  `allow_plain_text_fallback=False` so a non-JSON refusal becomes an explicit error.
- **Dead Critic `generation_view` widget removed** (the Critic is always a vision model and never
  streams).

---

## [1.0.5] — 2026-08-13

### Fixed
- **`reuse_last_prompt` broke when the user swapped the checkpoint.** The Writer cache key
  included `family`, which is driven by the loaded checkpoint (via Smart Loader's
  `detected_family`). So switching to a different checkpoint changed the key, forced a cache
  miss, and regenerated the prompt instead of reusing the old one. The cache key is now
  `(unique_id, prompt_mode)` — a mode switch still regenerates (as before), but a checkpoint
  swap no longer invalidates the cache, so the previously generated prompts are carried over
  to the new checkpoint as intended.

---

## [1.0.4] — 2026-08-13

### Fixed
- **Model preload failed on LM Studio versions that reject the snake_case `gpu_offload`
  key (HTTP 400 `Unrecognized key(s) in object: 'gpu_offload'`).** The 1.0.3 change switched
  the native v1 load body to `gpu_offload`, but some LM Studio builds only accept the
  camelCase `gpuOffload`. `load_model` now tries `gpu_offload` first and, if the server
  rejects that specific key, retries the v1 load once with `gpuOffload` — so the model
  preloads with the user's `context_length` / `gpu_offload` settings on either server
  variant. A genuine rejection (non-gpu-key) still returns `False` without masking the
  failure via the legacy route.

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
- **Writer cache key** is `(unique_id, prompt_mode)` — `reuse_last_prompt` never returns a
  result from a different mode, but a checkpoint swap (which changes `family`) no longer
  invalidates the cache, so the same prompts carry over to the new checkpoint.
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