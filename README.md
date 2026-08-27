# ComfyUI LLM Prompt Studio

A ComfyUI custom-node pack that drives **architecture-aware prompt engineering** with a
**local LLM** served by [LM Studio](https://lmstudio.ai) or any **local OpenAI-compatible
server**. It adapts its output to the base model you load — SDXL, SD1.5, Pony,
Illustrious, Flux and SD3 — not just SDXL.

It closes the full loop: **generate a prompt → render the image → critique it with a
vision model → auto-revise → save approved results with full metadata**, plus a prompt
library, scene rebuilding from images, and smart model/LoRA loading.

No cloud, no API keys required — everything runs on your machine.

## Intended Usage

This pack is built for a **single-user, single-machine, single-ComfyUI-instance** workflow:

- **One user per machine** — it loads/unloads the LLM model on your local LM Studio server
  and tailors prompts to *your* loaded checkpoint. It is not designed for multi-tenant or
  shared-host use.
- **One ComfyUI instance** — the nodes share one LM Studio server and assume they own the
  loaded model; running several ComfyUI instances against the same server can race on model
  load/unload.
- **One workflow at a time** — run one prompt-generation/critique pipeline per session so the
  auto-revision loop and VRAM release behave predictably.
- **A local LLM in LM Studio** (or any local OpenAI-compatible server) with a model loaded
  before you queue the nodes.

---

## Features

- **Prompt Writer** — turns a free-form idea (any language) into a precise SDXL prompt,
  with optional face-inpainting prompts.
- **Image Critic** — a vision model scores the rendered image against the prompt and
  returns concrete revision notes.
- **Auto-revision loop** — when enabled, the critic's notes are fed back into the Writer
  and the workflow re-queues itself automatically until approved or `max_retries` is hit.
- **Smart Save** — saves only approved images as JPEG with prompts, model, LoRA and
  sampler params written into EXIF.
- **Prompt library** — reusable scenes stored in a JSON file; save & reload them.
- **Scene Builder** — two-stage: describe an image with a vision model, then compose a
  new prompt from that description (+ your edits).
- **Smart Loader** — loads a checkpoint, auto-detects its distillation family
  (DMD / LCM / Turbo / Hyper / Lightning / Flash, plus `schnell`/`tcd`/`pcm` mapped to their base
  families) and conditionally applies a distillation LoRA. It exposes `detected_family` (the
  **effective** family — a distilled LoRA applied to a base checkpoint folds its family in, so
  no-negative mode auto-enables downstream) and `detected_family_info` with the detection provenance
  (`filename` / `metadata` / `base` / `override`). The widget shows the checkpoint's **own** family
  and a notification when a distillation LoRA was applied, e.g.
   `family: base | source: filename | LoRA applied: dmd_lora.safetensors (distilled: dmd)`.
   It also reports the base **`architecture`** (SDXL / SD1.5 / Pony / Illustrious / Flux / SD3) via
   `detected_architecture` / `detected_architecture_info`; wire `detected_architecture` into the
   Writer / Scene Builder `architecture` input to tailor token style and negatives per base model
   (SDXL generation is unchanged when the input is left empty).
 - **Smart Parameters** — recommends KSampler `steps` / `cfg` / `sampler` / `scheduler` from the
   detected checkpoint family (wired from the Smart Loader's `detected_family`); emits them as COMBO
   outputs that plug straight into a `KSampler` (one full scheduler list including AYS SD1/SDXL/SVD and GITS).
- **LM Studio server-status indicator** — Writer / Image Critic / Scene Builder show a live
  `server_status` widget (`● Connected — <loaded model names>` / `● Connected (no model loaded)` /
  `● Server down`), polled from the pack's status route every few seconds.
 - **Smart Multi-Clip** — architecture-aware encoding of up to four prompt pairs with one CLIP
   and shared size settings; wire the Smart Loader's `detected_architecture` for non-SDXL models.
 - **Style presets (Styles/ folder)** — pick from 162 built-in styles across ~25 domains
   (Photography, Art Movements, Asian Art, Traditional Media, Digital & Contemporary,
   Fantasy & Horror, Anime & Manga, Cinematic, Period & Style, Basic Styles, and more); the
   combobox shows `Category > Name` labels. Each preset lives in its own JSON file under
   `Styles/` and may `extends` another for inheritance. A preset appends its style tags to the
    prompt and can override the system prompt. `prompt_mode` is the sole control for the negative
     prompt (see below). The other orthogonal generation toggles — `generate_face_prompts`, `nsfw`,
     `prompt_format` (natural / tags / weighted / structured / midjourney / booru) and `blend_styles`
    — are applied at assembly time, not stored per preset. The Writer / Scene Builder also expose an
    `architecture` input that adapts token style and default negatives for SD1.5 / Pony / Illustrious
    / Flux / SD3.
- **Advanced LM Studio v1 options** — `reasoning`, `flash_attention`, `offload_kv_cache_to_gpu`,
  `repeat_penalty`, `top_k`, `top_p`, `min_p` are forwarded to LM Studio's native v1 API when
  supported.
- **Load model profile** — a `load_model_profile` combobox (Writer / Critic / Scene Builder)
  applies a recommended sampling profile (`auto` picks one from a universal model-size
  heuristic, with no hard-coded model list). Model-load knobs (`context_length`,
  `gpu_offload`, `flash_attention`, `offload_kv_cache_to_gpu`) are hidden behind a
  **«⚙ Advanced settings»** button (next to `load_model_profile`) and collapsed by default;
  click it to show/hide them.
- **Debug logging** — opt-in (`debug.py`, OFF by default) logs node/HTTP/parse activity with
  API keys and image blobs masked.

---

## Breaking Changes in 1.1.0

- **Live streaming removed.** The `stream` / `generation_view` widgets, the SSE consumer
  (`_consume_sse`), the `on_delta` callback, the websocket bridge push path, the
  `STREAM_WATCHDOG_SEC` env var, and `test_stream_ws_push.py` are gone. Generation now returns
  the full prompt/description once, at completion — there is no live-token view anymore.
- **`load_model_profile` is now always present** (default `auto`) on Writer / Image Critic /
  Scene Builder. Workflows saved before this widget existed will load with the recommended `auto`
  profile applied until you choose `custom`.
 - **System/style prompts now live in the `Styles/` folder.** Base prompts are in
   `Styles/system_prompts.json` and the 162 style presets each live in their own direction file
   under `Styles/` (a mirror of the former single `presets_default.json`). On first run the whole
   `Styles/` folder is copied into the user-editable `llm_prompt_studio_styles/` folder inside the
   ComfyUI output directory (never overwritten — hand edits survive a refresh). Use the node's
   **Reload Styles** / **Reset Styles** buttons to re-read or restore the shipped styles. The legacy
   `presets_default.json` is kept only as a fallback source.

---

## Requirements

- ComfyUI — **recommended and tested on ComfyUI 0.19.3 (classic)**.
- Python packages: `requests`, `numpy`, `pillow`, `opencv-python-headless` (declared in
  `requirements.txt`; usually already present in a ComfyUI environment). `opencv` is required
  for the Face Detailer's default `haar` detection.
- **LM Studio** running a local server (default `http://localhost:1234/v1`).
  - A chat/instruct model for the Writer / Critic text tasks.
  - A **vision-capable** model (Qwen2.5-VL, LLaVA, Gemma-3, …) for the Critic and
    Scene Builder image analysis.
- Reasoning/thinking, if your model supports it, should be **disabled in LM Studio**.
  Note: when a model returns an empty `content` but a non-empty `reasoning_content`
  (thinking mode), the pack falls back to using that reasoning text as the prompt
  (see `lm_http.chat_completion`). To keep reasoning output out of your prompts and
  critiques, disable thinking for that model in LM Studio.

---

## Installation

1. Clone / copy this folder into `ComfyUI/custom_nodes/`:
   ```
   ComfyUI/custom_nodes/comfyui_llm_prompt_studio/
   ```
2. Install dependencies (if missing):
   ```
    pip install requests numpy pillow opencv-python-headless
   ```
3. Start LM Studio, load a model and enable the local server.
4. Restart ComfyUI. You should see in the console:
   ```
    [LLMPromptStudio] Package loaded: LLMPromptStudioWriter, LLMPromptStudioCritic,
    LLMPromptStudioSmartSave, LLMPromptStudioLibraryLoader, LLMPromptStudioSceneBuilder,
    LLMPromptStudioSmartLoader, LLMPromptStudioMultiClipSDXL, LLMPromptStudioSmartParameters,
    LLMPromptStudioKSamplerHiresFix, LLMPromptStudioFaceDetailer.
    ```

### Package layout

```
comfyui_llm_prompt_studio/
├── __init__.py               # logging setup + node registration
├── constants.py              # shared placeholder constants
├── debug.py                  # DEBUG_LEVEL / masked debug logging
├── presets.py                # legacy preset loading / fallback
├── styles.py                 # Styles/ folder loader + system-prompt assembler
├── lm_http.py                # HTTP client, model cache, SSRF guard
├── combos.py                 # combo lists (models / checkpoints / LoRA / VAE)
├── parsing.py                # JSON extraction / salvage / slugify
├── library.py                # prompt library, path guard, atomic writes
├── imaging.py                # image tensor → base64 JPEG
├── model_meta.py             # generation metadata, safetensors, family detection
├── server_routes.py          # /llm_prompt_studio/* routes used by the JS bridge
├── vram.py                   # LM Studio model release helpers
├── presets_default.json      # legacy fallback source (superseded by Styles/)
├── Styles/                   # style system: system_prompts.json + 25 direction files (162 presets)
├── nodes/
│   ├── __init__.py           # node mappings
│   ├── _contracts.py         # sampler/scheduler combo contract checks
│   ├── _defaults.py          # loads base prompts from Styles/system_prompts.json
│   ├── _distilled_presets.py # distilled-family preset tables
│   ├── _imgutils.py          # image helper utilities
│   ├── _ksample.py           # shared KSampler execution helper
│   ├── _latent_upscaler.py   # latent upscale helper
│   ├── model_recommendations.py  # sampling profiles + JSON schemas + resolve_profile
│   ├── writer.py
│   ├── critic.py
│   ├── smart_save.py
│   ├── library_loader.py
│   ├── scene_builder.py
│   ├── smart_loader.py
│   ├── multi_clip.py
│   ├── smart_parameters.py   # Smart Parameters node
│   ├── ksampler_hiresfix.py  # KSampler (Hires Fix) node
│   └── face_detailer.py      # Face Detailer node
├── docs/                     # QA checklist, API notes, ComfyUI custom-node guide
├── models/                   # bundled models (face detectors, resizers)
├── tests/                    # pytest golden tests
├── web/js/
│   ├── llm_prompt_studio_shared.js    # constants, helpers, shared state
│   ├── llm_prompt_studio_actions.js   # button handlers, re-queue helper
│   └── llm_prompt_studio_bridge.js    # entry point: extension + executed handler
├── requirements.txt
├── pyproject.toml
├── LICENSE
├── README.md
├── readme_ru.md
├── CHANGELOG.md
└── changelog_ru.md
```

---

## Nodes

All nodes live under the **`LLM Prompt Studio`** category.

### LLM Prompt Studio Writer

Generates an SDXL prompt from an idea.
  - **Inputs:** `server_url`, `api_key`, `model`, `load_model_profile`, `context_length`, `gpu_offload`,
    `system_prompt`, `idea`, `revision_notes`, `temperature`, `max_tokens`, `seed`,
     `reuse_last_prompt`, `generate_face_prompts`, `max_field_retries`,
     `face_prompt_instruction`, `prompt_mode`, `family`, `architecture`, `style_preset`,
       `use_preset_system_prompt`, `nsfw`, `prompt_format`,
       `blend_styles`,
      `reasoning`, `flash_attention`, `offload_kv_cache_to_gpu`, `repeat_penalty`,
     `top_k`, `top_p`, `min_p`, `server_status`, `release_vram_after_run`, `cv_scene_hint`.
  - **Optional `image` input:** connect an image and enable `cv_scene_hint` (default on) to feed a
    cheap OpenCV pre-analysis (scene type, lighting, face count — no extra vision call) into the brief,
    so the generated prompt fits the content (anime/illustration vs photo, natural skin for portraits).
  - **`prompt_mode`** selects how the negative prompt is handled: `auto` (default) switches
    to a no-negative system prompt when the checkpoint family is distilled (DMD / LCM /
    Turbo / Hyper / Lightning / Flash, **or** a distillation LoRA applied to a base checkpoint —
    the Smart Loader folds the LoRA family into `detected_family`); `standard` always emits a
    negative; `no_negative` always emits an empty negative. Distilled models sample at CFG ~1,
    where a negative prompt is mathematically ignored, so the no-negative prompt rephrases every
    constraint as a positive statement instead.
  - **`load_model_profile`** (always visible, right under `model`): picks the recommended
    **sampling + load profile** for the selected model. `auto` (default) applies the
    profile chosen by a *universal* heuristic — the model size is read from its id
    (`qwen2.5-14b-instruct` → 14B, `nvidia/nemotron-3-nano-4b` → 4B) and mapped to a
    profile with **no hard-coded model list**: ≥7B models get the `structured` profile
    (near-greedy ~0.1 temperature + structured JSON output for the highest schema parse rate), <7B models get `strict`, and unrecognized sizes fall back to a safe `baseline`.
    The other choices pick a fixed profile (`baseline` / `structured` / `creative` /
    `strict`), and `custom` keeps your individually-set sampling widgets (full backward
    compatibility with old workflows). When a profile is active, its sampling parameters
    (temperature / top_p / top_k / repeat_penalty / presence_penalty / min_p / reasoning)
    **override** the individual sampling widgets, which are then ignored (this is logged).
    Structured (JSON-schema `response_format`) is only ever used for `writer` / `compose`
    text output and **never** with an image (vision) input or the `describe` stage.
  - **`Advanced settings`** (button): `context_length`, `gpu_offload`, `flash_attention`,
    `offload_kv_cache_to_gpu` **and the sampling widgets** `temperature`, `max_tokens`,
    `repeat_penalty`, `top_k`, `top_p`, `min_p` are hidden behind a **«⚙ Advanced settings»**
    button (next to `load_model_profile`) and are collapsed by default. Click the button to
    show/hide them (the widget values are preserved, so the node always receives the params).
   - **`family`** (optional) — wire the Smart Loader's `detected_family` output here to let
    `auto` detect the distilled family automatically. Because the Smart Loader folds an applied
    distillation LoRA into `detected_family`, a base checkpoint with a DMD/LCM/Turbo/… LoRA also
    engages no-negative mode. Leave unconnected to use `standard`.
   - **`architecture`** (optional) — wire the Smart Loader's `detected_architecture` output (or
    set a value manually) to adapt the generated prompt to the base architecture: it switches the
    token style and appends architecture-specific default negatives for `Flux` / `SD3` / `SD1.5` /
    `Pony` / `Illustrious` (SDXL is unchanged when empty/unwired). `Flux` and `SD3` also force the
    no-negative path regardless of `prompt_mode`.
  - Loads the selected model on the LM Studio server **on demand** (via the load API) with
    `context_length` (default 8192), `gpu_offload` (1.0 = max GPU) and the advanced load options
    (`flash_attention`, `offload_kv_cache_to_gpu`), and **unloads every model currently resident
    on the server — from any node — before loading the selected one**, so only the model you picked
    stays in VRAM. You no longer need to pre-load the model in the LM Studio GUI.
 - **Outputs:** `positive`, `negative`, `raw`, `scene_name`, `face_positive`,
   `face_negative`.
 - **`release_vram_after_run`** (default on) unloads the LM Studio model after the node
   finishes so the diffusion pipeline reclaims the VRAM. Turn it off only when the LLM and the
   checkpoint fit in VRAM simultaneously.
- **`generate_face_prompts`** (default off): turn it on to also get short face-only prompts
  (`face_positive` / `face_negative`) for FaceDetailer/inpainting. When **turned off**, both
  face fields are forcibly cleared (even if the model returned them in the JSON), so FaceDetailer
  correctly falls back to the main `positive`/`negative`. When on, empty face fields automatically
  fall back to the main `positive`/`negative`.
- **`reuse_last_prompt`** returns the cached result without calling the LLM (useful to
  re-render without re-generating the prompt). Note: turn it OFF for the auto-revision
  loop to work.
- **`seed`** is forwarded to the LLM request (for servers that honor it).
- **`max_field_retries`** (default 2): if the model returns a JSON answer missing required
  fields, the node re-asks it for a complete answer up to this many times. Any `scene_name`
  (including a non-empty one, with quotes or "Scene: …") is normalized to a consistent lowercase
  slug via `slugify`; an empty value falls back to a slug of `positive`; empty
  `positive`/`negative` raise an error.
- Transient `chat_completion` failures (network/timeout) are caught and retried within the
  generation / auto-revision loop instead of aborting the node; each failure is logged as a
  warning so the run keeps going when the LM Studio server briefly hiccups.
- Button **🔄 Refresh models** re-reads the model list from the server.
 - **Style presets (`Styles/` system)** (`style_preset` + `use_preset_system_prompt`): pick a
   built-in style (from the `Styles/` folder, 162 presets) to append its style tags to the prompt.
   `use_preset_system_prompt` (default on) additionally overrides the system prompt with the
   preset's; turn it off to keep your own system prompt while still applying the preset's style tags.
   The selected style also shapes `face_positive` / `face_negative`, so a FaceDetailer-refined face
   matches the chosen style. Presets that opt out of no-negative mode (`disabled_in_no_negative_mode`)
   are skipped automatically in that mode, and a preset's `system_prompt_no_negative` variant is used
   when the negative prompt is off.
     - **Orthogonal generation toggles** (applied at assembly time, never stored per preset):
       - **`nsfw`** (default off) — safe-for-work by default; on permits explicit content only when the
       idea explicitly requests it.
     - **`prompt_format`** — `natural` (default) / `tags` / `weighted` / `structured` / `midjourney` /
       `booru`: how the model phrases the positive prompt.
     - **`blend_styles`** — comma-separated style ids/labels (max 3) whose tags and blend notes are
        merged additively onto the primary preset. Every listed style must appear visibly in the
        generated prompt (strict fuse), so all referenced looks are present at once.
      - **Reload Styles / Reset Styles buttons** — the node's *Reload Styles* button re-reads the
        `Styles/` folder from disk (picking up any hand edits you made to a style file) and refreshes
        the `style_preset` combo; *Reset Styles* deletes your user-editable `llm_prompt_studio_styles/`
        copy and falls back to the shipped `Styles/`.
- **Advanced options** (`reasoning`, `flash_attention`, `offload_kv_cache_to_gpu`,
  `repeat_penalty`, `top_k`, `top_p`, `min_p`): forwarded to LM Studio's native v1 API when the
  server supports them. Leaving them at defaults keeps the old
  OpenAI-compatible behavior. `reasoning` enables thinking-model reasoning; `flash_attention` /
  `offload_kv_cache_to_gpu` are model-load options.

### LLM Prompt Studio Image Critic

A vision model scores how well the image matches the prompt.
  - **Inputs:** `image`, `prompt`, `server_url`, `api_key`, `model`, `load_model_profile`,
    `context_length`, `gpu_offload`, `critic_prompt`, `threshold`, `image_max_size`,
    `temperature`, `max_tokens`, `clear_notes_on_approve`, `auto_loop`, `max_retries`,
    `vision_check`, `revision_view`, `cv_scene_hint`, `flash_attention`, `offload_kv_cache_to_gpu`,
    `server_status`, `release_vram_after_run`.
  - **`cv_scene_hint`** (default on): runs a cheap OpenCV pre-analysis (scene type, lighting,
    face count — no extra vision call) and adds it to the evaluation context, so the model critiques
    the image in the right frame (anime/illustration vs photo, natural skin for portraits). The hint
    is also surfaced in the node's `scene_hint` output.
  - **`load_model_profile`** works exactly as on the Writer (default `auto`, universal
    size-based recommendation, `custom` keeps the widget values). Because the Critic is
    always a vision model, its profile expands the `chat_completion` call with the
    recommended sampling parameters (temperature / top_p / top_k / repeat_penalty /
    min_p / presence_penalty), but structured JSON output is never used with an image.
  - **`Advanced settings`** (button): same as the Writer — `context_length`, `gpu_offload`,
    `flash_attention`, `offload_kv_cache_to_gpu` **and the sampling widgets** `temperature`,
    `max_tokens`, `repeat_penalty`, `top_k`, `top_p`, `min_p` are hidden behind a
    **«⚙ Advanced settings»** button (next to `load_model_profile`), collapsed by default;
    click it to show/hide them.
  - **Outputs:** `approved` (bool), `score` (int), `revision_notes`, `verdict`, `raw`.
 - **`release_vram_after_run`** (default on) unloads the LM Studio model after the node
   finishes so the diffusion pipeline reclaims the VRAM. Turn it off only when the LLM and the
   checkpoint fit in VRAM simultaneously.
- `approved = score >= threshold`. Wire `approved` into Smart Save to gate saving.
- **Auto-revision loop**: enable `auto_loop` to have the critic automatically feed its
  `revision_notes` back into the Writer and re-queue until the image passes or
  `max_retries` is reached. Requires `reuse_last_prompt` OFF on the Writer.
- The node title shows the live score and approval status.

### LLM Prompt Studio Smart Save

Saves approved images as JPEG, optionally with metadata in EXIF.
- **Inputs:** `image`, `approved`, `filename_prefix`, `save_dir`, `jpeg_quality`,
  `auto_save_to_library`, `save_metadata_to_exif`, `library_path`, plus optional
  `positive`, `negative`, `scene_name`, `face_positive`, `face_negative`.
- **Outputs:** none (output node).
- Only saves when `approved` is true. Filename:
  `[prefix_]checkpoint[_lora1_lora2...]_NNNNN.jpg` (`NNNNN` auto-increments).
- EXIF (written when `save_metadata_to_exif` is on, default) stores prompts, face prompts,
  checkpoint, LoRA and sampler params (both ASCII and Unicode tags).
- Turn **`save_metadata_to_exif` off** to write JPEGs without any metadata — smaller files
  and better privacy (e.g. when sharing images publicly).
- Button **💾 Save prompt to library** writes the last saved prompt into the library.

### LLM Prompt Studio Library Loader

Loads a saved scene from the prompt library.
- **Inputs:** `library_path`, `scene_name` (combo of saved scenes).
- **Outputs:** `positive`, `negative`, `scene_name`, `face_positive`, `face_negative`.
- Button **🔄 Refresh scene list** re-reads the library.

### LLM Prompt Studio Scene Builder

Two-stage scene construction from an image.
  - **Inputs:** `stage` (`1 - describe` / `2 - compose`), `image`, `server_url`,
    `api_key`, `model`, `load_model_profile`, `context_length`, `gpu_offload`,
    `describe_prompt`, `composer_prompt`, `user_changes`, `image_max_size`, `temperature`,
    `max_tokens`,     `max_field_retries`, `vision_check`, `description_view`, `prompt_mode`,
    `family`, `architecture`, `flash_attention`, `offload_kv_cache_to_gpu`, `reasoning`, `repeat_penalty`,
    `top_k`, `top_p`, `min_p`, `release_vram_after_run`.
  - **`load_model_profile`** works as on the Writer. The stage decides the "kind": stage 1
    (`describe`) is prose from a vision input, so it is always `baseline` with **no**
    structured output; stage 2 (`compose`) is the JSON writer contract and gets
    the `structured` profile (near-greedy ~0.1 + structured JSON) for ≥7B models, just like
    the Writer. `custom` keeps the widget sampling values. `vision_check` only applies to stage
    1 (the image-analysis stage) — stage 2 composes from the saved text description and does
    not require a vision model.
  - **`Advanced settings`** (button): same as the Writer — `context_length`, `gpu_offload`,
    `flash_attention`, `offload_kv_cache_to_gpu` **and the sampling widgets** `temperature`,
    `max_tokens`, `repeat_penalty`, `top_k`, `top_p`, `min_p` are hidden behind a
    **«⚙ Advanced settings»** button (next to `load_model_profile`), collapsed by default;
    click it to show/hide them.
   - **`prompt_mode`** and **`family`** behave exactly as on the Writer: `auto`
    switches to a no-negative composer prompt for distilled checkpoint families (wired `family`),
    which the Smart Loader reports including the case of a distillation LoRA applied to a base
    checkpoint (it folds the LoRA family into `detected_family`).
   - **`architecture`** (optional, stage 2 only) behaves as on the Writer: it adapts the token
    style and appends architecture-specific default negatives for `Flux` / `SD3` / `SD1.5` /
    `Pony` / `Illustrious`, and forces the no-negative path for `Flux` / `SD3`.
 - **Outputs:** `positive`, `negative`, `scene_name`, `prompt_view`, `description`.
 - **`release_vram_after_run`** (default on) unloads the LM Studio model after the node
   finishes so the diffusion pipeline reclaims the VRAM. Turn it off only when the LLM and the
   checkpoint fit in VRAM simultaneously.
  Stage 1 puts the vision description into `description`; stage 2 composes the prompt.
- **`max_field_retries`** (default 2): in stage 2, if the model's JSON omits required
  fields, the node re-asks for a complete answer; any `scene_name` (including a non-empty one)
  is normalized to a lowercase slug via `slugify`, and an empty value falls back to a
  slug of `positive`.
- Button **→ Send to Writer** copies the description/prompt into a Writer's `idea`.
- Button **🔄 Refresh models** (same as on the Writer/Critic) re-reads the model list from the server.

### LLM Prompt Studio Smart Loader

Loads a checkpoint and handles distillation LoRA automatically.
- **Inputs:** `ckpt_name`, `family_override`, `lora_name`, `apply_lora`,
  `strength_model`, `vae_user`.
- **Outputs:** `MODEL`, `CLIP`, `VAE_MODEL`, `VAE_USER`, `detected_family`,
  `detected_family_info`, `detected_architecture`, `detected_architecture_info`, `ckpt_name`.
- Detects the checkpoint family from the filename + safetensors metadata:
  `base`, `dmd`, `lcm`, `turbo`, `hyper`, `lightning`, `flash` (plus `schnell`/`tcd` → `turbo`,
  `pcm` → `lcm`). `family_override` forces it.
- `detected_family_info` is a STRING describing **why** that family was chosen, e.g.
  `family: turbo | source: filename` — useful when wiring `detected_family` into the Writer's
  `family` input and confirming the detector picked the right family.
- `apply_lora`: **`auto`** applies the LoRA only for `base` (non-distilled) models,
  **`always`** forces it, **`never`** skips it.
- `VAE_USER` falls back to the checkpoint's built-in VAE when `vae_user = [none]`.
  Use `VAE_USER` downstream so the output is never empty.
- The node title shows the detected family.

### LLM Prompt Studio Smart Parameters

Recommends KSampler `steps` / `cfg` / `sampler` / `scheduler` from the checkpoint family so you
don't have to look them up. There is a **single** node — it uses one full scheduler list
(standard ComfyUI schedulers plus `AYS SD1` / `AYS SDXL` / `AYS SVD` / `GITS`) that the studio's
own KSampler supports, so no `target`/"Efficient" split is needed.

- **Inputs:** `family_override` (auto / family), `preset` (`user` / `balanced` / `speed` / `quality`),
  `steps`, `cfg`, `sampler_name`, `scheduler`, plus optional `detected_family`
  (wire the Smart Loader's `detected_family` output here), `ckpt_name`, and `architecture`
  (wire the Smart Loader's `detected_architecture` output here).
- **Outputs:** `steps` (INT), `cfg` (FLOAT), `sampler_name` (COMBO), `scheduler` (COMBO), `info`
  (STRING).
- **Presets:** `user` passes your widget values through unchanged; `balanced` / `speed` / `quality`
  pick a recommended steps/cfg/sampler row (quality = highest step count, speed = lowest).
- **Family detection:** the effective family is resolved in this order — `detected_family` (if
  wired, from the Smart Loader; it already folds in a distillation LoRA), then `family_override`,
  then auto-detection from the checkpoint **filename**, its safetensors **metadata**, and its
  **parent folder name** (so a generically-named file inside a `Lightning/` folder is still
  recognized). For known Lightning/Hyper checkpoints the step count can also be read from the
  filename (`SDXL-Lightning_4step` → 4 steps).
- **AYS SDXL for distilled families:** for distilled families (lightning, hyper, dmd, turbo, lcm,
  tcd, pcm, flash, schnell) the node recommends **`AYS SDXL`** at the `balanced`/`speed` presets;
  any family present in `model_meta.FAMILY_MARKERS` that has no hand-written row gets sane generic
  distilled defaults automatically.
- **Architecture awareness:** when the resolved family is `base` and a known `architecture` is
  wired in (SD1.5 / Pony / Illustrious / Flux / SD3), the node recommends that base
  architecture's own sampler defaults (e.g. Flux → 24 steps, cfg 1.0, euler/simple; SD1.5 →
  cfg 7.0) instead of the SDXL base row, so a non-SDXL checkpoint is not washed out. Distilled
  families keep priority and are never overridden by the architecture.
- **Sentinels:** leave `steps`/`cfg` at `0`/`-1.0` and `sampler_name`/`scheduler` at `auto`
  to use the recommendation. Any value you type manually overrides the recommendation and is
  preserved on reload (the web UI only autofills widgets you have not edited).
- The web UI autofills the widgets from the `/llm_prompt_studio/sampler_params` route when the
  family/preset/checkpoint changes (the route also accepts `arch`).

### LLM Prompt Studio Smart Multi-Clip

Encodes up to four prompt pairs with one CLIP and shared size settings. It is **architecture-
aware**: wire the Smart Loader's `detected_architecture` into the `architecture` input so
non-SDXL checkpoints are conditioned correctly.
- **Inputs:** `clip`, `width`, `height`, `crop_w`, `crop_h`, and optional
  `target_width`, `target_height`, `positive1_g/l`, `negative1_g/l`,
  `positive2_g/l`, `negative2_g/l`, and `architecture`.
- **Outputs:** `clip` (pass-through), `positive1`, `negative1`, `positive2`,
  `negative2` (CONDITIONING).
- Each `*_l` falls back to its `*_g` when empty; `target_*` falls back to
  `width`/`height` when empty; an empty pair encodes an empty string.
- **Conditioning by architecture:**
  - **SDXL / Pony / Illustrious** — dual g/l conditioning (the original behavior).
  - **SD1.5** — encoded through its single encoder; the width/height metadata is attached but
    ignored by SD1.5 (harmless).
  - **Flux / SD3** — best-effort conditioning (a warning is logged recommending the core
    `CLIPTextEncodeFlux` / `CLIPTextEncodeSD3` for native-quality conditioning); the width/height
    metadata is attached but ignored by those models.
  - If `architecture` is left unwired, the node inspects the CLIP's own encoder shape (dual vs
    single) and branches accordingly, so existing SDXL graphs are byte-identical.
   - Replaces multiple `CLIPTextEncodeSDXL` nodes: `positive1/negative1` → main KSampler,
   `positive2/negative2` → FaceDetailer.

### LLM Prompt Studio KSampler (Hires Fix)

Single KSampler node that runs a base pass and an optional hires (upscale) pass. It supports the
studio's **full scheduler list** (standard ComfyUI schedulers plus `AYS SD1` / `AYS SDXL` / `AYS SVD`
/ `GITS`) via the shared AYS/GITS-aware sampler. The full quality path passes **LATENT** from this
node straight into **FaceDetailer**, so no extra VAE decode is needed between them.

- **Inputs:** `model`, `positive`, `negative`, `latent_image`, `seed`, `steps`, `cfg`,
  `sampler_name`, `scheduler`, `denoise`, `hires_enabled`, `hires_upscale_type`,
  `hires_upscale_method`, `hires_latent_upscale_model`, `hires_latent_upscale_factor`,
  `hires_upscale_iterations`, `hires_latent_upscale_tile`, `hires_steps`, `hires_cfg`,
  `hires_denoise`, `hires_sampler_name`,
   `hires_scheduler`, `hires_use_same_seed`, `hires_seed`, `vae_decode`, `preview_method`,
   `vae_tile_size`; optional `hires_upscale_model` (UPSCALE_MODEL), `hires_positive`,
   `hires_negative`, `optional_vae`.
 - **Outputs:** `LATENT`, `IMAGE`, `VAE_TILE_SIZE` (the IMAGE is only produced when `vae_decode`
   is on, otherwise it returns a 1×1 placeholder so downstream graphs stay valid; `VAE_TILE_SIZE`
   echoes the `vae_tile_size` input so it can be wired into FaceDetailer's `vae_tile_size`).
 - **`vae_tile_size`** (default 0 = whole frame) tiles the final VAE decode to cap VRAM on
   large hires latents, and is emitted on the `VAE_TILE_SIZE` output.
- **Seed control:** `seed` and `hires_seed` carry ComfyUI's `control_after_generate` toggle
  (randomize / increment / decrement / fixed) so the field auto-updates after each Generate.
  `hires_seed` is only used when `hires_use_same_seed` is off; otherwise the base `seed` drives
  the hires pass. Both `denoise` (base, default `1.0`) and `hires_denoise` (hires, default `0.5`)
  are forwarded to `KSampler().sample(...)`, so lowering them genuinely reduces how much the
  latent is re-noised.
- **Hires pass:** enabled when `hires_enabled` is true and the derived target size differs from
  the base latent. The hires output size is computed automatically as `base × per_pass_scale^iterations`,
  where `per_pass_scale` is `hires_latent_upscale_factor` for the latent types and the model scale
  for `pixel (model)`, and `hires_upscale_iterations` is the number of progressive upscale passes
  (this avoids fiddly explicit width/height fields). `hires_upscale_type` makes the upscaler
  explicit: `latent` = interpolate the latent (no model), `latent (model)` = a LatentUpscaleModel,
  `pixel (model)` = decode → super-res UPSCALE_MODEL → re-encode (requires the `hires_upscale_model`
  input). The hires pass reuses the base sampler/scheduler unless overridden by `hires_sampler_name`
  / `hires_scheduler` (`base` = reuse base).
- **Latent upscale (`latent` type):** `hires_upscale_type = latent` interpolates the latent in
  latent space (no model). It accepts fractional factors (e.g. `1.25`, `1.5`). Previously a
  fractional factor fed the cell dimensions into `LatentUpscale` as if they were pixels (which
  divides by 8 again), shrinking the latent to 1/64 of its area and aborting the UNet forward
  with `Fatal Python error: Aborted`; this is fixed — the target is now converted back to pixel
  dimensions before the call.
- **`latent (model)` model list:** the `hires_latent_upscale_model` dropdown only lists the
  project's latent resizers from `models/upscale_models` (ttl-nn `sd15_resizer.pt` /
  `sdxl_resizer.pt`). Pixel ESRGAN `.safetensors` are **not** valid as a `LatentUpscaleModel` and
  are no longer listed. The broken City96 `latent-upscaler-v2.1_*.safetensors` were removed.
- **Latent upscale memory:** the `latent (model)` upscalers apply the net directly to the latent on
  ComfyUI's compute device (falling back to CPU when VRAM is short). Their attention is evaluated in
  chunks, so its `tokens²` score matrix can no longer blow up (that was the
  `DefaultCPUAllocator: not enough memory ... 3600000000 bytes` crash at large hires targets).
  Oversized latents are upscaled in overlapping tiles instead; `hires_latent_upscale_tile` = 0 keeps
  that automatic (whole latent while it fits, best quality), and a positive value (e.g. `64`) forces
  tiling on a low-memory machine at the cost of some quality, because each tile is normalized on its
  own.
- **Preview method:** `preview_method` controls how the output `IMAGE` is generated (mirrors
  Efficient KSampler): `vae` = full VAE decode (most accurate, default), `latent2rgb` = fast
  approximate latent→RGB, `taesd` = TAESD preview if available (else VAE), `none` = no preview
  image. The `vae_decode` toggle independently gates whether any decode happens at all.
- **AYS / GITS schedulers** require an external custom node (ComfyUI-AlignYourSteps for AYS,
  KJNodes for GITS). When that node is not installed, the sampler logs a warning and falls back to
  a standard scheduler (`karras` for AYS, `simple` for GITS) so the studio still works
  out-of-the-box.

### LLM Prompt Studio Face Detailer

A port of Impact-Pack's `enhance_detail`: it finds faces in the frame, upscales each face's
crop to `guide_size`, re-renders the improved face (inpaint/refine) and composites it back with
a feathered mask. It accepts either a pixel `image` or a `latent` (optimized path — one shared
VAE decode for detection, crops straight from the latent). The `latent` input is the intended
input when chained after the Hires Fix node.

- **Inputs (required):** `model`, `vae`, `positive`, `negative`, `seed` (INT,
  randomize/increment/decrement/fixed), `steps` (INT, 20), `cfg` (FLOAT, 7.0), `sampler_name`,
  `scheduler`, `denoise` (FLOAT, 0.5), `guide_size` (INT, 512), `max_size` (INT, 1024),
   `crop_factor` (FLOAT, 1.5 — context margin kept around the face), `detection_method`
   (`haar` / `yolo` / `yolo_seg` / `yolo_bbox_seg`, default `haar`), `yolo_model_name` (YOLO bbox detector,
   default `face_yolov8s.pt`), `yolo_seg_model_name` (seg model for `yolo_seg` / `yolo_bbox_seg`, default `(none)`),
   `yolo_seg_class` (COMBO, `face`), `gender_filter` (`any` / `female` / `male`, default `any`),
   `gender_model_name` (gender classifier, default `(none)`), `gender_model_female_class`
   (COMBO, `female`), `gender_threshold` (FLOAT, 0.5), `detection_threshold` (FLOAT, 0.5),
  `drop_size` (INT, 0), `feather` (INT, 5 — mask feather radius in px), `mask_shape`
  (`square` / `oval`, default `square`), `bbox_scale` (FLOAT, 1.0), `iterations` (INT, 1),
   `inpaint_model` (BOOLEAN, false), `match_color` (COMBO, `luminance`).
- **Inputs (optional):** `image` (IMAGE, force input), `latent` (LATENT, force input),
  `face_positive` / `face_negative` (CONDITIONING), `vae_tile_size` (INT, 0 = whole frame).
- **Outputs:** `IMAGE` (the composited frame with the refined faces). All diagnostic info is
  written to the ComfyUI console (the node is an output node).
- **`yolo_seg_class`** (default `face`): which YOLO seg-model class to treat as a face. The
  widget is a combo that is **populated automatically from the selected `yolo_seg_model_name`** —
  picking a model lists its real class names (e.g. `face`, `hair`, `skin`), so you no longer type
  a raw index. Detections of other classes (person, car, …) are ignored — this removes the false
  positives that forced `detection_threshold` down to 0.1–0.2, so you can keep it at a normal ~0.5.
- **`detection_threshold`** (default 0.5): detector confidence cutoff; keep around 0.5 after
  enabling the class filter. It is forwarded to the YOLO model as its `conf`, so the node's
  slider is authoritative even for weak faces below ultralytics' built-in 0.25 default.
- **`gender_filter`** (default `any`): restrict refinement to one gender. `female` / `male`
  keep only faces of the chosen gender and skip the rest; `any` processes every detected face.
  It runs a lightweight ultralytics **gender classification model** (set in `gender_model_name`)
  on each detected face crop and drops faces whose predicted class does not match. The female
  class index is set in `gender_model_female_class` (default `female`). The widget is a combo that
  is **populated automatically from the selected `gender_model_name`** — picking a model lists its
  real class names (e.g. `female`, `male`). If no gender model is selected,
  the filter logs a warning and is disabled (all faces processed). Drop a `.pt` classifier into
  `models/ultralytics/gender/`.
- **`gender_threshold`** (default 0.5): minimum confidence of the gender classifier. Faces whose
  predicted gender confidence is below this are treated as `unknown` and are **kept** (not dropped
  by the gender filter) regardless of `gender_filter`. Only applies when `gender_filter != any`.
- **`drop_size`** (default 0): faces whose shorter side (px) is below this are skipped; `0` =
  process every face.
- **`guide_size`** / **`max_size`**: target size of the face's shorter side after upscaling the
  crop. Faces already at/above `guide_size` are skipped; `max_size` caps the crop from above,
  lowering the upscale ratio when needed.
- **Brightness match:** the refined face is matched to the surroundings' brightness/contrast
  within the feathered mask (`match_luminance`), removing the "face brighter than the frame" seam.
- **`match_color`** (default `luminance`): how the refined face is blended back into the frame.
  `luminance` matches only brightness/contrast; `lab` runs the same affine transfer in LAB space so
  hue/chroma seams are removed too; `histogram` does per-channel CDF matching against the surrounding
  original; `none` disables matching. OpenCV is imported lazily, so the node still loads if OpenCV is
  unavailable (falling back to `luminance`).
- **Logging:** the node writes to the ComfyUI console (prefix `[FaceDetailer]`) the start params
  (method, guide_size, max_size, drop_size, yolo_seg_class), each detected face (size in px), the
  upscale ratio, the max_size crop clamp, any skip (already ≥ guide_size / smaller than drop_size)
  and a final count of processed faces; the gender step logs the per-image count of detected
  genders (`N female, M male, K unknown`).
- **Detection:** `haar` uses the built-in cv2 cascade; `yolo` uses the YOLO bounding-box detector
  selected in `yolo_model_name`; `yolo_seg` uses `yolo_seg_model_name` to produce a per-face
  segmentation mask of the real shape instead of a rectangle (`yolo_seg_class` selects the face
  class); `yolo_bbox_seg` locates faces with the strong bbox model (`yolo_model_name`) and only
  borrows the seg model (`yolo_seg_model_name`) to shape the inpaint mask inside each located crop
  (matched by IoU, falls back to a rectangle). The detection model is loaded once and cached for the
  session, so repeated runs don't
  reload it on every pass. `mask_shape` (`square` / `oval`) sets the inpaint mask shape for
  `haar` / `yolo` (ignored for `yolo_seg`); `bbox_scale` (0.1–3.0, default 1.0) expands/contracts
  the crop around the face center (e.g. to include jaw/neck); `iterations` (1–10, default 1)
  repeats the refinement pass over the upscaled crop for stronger cleanup; `feather` (px) softens
  the mask edge when compositing back; `vae_tile_size` (px, default 0 = whole frame) tiles the VAE
  decode of the refined crop to cap VRAM on large faces; `inpaint_model` switches to
  `InpaintModelConditioning` so the model sees the original face structure while regenerating.
- **Per-face prompts:** when `face_positive` / `face_negative` are wired (e.g. from the Writer's
  `face_positive` / `face_negative` outputs via Smart Multi-Clip), they override `positive` /
  `negative` for each cropped face; otherwise the main conditioning is used. Each face crop is
  upscaled so its shorter side reaches `guide_size`, then refined with the node's own KSampler and
  pasted back with a feathered mask.

---

## Prompt templates and style presets (`Styles/`)

All default system prompts and all style presets live in the **`Styles/`** folder at the package
root — edit them without touching Python.

- **`Styles/system_prompts.json`** — the base system prompts and the orthogonal assembly fragments
  (nsfw policy, prompt_format, negative handling, face instruction, architecture addenda, reasoning
  hint). Keys: `writer_system`, `face_instruction`, `critic_system`, `describe`, `composer`,
  `reasoning_hint`, `nsfw_policy` (`on`/`off`), `prompt_format` (`natural`/`tags`/`weighted`/
  `structured`/`midjourney`/`booru`), `negative` (`on`/`off`), `face_prompt` (`on`/`off`),
  `architecture_guidance` (per-arch `system_addendum` / `default_negative` / `force_no_negative`).

- **`Styles/<direction>.json`** — 25 direction files, each containing a `presets` list. Together they
  provide **162 style presets** (see [Style presets](#style-presets)). A preset may `extends`
  another preset id (inheritance: child merges `system_prompt` + `system_prompt_suffix` and unions
  `style_tags_*`, up to depth 3). Each preset supports: `id`, `name`, `category`, `description`,
  `system_prompt`, `system_prompt_no_negative` (used when the negative prompt is off or the arch
  forces no-negative), `system_prompt_suffix`, `style_tags_positive`, `style_tags_negative`,
  `blend_note`, `disabled_in_no_negative_mode`.

On first run the whole `Styles/` folder is copied into a user-editable
**`llm_prompt_studio_styles/`** folder inside the ComfyUI output directory (never overwritten — hand
edits survive a refresh). Use the node **Reload Styles** / **Reset Styles** buttons to re-read disk or
restore the shipped set. The legacy `presets_default.json` is kept only as a fallback source.
Every node also exposes its prompt as an editable widget, so you can override per-node.

---

## Prompt library

Approved prompts can be stored in a JSON library (default
`output/llm_prompt_studio_library.json`). Each entry keeps `prompt`, `negative_prompt`,
`face_positive` and `face_negative`. A save is treated as a duplicate (and skipped) only when
**all four** fields match an existing entry, so two prompts that share the same positive but
have different face prompts are both kept. Writes are serialized by a per-file lock and are
atomic (`.tmp` + rename); the previous file is kept as `*.json.bak` for recovery. Load them
back with **Library Loader**.

---

## Style presets

The Writer's `style_preset` combobox lists **162 built-in styles** across ~25 domains
(Photography, Art Movements, Asian Art, Traditional Media, Digital & Contemporary,
Fantasy & Horror, Anime & Manga, Cinematic, Period & Style, Basic Styles, and more); each
entry is shown as `Category > Name`. Presets are loaded from the `Styles/` folder (see
[Prompt templates and style presets](#prompt-templates-and-style-presets-styles)). Selecting one:

- appends the preset's `style_tags_positive` / `style_tags_negative` to the generated
  `positive` / `negative` (and to `face_positive` / `face_negative`, so a refined face matches
  the style); a style tag already present in the prompt (or in `face_positive`) is not appended
  again — duplicates are skipped (this already applied to the negative tags); the preset's
  negative tags are skipped in no-negative mode;
- overrides the system prompt with the preset's `system_prompt` **only when** the system-prompt
  widget is left at its default. In no-negative mode the preset's dedicated
  `system_prompt_no_negative` variant is used instead (it requires an empty `negative` /
  `face_negative`), so the preset's style still applies without contradicting the mode.

Every preset's system prompt carries an **Engineering rules** block (derived from SDXL prompt-engineering
practice): compose in clear layers (subject → appearance → action/pose → object relationships → environment →
composition → lighting → color palette → mood → style), lead with the main subject, prefer concrete
visual descriptors over abstract booster adjectives, keep negatives short and targeted, and (optionally) weight only
the most important trait as `(concept:1.1)`–`(concept:1.3)`. NSFW is fully authorized — intimate content must be
described with anatomically correct, precise terms and is never censored.

Presets are copied from the `Styles/` folder into an editable `llm_prompt_studio_styles/`
folder in the ComfyUI output directory on first run (never overwritten — hand edits survive a
refresh). Use the node **Reload Styles** button to re-read the folder after editing, and
**Reset Styles** to delete the user copy and fall back to the shipped `Styles/`. A preset with
`disabled_in_no_negative_mode` set is automatically skipped in no-negative mode.

---

## Auto-revision loop

With `auto_loop` enabled on the Critic, the pack closes the revision loop automatically:

1. Writer generates a prompt → image is rendered → Critic scores it.
2. If rejected, the Critic's `revision_notes` are injected into the Writer's
   `revision_notes` widget and the workflow is re-queued.
3. The Writer regenerates a corrected prompt, the image re-renders, the Critic re-scores.
4. Repeats until `approved` or `max_retries` is reached (counter resets on approval).

Requirements: `reuse_last_prompt` OFF on the Writer, and `revision_notes` NOT wired as a
graph edge from Critic to Writer (the feedback is injected via the widget to avoid a
cycle in the execution graph).

---

## Security

External input is validated to reduce SSRF and path-traversal risk. Both guards are
strict by default and can be relaxed with a flag if you need remote servers or paths
outside the output folder.

- **`server_url` (SSRF guard)** — only `http`/`https` and local/private hosts
  (`localhost`, loopback, RFC-1918, link-local) are allowed. Other local hostnames
  (e.g. `nas.local`, `my-pc`) are resolved via `socket.getaddrinfo` and accepted only
  when *every* resolved IP is loopback / private / link-local; otherwise the URL is
  rejected. To use a remote/public LM Studio server, set the environment variable
  `LLM_PROMPT_STUDIO_ALLOW_PUBLIC=true` (or assign `ALLOW_PUBLIC_SERVER_URLS = True`
  directly in `lm_http.py`).
- **No internal error leakage** — the server routes (`/models`, `/library/save`,
  `/sampler_params`) return generic error messages to the client and log the real
  exception server-side, so internal paths/causes are never exposed to the caller.
- **`library_path` / `save_dir` (path-traversal guard)** — confined to the ComfyUI
  output directory. To allow paths outside it, set `RESTRICT_PATHS_TO_OUTPUT = False`
  in `library.py`.
- **Global LLM-response cache** — OFF by default. Set `LLM_PROMPT_STUDIO_LLM_CACHE=true`
  to reuse identical chat-completion responses within a session (bounded LRU, ~256 entries).
  The cache key excludes the API key and lives below the per-node `reuse_last_prompt` cache.
- **VRAM release after LLM run** — ON by default. Each LLM node (Writer, Image Critic,
  Scene Builder) unloads its LM Studio model when it finishes, so the diffusion pipeline
  (KSampler Hires Fix / Face Detailer) gets the VRAM back and the vision model can load
  after a sampling pass. The boolean widget **`release_vram_after_run`** (in ⚙ Advanced
  settings, default **true**) controls this per node; set it to **false** to keep the model
  resident (e.g. when you have enough VRAM for the LLM and the checkpoint at once). A global
  kill switch, `LLM_PROMPT_STUDIO_KEEP_MODEL_LOADED=1`, disables the release everywhere
  without editing any node. Because each release costs one reload (~5–10 s on typical
  hardware, the GGUF stays in OS page cache), repeated runs trade a little latency for
  avoiding CUDA OOM on small GPUs.
- The prompt library is written atomically (`.tmp` then rename) to avoid corruption.

---

## Tests

Golden tests cover JSON parsing, the salvage logic, `slugify` and checkpoint-family
detection:

    pip install pytest
    python -m pytest tests/ -v

The ComfyUI-specific `folder_paths` module is stubbed in `tests/conftest.py`, so tests
run without ComfyUI (but need `requests`, `numpy`, `pillow`).

---

## Debug logging & advanced options

### Debug logging

Logging is OFF by default. To enable, edit `debug.py` at the package root and set `DEBUG_LEVEL`
(requires a ComfyUI restart):

- `MINIMAL` — logs node enter / exit / error.
- `FULL` — adds HTTP request/response and JSON-parse attempts.

The rotating log is written to `llm_prompt_studio.log` in the ComfyUI output directory only when
logging is active. API keys (Bearer tokens, `api_key` fields) and base64 image blobs are masked
or truncated and never written in full.

### Native LM Studio v1 API & advanced options

The nodes call LM Studio's native `/api/v1/chat` endpoint (and `/api/v1/models/load` for
`flash_attention` / `offload_kv_cache_to_gpu`) for both text and vision prompts. The native
endpoint sends the system prompt as a top-level `system_prompt`, renders message content as typed
`input` parts (text + `data_url` images), and sets `store: false` so LM Studio keeps no server-side
conversation state. The advanced widgets — `reasoning`, `flash_attention`, `offload_kv_cache_to_gpu`,
`repeat_penalty`, `top_k`, `top_p`, `min_p` — are forwarded to those endpoints when the server
supports them. `reasoning` is only sent when the model actually exposes it (detected via
`capabilities.reasoning.allowed_options`); if a server rejects it, the call retries once without it.
The OpenAI `/chat/completions` route is kept only as a graceful fallback (e.g. when native vision is
rejected). Model loads are resilient: if a server rejects an optional load parameter
(e.g. `gpu_offload` on some builds), that key is dropped and the load still succeeds; before loading
a model, **every currently loaded instance is unloaded via its `instance_id`**, so only the selected
model remains resident.

The model list is cached **per LM Studio server URL** (the cache key is the URL only — the API
  key is never persisted). **🔄 Refresh models** populates the combo for that server; each node
  patches only its own server's list, so multiple servers don't cross-contaminate. The same
  refresh no longer overwrites the model you currently have selected.

**Live server status.** Writer / Image Critic / Scene Builder show a `server_status` display widget
(enabled by default) that polls the pack's `GET /llm_prompt_studio/status` route every few seconds
and reports `● Connected — <loaded model names>`, `● Connected (no model loaded)`, or
`● Server down`. Embedding
models returned by LM Studio's model list are filtered out, so they never appear as chat models in
the combo.

---

## Typical workflow

1. **Writer** generates a prompt from your idea.
 2. Prompt → your SDXL sampling graph (use **Smart Loader** + **Smart Multi-Clip**).
3. Rendered image → **Critic** (vision model scores it).
4. `approved` → **Smart Save** (saves only good images).
5. Enable **`auto_loop`** on the Critic to refine automatically, or wire
   `revision_notes` manually for hands-on control.
6. Optionally save the prompt to the **library** and reload it later.

---

## Notes & troubleshooting

- **No model in the combo?** Click **🔄 Refresh models** on the Writer/Critic, and make
  sure the LM Studio server is running at `server_url`.
- **Critic says the model is not vision-capable?** Pick a VL model or uncheck
  `vision_check`.
- **Reasoning text leaking into results?** The pack falls back to a model's
  `reasoning_content` when the normal `content` field is empty, so thinking output can
  end up in results. Disable thinking/reasoning for that model in LM Studio to get clean
  output.
- **Placeholders** like `— server unavailable —` mean the server could not be reached.
- **Model forgets to generate some JSON fields?** Writer and Scene Builder (stage 2) detect
  missing required fields (`positive`, `negative`, `scene_name`, and face fields when
  requested) and automatically re-ask the model up to `max_field_retries` times for a
  complete answer. If `scene_name` is still missing it falls back to a slug of `positive`;
  if `positive`/`negative` are still missing the node raises an error. If your model keeps
  omitting fields, increase `max_field_retries` or switch to a model with better instruction
  following (the `writer_system` / `composer` prompts already require all fields).
- Logs use Python `logging` under the `llm_prompt_studio` logger with a `[LLMPromptStudio]`
  prefix. Lower that logger's level to DEBUG to see LLM call timings and cache activity.

---

## Acknowledgements

This project builds on ideas and inspiration from other open-source ComfyUI projects
and their authors:

- [ComfyUI-SmartPromptCrafter](https://github.com/jideka/ComfyUI-SmartPromptCrafter)
- [comfyui-llm-prompt-enhancer](https://github.com/pinkpixel-dev/comfyui-llm-prompt-enhancer)
- [efficiency-nodes-comfyui](https://github.com/jags111/efficiency-nodes-comfyui)
- [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)

Thanks to their authors for the concepts that informed this node pack.
