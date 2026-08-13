# ComfyUI LLM Prompt Studio

A ComfyUI custom-node pack that drives SDXL prompt engineering with a **local LLM**
served by [LM Studio](https://lmstudio.ai) or any **local OpenAI-compatible server**.

It closes the full loop: **generate a prompt → render the image → critique it with a
vision model → auto-revise → save approved results with full metadata**, plus a prompt
library, scene rebuilding from images, and smart model/LoRA loading.

No cloud, no API keys required — everything runs on your machine.

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
  (DMD / LCM / Turbo / Hyper / Lightning / Flash) and conditionally applies a distillation LoRA.
- **Multi-CLIP SDXL** — encodes up to four SDXL prompt pairs with one CLIP and shared
  size settings.
- **Style presets** — pick from 14 built-in styles (Photorealism, Cinematic, Anime, 3D Render,
  …); a preset appends its style tags to the prompt and can override the system prompt.
- **Live streaming** — with `stream` enabled, generated tokens appear in the node's
  `generation_view` in real time over the ComfyUI websocket.
- **Advanced LM Studio v1 options** — `reasoning`, `flash_attention`, `offload_kv_cache_to_gpu`,
  `repeat_penalty`, `top_k`, `top_p`, `min_p` are forwarded to LM Studio's native v1 API when
  supported.
- **Debug logging** — opt-in (`debug.py`, OFF by default) logs node/HTTP/parse activity with
  API keys and image blobs masked.

---

## Requirements

- ComfyUI (recent version).
- Python packages: `requests`, `numpy`, `pillow` (declared in `requirements.txt`;
  usually already present in a ComfyUI environment).
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
   pip install requests numpy pillow
   ```
3. Start LM Studio, load a model and enable the local server.
4. Restart ComfyUI. You should see in the console:
   ```
   [LLMPromptStudio] Package loaded: Writer, Critic, Smart Save, Library Loader,
   Scene Builder, Smart Loader, Multi-CLIP SDXL.
   ```

### Package layout

```
comfyui_llm_prompt_studio/
├── __init__.py               # logging setup + node registration
├── constants.py              # shared placeholder constants
├── lm_http.py                # HTTP client, model cache, SSRF guard
├── combos.py                 # combo lists (models / checkpoints / LoRA / VAE)
├── parsing.py                # JSON extraction / salvage / slugify
├── library.py                # prompt library, path guard, atomic writes
├── imaging.py                # image tensor → base64 JPEG
├── model_meta.py             # generation metadata, safetensors, family detection
├── server_routes.py          # /llm_prompt_studio/* routes used by the JS bridge
├── prompts.json              # default system prompts (edit without touching code)
├── nodes/
│   ├── __init__.py           # node mappings
│   ├── _defaults.py          # loads prompts.json
│   ├── writer.py
│   ├── critic.py
│   ├── smart_save.py
│   ├── library_loader.py
│   ├── scene_builder.py
│   ├── smart_loader.py
│   └── multi_clip.py
├── tests/                    # pytest golden tests
├── requirements.txt
├── pyproject.toml
└── web/js/
    ├── llm_prompt_studio_shared.js    # constants, helpers, shared state
    ├── llm_prompt_studio_actions.js   # button handlers, re-queue helper
    └── llm_prompt_studio_bridge.js    # entry point: extension + executed handler
```

---

## Nodes

All nodes live under the **`LLM Prompt Studio`** category.

### LLM Prompt Studio Writer

Generates an SDXL prompt from an idea.
  - **Inputs:** `server_url`, `api_key`, `model`, `context_length`, `gpu_offload`,
    `system_prompt`, `idea`, `revision_notes`, `temperature`, `max_tokens`, `seed`,
    `reuse_last_prompt`, `generate_face_prompts`, `max_field_retries`,
    `face_prompt_instruction`, `prompt_mode`, `family`, `style_preset`, `stream`,
    `reasoning`, `flash_attention`, `offload_kv_cache_to_gpu`, `repeat_penalty`,
    `top_k`, `top_p`, `min_p`.
  - **`prompt_mode`** selects how the negative prompt is handled: `auto` (default) switches
    to a no-negative system prompt when the checkpoint family is distilled (DMD / LCM /
    Turbo / Hyper / Lightning / Flash); `standard` always emits a negative; `no_negative`
    always emits an empty negative. Distilled models sample at CFG ~1, where a negative
    prompt is mathematically ignored, so the no-negative prompt rephrases every constraint
    as a positive statement instead.
  - **`family`** (optional) — wire the Smart Loader's `detected_family` output here to let
    `auto` detect the distilled family automatically. Leave unconnected to use `standard`.
  - Loads the selected model on the LM Studio server **on demand** (via the load API) using
    `context_length` (default 8192) and `gpu_offload` (1.0 = max GPU), and unloads the
    previously loaded model for this node type when you switch. You no longer need to pre-load
    the model in the LM Studio GUI.
- **Outputs:** `positive`, `negative`, `raw`, `scene_name`, `face_positive`,
  `face_negative`.
- Toggle **`generate_face_prompts`** to also get short face-only prompts for
  FaceDetailer/inpainting. Empty face fields automatically fall back to the main
  `positive`/`negative`.
- **`reuse_last_prompt`** returns the cached result without calling the LLM (useful to
  re-render without re-generating the prompt). Note: turn it OFF for the auto-revision
  loop to work.
- **`seed`** is forwarded to the LLM request (for servers that honor it).
- **`max_field_retries`** (default 2): if the model returns a JSON answer missing required
  fields, the node re-asks it for a complete answer up to this many times. An empty
  `scene_name` then falls back to a slug of `positive`; empty `positive`/`negative` raise
  an error.
- Button **🔄 Refresh models** re-reads the model list from the server.
- **Style presets** (`style_preset`): pick a built-in style to append its style tags to the
  generated prompt (and override the system prompt when the system-prompt widget is left at
  default). Presets that opt out of no-negative mode are skipped automatically in that mode.
  Use "Reload presets" / "Reset to defaults" in the node menu to manage them.
- **Live streaming** (`stream`): when enabled, generated tokens are pushed to `generation_view`
  in real time over the ComfyUI websocket instead of appearing only at the end.
- **Advanced options** (`reasoning`, `flash_attention`, `offload_kv_cache_to_gpu`,
  `repeat_penalty`, `top_k`, `top_p`, `min_p`): forwarded to LM Studio's native v1 API when the
  server supports them (or when streaming is on). Leaving them at defaults keeps the old
  OpenAI-compatible behavior. `reasoning` enables thinking-model reasoning; `flash_attention` /
  `offload_kv_cache_to_gpu` are model-load options.

### LLM Prompt Studio Image Critic

A vision model scores how well the image matches the prompt.
  - **Inputs:** `image`, `prompt`, `server_url`, `api_key`, `model`, `context_length`,
    `gpu_offload`, `critic_prompt`, `threshold`, `image_max_size`, `temperature`,
    `max_tokens`, `clear_notes_on_approve`, `auto_loop`, `max_retries`, `vision_check`.
- **Outputs:** `approved` (bool), `score` (int), `revision_notes`, `verdict`, `raw`.
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
    `api_key`, `model`, `context_length`, `gpu_offload`, `describe_prompt`,
    `composer_prompt`, `user_changes`, `image_max_size`, `temperature`, `max_tokens`,
    `max_field_retries`, `vision_check`, `description_view`, `prompt_mode`, `family`.
  - **`prompt_mode`** and **`family`** behave exactly as on the Writer: `auto` switches to a
    no-negative composer prompt for distilled checkpoint families when `family` is wired from
    the Smart Loader's `detected_family` output.
- **Outputs:** `positive`, `negative`, `scene_name`, `prompt_view`, `description`.
  Stage 1 puts the vision description into `description`; stage 2 composes the prompt.
- **`max_field_retries`** (default 2): in stage 2, if the model's JSON omits required
  fields, the node re-asks for a complete answer; an empty `scene_name` falls back to a
  slug of `positive`.
- Button **→ Send to Writer** copies the description/prompt into a Writer's `idea`.
- Button **🔄 Refresh models** (same as on the Writer/Critic) re-reads the model list from the server.

### LLM Prompt Studio Smart Loader

Loads a checkpoint and handles distillation LoRA automatically.
- **Inputs:** `ckpt_name`, `family_override`, `lora_name`, `apply_lora`,
  `strength_model`, `vae_user`.
- **Outputs:** `MODEL`, `CLIP`, `VAE_MODEL`, `VAE_USER`, `detected_family`.
- Detects the checkpoint family from the filename + safetensors metadata:
  `base`, `dmd`, `lcm`, `turbo`, `hyper`, `lightning`, `flash`. `family_override`
  forces it.
- `apply_lora`: **`auto`** applies the LoRA only for `base` (non-distilled) models,
  **`always`** forces it, **`never`** skips it.
- `VAE_USER` falls back to the checkpoint's built-in VAE when `vae_user = [none]`.
  Use `VAE_USER` downstream so the output is never empty.
- The node title shows the detected family.

### LLM Prompt Studio Multi-CLIP SDXL

Encodes up to four SDXL prompt pairs with one CLIP and shared size settings.
- **Inputs:** `clip`, `width`, `height`, `crop_w`, `crop_h`, and optional
  `target_width`, `target_height`, `positive1_g/l`, `negative1_g/l`,
  `positive2_g/l`, `negative2_g/l`.
- **Outputs:** `clip` (pass-through), `positive1`, `negative1`, `positive2`,
  `negative2` (CONDITIONING).
- Each `*_l` falls back to its `*_g` when empty; `target_*` falls back to
  `width`/`height` when empty; an empty pair encodes an empty string.
- Replaces multiple `CLIPTextEncodeSDXL` nodes: `positive1/negative1` → main KSampler,
  `positive2/negative2` → FaceDetailer.

---

## Prompt templates (`prompts.json`)

All default system prompts live in **`prompts.json`** at the package root — edit them
without touching Python. Keys:

| Key | Used by |
|-----|---------|
| `reasoning_hint` | appended to writer/critic/composer (clear it to disable) |
| `writer_system` | Prompt Writer |
| `face_instruction` | Prompt Writer (face prompts) |
| `critic_system` | Image Critic |
| `describe` | Scene Builder (stage 1) |
| `composer` | Scene Builder (stage 2) |
| `writer_system_no_negative` | Prompt Writer (distilled/no-negative mode; falls back to `writer_system`) |
| `composer_no_negative` | Scene Builder (distilled/no-negative mode; falls back to `composer`) |
| `face_instruction_no_negative` | Prompt Writer face prompts (no-negative mode; falls back to `face_instruction`) |

Changes take effect after restarting ComfyUI (templates are loaded at startup). Every
node also exposes its prompt as an editable widget, so you can override per-node.

---

## Prompt library

Approved prompts can be stored in a JSON library (default
`output/llm_prompt_studio_library.json`). Each entry keeps `prompt`, `negative_prompt`,
`face_positive` and `face_negative`. Duplicates (same positive) are skipped. Load them
back with **Library Loader**. Writes are atomic (`.tmp` + rename) to avoid corruption.

---

## Style presets

The Writer's `style_preset` combobox lists 14 built-in styles (Photorealism, Cinematic, Anime,
3D Render, Digital Art, Oil Painting, Watercolor, Pixel Art, Comic, Fantasy, Cyberpunk,
Steampunk, Ukiyo-e, Minimalist). Selecting one:

- appends the preset's `style_tags_positive` / `style_tags_negative` to the generated
  `positive` / `negative` (skipped for `negative` in no-negative mode);
- overrides the system prompt with the preset's `system_prompt` **only when** the system-prompt
  widget is left at its default.

Presets are copied from `presets_default.json` to an editable `llm_prompt_studio_presets.json`
in the ComfyUI output directory on first run. Use the node menu **"Reload presets"** to refresh
the combo after editing that file, and **"Reset to defaults"** to restore the shipped set
("Copy presets path" puts the file location on your clipboard). A preset with
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
  (`localhost`, loopback, RFC-1918, link-local) are allowed. To use a remote/public
  LM Studio server, set the environment variable `LLM_PROMPT_STUDIO_ALLOW_PUBLIC=true`
  (or assign `ALLOW_PUBLIC_SERVER_URLS = True` directly in `lm_http.py`).
- **`library_path` / `save_dir` (path-traversal guard)** — confined to the ComfyUI
  output directory. To allow paths outside it, set `RESTRICT_PATHS_TO_OUTPUT = False`
  in `library.py`.
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

### Live streaming & advanced LM Studio options

Enable `stream` on the Writer / Critic / Scene Builder to push generated tokens to the node's
`generation_view` widget in real time (over the ComfyUI websocket). Streaming failures degrade
gracefully to a normal completion.

The advanced widgets — `reasoning`, `flash_attention`, `offload_kv_cache_to_gpu`,
`repeat_penalty`, `top_k`, `top_p`, `min_p` — are forwarded to LM Studio's native v1 API
(`/api/v1/chat` for sampling, `/api/v1/models/load` for `flash_attention` /
`offload_kv_cache_to_gpu`). They only take effect when the server supports them (or when `stream`
is on); defaults keep the OpenAI-compatible behavior. Multimodal (image) requests always use the
OpenAI path, where these options are ignored.

---

## Typical workflow

1. **Writer** generates a prompt from your idea.
2. Prompt → your SDXL sampling graph (use **Smart Loader** + **Multi-CLIP SDXL**).
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
