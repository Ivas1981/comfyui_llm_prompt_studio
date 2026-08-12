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
    `face_prompt_instruction`, `prompt_mode`, `family`.
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

### LLM Prompt Studio Smart Loader

I will now call the tool to update README with the recommended environment var guidance in the Security section. This will be committed to main. Proceeding to write the change. If you want it in a separate branch instead, tell me now.