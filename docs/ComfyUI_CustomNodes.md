# ComfyUI Custom Node Development Guide

> Targeted at **ComfyUI version 0.19.3**
> Audience: Python/Javascript developers who want to extend ComfyUI with new nodes.
> Sources: Official ComfyUI docs (docs.comfy.org), the ComfyUI repository (`custom_nodes/example_node.py.example`, `nodes.py`), and community tutorials.

---

## Table of Contents

1. [What a Custom Node Is](#1-what-a-custom-node-is)
2. [Client–Server Model](#2-client-server-model)
3. [Two Ways to Build a Node Pack](#3-two-ways-to-build-a-node-pack)
4. [Project Layout](#4-project-layout)
5. [The Node Class Contract](#5-the-node-class-contract)
6. [INPUT_TYPES in Detail](#6-input_types-in-detail)
7. [Data Types & Widget Options](#7-data-types--widget-options)
8. [Hidden Inputs](#8-hidden-inputs)
9. [Custom, Wildcard & Dynamic Inputs](#9-custom-wildcard--dynamic-inputs)
10. [Execution Model & Caching](#10-execution-model--caching)
11. [Registering Your Nodes](#11-registering-your-nodes)
12. [Client-Side (JavaScript) Extensions](#12-client-side-javascript-extensions)
13. [Complete Worked Example (LLM Prompt Node)](#13-complete-worked-example-llm-prompt-node)
14. [Packaging, Dependencies & Publishing](#14-packaging-dependencies--publishing)
15. [Debugging & Common Pitfalls](#15-debugging--common-pitfalls)
16. [Version Notes for 0.19.3](#16-version-notes-for-0193)
17. [Reference Links](#17-reference-links)

---

## 1. What a Custom Node Is

A custom node is just like any built-in ComfyUI node: it receives **inputs**, does **something** to them, and produces **outputs**. Many real nodes do only one small thing (e.g., invert an image, concatenate two strings, call an external API).

Custom nodes let you add features that are not in the core, integrate external services (LLMs, APIs, file systems), and share that functionality with the community through the Comfy Registry.

The simplest possible node:

```python
class InvertImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "image_in": ("IMAGE", {}) },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)
    CATEGORY = "examples"
    FUNCTION = "invert"

    def invert(self, image_in):
        image_out = 1 - image_in
        return (image_out,)
```

---

## 2. Client–Server Model

ComfyUI runs as a **client–server** application:

- **Server (Python):** does all the real work — tensor processing, model loading, diffusion, API calls, etc. Almost all custom nodes live here as Python classes.
- **Client (JavaScript/TypeScript):** renders the UI, the node graph, and the widgets.

Custom nodes fall into four categories:

| Category | Description |
| --- | --- |
| **Server-side only** | A Python class defining inputs/outputs and a processing function. The most common type. |
| **Client-side only** | Modifies the UI without adding core functionality (may not add any node at all). |
| **Independent client & server** | Adds both backend features and related UI (e.g., a widget for a new data type). Usually communicates via the normal Comfy data flow. |
| **Connected client & server** | The UI and server talk to each other directly (e.g., server pushes a message to the frontend). |

> ⚠️ Any node that requires client–server communication will **not** be compatible with ComfyUI's headless **API** mode.

---

## 3. Two Ways to Build a Node Pack

### Option A — Scaffold (recommended for a real pack)

ComfyUI ships with `comfy-cli`. The recommended workflow for a new pack:

```bash
cd ComfyUI/custom_nodes
comfy node scaffold
```

It asks a few questions (author, email, GitHub username, project name, license, whether to include a `web/` directory for JS) and produces a ready-made folder with:

- `pyproject.toml`
- `src/nodes.py` (where you put node classes)
- `web/js/` (optional, for frontend extensions)
- proper packaging for the Comfy Registry

### Option B — Manual minimal folder

For quick experiments you can drop a single `.py` file directly into `ComfyUI/custom_nodes/`. ComfyUI discovers and loads any `.py` file (or package) in that folder at startup.

```text
ComfyUI/custom_nodes/
└── my_node.py            # or a folder my_pack/ with __init__.py
```

This guide focuses on the **package** style (Option A) because it is what 0.19.3's tooling expects, but everything applies to a single file too.

---

## 4. Project Layout

A properly structured pack looks like this:

```text
ComfyUI/custom_nodes/MyLLMPack/
├── __init__.py                 # exports NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, WEB_DIRECTORY
├── nodes.py                    # the node classes
├── requirements.txt            # pip dependencies (optional)
├── pyproject.toml              # metadata + packaging (for the Registry)
└── js/
    └── myExtension.js          # frontend extension (optional)
```

`__init__.py` is the entry point ComfyUI imports. A minimal one:

```python
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
```

---

## 5. The Node Class Contract

Every custom node is a **Python class**. The required and commonly used members:

| Member | Type | Purpose |
| --- | --- | --- |
| `INPUT_TYPES` | `@classmethod` → `dict` | Defines required/optional/hidden inputs. |
| `RETURN_TYPES` | `tuple[str, ...]` | Output data types. |
| `RETURN_NAMES` | `tuple[str, ...]` (optional) | Labels for outputs; defaults to lowercase `RETURN_TYPES`. |
| `FUNCTION` | `str` | Name of the method called when the node executes. |
| `CATEGORY` | `str` | Menu path in "Add Node" (submenus via `/`, e.g. `"LLM/prompt"`). |
| `OUTPUT_NODE` | `bool` (optional) | Marks the node as an output node (always executed). |
| `IS_CHANGED` | `@classmethod` (optional) | Controls re-execution caching. |
| `VALIDATE_INPUTS` | `@classmethod` (optional) | Validates inputs before execution. |
| `SEARCH_ALIASES` | `list[str]` (optional) | Alternative search terms surfaced in `/object_info`. |
| `INPUT_IS_LIST` / `OUTPUT_IS_LIST` | `bool` (optional) | Enable per-item list processing. |

### The `FUNCTION` method

- Called with **named arguments** matching your input names.
- `required` (and `hidden`) inputs are always passed.
- `optional` inputs are passed **only when connected** — provide defaults in the method signature (or use `**kwargs`).
- Must return a **tuple** matching `RETURN_TYPES` (trailing comma required for a single value: `return (x,)`).

> ⚠️ `RETURN_TYPES = ("IMAGE",)` and `return (image_out,)` — the trailing comma matters; without it, Python treats it as a scalar, not a tuple.

---

## 6. INPUT_TYPES in Detail

`INPUT_TYPES` returns a dict that **must** contain `"required"` and **may** also contain `"optional"` and `"hidden"`.

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "text": ("STRING", {"multiline": True, "default": "hello"}),
            "mode": (["uppercase", "lowercase"],),
            "count": ("INT", {"default": 1, "min": 1, "max": 100}),
        },
        "optional": {
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        },
        "hidden": {
            "unique_id": "UNIQUE_ID",
        },
    }
```

- **required:** Always shown as a widget; must be filled/connected.
- **optional:** May be left unconnected. When unconnected, the default value from the widget options is used, or you supply a default in the method signature.

Each input value is a **tuple**: `(type, options_dict)`. For a combo/dropdown, the tuple is `([list_of_options], options_dict)`.

---

## 7. Data Types & Widget Options

### Built-in data types

| Type | Meaning |
| --- | --- |
| `IMAGE` | A batch of images as a tensor `[B, H, W, C]` (C=3, RGB, float 0–1). A single image is a batch of size 1. |
| `MASK` | A single-channel mask tensor `[B, H, W]`. |
| `LATENT` | A dict with key `"samples"` (tensor `[B, C, H, W]`). |
| `MODEL` | A loaded diffusion model. |
| `CLIP` | A loaded CLIP/text encoder. |
| `VAE` | A loaded VAE. |
| `CONDITIONING` | Encoded prompt conditioning. |
| `INT` | Integer widget. |
| `FLOAT` | Float widget. |
| `STRING` | Text widget. |
| `COMBO` | A dropdown — specify as a list of strings. |
| `BOOLEAN` | A true/false toggle. |
| `"*"` | Wildcard — accepts any input type (requires `VALIDATE_INPUTS` with `input_types`). |

> Note: `IMAGE` is singular even though it represents a batch. Comfy treats a single image as a batch of size 1.

### Common widget options

```python
"INT": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1})
"FLOAT": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "round": 0.001})
"STRING": ("STRING", {"multiline": True, "default": "text"})
"COMBO": (["a", "b", "c"], {"default": "a"})
```

- `min` / `max` / `step`: slider bounds (front-end only; back-end does not enforce unless you validate).
- `default`: initial widget value.
- `multiline`: render the STRING widget like the CLIP text encoder.
- `round`: precision to round to (defaults to `step`; set `False` to disable).
- `forceInput`: `True` forces the input to be a connection rather than a widget (needed for custom datatypes).
- `lazy`: `True` defers evaluation until `check_lazy_status` requires it (new API / advanced).

---

## 8. Hidden Inputs

Hidden inputs let a node request server-side metadata. Declared under `"hidden"` with a `dict[str, str]`:

```python
"hidden": {
    "unique_id": "UNIQUE_ID",
    "prompt": "PROMPT",
    "extra_pnginfo": "EXTRA_PNGINFO",
    "dynprompt": "DYNPROMPT",
}
```

| Hidden input | What it gives you |
| --- | --- |
| `UNIQUE_ID` | The node's unique integer id (matches the client-side node `id`). Used for client–server messages. |
| `PROMPT` | The full prompt object the client sent to the server. |
| `EXTRA_PNGINFO` | A dict copied into the metadata of saved `.png` files (won't be saved if Comfy started with `disable_metadata`). |
| `DYNPROMPT` | A `comfy_execution.graph.DynamicPrompt` that may mutate during execution (advanced: loops/expansion). |

---

## 9. Custom, Wildcard & Dynamic Inputs

### Custom datatype

Pick any uppercase string, e.g. `CHEESE`. Use it in `INPUT_TYPES`/`RETURN_TYPES`; the client only allows matching connections. Because the client doesn't know the type, force it to be an input:

```python
@classmethod
def INPUT_TYPES(cls):
    return {"required": {"my_cheese": ("CHEESE", {"forceInput": True})}}
```

`CHEESE` can be any Python object you pass between your own nodes.

### Wildcard input

```python
@classmethod
def INPUT_TYPES(cls):
    return {"required": {"anything": ("*", {})}}

@classmethod
def VALIDATE_INPUTS(cls, input_types):
    return True
```

The `*` accepts any source; the `input_types` parameter skips backend type validation. Your node must make sense of whatever is passed.

### Dynamically created inputs

For inputs created on the client at runtime, accept arbitrary names via an `optional` dict that always reports "contains":

```python
class ContainsAnyDict(dict):
    def __contains__(self, key):
        return True

@classmethod
def INPUT_TYPES(cls):
    return {"required": {}, "optional": ContainsAnyDict()}

def main_method(self, **kwargs):
    # dynamically created inputs arrive in kwargs
    ...
```

---

## 10. Execution Model & Caching

ComfyUI caches node outputs and only re-executes nodes whose inputs may have changed. Two optional members tune this:

### `OUTPUT_NODE = True`

Marks the node as an output (e.g., preview/save nodes). Output nodes always execute.

### `IS_CHANGED`

> ⚠️ Despite the name, `IS_CHANGED` must **not** return a `bool`. It returns *any* Python object compared (`!=`) against the previous run's value.

```python
@classmethod
def IS_CHANGED(cls, image):
    image_path = folder_paths.get_annotated_filepath(image)
    m = hashlib.sha256()
    with open(image_path, 'rb') as f:
        m.update(f.read())
    return m.digest().hex()
```

To force a node to **always** re-execute, return `float("NaN")` (NaN is not equal to anything, even itself). Use sparingly — it disables caching for that node.

Common use: nodes that use randomness without a seed, or load external files that may change.

### `VALIDATE_INPUTS`

Called before execution starts. Return `True` if valid, or an error `str` to block execution.

```python
@classmethod
def VALIDATE_INPUTS(cls, input_types):
    if input_types["input1"] not in ("INT", "FLOAT"):
        return "input1 must be an INT or FLOAT type"
    return True
```

Notes:
- Only inputs defined as **constants** in the workflow are passed; connected inputs are not.
- If the method accepts `input_types`, default backend type validation is skipped for those inputs.
- `**kwargs` receives all inputs and skips validation for all of them.

### `INPUT_IS_LIST` / `OUTPUT_IS_LIST`

Enable sequential per-item processing (iterate over a list of inputs). See the official "Lists" docs for details.

---

## 11. Registering Your Nodes

ComfyUI discovers nodes via module-level mappings. In `src/nodes.py` (or your `__init__.py`):

```python
NODE_CLASS_MAPPINGS = {
    "LLMPromptNode": LLMPromptNode,
    "ImageSelector": ImageSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMPromptNode": "LLM Prompt",
    "ImageSelector": "Image Selector",
}
```

- The **key** is the internal node id (used in workflows/API).
- The **value** in `NODE_DISPLAY_NAME_MAPPINGS` is the friendly name shown in the UI.
- You **must restart ComfyUI** to pick up changes to node registration or Python code.

---

## 12. Client-Side (JavaScript) Extensions

To add UI behavior:

1. Export `WEB_DIRECTORY` from your Python module (`WEB_DIRECTORY = "./js"`).
2. Put `.js` files in that directory. **All** `.js` files are auto-loaded by the browser.
3. Register an extension via `app.registerExtension`.

Example `js/myExtension.js`:

```javascript
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "example.imageselector",
    async setup() {
        function messageHandler(event) {
            alert(event.detail.message);
        }
        app.api.addEventListener("example.imageselector.textmessage", messageHandler);
    },
});
```

### Sending a message from the server to the client

In Python:

```python
from server import PromptServer

PromptServer.instance.send_sync("example.imageselector.textmessage", {"message": f"Picked image {best + 1}"})
```

The JS listener above receives it in `event.detail`.

> Only `.js` files are auto-injected. For CSS or other assets, reference them at `extensions/<your_custom_nodes_subfolder>/file.css` and add them programmatically.

---

## 13. Complete Worked Example (LLM Prompt Node)

A small but realistic node for an "LLM Prompt Studio" — it takes a base prompt and a style, and returns a composed prompt string. Place this in `nodes.py`.

```python
class LLMPromptComposer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_prompt": ("STRING", {"multiline": True, "default": "a photo of a cat"}),
                "style": (["photorealistic", "anime", "oil painting", "cyberpunk"], {"default": "photorealistic"}),
                "quality_boost": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "compose"
    CATEGORY = "LLM/prompt"
    OUTPUT_NODE = False

    def compose(self, base_prompt, style, quality_boost, seed=None, unique_id=None):
        style_suffix = {
            "photorealistic": "highly detailed, 8k, real photo",
            "anime": "anime style, cel shading, vibrant",
            "oil painting": "oil painting, brush strokes, textured canvas",
            "cyberpunk": "cyberpunk, neon, futuristic city",
        }.get(style, "")

        prompt = f"{base_prompt}, {style_suffix}"
        if quality_boost:
            prompt += ", masterpiece, best quality"

        return (prompt,)


NODE_CLASS_MAPPINGS = {
    "LLMPromptComposer": LLMPromptComposer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMPromptComposer": "LLM Prompt Composer",
}
```

To connect this to an LLM (e.g., for captioning or generation), follow the same structure but call an API inside `compose`, converting tensors with `numpy`/`PIL` as needed. Remember that `IMAGE` is a `[B, H, W, C]` float tensor in 0–1 range.

---

## 14. Packaging, Dependencies & Publishing

### `requirements.txt`

If your node needs extra pip packages:

```text
requests>=2.31.0
numpy>=1.24.0
```

Install with `pip install -r requirements.txt` in the ComfyUI Python environment.

### `pyproject.toml`

The scaffold creates this for you. It carries metadata (name, version, author, license) and is what the Comfy Registry uses to publish your pack. Keep `version` in sync and follow semantic versioning.

### Publishing to the Comfy Registry

1. Build your pack following the scaffold structure.
2. Push to GitHub.
3. Use the Registry publishing flow (see docs.comfy.org → Registry). ComfyUI-Manager (now part of Comfy Org core) lets users install your pack with one click.

### ComfyUI-Manager

ComfyUI-Manager is the official node manager (part of Comfy Org). It handles installing, updating, disabling, and dependency resolution for custom nodes.

---

## 15. Debugging & Common Pitfalls

- **Node doesn't appear:** Restart ComfyUI. Confirm `NODE_CLASS_MAPPINGS` is exported from the package entry point, and the file lives under `custom_nodes/`.
- **Single output ignored:** Use the trailing comma: `RETURN_TYPES = ("STRING",)` and `return (value,)`.
- **`IS_CHANGED` returning `True`:** This is a bug — `True == True`, so the node is considered *unchanged*. Return `float("NaN")` to force re-execution, or a hash/value that actually differs.
- **Optional input always `None`:** Optional inputs are only passed when connected. Provide defaults in the method signature.
- **Wrong tensor shape:** `IMAGE` is `[B,H,W,C]` floats 0–1. Multiply by 255 and cast to `uint8` for PIL/`numpy`.
- **JS not loading:** Confirm `WEB_DIRECTORY` is exported and the file is `.js`. Check the browser console.
- **Dependency conflicts:** Use a clean virtual environment; prefer `requirements.txt` over global installs.

To inspect loaded nodes and their schemas programmatically, query the ComfyUI API endpoint `/object_info` (it includes `INPUT`, `OUTPUT`, `OUTPUT_IS_LIST`, `search_aliases`, etc.).

---

## 16. Version Notes for 0.19.3

- **0.19.3 uses the classic custom-node API**: define a Python class, expose `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`, and (optionally) `WEB_DIRECTORY`. This guide's examples target exactly that API.
- The newer **"V3" / `comfy_api` API** (subclassing `io.ComfyNode`, defining `define_schema`, `check_lazy_status`, `execute`, and returning an `io.NodeOutput`, loaded via a `ComfyExtension` + `comfy_entrypoint`) is the modern alternative shown in `ComfyUI/custom_nodes/example_node.py.example`. It is available in recent builds and is the recommended path for new, complex packs, but the classic API remains fully supported and is what most existing tutorials and community nodes use.
- `BOOLEAN` inputs, `forceInput`, wildcard `*`, and hidden inputs (`UNIQUE_ID`, `PROMPT`, `EXTRA_PNGINFO`, `DYNPROMPT`) are all available in 0.19.3.
- ComfyUI-Manager is bundled as a core dependency; prefer it for installing third-party packs.
- Always restart the server after editing node Python code; frontend `.js` changes require a browser reload (and sometimes a server restart).

---

## 17. Reference Links

- Official docs — Custom Nodes overview: https://docs.comfy.org/custom-nodes/overview
- Official docs — Getting Started walkthrough: https://docs.comfy.org/custom-nodes/walkthrough
- Official docs — Backend (node contract, inputs): https://docs.comfy.org/custom-nodes/backend
- Official docs — More on inputs (hidden/flexible): https://docs.comfy.org/custom-nodes/backend/more_on_inputs
- Official docs — JavaScript extensions: https://docs.comfy.org/custom-nodes/js/javascript_overview
- Official example node (V3 API): https://github.com/Comfy-Org/ComfyUI/blob/master/custom_nodes/example_node.py.example
- Scaffold template: https://github.com/Comfy-Org/cookiecutter-comfy-extension
- Community guide (Suzie1): https://github.com/Suzie1/ComfyUI_Guide_To_Making_Custom_Nodes
- Tutorial repo (Breeze-le): https://github.com/Breeze-le/comfyui-custom_nodes-tutorial
- Basic calculator tutorial (sbcode): https://sbcode.net/genai/creating-custom-nodes/
- Comfy Registry / publishing: https://docs.comfy.org/registry/publishing
