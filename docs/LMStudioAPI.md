# LM Studio API — Interaction Guide

> **Target version:** LM Studio **0.4.21 (Build 2)** (released 2026‑08‑12)
> **Scope:** Interacting with the LM Studio local API server — REST (native v1 + legacy v0), OpenAI‑compatible, Anthropic‑compatible endpoints, and the official SDKs.
> **Default base URL:** `http://localhost:1234`
> **Native REST base:** `http://localhost:1234/api/v1`
> **OpenAI‑compatible base:** `http://localhost:1234/v1`

---

## Table of Contents

1. [Overview](#1-overview)
2. [Starting the Server](#2-starting-the-server)
3. [Server Settings & Authentication](#3-server-settings--authentication)
4. [Native REST API v1 (`/api/v1`)](#4-native-rest-api-v1-apiv1)
5. [Legacy REST API v0 (`/api/v0`)](#5-legacy-rest-api-v0-apiv0)
6. [OpenAI‑Compatible Endpoints (`/v1`)](#6-openai-compatible-endpoints-v1)
7. [Anthropic‑Compatible Endpoints (`/v1/messages`)](#7-anthropic-compatible-endpoints-v1messages)
8. [Inference Parameters](#8-inference-parameters)
9. [Vision / Image Inputs](#9-vision--image-inputs)
10. [Tool Use & MCP Integration](#10-tool-use--mcp-integration)
11. [Streaming & SSE Events](#11-streaming--sse-events)
12. [Structured Output](#12-structured-output)
13. [Idle TTL & Auto‑Evict](#13-idle-ttl--auto-evict)
14. [Official SDKs (Python / TypeScript / CLI)](#14-official-sdks-python--typescript--cli)
15. [Quick Reference & Project Notes](#15-quick-reference--project-notes)

---

## 1. Overview

LM Studio 0.4.21 exposes a local API server that lets applications drive local LLMs
(either on `localhost` or on the local network). There are **three API surfaces**:

| Surface | Base path | Best for |
| --- | --- | --- |
| **Native REST API v1** | `/api/v1/*` | Stateful chats, model load/unload/download, MCP, rich stats. **Recommended for new projects** (available since 0.4.0). |
| **Legacy REST API v0** | `/api/v0/*` | OpenAI‑shaped `/chat/completions`, `/embeddings`, `/completions`, plus loaded/unloaded model listing with `stats`. |
| **OpenAI‑compatible** | `/v1/*` | Drop‑in reuse of OpenAI clients (`/v1/chat/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/models`, `/v1/completions`). |
| **Anthropic‑compatible** | `/v1/messages` | Claude‑style Messages API (added in 0.4.1). |

The Python SDK (`lmstudio` / `lmstudio-python`) and the TypeScript SDK (`lmstudio-js`)
wrap these APIs and also communicate over an internal websocket/HTTP channel.

> **Version note for 0.4.21 (Build 2):**
> - Improved `llama.cpp` model‑load error messages.
> - Updated advanced load settings for `mmap`, `mlock`, and `direct io` for `llama.cpp` engines **2.28.1 and newer**.
> - Build 1 added support for a **local server API key in enterprise internal network endpoint mode**.
>
> All endpoints documented below are stable across 0.4.x. Authentication/API tokens, the v1 REST API, MCP, and the Anthropic endpoint were all introduced at or before 0.4.1 and remain current.

---

## 2. Starting the Server

### GUI
Open LM Studio → **Developer** tab → toggle **"Start Server"**.

### CLI (`lms`)
The `lms` CLI is installed separately. Install it once with:

```bash
npx lmstudio install-cli
```

Then:

```bash
# Start on the default port (last used port, usually 1234)
lms server start

# Custom port
lms server start --port 3000

# Expose on the local network (binds 0.0.0.0)
lms server start --bind 0.0.0.0

# Enable CORS (needed for some browser / VS Code extension clients)
lms server start --cors

# Check status
lms server status
```

> Any bind other than `127.0.0.1` exposes the server beyond `localhost`. LM Studio
> recommends enabling authentication before doing this.

---

## 3. Server Settings & Authentication

### Server Settings (Developer → Server Settings)

| Setting | Type | Description |
| --- | --- | --- |
| `Server Port` | Integer | Port on which the API server listens. |
| `Require Authentication` | Switch | Require clients to send a valid API token in the `Authorization` header. |
| `Serve on Local Network` | Switch | Allow other devices on the same network to reach the API. |
| `Allow per-request MCPs` | Switch | Allow API clients to use **ephemeral** MCP servers (not in `mcp.json`). Only remote MCPs are supported. |
| `Allow calling servers from mcp.json` | Switch | Allow API clients to use your configured `mcp.json` servers. **Requires `Require Authentication` enabled.** |
| `Enable CORS` | Switch | Allow cross‑origin browser requests. |
| `Just in Time Model Loading` | Switch | Load models dynamically at request time to save memory. |
| `Auto Unload Unused JIT Models` | Switch | Unload JIT‑loaded models when no longer in use. |
| `Only Keep Last JIT Loaded Model` | Switch | Keep only the most recently used JIT‑loaded model in memory. |

### API Tokens (requires LM Studio 0.4.0+)
By default, the API server requires **no** authentication. To enable it, turn on
**Require Authentication** and create tokens in **Manage Tokens** (Server Settings).

When authentication is on, include the token in the `Authorization` header:

```http
Authorization: Bearer $LM_API_TOKEN
```

For the **Anthropic‑compatible** endpoint, LM Studio also accepts the `x-api-key` header.

```bash
curl -X POST http://localhost:1234/api/v1/chat \
  -H "Authorization: Bearer $LM_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{ "model": "ibm/granite-4-micro", "input": "Hello" }'
```

---

## 4. Native REST API v1 (`/api/v1`)

Available since 0.4.0. **Recommended for new integrations.**

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/v1/chat` | POST | Stateful chat (auto‑managed conversation history, MCP, streaming). |
| `/api/v1/models` | GET | List all local models (LLM + embedding) with rich metadata. |
| `/api/v1/models/load` | POST | Load an LLM/embedding model into memory with config. |
| `/api/v1/models/unload` | POST | Unload a loaded model instance. |
| `/api/v1/models/download` | POST | Download a model by catalog id or Hugging Face URL. |
| `/api/v1/models/download/status/{job_id}` | GET | Poll download progress. |

### 4.1 `POST /api/v1/chat`

Send a message (or message array) and receive a response. **Stateful by default.**

**Request body**

| Field | Type | Notes |
| --- | --- | --- |
| `model` | string | **Required.** Unique model identifier. |
| `input` | string \| array | Text string, or array of typed items (`{"type":"message","content":...}`, `{"type":"image","data_url":...}`). |
| `system_prompt` | string | Optional system message. |
| `integrations` | array | Optional list of MCP integrations (see §10). |
| `stream` | boolean | Stream partial output via SSE. Default `false`. |
| `temperature` | number | `[0,1]`. `0` deterministic. |
| `top_p` | number | Cumulative probability cutoff `[0,1]`. |
| `top_k` | integer | Limit next tokens to top‑k. |
| `min_p` | number | Minimum base probability `[0,1]`. |
| `repeat_penalty` | number | `1` = no penalty. |
| `max_output_tokens` | integer | Max tokens to generate. |
| `reasoning` | `"off"\|"low"\|"medium"\|"high"\|"on"` | Reasoning effort. Errors if unsupported by model. |
| `context_length` | integer | Token context window. Higher recommended for MCP. |
| `store` | boolean | Store the chat and return `response_id`. Default `true`. |
| `previous_response_id` | string | Continue a prior conversation (must start with `resp_`). |

**Example**

```bash
curl http://localhost:1234/api/v1/chat \
  -H "Authorization: Bearer $LM_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ibm/granite-4-micro",
    "input": "Write a short haiku about sunrise.",
    "temperature": 0.7
  }'
```

**Response fields**

- `model_instance_id` — loaded model instance that produced the response.
- `output` — array of items: `message`, `tool_call`, `reasoning`, or `invalid_tool_call`.
- `stats` — `input_tokens`, `total_output_tokens`, `reasoning_output_tokens`, `tokens_per_second`, `time_to_first_token_seconds`, optional `model_load_time_seconds`.
- `response_id` — present when `store` is `true`; use it as `previous_response_id` to continue.

```json
{
  "model_instance_id": "ibm/granite-4-micro",
  "output": [
    { "type": "message", "content": "Soft light on the hill..." }
  ],
  "stats": {
    "input_tokens": 12,
    "total_output_tokens": 18,
    "tokens_per_second": 41.2,
    "time_to_first_token_seconds": 0.81
  },
  "response_id": "resp_02b2017dbc06c12bfc353a2ed6c2b802f8cc682884bb5716"
}
```

### 4.2 Stateful conversations

```bash
# 1) First turn — server returns response_id
curl http://localhost:1234/api/v1/chat -H "Content-Type: application/json" \
  -d '{ "model": "ibm/granite-4-micro", "input": "My favorite color is blue." }'

# 2) Continue — pass previous_response_id; no need to resend history
curl http://localhost:1234/api/v1/chat -H "Content-Type: application/json" \
  -d '{ "model": "ibm/granite-4-micro",
        "input": "What color did I just mention?",
        "previous_response_id": "resp_abc123..." }'

# Stateless one-off
curl http://localhost:1234/api/v1/chat -H "Content-Type: application/json" \
  -d '{ "model": "ibm/granite-4-micro", "input": "Tell me a joke.", "store": false }'
```

### 4.3 `GET /api/v1/models`

No parameters. Returns all local models (LLM + embedding) with rich metadata:
`type`, `publisher`, `key`, `display_name`, `architecture`, `quantization`,
`size_bytes`, `params_string`, `loaded_instances[]`, `max_context_length`, `format`,
and `capabilities` (`vision`, `trained_for_tool_use`, `reasoning.allowed_options/default`).
Embedding models omit `capabilities`. Multi‑variant models include `variants`/`selected_variant`.

```bash
curl http://localhost:1234/api/v1/models -H "Authorization: Bearer $LM_API_TOKEN"
```

### 4.4 `POST /api/v1/models/load`

| Field | Type | Notes |
| --- | --- | --- |
| `model` | string | **Required.** Model identifier (LLM or embedding). |
| `context_length` | number | Max tokens the model will consider. |
| `eval_batch_size` | number | llama.cpp engine only. |
| `flash_attention` | boolean | llama.cpp engine only; can lower VRAM and speed up. |
| `num_experts` | number | MoE models only. |
| `offload_kv_cache_to_gpu` | boolean | llama.cpp engine only. |
| `echo_load_config` | boolean | Echo final config under `load_config`. Default `false`. |

```bash
curl http://localhost:1234/api/v1/models/load \
  -H "Authorization: Bearer $LM_API_TOKEN" -H "Content-Type: application/json" \
  -d '{ "model": "openai/gpt-oss-20b", "context_length": 16384,
        "flash_attention": true, "echo_load_config": true }'
```

Response: `type`, `instance_id`, `load_time_seconds`, `status: "loaded"`, and
optional `load_config`.

### 4.5 `POST /api/v1/models/unload`

```bash
curl http://localhost:1234/api/v1/models/unload \
  -H "Authorization: Bearer $LM_API_TOKEN" -H "Content-Type: application/json" \
  -d '{ "instance_id": "openai/gpt-oss-20b" }'
```

Response: `{ "instance_id": "openai/gpt-oss-20b" }`

### 4.6 `POST /api/v1/models/download`

| Field | Type | Notes |
| --- | --- | --- |
| `model` | string | **Required.** Catalog id (e.g. `ibm/granite-4-micro`) or exact HF URL (e.g. `https://huggingface.co/lmstudio-community/gpt-oss-20b-GGUF`). |
| `quantization` | string | Optional; only for HF links (e.g. `Q4_K_M`). |

```bash
curl http://localhost:1234/api/v1/models/download \
  -H "Authorization: Bearer $LM_API_TOKEN" -H "Content-Type: application/json" \
  -d '{ "model": "ibm/granite-4-micro" }'
```

Response returns a job status: `job_id`, `status` (`downloading`/`completed`/`already_downloaded`/...),
`total_size_bytes`, `started_at`, `completed_at`.

### 4.7 `GET /api/v1/models/download/status/{job_id}`

```bash
curl -H "Authorization: Bearer $LM_API_TOKEN" \
  http://localhost:1234/api/v1/models/download/status/job_493c7c9ded
```

Status object includes `bytes_per_second`, `estimated_completion`, `downloaded_bytes`, etc.

---

## 5. Legacy REST API v0 (`/api/v0`)

Still supported; uses OpenAI‑style request/response shapes and adds rich `stats`.

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/v0/models` | GET | List all loaded/downloaded models. |
| `/api/v0/models/{model}` | GET | Info about one model. |
| `/api/v0/chat/completions` | POST | Chat completions (messages → assistant response). |
| `/api/v0/completions` | POST | Text completions (prompt → completion). |
| `/api/v0/embeddings` | POST | Text embeddings. |

**Example — Chat completions:**

```bash
curl http://localhost:1234/api/v0/chat/completions \
  -H "Authorization: Bearer $LM_API_TOKEN" -H "Content-Type: application/json" \
  -d '{
    "model": "granite-3.0-2b-instruct",
    "messages": [
      { "role": "system", "content": "Always answer in rhymes." },
      { "role": "user", "content": "Introduce yourself." }
    ],
    "temperature": 0.7,
    "max_tokens": -1,
    "stream": false
  }'
```

The v0 response includes an extra `stats` block:

```json
"stats": {
  "tokens_per_second": 51.43,
  "time_to_first_token": 0.111,
  "generation_time": 0.954,
  "stop_reason": "eosFound"
}
```

plus `model_info` (arch, quant, format, context_length) and `runtime` (engine name/version).

---

## 6. OpenAI‑Compatible Endpoints (`/v1`)

To reuse existing OpenAI clients, point the base URL at LM Studio:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
```

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/v1/models` | GET | List models visible to the server. |
| `/v1/chat/completions` | POST | Chat with history (text + images). |
| `/v1/responses` | POST | Stateful Responses API (reasoning, prior state, remote MCP). |
| `/v1/embeddings` | POST | Embedding vectors. |
| `/v1/completions` | POST | Legacy text completion (base models). |

### 6.1 `/v1/chat/completions`

**Supported payload parameters:** `model`, `messages`, `temperature`, `top_p`, `top_k`,
`max_tokens`, `stream`, `stop`, `presence_penalty`, `frequency_penalty`, `logit_bias`,
`repeat_penalty`, `seed`, plus `tools`, `tool_choice`, `response_format` (see §10, §12).

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

completion = client.chat.completions.create(
    model="model-identifier",
    messages=[
        {"role": "system", "content": "Always answer in rhymes."},
        {"role": "user",   "content": "Introduce yourself."}
    ],
    temperature=0.7,
)
print(completion.choices[0].message)
```

### 6.2 `/v1/responses`

OpenAI‑compatible Responses endpoint (stateful, supports reasoning + remote MCP).

```bash
curl http://localhost:1234/v1/responses -H "Content-Type: application/json" \
  -d '{ "model": "openai/gpt-oss-20b",
        "input": "Provide a prime number less than 50",
        "reasoning": { "effort": "low" } }'
```

- Stateful follow‑up: set `previous_response_id` to a prior `id`.
- Streaming: `"stream": true` → SSE events `response.created`, `response.output_text.delta`, `response.completed`.
- Remote MCP (opt‑in, enable in Developer → Settings):
  ```json
  "tools": [ { "type": "mcp", "server_label": "huggingface",
               "server_url": "https://huggingface.co/mcp",
               "allowed_tools": ["model_search"] } ]
  ```

### 6.3 `/v1/embeddings`

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def get_embedding(text, model="model-identifier"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding
```

### 6.4 `/v1/completions` (Legacy)

Text completion for **base models**. LM Studio continues to support this even though
OpenAI deprecated it. The prompt template is **not** applied; using it with chat‑tuned
models may produce unexpected tokens — prefer a base model.

### 6.5 `/v1/models`

```bash
curl http://localhost:1234/v1/models
```

Returns models visible to the server (may include all downloaded models when JIT loading is on).

---

## 7. Anthropic‑Compatible Endpoints (`/v1/messages`)

Added in 0.4.1. Use Claude‑style clients or HTTP requests.

```bash
curl http://localhost:1234/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $LM_API_TOKEN" \
  -d '{ "model": "ibm/granite-4-micro",
        "max_tokens": 256,
        "messages": [ {"role": "user", "content": "Write a haiku about local LLMs."} ] }'
```

- With authentication off, the `x-api-key` header is optional.
- Streaming: `"stream": true` → SSE events `message_start`, `content_block_start`,
  `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`.
- Tools: pass `tools` (each with `name`, `description`, `input_schema`) and optional
  `tool_choice` (`{"type":"any"}`).

**Claude Code** can point at LM Studio:

```bash
export ANTHROPIC_BASE_URL=http://localhost:1234
export ANTHROPIC_AUTH_TOKEN=lmstudio
claude --model openai/gpt-oss-20b
```

**Python (Anthropic SDK):**

```python
from anthropic import Anthropic
client = Anthropic(base_url="http://localhost:1234", api_key="lmstudio")
message = client.messages.create(
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello from LM Studio"}],
    model="ibm/granite-4-micro",
)
print(message.content)
```

---

## 8. Inference Parameters

Common generation parameters (supported across `/api/v1/chat`, `/api/v0/*`, `/v1/chat/completions`):

| Parameter | Type | Meaning |
| --- | --- | --- |
| `temperature` | number `[0,1]` | Randomness. `0` = deterministic. |
| `top_p` | number `[0,1]` | Nucleus sampling cutoff. |
| `top_k` | integer | Restrict to top‑k probable tokens. |
| `min_p` | number `[0,1]` | Minimum base probability for a token. |
| `repeat_penalty` | number | `1` = no penalty; higher discourages repetition. |
| `max_tokens` / `max_output_tokens` | integer | Max tokens to generate (`-1` may mean unlimited on v0). |
| `stop` | string/array | Stop sequences. |
| `presence_penalty` / `frequency_penalty` | number | OpenAI‑style penalties. |
| `logit_bias` | object | Per‑token bias. |
| `seed` | integer | Reproducibility seed. |
| `reasoning` | `"off"\|"low"\|"medium"\|"high"\|"on"` | Reasoning effort (v1 chat; model must support it). |
| `context_length` | integer | Context window for the request (v1 chat). |
| `ttl` | integer (seconds) | Idle TTL for JIT‑loaded models (see §13). |

> **Reasoning content** (e.g. DeepSeek‑R1, gpt‑oss): for `/v1/chat/completions` it is
> returned in `choices[0].message.reasoning` (non‑streaming) / `choices[0].delta.reasoning`
> (streaming), separate from `content`.

---

## 9. Vision / Image Inputs

Vision‑capable models (`capabilities.vision: true` from `/api/v1/models`) accept images.

- **OpenAI‑compatible** (`/v1/chat/completions`): use the standard
  `{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}` content part.
- **Native v1** (`/api/v1/chat`): use typed input items
  `{"type":"image","data_url":"data:image/jpeg;base64,..."}`.

```python
import base64
b64 = base64.b64encode(image_bytes).decode()
messages = [{
    "role": "user",
    "content": [
        {"type": "text",      "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
    ]
}]
```

> In `comfyui_llm_prompt_studio` the request layer converts `image_url` parts into the
> native `{"type":"image","data_url":...}` form before calling `/api/v1/chat` (see
> `lm_http.py:chat_completion` / `lm_http.py:ensure_model_loaded`).

---

## 10. Tool Use & MCP Integration

### 10.1 Tool Use (Function Calling)
Via `/v1/chat/completions` and `/v1/responses` using OpenAI‑compatible `tools`:

```bash
curl http://localhost:1234/v1/chat/completions -H "Content-Type: application/json" \
  -d '{
    "model": "lmstudio-community/qwen2.5-7b-instruct",
    "messages": [{"role":"user","content":"What dell products under $50 in electronics?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "search_products",
        "description": "Search the product catalog.",
        "parameters": {
          "type": "object",
          "properties": {
            "query":    {"type":"string"},
            "category": {"type":"string","enum":["electronics","clothing","home","outdoor"]},
            "max_price":{"type":"number"}
          },
          "required": ["query"]
        }
      }
    }]
  }'
```

The model returns `choices[0].message.tool_calls` with `finish_reason: "tool_calls"`.
When **streaming**, tool names/arguments arrive in chunks via
`chunk.choices[0].delta.tool_calls[].function.name` / `.arguments` and must be accumulated.

Models support **Native** (hammer‑badge; trained for tool use — e.g. Qwen2.5‑Instruct,
Llama‑3.1/3.2, Mistral) or **Default** tool use (system‑prompt‑based fallback for
all other models). `tool_choice` supports `"auto"`, `"none"`, `"required"` (llama.cpp).

### 10.2 MCP via the native `/api/v1/chat` endpoint
Two ways to attach MCP servers (requires 0.4.0+):

| Kind | `integrations` entry | Requires |
| --- | --- | --- |
| Ephemeral (per‑request) | `{ "type":"ephemeral_mcp", "server_label":..., "server_url":..., "allowed_tools":[...], "headers":{...} }` | `Allow per-request MCPs` |
| From `mcp.json` | `{ "type":"plugin", "id":"mcp/playwright", "allowed_tools":[...] }` or shorthand `"mcp/playwright"` | `Allow calling servers from mcp.json` (+ auth) |

```bash
curl http://localhost:1234/api/v1/chat \
  -H "Authorization: Bearer $LM_API_TOKEN" -H "Content-Type: application/json" \
  -d '{
    "model": "ibm/granite-4-micro",
    "input": "What is the top trending model on hugging face?",
    "integrations": [
      { "type": "ephemeral_mcp", "server_label": "huggingface",
        "server_url": "https://huggingface.co/mcp", "allowed_tools": ["model_search"] }
    ],
    "context_length": 8000
  }'
```

Output items include `type:"tool_call"` with `tool`, `arguments`, `output`, and `provider_info`
(`type:"ephemeral_mcp"` or `"plugin"`).

---

## 11. Streaming & SSE Events

### Native v1 chat (`POST /api/v1/chat`, `stream:true`)
Server‑Sent Events, always beginning with `chat.start` and ending with `chat.end`
(which carries the full aggregated response). Event sequence:

`chat.start` → (`model_load.start` → `model_load.progress` → `model_load.end`) →
`prompt_processing.start` → `prompt_processing.progress` → `prompt_processing.end` →
(`reasoning.start` → `reasoning.delta`* → `reasoning.end`) →
(`tool_call.start` → `tool_call.arguments` → `tool_call.success`/`tool_call.failure`) →
`message.start` → `message.delta`* → `message.end` → (`error`) → `chat.end`

Raw wire format:

```
event: <event type>
data: <JSON event data>
```

Key event data shapes:
- `chat.start` / `model_load.*` / `prompt_processing.*` carry identifiers and `progress` floats `[0,1]`.
- `reasoning.delta` → `{ "type":"reasoning.delta", "content":"..." }`
- `message.delta` → `{ "type":"message.delta", "content":"..." }`
- `chat.end` → `{ "type":"chat.end", "result": { model_instance_id, output, stats, response_id } }`

### OpenAI‑compatible streaming
Use `stream:true` on `/v1/chat/completions` (OpenAI SSE `data:` chunks) or `/v1/responses`
(`response.created`, `response.output_text.delta`, `response.completed`). Set
`stream_options.include_usage:true` to get token usage in the final chunk.

---

## 12. Structured Output

Enforce a JSON schema on `/v1/chat/completions` (works via any OpenAI client):

```bash
curl http://localhost:1234/v1/chat/completions -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [{"role":"user","content":"Tell me a joke."}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "joke_response",
        "strict": "true",
        "schema": { "type":"object",
                    "properties": { "joke": {"type":"string"} },
                    "required": ["joke"] }
      }
    },
    "temperature": 0.7,
    "max_tokens": 50,
    "stream": false
  }'
```

- The JSON object is returned as a **string** in `choices[0].message.content`; parse it client‑side.
- Backed by `llama.cpp` grammar sampling (GGUF) or Outlines (MLX).
- **Not all models** can do structured output — generally avoid for models < 7B.

---

## 13. Idle TTL & Auto‑Evict

Relevant when **JIT loading** is enabled.

- **JIT loading** — models load automatically on first request. Default **on**.
- **Idle TTL** — how long a model may stay loaded without requests before auto‑unload.
  Default **60 minutes (3600s)**. Set per request with the `ttl` field (seconds):
  ```bash
  curl http://localhost:1234/api/v0/chat/completions -H "Content-Type: application/json" \
    -d '{ "model":"deepseek-r1-distill-qwen-7b", "ttl": 300,
          "messages":[{"role":"user","content":"Hi"}] }'
  ```
  Or via CLI: `lms load <model> --ttl 3600`. (`lms load` has **no TTL** by default.)
- **Auto‑Evict** — unload previously JIT‑loaded models before loading new ones.
  Default **on**; keeps at most 1 JIT model in memory. When off, models persist until
  TTL expiry or manual unload.

---

## 14. Official SDKs (Python / TypeScript / CLI)

### Python — `lmstudio` (lmstudio-python)
```bash
pip install lmstudio
```
```python
import lmstudio as lms

# Convenience API
model = lms.llm("qwen/qwen3-4b-2507")
print(model.respond("What is the meaning of life?"))

# Streaming
for fragment in model.respond_stream("Tell me a story"):
    print(fragment.content, end="", flush=True)

# Multi-turn chat
chat = lms.Chat("You are a helpful assistant.")
chat.add_user_message("Hello")
print(model.respond(chat))

# Scoped / async client
with lms.Client() as client:
    m = client.llm.model("openai/gpt-oss-20b")
    print(m.respond("Who are you?"))
```
Configure a custom host: `lms.configure_default_client("localhost:1234")`.
SDK v1.5.0+ auto‑discovers the local API port. Use `lms.Client.is_valid_api_host(...)`
or `lms.Client.find_default_local_api_host()` to check connectivity.

### TypeScript — `lmstudio-js`
```bash
npm install @lmstudio/sdk
```
```typescript
import { LMStudioClient } from "@lmstudio/sdk";
const client = new LMStudioClient();
const model = client.llm.model("model-identifier");
const prediction = await model.respond("Hello!");
console.log(prediction.content);
```

### CLI — `lms`
`lms server start`, `lms load`, `lms unload`, `lms ps`, `lms get <model>`,
`lms ls --variants`, `lms log stream`, `lms server status`.

---

## 15. Quick Reference & Project Notes

### Minimal curl — chat (stateful v1)
```bash
curl http://localhost:1234/api/v1/chat -H "Content-Type: application/json" \
  -d '{ "model":"<model-id>", "input":"Hello" }'
```

### Minimal Python — OpenAI client
```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
r = c.chat.completions.create(
    model="<model-id>",
    messages=[{"role":"user","content":"Hello"}],
)
print(r.choices[0].message.content)
```

### Endpoint cheat‑sheet
| Need | Use |
| --- | --- |
| Stateful chat + MCP + stats | `POST /api/v1/chat` |
| List models + capabilities | `GET /api/v1/models` |
| Load / unload model | `POST /api/v1/models/load` · `POST /api/v1/models/unload` |
| Download model | `POST /api/v1/models/download` |
| Drop‑in OpenAI chat | `POST /v1/chat/completions` |
| Stateful Responses API | `POST /v1/responses` |
| Embeddings | `POST /v1/embeddings` |
| Claude‑style | `POST /v1/messages` |
| Legacy OpenAI‑shaped + stats | `POST /api/v0/chat/completions` |

### Notes for `comfyui_llm_prompt_studio`
The project (`lm_http.py`) already integrates LM Studio and:
- Defaults to `http://localhost:1234/v1` (`DEFAULT_SERVER`).
- Lists models via **native** `GET /api/v1/models` and derives vision capability from `capabilities.vision`.
- Loads models via `POST /api/v1/models/load` first, with fallbacks to legacy
  `POST /v1/models/{id}/load` and `POST /api/v0/models/{id}/load` (handling casing
  differences like `gpu_offload` vs `gpuOffload` across LM Studio versions).
- Converts OpenAI `image_url` content parts into native `{"type":"image","data_url":...}`
  items when calling `/api/v1/chat`.
- Supports streaming with a watchdog (`LLM_PROMPT_STUDIO_STREAM_WATCHDOG_SEC`) and
  JSON‑Schema `response_format` for structured outputs.

When extending the project, prefer the **native v1** endpoints for richer metadata
(MCP, reasoning, stats, stateful `response_id`) and the **OpenAI‑compatible** layer
for maximum client‑library reuse. Enable **Require Authentication** and create an
API token before exposing the server beyond `localhost`.

---

### References
- Native REST API: https://lmstudio.ai/docs/developer/rest
- OpenAI compatibility: https://lmstudio.ai/docs/developer/openai-compat
- Anthropic compatibility: https://lmstudio.ai/docs/developer/anthropic-compat
- Python SDK: https://lmstudio.ai/docs/python · TypeScript SDK: https://lmstudio.ai/docs/typescript
- CLI: https://lmstudio.ai/docs/cli · Changelog: https://lmstudio.ai/changelog/lmstudio
- LM Studio 0.4.21 release notes: https://lmstudio.ai/changelog/lmstudio-v0.4.21
