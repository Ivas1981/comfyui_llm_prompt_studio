import { api } from "/scripts/api.js";

// Placeholders - MUST match the values in constants.py
export const PH_DOWN   = "— server unavailable —";
export const PH_EMPTY  = "— no models on server —";
export const LIB_EMPTY = "— library is empty —";

// ---------------------------------------------------------------------------
// Widget / node helpers
// ---------------------------------------------------------------------------
export function getW(node, name) {
    return node.widgets ? node.widgets.find(w => w.name === name) : null;
}

export function cls(node) {
    return node.comfyClass || node.type || "";
}

export function isWriter(node) {
    if (cls(node) === "LLMPromptStudioWriter") return true;
    return !!(getW(node, "idea") && getW(node, "system_prompt") &&
              getW(node, "server_url") && !getW(node, "critic_prompt") &&
              !getW(node, "stage"));
}

export function isCritic(node) {
    if (cls(node) === "LLMPromptStudioCritic") return true;
    return !!(getW(node, "critic_prompt") && getW(node, "threshold") &&
              getW(node, "server_url"));
}

export function isSmartSave(node) {
    if (cls(node) === "LLMPromptStudioSmartSave") return true;
    return !!(getW(node, "jpeg_quality") && getW(node, "library_path") &&
              getW(node, "filename_prefix"));
}

export function isLoader(node) {
    if (cls(node) === "LLMPromptStudioLibraryLoader") return true;
    return !!(getW(node, "scene_name") && getW(node, "library_path") &&
              !getW(node, "jpeg_quality"));
}

export function isScene(node) {
    if (cls(node) === "LLMPromptStudioSceneBuilder") return true;
    return !!(getW(node, "stage") && getW(node, "describe_prompt") &&
              getW(node, "composer_prompt"));
}

export function isSmartLoader(node) {
    if (cls(node) === "LLMPromptStudioSmartLoader") return true;
    return !!(getW(node, "family_override") && getW(node, "apply_lora") &&
              getW(node, "ckpt_name"));
}

export function isSmartParams(node) {
    if (cls(node) === "LLMPromptStudioSmartParameters") return true;
    return !!(getW(node, "preset") && getW(node, "family_override") &&
              getW(node, "sampler_name") && getW(node, "scheduler"));
}

export function isKSampler(node) {
    if (cls(node) === "LLMPromptStudioKSamplerHiresFix") return true;
    return !!(getW(node, "latent_image") && getW(node, "hires_enabled") &&
              getW(node, "hires_upscale_type"));
}

// ---------------------------------------------------------------------------
// API wrappers
// ---------------------------------------------------------------------------
export async function postJSON(route, body) {
    const res = await api.fetchApi(route, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {}),
    });
    if (!res.ok) {
        // Try to return JSON error when possible, otherwise text
        const text = await res.text();
        let parsed = null;
        try { parsed = JSON.parse(text); } catch (_) { parsed = null; }
        if (parsed && parsed.error) throw new Error(parsed.error);
        throw new Error(`API ${route} returned HTTP ${res.status}: ${text}`);
    }
    const contentType = res.headers.get("Content-Type") || "";
    if (!contentType.includes("application/json")) {
        const txt = await res.text();
        throw new Error(`API ${route} returned non-JSON response: ${txt}`);
    }
    return res.json();
}

export async function getJSON(route) {
    const res = await api.fetchApi(route, { method: "GET" });
    if (!res.ok) {
        const text = await res.text();
        let parsed = null;
        try { parsed = JSON.parse(text); } catch (_) { parsed = null; }
        if (parsed && parsed.error) throw new Error(parsed.error);
        throw new Error(`API ${route} returned HTTP ${res.status}: ${text}`);
    }
    const contentType = res.headers.get("Content-Type") || "";
    if (!contentType.includes("application/json")) {
        const txt = await res.text();
        throw new Error(`API ${route} returned non-JSON response: ${txt}`);
    }
    return res.json();
}

// ---------------------------------------------------------------------------
// Shared mutable state
// ---------------------------------------------------------------------------
// Last saved prompt data per Smart Save node (filled on execute).
export const lastSaveData = new Map();
// Auto-revision loop counters: Critic node id -> number of retries done.
export const loopCounters = new Map();
